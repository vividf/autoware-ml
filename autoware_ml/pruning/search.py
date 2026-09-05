# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastNAS width search over a model's :class:`~autoware_ml.pruning.spec.PruningSpec`.

The search (NVIDIA ModelOpt ``fastnas``) runs on the spec's subtree only; the rest of the
model is untouched. Two facts the code below encodes, both learnt the hard way (M1 probe,
2026-09-07):

1. FastNAS re-calibrates BatchNorm on the provided frames for EVERY candidate, the
   un-pruned one included. A score reference taken before the conversion therefore makes
   the un-pruned subnet itself look degraded and every pruning "free" (FastNAS'
   ``reach_zero`` shortcut zeroes the sensitivity map and the binary search collapses to
   the minimum width). The proxy score takes its reference lazily on the first call.
2. ModelOpt 0.46's ``search`` already exports the best subnet in place; a second
   ``mtn.export`` fails on an empty export stack. Nothing here calls export.
"""

from __future__ import annotations

import logging
import math
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from autoware_ml.deployment.pipeline import PipelineCache, StagedPipeline
from autoware_ml.evaluation.evaluator import evaluate_backend
from autoware_ml.pruning.channels import ChannelTable
from autoware_ml.pruning.config import PruningConfig
from autoware_ml.pruning.spec import PruningSpec
from autoware_ml.types.backend import Backend

logger = logging.getLogger(__name__)

_MAC_LAYERS = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)


@dataclass(frozen=True)
class SearchResult:
    """What the search produced, for logs and the MLflow record."""

    channel_table: ChannelTable
    baseline_table: ChannelTable
    baseline_gmac: float
    pruned_gmac: float
    score_evaluations: int
    seconds: float

    def changed_layers(self) -> dict[str, tuple[dict, dict]]:
        """Layers whose width changed: ``{name: (before, after)}``."""
        return self.channel_table.changed_from(self.baseline_table)


def subtree_macs(subtree: nn.Module, x: torch.Tensor) -> int:
    """MACs of the conv/linear layers for one forward on ``x`` (forward hooks)."""
    total = [0]

    def hook(m, inputs, output):
        if isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features * math.prod(output.shape[:-1])
        else:
            spatial = math.prod(
                inputs[0].shape[2:] if isinstance(m, nn.ConvTranspose2d) else output.shape[2:]
            )
            total[0] += (
                math.prod(m.kernel_size) * m.in_channels * m.out_channels * spatial // m.groups
            )

    handles = [
        m.register_forward_hook(hook) for m in subtree.modules() if isinstance(m, _MAC_LAYERS)
    ]
    with torch.no_grad():
        subtree(x)
    for h in handles:
        h.remove()
    return total[0]


def cache_stage_inputs(
    model, datamodule, stage_input: str, num_frames: int, device
) -> list[torch.Tensor]:
    """Run the pytorch stage graph on the first ``num_frames`` predict batches; keep ``stage_input``."""
    pipeline = StagedPipeline(model.build_stages(), backend=Backend.PYTORCH, device=device)
    cached: list[torch.Tensor] = []
    for i, batch in enumerate(datamodule.predict_dataloader()):
        if i >= num_frames:
            break
        _, context = pipeline.run(model.preprocess_batch(batch, device))
        cached.append(context[stage_input].detach().clone())
    if not cached:
        raise RuntimeError("The predict dataloader yielded no batch to cache for the search.")
    return cached


def _headline(result) -> float:
    name = result.headline_metrics[0]
    key = next(k for k in result.metrics if k.endswith(f"/{name}"))
    return float(result.metrics[key])


def search_channel_table(
    model,
    spec: PruningSpec,
    datamodule,
    config: PruningConfig,
    device: torch.device,
) -> SearchResult:
    """Prune ``spec.subtree`` in place with FastNAS and return its channel table.

    Args:
        model: Full model (FP weights loaded, eval mode). Its ``spec.submodules`` are
            narrowed in place; nothing else changes.
        spec: The model's pruning declaration.
        datamodule: Prepared datamodule (``setup("predict")`` done); the predict split
            feeds the frame cache and the ``map`` score.
        config: Parsed ``pruning`` section.
        device: CUDA device.

    Raises:
        ImportError: When ``torchprofile`` (ModelOpt's FLOPs counter) is missing.
    """
    try:
        import torchprofile  # noqa: F401  (modelopt.nas FLOPs counting; needs 0.0.4's profile.handlers)
    except ImportError as error:
        raise ImportError(
            "pruning needs `torchprofile==0.0.4` (nvidia-modelopt's FLOPs counter); "
            "install it into the environment."
        ) from error
    import modelopt.torch.prune as mtp

    subtree = spec.subtree.eval()
    cached = cache_stage_inputs(model, datamodule, spec.stage_input, config.calib_frames, device)
    x0 = cached[0]
    baseline_table = ChannelTable.record(model, spec.submodules)
    baseline_macs = subtree_macs(subtree, x0)
    logger.info(
        "Pruning search on %s: %d layers, %.1f GMAC, %d cached frames of %s %s — %s",
        list(spec.submodules),
        len(baseline_table),
        baseline_macs / 1e9,
        len(cached),
        spec.stage_input,
        tuple(x0.shape),
        config.describe(),
    )

    ss = mtp.fastnas.FastNASConfig()
    for key in ("nn.Conv1d", "nn.Conv2d", "nn.Conv3d", "nn.ConvTranspose2d"):
        ss[key]["*"]["channels_ratio"] = config.channels_ratio
        ss[key]["*"]["channel_divisor"] = config.channel_divisor
    ss["nn.Linear"]["*"]["features_ratio"] = config.channels_ratio
    ss["nn.Linear"]["*"]["feature_divisor"] = config.channel_divisor

    evaluations = [0]
    teacher: list[tuple[torch.Tensor, ...]] = []

    def score_proxy(m: nn.Module) -> float:
        evaluations[0] += 1
        with torch.no_grad():
            if not teacher:  # reference = the un-pruned subnet, AFTER FastNAS' BN re-calibration
                teacher.extend(tuple(t.detach().clone() for t in m(x)) for x in cached)
                return 0.0
            err = 0.0
            for x, ref in zip(cached, teacher):
                outs = m(x)
                err += sum(
                    ((o.float() - r.float()).norm() / (r.float().norm() + 1e-6)).item()
                    for o, r in zip(outs, ref)
                ) / len(ref)
        return -err / len(cached)

    def score_map(m: nn.Module) -> float:
        evaluations[0] += 1
        del m  # the full model shares the patched modules; score through the deploy loop
        pipe = PipelineCache(
            model.build_stages(), tempfile.mkdtemp(), assemble=model.assemble_predictions
        ).get(Backend.PYTORCH, device)
        result = evaluate_backend(
            model,
            datamodule.predict_dataloader(),
            pipe,
            device,
            num_samples=config.score_samples,
            num_warmup=0,
        )
        return _headline(result)

    score_func: Callable[[nn.Module], float] = score_map if config.score == "map" else score_proxy
    t0 = time.time()
    pruned, _ = mtp.prune(
        model=subtree,
        mode=[("fastnas", ss)],
        constraints={"flops": config.flops},
        dummy_input=x0,
        config={
            "data_loader": cached,  # BN re-calibration of every candidate subnet
            "collect_func": lambda b: b,
            "score_func": score_func,
            "verbose": True,
        },
    )
    assert pruned is subtree, "modelopt prune() must operate in place on the spec subtree"
    seconds = time.time() - t0

    channel_table = ChannelTable.record(model, spec.submodules)
    pruned_macs = subtree_macs(subtree, x0)
    result = SearchResult(
        channel_table=channel_table,
        baseline_table=baseline_table,
        baseline_gmac=baseline_macs / 1e9,
        pruned_gmac=pruned_macs / 1e9,
        score_evaluations=evaluations[0],
        seconds=seconds,
    )
    logger.info(
        "Pruning search done in %.0fs (%d score evaluations): %.1f -> %.1f GMAC (%.0f%%), "
        "%d layers changed",
        seconds,
        evaluations[0],
        result.baseline_gmac,
        result.pruned_gmac,
        100.0 * pruned_macs / max(baseline_macs, 1),
        len(result.changed_layers()),
    )
    for name, (before, after) in result.changed_layers().items():
        logger.info("  %-50s %s -> %s", name, _fmt(before), _fmt(after))
    return result


def _fmt(shape: dict) -> str:
    values = list(shape.values())
    return "->".join(str(v) for v in values) if len(values) > 1 else str(values[0])
