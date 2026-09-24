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

"""Post-export steps over a model's stage graph: verification and per-backend evaluation.

Both need the same things — the exported artifacts, pipelines per backend, batches from
the datamodule preprocessed the way the model's ``on_after_batch_transfer`` does — so
they are wired here rather than in ``scripts/deploy.py``, which only decides whether to
call in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lightning as L
import torch

from autoware_ml.deployment.config import (
    EvaluationConfig,
    PostExportConfig,
    VerificationConfig,
)
from autoware_ml.deployment.pipeline import PipelineCache, available_backends
from autoware_ml.deployment.verification.backend_verifier import BackendVerifier
from autoware_ml.evaluation.evaluator import (
    EvaluationResult,
    evaluate_backend,
    flatten_results,
    log_comparison,
)
from autoware_ml.metrics.base import EvalStage
from autoware_ml.types.backend import Backend
from autoware_ml.utils.deploy import move_to_device

logger = logging.getLogger(__name__)


def preprocessed_batches(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
    *,
    split: str,
    limit: int,
) -> list[Any]:
    """Load up to ``limit`` batches of ``split`` and preprocess them like Lightning would.

    ``limit < 0`` loads the whole split. The batch goes through
    ``model.on_after_batch_transfer`` — the one place preprocessing runs — so a stage
    graph sees exactly what ``forward()`` sees.
    """
    datamodule.setup({"test": "test", "val": "validate", "predict": "predict"}[split])
    dataloader = getattr(datamodule, f"{split}_dataloader")()
    batches: list[Any] = []
    for index, batch in enumerate(dataloader):
        if 0 <= limit <= index:
            break
        batch = move_to_device(batch, device)
        batches.append(model.on_after_batch_transfer(batch, dataloader_idx=0))
    return batches


def run_verification(
    cfg: VerificationConfig,
    model: L.LightningModule,
    pipelines: PipelineCache,
    available: set[Backend],
    datamodule: L.LightningDataModule,
    device: torch.device,
) -> None:
    """Compare the backends' raw graph outputs on the first predict batches; raise on failure.

    A model may declare ``verification_caveat`` (a sentence) when its raw outputs are
    incomparable across backends by construction; verification is then skipped loudly
    and per-backend evaluation is the meaningful gate.
    """
    caveat = getattr(model, "verification_caveat", None)
    if caveat:
        logger.warning(
            "Verification SKIPPED: %s declares its raw graph outputs incomparable across "
            "backends — %s Per-backend metrics (deploy.evaluation) are the meaningful gate.",
            type(model).__name__,
            caveat,
        )
        return

    batches = preprocessed_batches(
        datamodule, model, device, split="predict", limit=cfg.num_verify_batches
    )
    if not batches:
        raise ValueError("Verification produced zero batches from the predict dataloader.")

    verifier = BackendVerifier(pipelines, tolerance=cfg.tolerance)
    if not verifier.run(batches, cfg.scenarios, available):
        raise RuntimeError(
            "Backend verification FAILED — exported artifacts diverge from the reference "
            "beyond tolerance. See the scenario logs above."
        )
    logger.info("Backend verification passed.")


def run_evaluation(
    cfg: EvaluationConfig,
    model: L.LightningModule,
    pipelines: PipelineCache,
    available: set[Backend],
    datamodule: L.LightningDataModule,
    device: torch.device,
) -> list[EvaluationResult]:
    """Score every enabled backend with artifacts against ground truth; log the comparison.

    The ``test`` split scores the test dataloader (with ground truth); ``val`` scores the
    validation dataloader and reports under ``val/...``.
    """
    if cfg.split == "val":
        datamodule.setup("validate")
        make_dataloader, stage = datamodule.val_dataloader, EvalStage.VAL
        logger.info("Evaluation split: val (metric keys report under val/...).")
    else:
        datamodule.setup("test")
        make_dataloader, stage = datamodule.test_dataloader, EvalStage.TEST

    results: list[EvaluationResult] = []
    with _limited_cpu_threads(cfg.cpu_threads):
        results = _evaluate_enabled_backends(
            cfg, model, pipelines, available, make_dataloader, stage, device
        )
    log_comparison(results)
    return results


@contextmanager
def _limited_cpu_threads(count: int) -> Iterator[None]:
    """Cap the process's CPU worker pools while backends are timed.

    See ``EvaluationConfig.cpu_threads``: busy-waiting BLAS / OpenMP workers left behind by
    a CPU op inflate the measured time of engines with host-side shape syncs. Covers
    PyTorch's intra-op pool and every pool threadpoolctl finds (numpy's OpenBLAS above
    all); ``count <= 0`` changes nothing.
    """
    if count <= 0:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    logger.info(
        "Evaluation caps CPU worker threads at %d (torch intra-op was %d) so host threads "
        "do not distort the graph latency.",
        count,
        previous,
    )
    try:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # pragma: no cover - threadpoolctl ships with the environment
            logger.warning("threadpoolctl unavailable: BLAS worker pools stay at their defaults.")
            yield
        else:
            with threadpool_limits(limits=count):
                yield
    finally:
        torch.set_num_threads(previous)


def _evaluate_enabled_backends(
    cfg: EvaluationConfig,
    model: L.LightningModule,
    pipelines: PipelineCache,
    available: set[Backend],
    make_dataloader: Callable[[], Any],
    stage: EvalStage,
    device: torch.device,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for backend, backend_cfg in cfg.enabled_backends():
        if backend not in available:
            logger.warning(
                "Skipping evaluation of backend '%s': artifacts not available in this run.",
                backend.value,
            )
            continue
        logger.info("=" * 70)
        logger.info(
            "Evaluating backend '%s' on %s (num_samples=%d, num_warmup=%d)",
            backend.value,
            backend_cfg.device,
            cfg.num_samples,
            cfg.num_warmup,
        )
        results.append(
            evaluate_backend(
                model,
                make_dataloader(),
                pipelines.get(backend, backend_cfg.device),
                device,
                num_samples=cfg.num_samples,
                num_warmup=cfg.num_warmup,
                stage=stage,
            )
        )
    return results


def run_post_export(
    deploy_cfg: Any,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    output_dir: str | Path,
    device: torch.device,
    log_metric: Callable[[str, float], None] | None = None,
    onnx_paths: Mapping[str, Path] | None = None,
) -> list[EvaluationResult]:
    """Run the enabled post-export steps of ``deploy_cfg`` over the artifacts in ``output_dir``.

    ``log_metric(key, value)`` receives every evaluation metric and mean latency (the
    deploy run's MLflow client, say).

    Nothing runs when both ``deploy.verification`` and ``deploy.evaluation`` are disabled
    (the default). Enabling either on a model without ``build_stages()`` is an error: the
    steps execute the stage graph, there is nothing else that could run the artifacts
    together with the model's glue.
    """
    cfg = PostExportConfig.from_deploy_cfg(deploy_cfg)
    if not cfg.any_enabled:
        return []
    stages = model.build_stages()
    if stages is None:
        raise ValueError(
            "deploy.verification / deploy.evaluation need a stage graph, but "
            f"{type(model).__name__} does not implement build_stages(). Declare the stage "
            "graph (see autoware_ml.deployment.stages) or disable these sections."
        )
    output_dir = Path(output_dir)
    tensorrt_cfg = (deploy_cfg or {}).get("tensorrt") or {}
    plugin_libraries = tuple(str(p) for p in (tensorrt_cfg.get("plugin_libraries") or ()))
    # ``onnx_paths``: the graphs the export loop shipped (a config modifier may write under
    # another name than <stage>.onnx); the engines are always <stage>.engine.
    pipelines = PipelineCache(
        stages, output_dir, plugin_libraries=plugin_libraries, onnx_paths=onnx_paths
    )
    available = available_backends(stages, output_dir, onnx_paths)
    if cfg.verification.enabled:
        run_verification(cfg.verification, model, pipelines, available, datamodule, device)
    results: list[EvaluationResult] = []
    if cfg.evaluation.enabled:
        results = run_evaluation(cfg.evaluation, model, pipelines, available, datamodule, device)
        if log_metric is not None:
            for key, value in flatten_results(results).items():
                log_metric(key, value)
    return results
