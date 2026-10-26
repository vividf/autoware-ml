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

"""Ground-truth evaluation of one backend's pipeline.

Preprocessing, ``build_deployed_eval_output`` and the metric suites are the model's own, so a
full-split ``pytorch`` run reproduces ``trainer.test`` and the backends differ only in the
pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from autoware_ml.deployment.pipeline import StagedPipeline
from autoware_ml.deployment.evaluation.latency import LatencyStats
from autoware_ml.deployment.evaluation.report import is_headline, latency_key, metric_key
from autoware_ml.metrics.reporting import check_required_keys, collect_suite_results
from autoware_ml.metrics.base import EvalStage
from autoware_ml.types.backend import Backend

logger = logging.getLogger(__name__)

PREPROCESS_STAGE = "preprocess"
POSTPROCESS_STAGE = "decode_and_metrics"
MODEL_STAGE = "model_graphs"
RESULTS_FILE_NAME = "evaluation.json"


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation outcome of one backend.

    Attributes:
        backend: Backend that ran the graph stages.
        device: Device the graph stages ran on.
        split: ``test`` or ``val``.
        metrics: Metrics keyed ``{split}/{backend}/{prefix}/{name}``.
        latency: Latency per stage; ``model_graphs`` sums the graph stages only.
        num_samples: Number of evaluated samples.
        headline_metrics: The suites' ``headline_metrics``, the rows a report leads with.
        fallback_stages: Graph stages that ran their PyTorch module on this backend, which
            the reports flag.
    """

    backend: Backend
    device: str
    split: str
    metrics: dict[str, float]
    latency: dict[str, LatencyStats]
    num_samples: int
    headline_metrics: tuple[str, ...] = ()
    fallback_stages: tuple[str, ...] = ()


def _batch_size(model: Any, batch: Any) -> int:
    size = model.get_log_batch_size(batch)
    return int(size) if size else 1


def evaluate_backend(
    model: Any,
    dataloader: Iterable[Any],
    pipeline: StagedPipeline,
    device: torch.device,
    *,
    num_samples: int = -1,
    num_warmup: int = 2,
    stage: EvalStage = EvalStage.TEST,
) -> EvaluationResult:
    """Score one backend against ground truth and time each of its stages.

    Args:
        model: Model exposing ``on_after_batch_transfer``, ``build_deployed_eval_output``,
            ``clone_metrics`` and ``get_log_batch_size`` (:class:`ModuleBaseModel`).
        dataloader: Batches of the split, with ground truth.
        pipeline: The backend's pipeline over the model's stages.
        device: Device of preprocessing and metrics.
        num_samples: Samples to evaluate, -1 for all; checked per batch.
        num_warmup: Discarded runs of the first batch before its scored run.
        stage: Stage whose metric suites are cloned; its name is the split in the keys.

    Raises:
        ValueError: When zero samples were processed.
    """
    device = torch.device(device)
    suites = [suite.to(device) for suite in model.clone_metrics(stage)]
    for suite in suites:
        suite.reset()

    stage_samples: dict[str, list[float]] = {}
    evaluated = 0
    for index, batch in enumerate(dataloader):
        if num_samples >= 0 and evaluated >= num_samples:
            break

        times: dict[str, float] = {}
        start = time.perf_counter()
        batch = model.on_after_batch_transfer(batch.to_device(device), dataloader_idx=0)
        _sync(device)
        times[PREPROCESS_STAGE] = (time.perf_counter() - start) * 1000.0

        if index == 0:
            for _ in range(num_warmup):
                pipeline.infer(batch)

        result = pipeline.infer(batch)

        start = time.perf_counter()
        outputs = pipeline.assemble(result, device=device)
        eval_out = model.build_deployed_eval_output(batch, outputs)
        if index == 0:
            check_required_keys(
                suites, eval_out, producer=f"{type(model).__name__}.build_deployed_eval_output"
            )
        for suite in suites:
            suite.update(eval_out)
        _sync(device)
        times[POSTPROCESS_STAGE] = (time.perf_counter() - start) * 1000.0

        evaluated += _batch_size(model, batch)
        for name, value in result.stage_times_ms.items():
            stage_samples.setdefault(name, []).append(value)
        stage_samples.setdefault(MODEL_STAGE, []).append(result.model_ms)
        stage_samples.setdefault(PREPROCESS_STAGE, []).append(times[PREPROCESS_STAGE])
        stage_samples.setdefault(POSTPROCESS_STAGE, []).append(times[POSTPROCESS_STAGE])

    if evaluated == 0:
        raise ValueError(
            "Evaluation processed zero samples — check num_samples and the dataloader."
        )

    result = EvaluationResult(
        backend=pipeline.backend,
        device=str(pipeline.device),
        split=stage.value,
        metrics=collect_suite_results(
            suites,
            stage,
            key_of=lambda prefix, name: metric_key(stage.value, pipeline.backend, prefix, name),
        ),
        latency={
            name: LatencyStats.from_samples(samples) for name, samples in stage_samples.items()
        },
        num_samples=evaluated,
        headline_metrics=tuple(
            dict.fromkeys(name for suite in suites for name in suite.headline_metrics)
        ),
        fallback_stages=tuple(getattr(pipeline, "fallback_stage_names", ())),
    )
    log_backend_report(result)
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def log_backend_report(result: EvaluationResult) -> None:
    """Log one backend's latency table and headline metrics."""
    logger.info(
        "Backend '%s' (%s) evaluated on %d sample(s).",
        result.backend.value,
        result.device,
        result.num_samples,
    )
    if result.fallback_stages:
        logger.warning(
            "  NOTE: stage(s) %s ran their PyTorch module on this backend "
            "(declared torch fallback) — this column is not pure %s.",
            ", ".join(result.fallback_stages),
            result.backend.value,
        )
    logger.info("  Latency [ms]:")
    for name, stats in sorted(result.latency.items()):
        logger.info(
            "    %-28s mean=%8.3f std=%7.3f min=%8.3f max=%8.3f median=%8.3f",
            name,
            stats.mean,
            stats.std,
            stats.min,
            stats.max,
            stats.median,
        )
    logger.info("  Headline metrics:")
    for key, value in sorted(
        (k, v) for k, v in result.metrics.items() if is_headline(k, result.headline_metrics)
    ):
        logger.info("    %-48s %.4f", key, value)


def log_comparison(results: Sequence[EvaluationResult]) -> None:
    """Log the shared headline metrics and the model latency of the backends side by side."""
    if len(results) < 2:
        return
    by_backend = {result.backend: result for result in results}

    def strip_backend(key: str, backend: Backend) -> str:
        split, rest = key.split("/", 1)
        return f"{split}/{rest.removeprefix(backend.value + '/')}"

    rows = None
    for result in results:
        keys = {
            strip_backend(k, result.backend)
            for k in result.metrics
            if is_headline(k, result.headline_metrics)
        }
        rows = keys if rows is None else rows & keys

    def label(result: EvaluationResult) -> str:
        return result.backend.value + ("*" if result.fallback_stages else "")

    logger.info("=" * 70)
    logger.info("Cross-backend comparison (%s):", ", ".join(label(r) for r in results))
    logger.info("    %-40s%s", "metric", "".join(f"{label(r):>12}" for r in results))
    for row in sorted(rows or ()):
        split, rest = row.split("/", 1)
        values = "".join(
            f"{by_backend[r.backend].metrics[f'{split}/{r.backend.value}/{rest}']:>12.4f}"
            for r in results
        )
        logger.info("    %-40s%s", row, values)
    latency_row = "".join(
        f"{(r.latency[MODEL_STAGE].mean if MODEL_STAGE in r.latency else 0.0):>12.3f}"
        for r in results
    )
    logger.info("    %-40s%s", MODEL_STAGE + " mean [ms]", latency_row)
    for result in results:
        if result.fallback_stages:
            logger.info(
                "    * %s: stage(s) %s ran in PyTorch (declared torch fallback).",
                result.backend.value,
                ", ".join(result.fallback_stages),
            )


def flatten_results(results: Sequence[EvaluationResult]) -> dict[str, float]:
    """Return the metrics and mean latencies of every result in one mapping."""
    flat: dict[str, float] = {}
    for result in results:
        flat.update(result.metrics)
        for name, stats in result.latency.items():
            flat[latency_key(result.backend, f"{name}_mean_ms")] = stats.mean
    return flat


def write_results_json(results: Sequence[EvaluationResult], path: str | Path) -> Path:
    """Write every backend's metrics and latency stats to ``path`` as JSON.

    The log shows the headline metrics only and a tracker may not be configured, so this file
    keeps the full report. Non-finite values (a class without support) are written as
    ``null`` to keep the file strict JSON.
    """

    def finite(value: float) -> float | None:
        value = float(value)
        return value if math.isfinite(value) else None

    entries = [
        {
            "backend": result.backend.value,
            "device": result.device,
            "split": result.split,
            "num_samples": result.num_samples,
            "fallback_stages": list(result.fallback_stages),
            "metrics": {key: finite(value) for key, value in sorted(result.metrics.items())},
            "latency_ms": {
                name: {field: finite(value) for field, value in asdict(stats).items()}
                for name, stats in result.latency.items()
            },
        }
        for result in results
    ]
    path = Path(path)
    path.write_text(json.dumps({"results": entries}, indent=2) + "\n")
    return path
