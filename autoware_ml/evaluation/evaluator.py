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

"""The evaluate loop: dataloader -> preprocess -> backend inference -> decode -> metrics.

:func:`evaluate_backend` is the one loop. It is parameterized by a
:class:`~autoware_ml.deployment.pipeline.StagedPipeline` (which already knows its
backend), preprocesses through the model's ``preprocess_batch``, decodes through
the model's ``build_eval_output_from_predictions``, and scores with a fresh clone of its own
metric suites — so a full-split ``pytorch`` run reproduces ``trainer.test`` and a
``tensorrt`` run differs from it only by the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Iterable, Sequence

import torch

from autoware_ml.deployment.pipeline import StagedPipeline
from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.report import check_required_keys, collect_suite_results, latency_key
from autoware_ml.evaluation.latency import LatencyStats
from autoware_ml.types.backend import Backend

logger = logging.getLogger(__name__)

PREPROCESS_STAGE = "preprocess"
POSTPROCESS_STAGE = "decode_and_metrics"
MODEL_STAGE = "model_graphs"


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation outcome of one backend.

    Attributes:
        backend: Backend that ran the exportable stages.
        device: Device string the exportable stages ran on.
        split: Evaluated split (``test``).
        metrics: Canonical-keyed metric report (``{split}/{backend}/{prefix}/{name}``).
        latency: Per-stage latency stats; ``model_graphs`` sums the exportable stages
            only (pure GPU time for TensorRT).
        num_samples: Number of evaluated samples/frames.
        headline_metrics: Metric names the evaluated suites declare as their headline
            ones (``MetricSuite.headline_metrics``) — the rows a report leads with.
        fallback_stages: Graph stages that ran their PyTorch module on this backend
            (declared ``torch_fallback_backends``) — reported, so a backend column is
            never silently the pytorch numbers under another name.
    """

    backend: Backend
    device: str
    split: str
    metrics: dict[str, float]
    latency: dict[str, LatencyStats]
    num_samples: int
    headline_metrics: tuple[str, ...] = ()
    fallback_stages: tuple[str, ...] = ()


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
    """Score one backend against ground truth and collect its latency breakdown.

    Args:
        model: Model exposing ``preprocess_batch``, ``build_eval_output_from_predictions``,
            ``clone_metrics``.
        dataloader: Batches of the split to evaluate (the predict dataloader).
        pipeline: Backend pipeline built from the model's stages (carries its backend).
        device: Preprocessing / ground-truth / metrics device.
        num_samples: Samples to evaluate (-1 = all). Checked at batch granularity.
        num_warmup: Extra re-runs of the first batch priming the GPU / TensorRT; their
            results and timing are discarded, every dataloader batch still counts.
        stage: Metric stage whose suites are cloned and whose name is the ``split`` key.

    Returns:
        The backend's metrics and latency.

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
        batch_size = int(batch.infer_batch_size())

        times: dict[str, float] = {}
        start = time.perf_counter()
        batch_inputs = model.preprocess_batch(batch, device)
        _sync(device)
        times[PREPROCESS_STAGE] = (time.perf_counter() - start) * 1000.0

        if index == 0:
            for _ in range(num_warmup):
                pipeline.infer(batch_inputs)

        result = pipeline.infer(batch_inputs)

        start = time.perf_counter()
        predictions = pipeline.assemble(result, device=device)
        eval_out = model.build_eval_output_from_predictions(batch_inputs, predictions)
        if index == 0:
            check_required_keys(suites, eval_out, producer=type(model).__name__)
        for suite in suites:
            suite.update(eval_out)
        _sync(device)
        times[POSTPROCESS_STAGE] = (time.perf_counter() - start) * 1000.0

        evaluated += batch_size
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
        metrics=collect_suite_results(suites, stage, backend=pipeline.backend),
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


def _is_headline(key: str, headline_metrics: Sequence[str]) -> bool:
    return bool(headline_metrics) and key.rsplit("/", 1)[-1].startswith(tuple(headline_metrics))


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
        (k, v) for k, v in result.metrics.items() if _is_headline(k, result.headline_metrics)
    ):
        logger.info("    %-48s %.4f", key, value)


def log_comparison(results: Sequence[EvaluationResult]) -> None:
    """Log a compact cross-backend table of the shared headline metrics and model latency.

    Metric keys carry the backend, so rows are aligned on the key *minus* its backend
    segment.
    """
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
            if _is_headline(k, result.headline_metrics)
        }
        rows = keys if rows is None else rows & keys
    def label(result: EvaluationResult) -> str:
        return result.backend.value + ("*" if result.fallback_stages else "")

    logger.info("=" * 70)
    logger.info("Cross-backend comparison (%s):", ", ".join(label(r) for r in results))
    logger.info("    %-40s" % "metric" + "".join(f"{label(r):>12}" for r in results))
    for row in sorted(rows or ()):
        split, rest = row.split("/", 1)
        values = "".join(
            f"{by_backend[r.backend].metrics[f'{split}/{r.backend.value}/{rest}']:>12.4f}"
            for r in results
        )
        logger.info(f"    {row:<40}{values}")
    latency_row = "".join(
        f"{(r.latency.get(MODEL_STAGE).mean if r.latency.get(MODEL_STAGE) else 0.0):>12.3f}"
        for r in results
    )
    logger.info(f"    {MODEL_STAGE + ' mean [ms]':<40}{latency_row}")
    for result in results:
        if result.fallback_stages:
            logger.info(
                "    * %s: stage(s) %s ran in PyTorch (declared torch fallback).",
                result.backend.value,
                ", ".join(result.fallback_stages),
            )


def log_results_to_mlflow(client: Any, run_id: str, results: Sequence[EvaluationResult]) -> None:
    """Log every result's metrics (canonical keys) and mean latencies to one MLflow run."""
    for result in results:
        for key, value in result.metrics.items():
            client.log_metric(run_id, key, value)
        for name, stats in result.latency.items():
            client.log_metric(run_id, latency_key(result.backend, f"{name}_mean_ms"), stats.mean)
