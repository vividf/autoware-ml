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

"""Post-export verification and per-backend evaluation over a model's stage graph."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
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
from autoware_ml.deployment.evaluation.evaluator import (
    RESULTS_FILE_NAME,
    EvaluationResult,
    evaluate_backend,
    flatten_results,
    log_comparison,
    write_results_json,
)
from autoware_ml.metrics.base import EvalStage
from autoware_ml.types.backend import Backend
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)


def predict_batches(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
    *,
    limit: int,
) -> list[Any]:
    """Load up to ``limit`` predict batches, all if negative, preprocessed as Lightning does."""
    # A cache hit after the export; a stand-alone call builds the record tables here.
    datamodule.prepare_data()
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()
    batches: list[Any] = []
    for index, batch in enumerate(dataloader):
        if 0 <= limit <= index:
            break
        batches.append(model.on_after_batch_transfer(batch.to_device(device), dataloader_idx=0))
    return batches


def run_verification(
    cfg: VerificationConfig,
    model: L.LightningModule,
    pipelines: PipelineCache,
    available: set[Backend],
    datamodule: L.LightningDataModule,
    device: torch.device,
) -> None:
    """Compare the backends' raw model outputs on the first predict batches; raise on failure."""
    batches = predict_batches(datamodule, model, device, limit=cfg.num_verify_batches)
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
    """Score every enabled backend that has artifacts, then log the comparison."""
    datamodule.prepare_data()
    if cfg.split == "val":
        datamodule.setup("validate")
        split, stage = SplitType.VAL, EvalStage.VAL
        logger.info("Evaluation split: val (metric keys report under val/...).")
    else:
        datamodule.setup("test")
        split, stage = SplitType.TEST, EvalStage.TEST

    def make_dataloader() -> Any:
        return _split_dataloader(datamodule, split, cfg.batch_size)

    results: list[EvaluationResult] = []
    with _limited_cpu_threads(cfg.cpu_threads):
        results = _evaluate_enabled_backends(
            cfg, model, pipelines, available, make_dataloader, stage, device
        )
    log_comparison(results)
    return results


def _split_dataloader(datamodule: Any, split: SplitType, batch_size: int) -> Any:
    attribute = {
        SplitType.VAL: "validation_dataloader_config",
        SplitType.TEST: "test_dataloader_config",
    }[split]
    config = getattr(datamodule, attribute)
    return datamodule.build_dataloader(split, replace(config, batch_size=batch_size))


@contextmanager
def _limited_cpu_threads(count: int) -> Iterator[None]:
    """Cap the PyTorch and BLAS / OpenMP thread pools (see ``EvaluationConfig.cpu_threads``)."""
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
            "Evaluating backend '%s' on %s (num_samples=%d, batch_size=%d, num_warmup=%d)",
            backend.value,
            backend_cfg.device,
            cfg.num_samples,
            cfg.batch_size,
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
    cfg: PostExportConfig,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    output_dir: str | Path,
    device: torch.device,
    log_metric: Callable[[str, float], None] | None = None,
    plugin_libraries: Sequence[str] = (),
) -> list[EvaluationResult]:
    """Run the enabled ``deploy.verification`` and ``deploy.evaluation`` over ``output_dir``.

    Evaluation writes every metric and latency to ``<output_dir>/evaluation.json`` and passes
    each to ``log_metric(key, value)`` when given. ``plugin_libraries`` are the TensorRT
    plugins the engines were built with. Enabling either step on a model without
    ``build_stages()`` is an error.
    """
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
    pipelines = PipelineCache(stages, output_dir, plugin_libraries=plugin_libraries)
    available = available_backends(stages, output_dir)
    if cfg.verification.enabled:
        run_verification(cfg.verification, model, pipelines, available, datamodule, device)
    results: list[EvaluationResult] = []
    if cfg.evaluation.enabled:
        results = run_evaluation(cfg.evaluation, model, pipelines, available, datamodule, device)
        report = write_results_json(results, output_dir / RESULTS_FILE_NAME)
        logger.info("Every metric and latency of the evaluated backends: %s", report)
        if log_metric is not None:
            for key, value in flatten_results(results).items():
                log_metric(key, value)
    return results
