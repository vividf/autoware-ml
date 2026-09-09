"""Metric lifecycle for models.

``MetricEvalMixin`` is mixed into ``BaseModel`` and drives the validation and
test metric lifecycle for a list of :class:`~autoware_ml.metrics.base.MetricSuite`
objects. A model only implements ``build_eval_output``. The mixin resets each
suite at epoch start, calls ``update`` per batch, and ``result`` at epoch end,
logging under the canonical ``{split}/{backend}/{prefix}/{key}`` convention of
:mod:`autoware_ml.metrics.report` with ``backend=pytorch`` — the trainer is just one
more backend, so deployment evaluation of an ONNX/TensorRT export of the same model
lands next to it in MLflow.

Each suite is cloned per stage and registered as a submodule, so Lightning moves
its state to the right device. torchmetrics owns the cross-GPU sync, which runs
inside ``result`` at epoch end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn

from autoware_ml.metrics.base import EvalStage, MetricSuite
from autoware_ml.metrics.report import check_required_keys, collect_suite_results
from autoware_ml.types.backend import Backend


class MetricEvalMixin:
    """Owns the metric suites and the validation/test epoch lifecycle."""

    def __init__(
        self, *args: Any, metrics: Sequence[MetricSuite] | None = None, **kwargs: Any
    ) -> None:
        """Clone the metric suites per stage and register them as submodules.

        Args:
            metrics: Suites attached from config. Empty means only losses are
                logged.
            *args: Positional arguments forwarded to the next base.
            **kwargs: Keyword arguments forwarded to the next base.
        """
        super().__init__(*args, **kwargs)
        prototypes = list(metrics) if metrics else []
        self._metrics_by_stage = nn.ModuleDict(
            {
                EvalStage.VAL.value: nn.ModuleList([metric.clone() for metric in prototypes]),
                EvalStage.TEST.value: nn.ModuleList([metric.clone() for metric in prototypes]),
            }
        )

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Map raw forward outputs and the batch to the flat dict metrics read.

        Override in a model that attaches metrics. The default produces nothing,
        which is correct for a model with no metrics.
        """
        return {}

    def _stage_metrics(self, stage: EvalStage) -> nn.ModuleList:
        return self._metrics_by_stage[stage.value]

    def clone_metrics(self, stage: EvalStage) -> list[MetricSuite]:
        """Return fresh clones of this model's metric suites for one stage.

        Deployment evaluation scores each exported backend with its own clone of
        the model's suites, so per-backend state never cross-contaminates and the
        numbers are computed by exactly the same metric code as ``trainer.test``.
        """
        return [metric.clone() for metric in self._stage_metrics(stage)]

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metric state for a fresh epoch."""
        for metric in self._stage_metrics(EvalStage.VAL):
            metric.reset()

    def on_test_epoch_start(self) -> None:
        """Reset the test metric state for a fresh epoch."""
        for metric in self._stage_metrics(EvalStage.TEST):
            metric.reset()

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one validation batch into every metric."""
        self._update_metrics(EvalStage.VAL, outputs, batch, batch_idx)

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one test batch into every metric."""
        self._update_metrics(EvalStage.TEST, outputs, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        """Combine, compute, and log the validation metrics."""
        self._log_metrics(EvalStage.VAL)

    def on_test_epoch_end(self) -> None:
        """Combine, compute, and log the test metrics."""
        self._log_metrics(EvalStage.TEST)

    def _update_metrics(self, stage: EvalStage, outputs: Any, batch: Any, batch_idx: int) -> None:
        metrics = self._stage_metrics(stage)
        if not len(metrics):
            return
        raw_outputs = (
            outputs["model_outputs"]
            if isinstance(outputs, Mapping) and "model_outputs" in outputs
            else outputs
        )
        eval_out = self.build_eval_output(batch, raw_outputs)
        if batch_idx == 0:
            check_required_keys(metrics, eval_out, producer=type(self).__name__)
        for metric in metrics:
            metric.update(eval_out)

    def _log_metrics(self, stage: EvalStage) -> None:
        report = collect_suite_results(self._stage_metrics(stage), stage, backend=Backend.PYTORCH)
        if not report:
            return
        # Values are already global and identical on every rank after sync, so no sync_dist.
        self.log_dict(report, on_step=False, on_epoch=True, logger=True)
