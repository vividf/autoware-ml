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

"""Base model classes for Autoware-ML.

This module defines shared Lightning model interfaces and helper abstractions
used by task-specific model wrappers throughout the framework.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, final
from types import MappingProxyType

from jaxtyping import Float32
import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTBatch
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.eval_mixin import MetricEvalMixin
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.types.dataset import SplitType
from autoware_ml.utils.optimizer import build_lightning_optimizer_config


class LogDictConfigs(NamedTuple):
    """Configuration for logging metrics in Lightning.

    Attributes:
        prog_bar: Whether to log to the progress bar.
        logger: Whether to log to the logger.
        on_step: Whether to log on each step.
        on_epoch: Whether to log on each epoch.
        reduce_fx: Reduction function to apply to the logged metric.
        sync_dist: Whether to synchronize across distributed processes.
        batch_size: Optional batch size to use for logging. If None, the batch size will be inferred.
        rank_zero_only: Whether to log only on rank zero in a distributed environment.
    """

    prog_bar: bool
    logger: bool | None = None
    on_step: bool | None = None
    on_epoch: bool | None = None
    reduce_fx: str = "mean"
    sync_dist: bool = False
    batch_size: int | None = None
    rank_zero_only: bool = False


class MultiTaskBaseModel(MetricEvalMixin, L.LightningModule):
    """Base Lightning Module for all Autoware-ML models.

    Provides common functionality for training, validation, and testing with
    built-in support for flexible optimizer and scheduler configuration.
    All parameters are explicitly typed for IDE support and type checking.
    """

    #: Why this model's raw graph outputs cannot be compared across backends, or
    #: ``None`` when they can. Cross-backend verification (deploy.verification) compares
    #: the final raw tensors element-wise; a model for which that comparison is invalid
    #: *by construction* — a PyTorch reference that is stochastic at inference, decoded
    #: outputs whose proposal ties reorder — declares the reason here, and the gate
    #: skips loudly instead of the reason living in a config comment. Per-backend
    #: ground-truth metrics (deploy.evaluation) are the gate that remains meaningful.
    verification_caveat: str | None = None

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ):
        """Initialize base model.

        Args:
            data_preprocessor: Data preprocessor to preprocess input batches.
            log_dict_configs: Configuration for logging metrics in Lightning.
            optimizer: Callable that returns an optimizer when given model parameters.
            scheduler: Callable that returns a scheduler when given the optimizer.
            log_dict_configs: Configuration for logging metrics in Lightning.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
            metrics: Task metrics accumulated during validation and test. Empty
                or ``None`` logs only losses.
        """
        super().__init__(metrics=metrics)
        self._data_preprocessor = data_preprocessor
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.optimizer_group_overrides = (
            dict(optimizer_group_overrides) if optimizer_group_overrides else None
        )
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {}
        self.log_dict_configs = log_dict_configs

    def on_after_batch_transfer(
        self, batch: MultiTaskGTBatch, dataloader_idx: int
    ) -> MultiTaskBatchInputs:
        """Apply runtime preprocessing after Lightning moves a batch to device.

        Args:
            batch: Collated batch of type :class:`MultiTaskGTBatch` on the target device.
            dataloader_idx: Lightning dataloader index.

        Returns:
            Batch of type :class:`MultiTaskBatchInputs` after runtime preprocessing.
        """
        return self._data_preprocessor(batch, is_training=self.training)

    def decode_outputs(self, outputs: MultiTaskOutputs) -> MultiTaskPredictions:
        """Convert raw model outputs into task-level predictions.

        Task wrappers must override this when prediction-time outputs differ from
        training-time outputs, for example to convert logits into probabilities
        and labels.

        Args:
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            Task-level predictions.
        """
        raise NotImplementedError("Model must implement decode_outputs()")

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Return structural optimizer groups for the model.

        Models that do not need custom grouping use a single ``default`` group.
        Models with optimizer-group-specific tuning can override this hook.

        Returns:
            Mapping from optimizer group names to parameter sequences.
        """
        return {
            "default": [parameter for parameter in self.parameters() if parameter.requires_grad]
        }

    def forward(self, multi_task_batch_inputs: MultiTaskBatchInputs) -> MultiTaskOutputs:
        """Forward pass of the model.

        Subclasses must follow the signature of this method and return a dataclass containing
        all model outputs. Users must add new implementations to MultiTaskBatchInputs and MultiTaskOutputs
        when supporting new types of tasks to the multi-task model.
        The default implementation raises a NotImplementedError.

        Args:
            multi_task_batch_inputs: Batch of type :class:`MultiTaskBatchInputs` containing
                multi-task inputs.

        Returns:
            Model outputs.
        """
        raise NotImplementedError("Model must implement forward()")

    def compute_metrics(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, multi_task_outputs: MultiTaskOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " 1"]]:
        """Compute metrics.

        Args:
            multi_task_batch_inputs: Full batch dictionary after runtime preprocessing.
            multi_task_outputs: Model outputs from forward().

        Returns:
            MappingProxyType[str, Float32[torch.Tensor, " 1"]]: Dictionary of metrics.
            The ``loss`` field is required. Note that users must register metrics in
            the dictionary for proper logging and checkpointing.
        """
        raise NotImplementedError("Model must implement compute_metrics()")

    def get_log_batch_size(self, multi_task_batch_inputs: MultiTaskBatchInputs) -> int | None:
        """Infer the effective sample batch size for logging.

        It searches for all available inputs in ``MultiTaskFeatures`` to infer the sample count.
        Models with non-standard input structures should override this hook to provide an
        explicit sample count.

        Args:
            multi_task_batch_inputs: Full batch dictionary from the dataloader.

        Returns:
            Sample batch size when it can be inferred, otherwise ``None``.
        """
        return multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size()

    def _core_step(
        self,
        multi_task_batch_inputs: MultiTaskBatchInputs,
        step_prefix: str,
    ) -> tuple[MappingProxyType[str, Float32[torch.Tensor, " 1"]], MultiTaskOutputs]:
        """
        Core step shared by training, validation, and test. It runs one forward pass,
        computes metrics, and logs them.

        Args:
            multi_task_batch_inputs: Full batch dictionary from the dataloader.
            step_prefix: Prefix for logging (train, val, test).

        Returns:
            Tuple of the metric dictionary (as a ``MappingProxyType``) and the raw model outputs.
            The metric dictionary contains at least a ``"loss"`` key.
        """
        outputs = self(multi_task_batch_inputs)
        metrics = self.compute_metrics(multi_task_batch_inputs, outputs)
        if "loss" not in metrics:
            raise ValueError("compute_metrics() must return a dict containing a 'loss' key.")
        batch_size = self.get_log_batch_size(multi_task_batch_inputs)
        # Step-level logging is training-only. Enabling it for val/test would make Lightning
        # suffix the keys (``val/loss_step``/``val/loss_epoch``) and break callbacks that
        # monitor ``val/loss``.
        on_step = self.log_dict_configs.on_step if step_prefix == SplitType.TRAIN else False
        logged_values: dict[str, Any] = {f"{step_prefix}/{k}": v for k, v in metrics.items()}
        self.log_dict(
            logged_values,
            prog_bar=self.log_dict_configs.prog_bar,
            logger=self.log_dict_configs.logger,
            on_step=on_step,
            on_epoch=self.log_dict_configs.on_epoch,
            reduce_fx=self.log_dict_configs.reduce_fx,
            sync_dist=self.log_dict_configs.sync_dist,
            batch_size=batch_size,
            rank_zero_only=self.log_dict_configs.rank_zero_only,
        )
        return metrics, outputs

    @final
    def training_step(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, batch_idx: int
    ) -> Float32[torch.Tensor, " 1"]:
        """
        Training step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Current batch index.

        Returns:
            Total loss tensor required for backpropagation.
        """
        metrics, _ = self._core_step(
            multi_task_batch_inputs,
            step_prefix=SplitType.TRAIN.value,
        )
        return metrics["loss"]

    @final
    def validation_step(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, batch_idx: int
    ) -> dict[str, Any]:
        """
        Validation step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Current batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs. The raw outputs are available
            to ``on_validation_batch_end`` for epoch-level metric accumulation.
        """
        metrics, outputs = self._core_step(
            multi_task_batch_inputs,
            step_prefix=SplitType.VAL.value,
        )
        # TODO (KokSeang): Return a dataclass instead of a dict for better type safety and clarity.
        return {**metrics, "model_outputs": outputs}

    @final
    def test_step(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, batch_idx: int
    ) -> dict[str, Any]:
        """Test step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs.
        """
        metrics, outputs = self._core_step(
            multi_task_batch_inputs,
            step_prefix=SplitType.TEST.value,
        )
        # TODO (KokSeang): Return a dataclass instead of a dict for better type safety and clarity.
        return {**metrics, "model_outputs": outputs}

    @final
    def predict_step(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, batch_idx: int
    ) -> MultiTaskPredictions:
        """Prediction step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Batch index.

        Returns:
            Predictions after decoding the raw model outputs.
        """
        outputs = self(multi_task_batch_inputs)
        return self.decode_outputs(outputs)

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        """Configure optimizers and schedulers.

        Scheduler behavior such as ``interval``, ``frequency``, and ``monitor``
        is configured explicitly through ``scheduler_config``. The framework
        only auto-fills ``total_steps`` when the configured scheduler declares
        that argument and it was not already bound in the scheduler factory.

        Returns:
            Optimizer instance or Lightning optimizer configuration dictionary.
        """
        if self.optimizer_partial is None:
            raise ValueError("Optimizer must be provided.")
        return build_lightning_optimizer_config(
            self,
            self.optimizer_partial,
            self.scheduler_partial,
            optimizer_group_overrides=self.optimizer_group_overrides,
            scheduler_config=self.scheduler_config,
            estimated_stepping_batches=self.trainer.estimated_stepping_batches
            if self._trainer is not None
            else None,
        )

    def preprocess_batch(
        self, batch: MultiTaskGTBatch, device: torch.device
    ) -> MultiTaskBatchInputs:
        """Move a collated batch to ``device`` and apply runtime preprocessing.

        The single spelling of "device transfer + :meth:`on_after_batch_transfer`" used
        outside the Lightning loop: deployment export/verification/evaluation and
        quantization calibration all preprocess through here, so every consumer feeds
        the model exactly the tensors the trainer would.

        Args:
            batch: Collated batch from a dataloader.
            device: Device the preprocessing (e.g. voxelization) runs on.

        Returns:
            Preprocessed model inputs.
        """
        return self.on_after_batch_transfer(batch.to_device(device), 0)

    def build_stages(self) -> Sequence[Any]:
        """Declare the model's inference stage graph for deployment.

        The stage graph is the *one* declaration deployment derives everything from:
        ONNX export units, artifact names, backend inference, verification, and
        evaluation (see :mod:`autoware_ml.deployment.stages`). Exportable stages
        (:class:`~autoware_ml.deployment.stages.GraphStage`) become one ONNX/TensorRT
        artifact each; the rest (:class:`~autoware_ml.deployment.stages.TorchStage`)
        always run in PyTorch on every backend. :meth:`forward` stays hand-written; a
        parity test pins it to the PyTorch run of these stages.

        Models that support deployment override this.

        Returns:
            Ordered stages, ending with the ``GraphStage`` whose outputs feed
            :meth:`assemble_outputs`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support deployment: build_stages() is not implemented."
        )

    def assemble_outputs(self, outputs: Mapping[str, torch.Tensor]) -> MultiTaskOutputs:
        """Wrap the final stage's named output tensors into the model's typed outputs.

        For a deployed graph that emits the head's raw output maps: the backend returns
        the final ``GraphStage``'s tensors keyed by dataclass *field* name (the stage
        declares the ONNX-name -> field mapping), and this hook rebuilds the
        :class:`MultiTaskOutputs` the model's own decode consumes. A graph that performs
        the decoding itself never produces head outputs — such a model overrides
        :meth:`assemble_predictions` instead and leaves this hook alone.

        Args:
            outputs: Field name -> tensor, as declared by the final stage.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support deployment: assemble_outputs() is not implemented."
        )

    def assemble_predictions(self, outputs: Mapping[str, torch.Tensor]) -> MultiTaskPredictions:
        """Turn a deployed graph's raw output tensors into task-level predictions.

        The one hook deployment needs, because predictions are what evaluation consumes.
        The default covers graphs that emit the head's raw maps by composing the model's
        own two steps, so those models implement nothing extra. A graph whose runtime ABI
        decodes in-graph overrides this and skips :meth:`assemble_outputs` entirely —
        there is no intermediate head output to fabricate.

        Args:
            outputs: Field name -> tensor, as declared by the final stage.

        Returns:
            Task-level predictions, as :meth:`decode_outputs` would produce.
        """
        return self.decode_outputs(self.assemble_outputs(outputs))

    def build_eval_output_from_predictions(
        self, batch: MultiTaskBatchInputs, predictions: MultiTaskPredictions
    ) -> dict[str, Any]:
        """Pair decoded predictions with ground truth for the metric suites.

        Both evaluation paths converge here: training and validation reach it through
        :meth:`build_eval_output` after decoding the model's outputs, and deployment
        reaches it directly with the predictions a backend produced. Keeping the pairing
        in one place is what lets the two paths be scored identically.

        Args:
            batch: The preprocessed batch holding the ground truth.
            predictions: Decoded predictions from either path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not report metrics: "
            "build_eval_output_from_predictions() is not implemented."
        )

    def build_eval_output(  # type: ignore[override]
        self, batch: MultiTaskBatchInputs, outputs: MultiTaskOutputs
    ) -> dict[str, Any]:
        """Decode model outputs and pair them with ground truth (training / validation)."""
        return self.build_eval_output_from_predictions(batch, self.decode_outputs(outputs))
