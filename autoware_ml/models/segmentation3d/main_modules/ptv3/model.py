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

"""Native PTv3 3D semantic segmentation model (multi-task interface).

A thin wrapper over the parity-verified PTv3 modules: the encoder and the decoder head
are untouched, and this module only adapts between the framework's typed containers and
their point-dict API.

Input contract: ``MultiTaskBatchInputs.points_data`` — a grid-quantized point batch. The
quantization itself is upstream (``GridSample`` transform, which also drops the matching
targets) and ``PointGridPreprocessor`` only assembles the batch, so this model never
re-quantizes.

Two levels matter and must not be confused: the loss is computed on the model's input
points, while metrics score the *original* cloud, reached by scattering predictions
through ``Segmentation3DGTBatch.inverse``.

Deployment (stage graph) lives in :mod:`.stages`; quantization in :mod:`.quantization`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Any, Mapping

from jaxtyping import Float32
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.dataclasses.segmentation3d.predictions import Segmentation3DPredictions
from autoware_ml.deployment.stages import Stage
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.models.multi_task_base_model import LogDictConfigs, MultiTaskBaseModel
from autoware_ml.models.segmentation3d.main_modules.ptv3.quantization import (
    build_ptv3_quantization_plan,
)
from autoware_ml.models.segmentation3d.main_modules.ptv3.stages import build_ptv3_seg_stages
from autoware_ml.models.segmentation3d.main_modules.ptv3.export_modules import (
    split_block_parameters,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan
from autoware_ml.utils.point_cloud.structures import bit_length_tensor


class PTv3SegmentationModel(MultiTaskBaseModel):
    """PTv3 point-wise semantic segmentation on the multi-task interface."""

    #: Ordered ONNX outputs of the segmentation head graph. The graph carries the argmax
    #: and the softmax, so the deployed outputs are already predictions.
    EXPORT_OUTPUT_NAMES = ("pred_labels", "pred_probs")

    #: Serialization orders the encoder interleaves across its attention blocks. Fixed
    #: rather than configurable because the trained weights are tied to this pairing.
    EXPORT_ORDER = ("z", "z-trans")

    verification_caveat = (
        "the PyTorch reference interleaves the four trained serialization orders, "
        "shuffled per forward (encoder.shuffle_orders uses torch.randperm, not gated by "
        "training mode), while the exported graph fixes the two of EXPORT_ORDER — the "
        "raw outputs differ by construction."
    )

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        encoder: torch.nn.Module,
        seg3d_head: torch.nn.Module,
        grid_size: float,
        point_cloud_range: Sequence[float],
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize the PTv3 segmentation model.

        Args:
            data_preprocessor: Multi-task preprocessor; must produce ``points_data``.
            encoder: PTv3 encoder (untouched).
            seg3d_head: PTv3 decoder head owning the segmentation losses (untouched).
            grid_size: Cell size the cloud was quantized with; with
                ``point_cloud_range`` it fixes the sparse shape and serialization depth.
            point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            optimizer_group_overrides: Per-group optimizer settings, keyed by the group
                names :meth:`build_optimizer_groups` returns (``default`` / ``block``).
            scheduler_config: Lightning scheduler metadata.
            metrics: Segmentation metrics accumulated during validation and test.
        """
        super().__init__(
            data_preprocessor=data_preprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            metrics=metrics,
            log_dict_configs=log_dict_configs,
        )
        self.encoder = encoder
        self.seg3d_head = seg3d_head
        self.grid_size = grid_size
        self.point_cloud_range = list(point_cloud_range)

    # ------------------------------------------------------------------ geometry facts

    @property
    def sparse_shape(self) -> torch.Tensor:
        """Grid extent implied by the range and cell size."""
        extents = self._axis_extents()
        return torch.round(extents).to(dtype=torch.long)

    @property
    def serialization_depth(self) -> torch.Tensor:
        """Bits the serialization curve needs to address the grid."""
        return bit_length_tensor(torch.max(self._axis_extents()))

    def _axis_extents(self) -> torch.Tensor:
        point_cloud_range = torch.tensor(self.point_cloud_range, dtype=torch.float32)
        return (point_cloud_range[3:] - point_cloud_range[:3]) / self.grid_size

    # ------------------------------------------------------------------ training hooks

    def forward(self, multi_task_batch_inputs: MultiTaskBatchInputs) -> MultiTaskOutputs:
        """Encode the point batch and decode point-wise logits."""
        points_data = multi_task_batch_inputs.points_data
        if points_data is None:
            raise ValueError(
                "MultiTaskBatchInputs must contain points_data for PTv3; add "
                "PointGridPreprocessor to data_preprocessor.preprocessor_modules."
            )
        point = self.encoder(points_data.as_point_dict())
        return MultiTaskOutputs(
            detection3d_head_outputs=None,
            segmentation3d_logits=self.seg3d_head(point),
        )

    def compute_metrics(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, multi_task_outputs: MultiTaskOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " 1"]]:
        """Compute the segmentation losses on the model's input points."""
        segmentation3d_gt_batch = (
            multi_task_batch_inputs.multi_task_gt_batch.segmentation3d_gt_batch
        )
        if segmentation3d_gt_batch is None:
            raise ValueError("MultiTaskBatchInputs must contain segmentation3d_gt_batch for PTv3.")
        logits = self._segmentation_logits(multi_task_outputs)
        return MappingProxyType(
            dict(self.seg3d_head.loss(logits, segmentation3d_gt_batch.gt_semantic_mask))
        )

    def decode_outputs(self, outputs: MultiTaskOutputs) -> MultiTaskPredictions:
        """Turn point-wise logits into labels and probabilities."""
        logits = self._segmentation_logits(outputs)
        return MultiTaskPredictions(
            detection3d_predictions=None,
            segmentation3d_predictions=Segmentation3DPredictions(
                pred_labels=logits.argmax(dim=1),
                pred_probs=logits.softmax(dim=1),
            ),
        )

    def build_eval_output_from_predictions(
        self, batch: MultiTaskBatchInputs, predictions: MultiTaskPredictions
    ) -> dict[str, Any]:
        """Scatter predictions back to the original cloud and pair them with its labels.

        Metrics score the points as they arrived, not the quantized subset the model saw,
        so both the predictions and the targets are taken at the original level.
        """
        segmentation3d_gt_batch = batch.multi_task_gt_batch.segmentation3d_gt_batch
        segmentation3d_predictions = predictions.segmentation3d_predictions
        if segmentation3d_gt_batch is None or segmentation3d_predictions is None:
            raise ValueError(
                "PTv3 metrics need both segmentation3d_gt_batch and segmentation3d_predictions."
            )
        if segmentation3d_gt_batch.inverse is None:
            raise ValueError(
                "PTv3 metrics score the original cloud, so the batch must carry `inverse` "
                "(produced by the GridSample transform)."
            )
        inverse = segmentation3d_gt_batch.inverse.to(segmentation3d_predictions.pred_labels.device)
        return {
            "seg_pred_labels": segmentation3d_predictions.pred_labels[inverse],
            "seg_target_labels": segmentation3d_gt_batch.origin_semantic_mask,
            "seg_coord": segmentation3d_gt_batch.origin_coord,
        }

    @staticmethod
    def _segmentation_logits(outputs: MultiTaskOutputs) -> torch.Tensor:
        logits = outputs.segmentation3d_logits
        if logits is None:
            raise ValueError("MultiTaskOutputs must contain segmentation3d_logits for PTv3.")
        return logits

    # ------------------------------------------------------------------ deployment hooks

    def get_export_output_names(self) -> list[str]:
        """Ordered ONNX output names of the head graph (the stage builder's contract)."""
        return list(self.EXPORT_OUTPUT_NAMES)

    def _prepare_encoder_export(self) -> torch.nn.Module:
        """An export-ready copy of the encoder, serializing with ``EXPORT_ORDER``."""
        return self.encoder.prepare_for_export(self.EXPORT_ORDER)

    def build_stages(self) -> tuple[Stage, ...]:
        """Declare the PTv3 segmentation stage graph (see :mod:`.stages`)."""
        return build_ptv3_seg_stages(self)

    def assemble_predictions(self, outputs: Mapping[str, torch.Tensor]) -> MultiTaskPredictions:
        """Wrap a backend's point-wise outputs into predictions.

        The exported head emits labels and probabilities directly (the argmax and softmax
        are in the graph), so there is nothing left to decode.
        """
        return MultiTaskPredictions(
            detection3d_predictions=None,
            segmentation3d_predictions=Segmentation3DPredictions(
                pred_labels=outputs["pred_labels"].long(),
                pred_probs=outputs["pred_probs"],
            ),
        )

    def build_quantization_plan(
        self, quantization_config: QuantizationConfig
    ) -> QuantizationPlan:
        """Bind PTv3's quantization rules to a parsed config (see :mod:`.quantization`)."""
        return build_ptv3_quantization_plan(quantization_config)

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Group parameters so attention blocks can take their own optimizer settings."""
        default_params, block_params = split_block_parameters(self)
        return {"default": default_params, "block": block_params}
