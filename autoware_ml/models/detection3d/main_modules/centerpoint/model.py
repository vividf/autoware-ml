# Copyright 2023 OpenMMLab.
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

"""Native CenterPoint lidar detector.

The task-level training / inference wrapper around the reusable PointPillars and
CenterPoint detection components. Deployment (stage graph, export wrappers) lives in
:mod:`.stages`, the quantization declaration in :mod:`.quantization`; this module
only wires the three hooks the framework calls.
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
from autoware_ml.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.dataclasses.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.deployment.stages import Stage
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import multi_task_eval_output
from autoware_ml.models.multi_task_base_model import LogDictConfigs, MultiTaskBaseModel
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import PillarFeatureNet
from autoware_ml.models.detection3d.encoders.pillars.point_pillar_scatter import PointPillarsScatter
from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.models.detection3d.main_modules.centerpoint.quantization import (
    build_centerpoint_quantization_plan,
)
from autoware_ml.models.detection3d.main_modules.centerpoint.stages import (
    assemble_centerpoint_outputs,
    build_centerpoint_stages,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan


class CenterPointDetectionModel(MultiTaskBaseModel):
    """Compose a CenterPoint detector from reusable lidar detection modules.

    The wrapper wires together pillar encoding, BEV feature extraction, and the
    CenterPoint dense head inside the shared :class:`BaseModel` interface.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        # TODO(KokSeang): Encoder and middle_encoder should be standardized to a common interface for all voxel encoders
        pts_voxel_encoder: PillarFeatureNet,
        pts_middle_encoder: PointPillarsScatter,
        pts_backbone: torch.nn.Module,
        pts_neck: torch.nn.Module,
        # TODO(KokSeang): bbox_head should be standardized to a common interface for all detection heads
        bbox_head: CenterHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """
        Initialize CenterPoint.

        Args:
            data_preprocessor: Multi-task data preprocessor.
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Sparse 3D or pillar-scatter middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: CenterPoint dense detection head.
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            scheduler_config: Lightning scheduler metadata (``interval`` / ``frequency`` /
                ``monitor``); per-step schedulers such as OneCycleLR need ``interval: step``.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(
            data_preprocessor=data_preprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_config=scheduler_config,
            metrics=metrics,
            log_dict_configs=log_dict_configs,
        )
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.bbox_head = bbox_head

    def build_eval_output_from_predictions(
        self, batch: MultiTaskBatchInputs, predictions: MultiTaskPredictions
    ) -> dict[str, Any]:
        """Pair decoded detections with ground truth for the metric suites."""
        return multi_task_eval_output(
            multi_task_predictions=predictions, multi_task_batch_inputs=batch
        )

    def forward(self, multi_task_batch_inputs: MultiTaskBatchInputs) -> MultiTaskOutputs:
        """Run the detector on voxelized lidar inputs.

        Args:
            multi_task_batch_inputs: MultiTaskBatchInputs containing the voxelized lidar inputs.

        Returns:
            Detection head outputs.
        """
        if multi_task_batch_inputs.voxels_data is None:
            raise ValueError(
                "MultiTaskBatchInputs must contain voxels_data for CenterPoint forward pass."
            )

        batch_size = multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size()
        pillar_features = self.pts_voxel_encoder(multi_task_batch_inputs.voxels_data)
        bev_features = self.pts_middle_encoder(
            pillar_features=pillar_features,
            coords=multi_task_batch_inputs.voxels_data.coords,
            batch_indices=multi_task_batch_inputs.voxels_data.batch_indices,
            batch_size=batch_size,
        )
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        head_outputs = self.bbox_head(bev_features)
        return MultiTaskOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=head_outputs, transfusion_head_outputs=None
            )
        )

    def compute_metrics(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, multi_task_outputs: MultiTaskOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " 1"]]:
        """Compute CenterPoint training losses."""
        if multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch is None:
            raise ValueError(
                "MultiTaskBatchInputs must contain detection3d_gt_batch for CenterPoint compute_metrics pass."
            )

        if multi_task_outputs.detection3d_head_outputs is None:
            raise ValueError(
                "MultiTaskOutputs must contain detection3d_head_outputs for CenterPoint compute_metrics pass."
            )

        gt_bboxes_3d = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch.gt_bboxes_3d
        gt_labels_3d = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch.gt_labels_3d
        gt_valid_bboxes = (
            multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch.gt_valid_bboxes
        )

        return self.bbox_head.loss(
            outputs=multi_task_outputs.detection3d_head_outputs,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
        )

    def decode_outputs(self, outputs: MultiTaskOutputs) -> MultiTaskPredictions:
        """Decode predictions for inference."""
        if outputs.detection3d_head_outputs is None:
            raise ValueError(
                "MultiTaskOutputs must contain detection3d_head_outputs for CenterPoint decode_outputs pass."
            )

        multi_task_predictions = self.bbox_head.decode_outputs(
            outputs=outputs.detection3d_head_outputs
        )
        return multi_task_predictions

    # ------------------------------------------------------------------ deployment hooks

    def build_stages(self) -> tuple[Stage, ...]:
        """Declare the CenterPoint inference stage graph (see :mod:`.stages`)."""
        return build_centerpoint_stages(self)

    def assemble_outputs(self, outputs: Mapping[str, torch.Tensor]) -> MultiTaskOutputs:
        """Wrap the head's named output tensors into :class:`MultiTaskOutputs`."""
        return assemble_centerpoint_outputs(outputs)

    def build_quantization_plan(self, quantization_config: QuantizationConfig) -> QuantizationPlan:
        """Bind CenterPoint's quantization rules to a parsed config (see :mod:`.quantization`)."""
        return build_centerpoint_quantization_plan(quantization_config)
