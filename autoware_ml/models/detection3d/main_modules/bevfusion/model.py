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

"""Native BEVFusion lidar detector (multi-task interface).

The multi-task wrapper around the AWML-parity-verified BEVFusion components
(``models/detection3d/bevfusion.py`` holds the legacy single-task wrapper until
Q5). The numerics live in the reused submodules and ``TransFusionHead``; this
module only adapts between the framework's typed containers and the head's dict
API — the head itself is untouched to preserve the verified parity.

Coordinate contract: ``voxels_data.coords`` arrives in ``(z, y, x)`` order (set
``voxelization_z_order_first: true`` on the PointPillarPreprocessor) — the same
layout the deployment runtime uses — and the batch column is prepended here.

Deployment (stage graph) lives in :mod:`.stages`; the checkpoint is the converted
native ``best_epoch_25_autoware_ml_native.pth`` (see the BEVFusion parity notes).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Any, Mapping

from jaxtyping import Float32
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
)
from autoware_ml.dataclasses.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.deployment.stages import Stage
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import multi_task_eval_output
from autoware_ml.models.detection3d.feature_extractors import LidarBEVFeatureExtractor
from autoware_ml.models.detection3d.main_modules.bevfusion.quantization import (
    build_bevfusion_quantization_plan,
)
from autoware_ml.models.detection3d.main_modules.bevfusion.stages import (
    build_bevfusion_lidar_stages,
    decode_packed_detections,
)
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan
from autoware_ml.models.multi_task_base_model import LogDictConfigs, MultiTaskBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

_HEAD_DICT_KEYS = (
    "center",
    "height",
    "dim",
    "rot",
    "vel",
    "heatmap",
    "dense_heatmap",
    "query_heatmap_score",
    "query_labels",
)


def head_dict_to_outputs(outputs: Mapping[str, torch.Tensor]) -> TransFusionHeadOutputs:
    """Wrap the TransFusion head's output dict into the typed container."""
    return TransFusionHeadOutputs(**{key: outputs.get(key) for key in _HEAD_DICT_KEYS})


def _as_predictions(samples: Sequence[Mapping[str, torch.Tensor]]) -> MultiTaskPredictions:
    """Wrap the head's per-sample detection dicts into the typed predictions container."""
    return MultiTaskPredictions(
        detection3d_predictions=[
            Detection3DSamplePredictions(
                bboxes_3d=sample["bboxes_3d"],
                scores_3d=sample["scores_3d"],
                labels_3d=sample["labels_3d"],
            )
            for sample in samples
        ]
    )


def outputs_to_head_dict(outputs: TransFusionHeadOutputs) -> dict[str, torch.Tensor]:
    """Unwrap the typed container back into the dict the head's loss/predict expect."""
    head_dict = {key: getattr(outputs, key) for key in _HEAD_DICT_KEYS}
    if head_dict["vel"] is None:
        del head_dict["vel"]
    return head_dict


class BEVFusionLidarDetectionModel(MultiTaskBaseModel):
    """Compose a lidar-only BEVFusion detector from the parity-verified modules."""

    verification_caveat = (
        "the dense graph packs its top-500 proposals, and the zero-padded heatmap "
        "borders produce mass score ties, so backends legitimately select different "
        "near-zero-score proposals — positional raw-output comparison is meaningless "
        "(measured 2026-09-01: high-score proposals align to 0.038 while the "
        "positional bbox_pred diff is 159)."
    )

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        pts_voxel_encoder: torch.nn.Module,
        pts_middle_encoder: torch.nn.Module,
        pts_backbone: torch.nn.Module,
        pts_neck: torch.nn.Module,
        bbox_head: torch.nn.Module,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize the lidar-only BEVFusion detector.

        Args:
            data_preprocessor: Multi-task data preprocessor (hard voxelization with
                ``voxelization_z_order_first: true``).
            pts_voxel_encoder: Lidar voxel feature encoder (HardSimpleVFE).
            pts_middle_encoder: Sparse (spconv) middle encoder producing the BEV map.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: TransFusion detection head (dict API, kept untouched).
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            scheduler_config: Lightning scheduler metadata.
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
        # The exact VFE+spconv composition the legacy wrapper (and the runtime) uses.
        self.lidar_feature_extractor = LidarBEVFeatureExtractor(
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_backbone=None,
            pts_neck=None,
        )

    def forward(self, multi_task_batch_inputs: MultiTaskBatchInputs) -> MultiTaskOutputs:
        """Run the detector on voxelized lidar inputs."""
        voxels_data = multi_task_batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError(
                "MultiTaskBatchInputs must contain voxels_data for BEVFusion forward pass."
            )
        batch_size = multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size()
        # coords arrive as (z, y, x); the sparse encoder consumes (batch, z, y, x).
        voxel_coords = torch.cat((voxels_data.batch_indices.view(-1, 1), voxels_data.coords), dim=1)
        bev_features = self.lidar_feature_extractor(
            voxels_data.voxels, voxels_data.num_points, voxel_coords, batch_size=batch_size
        )
        bev_features = self.pts_neck(self.pts_backbone(bev_features))
        head_dict = self.bbox_head(bev_features)
        return MultiTaskOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None,
                transfusion_head_outputs=head_dict_to_outputs(head_dict),
            )
        )

    def compute_metrics(
        self, multi_task_batch_inputs: MultiTaskBatchInputs, multi_task_outputs: MultiTaskOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " 1"]]:
        """Compute BEVFusion training losses through the head's per-sample-list API."""
        gt_batch = multi_task_batch_inputs.multi_task_gt_batch.detection3d_gt_batch
        if gt_batch is None:
            raise ValueError(
                "MultiTaskBatchInputs must contain detection3d_gt_batch for BEVFusion."
            )
        head_outputs = self._transfusion_outputs(multi_task_outputs)
        # The head's loss takes per-sample lists; slice the padded batch by valid counts.
        gt_boxes = [
            boxes[:count] for boxes, count in zip(gt_batch.gt_bboxes_3d, gt_batch.gt_valid_bboxes)
        ]
        gt_labels = [
            labels[:count].long()
            for labels, count in zip(gt_batch.gt_labels_3d, gt_batch.gt_valid_bboxes)
        ]
        return MappingProxyType(
            dict(self.bbox_head.loss(outputs_to_head_dict(head_outputs), gt_boxes, gt_labels))
        )

    def decode_outputs(self, outputs: MultiTaskOutputs) -> MultiTaskPredictions:
        """Decode predictions through the head's dict API into the typed container."""
        head_outputs = self._transfusion_outputs(outputs)
        return _as_predictions(self.bbox_head.predict(outputs_to_head_dict(head_outputs)))

    def build_eval_output_from_predictions(
        self, batch: MultiTaskBatchInputs, predictions: MultiTaskPredictions
    ) -> dict[str, Any]:
        """Pair decoded detections with ground truth for the metric suites."""
        return multi_task_eval_output(
            multi_task_predictions=predictions, multi_task_batch_inputs=batch
        )

    @staticmethod
    def _transfusion_outputs(outputs: MultiTaskOutputs) -> TransFusionHeadOutputs:
        head_outputs = outputs.detection3d_head_outputs
        if head_outputs is None or head_outputs.transfusion_head_outputs is None:
            raise ValueError(
                "MultiTaskOutputs must contain transfusion_head_outputs for BEVFusion."
            )
        return head_outputs.transfusion_head_outputs

    # ------------------------------------------------------------------ deployment hooks

    def build_quantization_plan(self, quantization_config: QuantizationConfig) -> QuantizationPlan:
        """Bind BEVFusion's quantization rules to a parsed config (see :mod:`.quantization`)."""
        return build_bevfusion_quantization_plan(quantization_config)

    def build_stages(self) -> tuple[Stage, ...]:
        """Declare the BEVFusion lidar split stage graph (see :mod:`.stages`)."""
        return build_bevfusion_lidar_stages(self)

    def assemble_predictions(self, outputs: Mapping[str, torch.Tensor]) -> MultiTaskPredictions:
        """Decode the deployed graph's packed tensors the way the runtime does.

        The dense graph performs the proposal selection itself, so a backend returns
        detections rather than head outputs — there is no :meth:`assemble_outputs` step
        for this model. The unpacking is BEVFusion's runtime ABI; the post-processing
        that follows is the head's own :meth:`decode_detections`, so the deployed
        behaviour cannot drift from the model's.
        """
        return _as_predictions(decode_packed_detections(self.bbox_head, outputs))
