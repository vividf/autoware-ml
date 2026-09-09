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

"""CenterPoint deployment stage graph — declared once, derived everywhere.

The deployed CenterPoint is two exported graphs with PyTorch glue between them
(the split the Autoware runtime expects):

    pillar_decorate (torch)  ->  pts_voxel_encoder (graph)  ->  scatter (torch)  ->  pts_backbone_neck_head (graph)

From this one declaration the framework derives the ONNX export units and their
trace inputs, the artifact names (``pts_voxel_encoder.onnx`` / ``.engine`` ...),
the per-backend inference pipeline, verification, evaluation, and the latency
breakdown. ``CenterPointDetectionModel.forward`` stays hand-written for training;
``tests/deployment/test_centerpoint_stages.py`` pins the two to each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from jaxtyping import Float32
import torch
from torch import nn

from autoware_ml.dataclasses.detection3d.head_outputs import (
    CenterHeadOutputs,
    Detection3DHeadOutputs,
)
from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.deployment.stages import GraphStage, Stage, StageContext, TorchStage
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import PillarFeatureNet
from autoware_ml.models.detection3d.heads.centerhead import CenterHead

if TYPE_CHECKING:
    from autoware_ml.models.detection3d.main_modules.centerpoint.model import (
        CenterPointDetectionModel,
    )

# Stage / artifact names. ``pts_voxel_encoder.onnx`` and ``pts_backbone_neck_head.onnx``
# are what the exporter writes and the Autoware runtime loads.
PILLAR_DECORATE_STAGE = "pillar_decorate"
VOXEL_ENCODER_STAGE = "pts_voxel_encoder"
SCATTER_STAGE = "scatter"
BACKBONE_NECK_HEAD_STAGE = "pts_backbone_neck_head"

# Context tensor names — these are the ONNX input/output names of the two graphs.
INPUT_FEATURES = "input_features"
PILLAR_FEATURES = "pillar_features"
SPATIAL_FEATURES = "spatial_features"

# ONNX output name -> CenterHeadOutputs field, in the order the deployed runtime expects.
# One table drives the export wrapper's return tuple, the graph's output names, and the
# reassembly into typed outputs. Do not reorder.
HEAD_OUTPUT_FIELDS: tuple[tuple[str, str], ...] = (
    ("heatmap", "heatmaps"),
    ("reg", "centers"),
    ("height", "heights"),
    ("dim", "dims"),
    ("rot", "rots"),
)
VELOCITY_FIELD: tuple[str, str] = ("vel", "vels")


class VoxelEncoderExportWrapper(nn.Module):
    """Export the PointPillars PFN stack from decorated input features."""

    def __init__(self, voxel_encoder: PillarFeatureNet) -> None:
        super().__init__()
        self.voxel_encoder = voxel_encoder

    def forward(
        self,
        input_features: Float32[torch.Tensor, "num_pillars max_num_points feature_channels"],
    ) -> Float32[torch.Tensor, "num_pillars 1 num_output_channels"]:
        """Encode already-decorated pillar features."""
        return self.voxel_encoder.encode_decorated(input_features)


class BackboneNeckHeadExportWrapper(nn.Module):
    """Export CenterPoint backbone, neck, and dense head from BEV features."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: CenterHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head.prepare_for_export()
        self.output_fields = head_output_fields(self.bbox_head)

    def forward(
        self,
        spatial_features: Float32[torch.Tensor, "batch_size num_channels height width"],
    ) -> tuple[torch.Tensor, ...]:
        """Run BEV feature extraction and flatten the head outputs for export."""
        bev_features = self.backbone(spatial_features)
        bev_features = self.neck(bev_features)
        center_head_outputs = self.bbox_head(bev_features)

        export_outputs: list[torch.Tensor] = []
        for onnx_name, field_name in self.output_fields:
            output = getattr(center_head_outputs, field_name)
            if output is None:
                raise ValueError(f"CenterHead produced no tensor for export output '{onnx_name}'.")
            export_outputs.append(output)
        return tuple(export_outputs)


def head_output_fields(bbox_head: CenterHead) -> tuple[tuple[str, str], ...]:
    """The head's ``(onnx_name, field)`` table, with velocity when the head predicts it."""
    fields = HEAD_OUTPUT_FIELDS
    if bbox_head.use_velocity:
        fields += (VELOCITY_FIELD,)
    return fields


def _batch_size_from_voxels(batch_indices: torch.Tensor) -> int:
    # The scatter canvas is indexed with the pillar batch indices, so the batch size is
    # derived from the voxels rather than from the ground truth: a deployment batch may
    # carry no ground truth at all.
    return int(batch_indices.max().item()) + 1 if batch_indices.numel() else 1


def build_centerpoint_stages(model: CenterPointDetectionModel) -> tuple[Stage, ...]:
    """Declare CenterPoint's stage graph over ``model``'s modules."""

    def pillar_decorate(context: StageContext) -> Mapping[str, torch.Tensor]:
        voxels_data = context.batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError("MultiTaskBatchInputs must contain voxels_data for CenterPoint.")
        # Pure tensor math, no learned parameters: runs where the data lives.
        return {INPUT_FEATURES: model.pts_voxel_encoder.encode(voxels_data)}

    def scatter(context: StageContext) -> Mapping[str, torch.Tensor]:
        voxels_data = context.batch_inputs.voxels_data
        device = context.device
        pillar_features = context[PILLAR_FEATURES].to(device).squeeze(1)
        spatial_features = model.pts_middle_encoder(
            pillar_features=pillar_features,
            coords=voxels_data.coords.to(device),
            batch_indices=voxels_data.batch_indices.to(device),
            batch_size=_batch_size_from_voxels(voxels_data.batch_indices),
        )
        return {SPATIAL_FEATURES: spatial_features.float()}

    head_wrapper = BackboneNeckHeadExportWrapper(
        model.pts_backbone, model.pts_neck, model.bbox_head
    )
    return (
        TorchStage(PILLAR_DECORATE_STAGE, run=pillar_decorate),
        GraphStage(
            VOXEL_ENCODER_STAGE,
            module=VoxelEncoderExportWrapper(model.pts_voxel_encoder),
            inputs=(INPUT_FEATURES,),
            outputs=(PILLAR_FEATURES,),
        ),
        TorchStage(SCATTER_STAGE, run=scatter),
        GraphStage(
            BACKBONE_NECK_HEAD_STAGE,
            module=head_wrapper,
            inputs=(SPATIAL_FEATURES,),
            outputs=tuple(onnx_name for onnx_name, _ in head_wrapper.output_fields),
            output_fields=head_wrapper.output_fields,
        ),
    )


def assemble_centerpoint_outputs(fields: Mapping[str, torch.Tensor]) -> MultiTaskOutputs:
    """Rebuild :class:`MultiTaskOutputs` from the head's field-named tensors."""
    kwargs: dict[str, torch.Tensor | None] = dict(fields)
    kwargs.setdefault("vels", None)
    return MultiTaskOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(
            center_head_outputs=CenterHeadOutputs(**kwargs),
            transfusion_head_outputs=None,
        )
    )
