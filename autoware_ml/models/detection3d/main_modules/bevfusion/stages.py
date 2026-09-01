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

"""BEVFusion (lidar-only) deployment stage graph — declared ahead of the migration.

Split form (the INT8 deployment line, mirroring AWML's
``bevfusion_split_int8_deployment`` artifacts):

    fetch_voxels (torch) -> bevfusion_sparse (graph) -> bevfusion_dense (graph)

Runtime ABI carried over from the AWML split artifacts:

- ``bevfusion_sparse``: ``voxels`` / ``coors`` / ``num_points_per_voxel`` in,
  ``lidar_bev`` out. The production INT8 ONNX is the libspconv format
  (``GetIndicePairsImplicitGemm`` / ``ImplicitGemmInt8`` custom ops with per-layer
  ``*_channel_scale`` / ``*_bias_scaled`` inputs) produced by a dedicated exporter —
  NOT by ``torch.onnx.export`` — hence ``external_onnx=True``. ONNX Runtime cannot
  execute spconv, hence ``torch_fallback_backends`` includes ``onnx``: the onnx
  backend of verification/evaluation runs this stage in PyTorch. ``tensorrt`` is a
  fallback too until the libspconv exporter lands (the TODO below) — the tensorrt
  backend then measures torch-sparse + TRT-dense, which is also what the dense
  INT8/FP16 milestones need to compare.
- ``bevfusion_dense``: ``lidar_bev`` in; ``bbox_pred`` / ``score`` / ``label_pred``
  out — the AWML dense graph DECODES in-graph (unlike CenterPoint's raw-map ABI),
  so the wrapper ends at the head's export decode.

Contract with the interface migration:

- **Submodules**: ``pts_voxel_encoder``, ``pts_middle_encoder`` (spconv),
  ``pts_backbone``, ``pts_neck``, ``bbox_head``.
- **Batch inputs**: ``MultiTaskBatchInputs.voxels_data`` provides voxel features,
  coordinates and per-voxel point counts (the ``_first_sample_voxel_inputs``
  tensors of the legacy export).
- **Backend evaluation decode**: ``assemble_bevfusion_outputs`` must rebuild
  per-sample predictions from the packed runtime outputs (the runtime-side decode
  of ``bbox_pred``'s raw channels) — lands with the evaluate milestone; until then
  verification (raw-output comparison) and trainer.test (pytorch forward) cover
  correctness.

.. todo:: TODO(vividf): port the libspconv INT8 sparse exporter (AWML
   ``projects/BEVFusion/deploy``) as the producer of ``bevfusion_sparse.onnx``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from autoware_ml.deployment.stages import GraphStage, Stage, StageContext, TorchStage
from autoware_ml.models.detection3d.feature_extractors import LidarBEVFeatureExtractor
from autoware_ml.types.backend import Backend

# Stage / artifact names (AWML split-deployment ABI: <name>.onnx / .engine).
FETCH_VOXELS_STAGE = "fetch_voxels"
SPARSE_STAGE = "bevfusion_sparse"
DENSE_STAGE = "bevfusion_dense"

# Context tensor names — the ONNX input/output names (AWML runtime ABI).
VOXELS = "voxels"
COORS = "coors"
NUM_POINTS_PER_VOXEL = "num_points_per_voxel"
LIDAR_BEV = "lidar_bev"
BBOX_PRED = "bbox_pred"
SCORE = "score"
LABEL_PRED = "label_pred"

# ONNX output name -> typed-outputs field (draft until the outputs dataclass lands).
OUTPUT_FIELDS: tuple[tuple[str, str], ...] = (
    (BBOX_PRED, "bbox_pred"),
    (SCORE, "score"),
    (LABEL_PRED, "label_pred"),
)


class BEVFusionSparseExportWrapper(nn.Module):
    """Voxel inputs -> dense lidar BEV features (VFE + spconv middle encoder).

    Single-sample graph in the runtime layout: ``coors`` is ``(z, y, x)`` without a
    batch column (the runtime voxelizes with spconv's Point2Voxel), so a zero batch
    column is prepended — the same adaptation the legacy ``_forward_export`` does.
    """

    def __init__(self, voxel_encoder: nn.Module, middle_encoder: nn.Module) -> None:
        super().__init__()
        self.extractor = LidarBEVFeatureExtractor(
            pts_voxel_encoder=voxel_encoder,
            pts_middle_encoder=middle_encoder,
            pts_backbone=None,
            pts_neck=None,
        )

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
    ) -> torch.Tensor:
        batch_column = torch.zeros((coors.shape[0], 1), dtype=coors.dtype, device=coors.device)
        voxel_coords = torch.cat((batch_column, coors), dim=1)
        return self.extractor(voxels, num_points_per_voxel, voxel_coords, batch_size=1)


class BEVFusionDenseExportWrapper(nn.Module):
    """Lidar BEV features -> packed runtime detections (backbone + neck + head).

    The output packing mirrors the legacy ``_export_detection_outputs`` (the
    runtime ABI): raw regression channels concatenated into ``bbox_pred`` plus the
    fused ``score`` and ``label_pred`` — the runtime decodes and NMS-filters
    itself, so no metric-space decoding happens in the graph.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        # Export-ready deep copy: decoder attention swapped for the exportable
        # equivalent (torch.onnx cannot trace nn.MultiheadAttention faithfully).
        self.bbox_head = (
            bbox_head.prepare_for_export()
            if hasattr(bbox_head, "prepare_for_export")
            else bbox_head
        )

    def forward(self, lidar_bev: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.bbox_head(self.neck(self.backbone(lidar_bev)))
        num_proposals = self.bbox_head.num_proposals
        query_labels = outputs["query_labels"]
        heatmap = outputs["heatmap"][..., -num_proposals:].sigmoid()
        one_hot = (
            F.one_hot(query_labels, num_classes=self.bbox_head.num_classes)
            .permute(0, 2, 1)
            .to(heatmap.dtype)
        )
        score = (heatmap * outputs["query_heatmap_score"] * one_hot)[0].max(dim=0).values
        if outputs.get("vel") is None:
            raise ValueError("BEVFusion export requires a velocity branch in the detection head.")
        bbox_pred = torch.cat(
            [
                outputs[key][0, :, -num_proposals:]
                for key in ("center", "height", "dim", "rot", "vel")
            ],
            dim=0,
        )
        return bbox_pred, score, query_labels[0]


def build_bevfusion_lidar_stages(model: Any) -> tuple[Stage, ...]:
    """Declare the lidar-only BEVFusion split stage graph over ``model``'s submodules."""

    def fetch_voxels(context: StageContext) -> Mapping[str, torch.Tensor]:
        voxels_data = context.batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError("MultiTaskBatchInputs must contain voxels_data for BEVFusion.")
        # Single-sample export graph: keep the first sample's voxels only.
        first_sample = voxels_data.batch_indices == 0
        return {
            VOXELS: voxels_data.voxels[first_sample],
            COORS: voxels_data.coords[first_sample].int().contiguous(),
            NUM_POINTS_PER_VOXEL: voxels_data.num_points[first_sample].int(),
        }

    return (
        TorchStage(FETCH_VOXELS_STAGE, run=fetch_voxels),
        GraphStage(
            SPARSE_STAGE,
            module=BEVFusionSparseExportWrapper(model.pts_voxel_encoder, model.pts_middle_encoder),
            inputs=(VOXELS, COORS, NUM_POINTS_PER_VOXEL),
            outputs=(LIDAR_BEV,),
            # ONNX Runtime cannot execute spconv; TensorRT falls back too until the
            # libspconv exporter (module TODO) produces a pluggable sparse engine.
            torch_fallback_backends=(Backend.ONNX, Backend.TENSORRT),
            # The INT8 sparse ONNX (libspconv format) comes from a dedicated exporter.
            external_onnx=True,
        ),
        GraphStage(
            DENSE_STAGE,
            module=BEVFusionDenseExportWrapper(model.pts_backbone, model.pts_neck, model.bbox_head),
            inputs=(LIDAR_BEV,),
            outputs=(BBOX_PRED, SCORE, LABEL_PRED),
            output_fields=OUTPUT_FIELDS,
        ),
    )


def decode_packed_detections(
    bbox_head: Any, packed: Any
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode packed runtime detections into (boxes, scores, labels).

    Mirrors the second half of ``TransFusionHead.predict``: the packed tensors
    already carry the fused per-proposal score and winning label, so the class
    scores are re-scattered for the bbox coder and the same score/range filter
    and circle NMS apply.
    """
    bbox_pred = packed.bbox_pred
    scores = packed.score
    labels = packed.label_pred.long()
    num_proposals = scores.shape[0]
    heatmap = bbox_pred.new_zeros((1, bbox_head.num_classes, num_proposals))
    heatmap[0, labels, torch.arange(num_proposals, device=bbox_pred.device)] = scores
    decoded = bbox_head.bbox_coder.decode(
        heatmap,
        bbox_pred[6:8].unsqueeze(0),
        bbox_pred[3:6].unsqueeze(0),
        bbox_pred[0:2].unsqueeze(0),
        bbox_pred[2:3].unsqueeze(0),
        bbox_pred[8:10].unsqueeze(0),
        filter_predictions=True,
    )[0]
    boxes = decoded["bboxes"]
    kept_scores = decoded["scores"]
    kept_labels = decoded["labels"]
    if boxes.numel() and bbox_head.nms_type == "circle":
        kept = bbox_head._apply_circle_nms(boxes, kept_scores, kept_labels)
        boxes, kept_scores, kept_labels = boxes[kept], kept_scores[kept], kept_labels[kept]
    return boxes, kept_scores, kept_labels


def assemble_bevfusion_outputs(outputs: Mapping[str, torch.Tensor]) -> Any:
    """Wrap the packed runtime tensors into typed outputs for backend evaluation.

    The packed tensors are the runtime ABI, not the head's dict — decoding them
    (bbox coder + NMS, mirroring the runtime) happens in
    :meth:`BEVFusionLidarDetectionModel.decode_outputs`.
    """
    from autoware_ml.dataclasses.detection3d.head_outputs import (
        Detection3DHeadOutputs,
        TransFusionPackedDetections,
    )
    from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs

    return MultiTaskOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(
            center_head_outputs=None,
            transfusion_head_outputs=None,
            transfusion_packed_detections=TransFusionPackedDetections(
                bbox_pred=outputs["bbox_pred"],
                score=outputs["score"],
                label_pred=outputs["label_pred"],
            ),
        )
    )
