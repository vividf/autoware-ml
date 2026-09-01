"""
Modules to save raw outputs from a detection3d head.
"""

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict

import torch


class TransFusionHeadOutputs(BaseModel):
    """Raw TransFusion head outputs — the head's output dict as a typed container.

    Field names and shapes mirror ``TransFusionHead.forward``'s dict exactly
    (``num_predictions`` = decoder layers x num_proposals); the model adapter
    converts between the two representations losslessly so the head's dict API
    (``loss`` / ``predict``) stays untouched.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    center: Float32[torch.Tensor, "batch_size 2 num_predictions"]
    height: Float32[torch.Tensor, "batch_size 1 num_predictions"]
    dim: Float32[torch.Tensor, "batch_size 3 num_predictions"]
    rot: Float32[torch.Tensor, "batch_size 2 num_predictions"]
    vel: Float32[torch.Tensor, "batch_size 2 num_predictions"] | None
    heatmap: Float32[torch.Tensor, "batch_size num_classes num_predictions"]
    dense_heatmap: Float32[torch.Tensor, "batch_size num_classes height width"]
    query_heatmap_score: Float32[torch.Tensor, "batch_size num_classes num_proposals"]
    query_labels: Int64[torch.Tensor, "batch_size num_proposals"]


class TransFusionPackedDetections(BaseModel):
    """BEVFusion's packed runtime detections (the deployed dense graph's ABI).

    Single-sample tensors as the runtime consumes them: ``bbox_pred`` stacks the
    raw regression channels (center 2, height 1, dim 3, rot 2, vel 2), ``score``
    is the fused per-proposal confidence, and ``label_pred`` the winning class.
    ``label_pred`` arrives as float because the pipeline normalizes every backend
    output to float32; decode casts it back to long.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    bbox_pred: Float32[torch.Tensor, "code_size num_proposals"]
    score: Float32[torch.Tensor, " num_proposals"]
    label_pred: Float32[torch.Tensor, " num_proposals"]


class CenterHeadOutputs(BaseModel):
    """
    Dataclass to save CenterHead-based outputs from a 3D detection model.

    Attributes:
      heatmaps: Heatmap to save probability for each class in a BEV heatmap.
      centers: Center_x and center_y translation from each cell in a BEV heatmap.
      heights: Height value from each cell in a BEV heatmap.
      dims: Dimension values (length, width, height) from each cell in a BEV heatmap.
      rots: Rotation values (sin, cos) from each cell in a BEV heatmap
      vels: Velocity values (vel_x, vel_y) from each cell in a BEV heatmap.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)
    heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"]
    centers: Float32[torch.Tensor, "batch_size 2 height width"]
    heights: Float32[torch.Tensor, "batch_size 1 height width"]
    dims: Float32[torch.Tensor, "batch_size 3 height width"]
    rots: Float32[torch.Tensor, "batch_size 2 height width"]
    vels: Float32[torch.Tensor, "batch_size 2 height width"] | None


class Detection3DHeadOutputs(BaseModel):
    """
    Dataclass to save outputs from 3D detection models.

    Attributes:
      center_head_outputs: Outputs from a CenterHead-based 3D detection model.
      transfusion_head_outputs: Outputs from a TransFusion-based 3D detection model.
      transfusion_packed_detections: BEVFusion's packed runtime detections, produced
        when outputs are reassembled from a deployed backend instead of the head.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    center_head_outputs: CenterHeadOutputs | None
    transfusion_head_outputs: TransFusionHeadOutputs | None
    transfusion_packed_detections: TransFusionPackedDetections | None = None
