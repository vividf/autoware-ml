"""
Modules to save raw outputs from multi-task models.
"""

from jaxtyping import Float32
from pydantic import BaseModel, ConfigDict
import torch

from autoware_ml.dataclasses.detection3d.head_outputs import Detection3DHeadOutputs


class MultiTaskOutputs(BaseModel):
    """
    Dataclass to save raw outputs from multi-task models.

    One slot per task: a model fills the slots its heads produce and leaves the rest
    ``None``.

    Attributes:
        detection3d_head_outputs: Raw outputs from a 3D detection head.
        segmentation3d_logits: Point-wise class logits from a 3D segmentation head,
            shape ``(num_points, num_classes)``, aligned with the model's input points.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    detection3d_head_outputs: Detection3DHeadOutputs | None
    segmentation3d_logits: Float32[torch.Tensor, "num_points num_classes"] | None = None

    # TODO (Kok Seang): Add outputs for other tasks in the future.
