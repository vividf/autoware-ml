from __future__ import annotations

from pydantic import BaseModel, ConfigDict, InstanceOf

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTBatch
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class MultiTaskBatchInputs(BaseModel):
    """Data class to represent the gt batch and data features for inputs to a multi-task model."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    # InstanceOf keeps pydantic from recursing into the NamedTuple's fields, whose jaxtyping
    # annotations use symbolic axes (e.g. "batch_size*num_points") that can only be resolved
    # inside a @jaxtyped scope.
    multi_task_gt_batch: InstanceOf[MultiTaskGTBatch]

    voxels_data: VoxelsData | None

    # TODO(Kok Seang): Add input features for 3D segmentation model.

    # TODO(Kok Seang): Add input features for 2D detection/segmentation model.
