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

"""PointPillars preprocessing for Detection3D models."""

from __future__ import annotations

from typing import Sequence

import torch


from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule
from autoware_ml.ops.voxelization.voxelization import hard_voxelize, VoxelsData


class PointPillarPreprocessor(DataPreprocessorModule):
    """Convert batched point clouds into padded pillars for PointPillars models.

    The preprocessor voxelizes each point cloud using
    :func:`~autoware_ml.ops.voxelization.hard_voxelize`, pads variable-size
    pillars to ``max_num_points``, and packages the tensors expected by
    PointPillars-style detectors.

    Args:
        voxel_size: Voxel size along each axis ``[dx, dy, dz]`` in meters.
        point_cloud_range: Spatial range ``[x_min, y_min, z_min, x_max, y_max, z_max]``
            in meters.
        max_num_points: Maximum number of points kept per pillar.
        max_voxels: Maximum number of pillars retained per sample.
        voxelization_z_order_first: If ``True``, this preprocessor will transpose [x, y, z]
            coordinates to [z, y, x] in coords from voxelization.
            This is used for backward-compatible, and will be removed very soon.
        default_point_channels: Default number of point channels to be used when no points
            are provided in the batch. Default is 4, which corresponds to (x, y, z, intensity).
    """

    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        max_num_points: int,
        max_voxels: int,
        voxelization_z_order_first: bool = False,
        default_point_channels: int = 4,
    ) -> None:
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.voxelization_z_order_first = voxelization_z_order_first
        self._default_point_channels = default_point_channels

    def __call__(
        self,
        multi_task_batch_inputs: MultiTaskBatchInputs,
        is_training: bool,
    ) -> MultiTaskBatchInputs:
        """
        Process batch data and convert to multi_task_input_features for downstream tasks.

        Args:
            multi_task_batch_inputs (MultiTaskBatchInputs): Batch data containing ground truths and
            input features.
            is_training (bool): Flag indicating whether the model is in training mode.

        Returns:
            MultiTaskBatchInputs: The processed input features for downstream tasks
            generating voxelization with VoxelData.
        """

        multi_task_gt_batch = multi_task_batch_inputs.multi_task_gt_batch
        if multi_task_gt_batch.point_cloud_gt_batch is None:
            raise ValueError("MultiTaskGTBatch must contain point cloud data for voxelization.")

        points = multi_task_gt_batch.point_cloud_gt_batch.points
        if not len(points):
            voxels_data = VoxelsData(
                voxels=torch.zeros(
                    (0, self.max_num_points, self._default_point_channels),
                ),
                num_points=torch.zeros((0,), dtype=torch.int32),
                coords=torch.zeros((0, 3), dtype=torch.int32),
                batch_indices=torch.zeros((0,), dtype=torch.int32),
            )
            # Return early if no points are available, but still return a valid VoxelsData object
            return multi_task_batch_inputs.model_copy(update={"voxels_data": voxels_data})

        device = points.device
        voxel_size = torch.tensor(self.voxel_size, device=device)
        point_cloud_range = torch.tensor(self.point_cloud_range, device=device)
        points_batch_indices = multi_task_gt_batch.point_cloud_gt_batch.batch_indices

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )

        # Handle the case where no voxels are generated
        if not len(voxels_data.voxels):
            voxels_data = VoxelsData(
                voxels=torch.zeros(
                    (0, self.max_num_points, points.shape[1]),
                    device=device,
                ),
                num_points=torch.zeros((0,), device=device, dtype=torch.int32),
                coords=torch.zeros((0, 3), device=device, dtype=torch.int32),
                batch_indices=torch.zeros((0,), device=device, dtype=torch.int32),
            )
            # Return since no voxels are available
            return multi_task_batch_inputs.model_copy(update={"voxels_data": voxels_data})

        # TODO (KokSeang): Remove this backward compatibility code in the future
        if self.voxelization_z_order_first:
            coords = voxels_data.coords[:, [2, 1, 0]].contiguous()
            # Re-create the VoxelsData with the updated coords
            voxels_data = VoxelsData(
                voxels=voxels_data.voxels,
                num_points=voxels_data.num_points,
                coords=coords,
                batch_indices=voxels_data.batch_indices,
            )

        return multi_task_batch_inputs.model_copy(update={"voxels_data": voxels_data})
