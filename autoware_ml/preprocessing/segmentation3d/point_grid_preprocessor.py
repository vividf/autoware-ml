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

"""Point-batch preprocessing for serialization-based models (PTv3)."""

from __future__ import annotations

from typing import Sequence

import torch

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.dataclasses.points_data import PointsData
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule
from autoware_ml.types.geometry import PointFieldIndex


class PointGridPreprocessor(DataPreprocessorModule):
    """Assemble the point batch a serialization-based backbone consumes.

    Three things happen here, all of them deterministic — the point *selection* already
    happened in the ``GridSample`` transform, because dropping points means dropping
    their targets too:

    - ``grid_coord``: the cell index of each point, from an origin fixed by
      ``point_cloud_range``. Using a fixed origin rather than the batch's own minimum is
      what lets the deployed graph recompute the same value from coordinates alone.
    - ``offset``: where each sample ends in the concatenated batch, which is how these
      models derive per-sample attention windows.
    - ``feat``: the feature columns the model was trained on, selected by index so a
      cloud carrying extra channels cannot silently change the input contract.

    Args:
        grid_size: Cell size, scalar or per-axis, matching the training transform.
        point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]``; its minimum
            corner fixes the cell origin.
        feature_indices: Point columns forming ``feat``, in order. Defaults to
            ``(x, y, z, intensity)``.
        intensity_scale: Multiplier applied to the intensity column of ``feat`` (ignored
            when that column is not selected). T4 stores intensity in ``[0, 255]`` while
            the segmentation checkpoints were trained on the normalized value, so the
            default rescales it to ``[0, 1]``.
    """

    def __init__(
        self,
        grid_size: float | Sequence[float],
        point_cloud_range: Sequence[float],
        feature_indices: Sequence[int] = (
            PointFieldIndex.X,
            PointFieldIndex.Y,
            PointFieldIndex.Z,
            PointFieldIndex.INTENSITY,
        ),
        intensity_scale: float = 1.0 / 255.0,
    ) -> None:
        super().__init__()
        self.grid_size = torch.as_tensor(grid_size, dtype=torch.float32).reshape(-1)
        if self.grid_size.numel() == 1:
            self.grid_size = self.grid_size.repeat(3)
        if self.grid_size.numel() != 3:
            raise ValueError(f"grid_size must be scalar or 3 values, got {grid_size!r}.")
        self.origin = torch.as_tensor(point_cloud_range[:3], dtype=torch.float32)
        self.feature_indices = tuple(int(index) for index in feature_indices)
        self.intensity_scale = float(intensity_scale)

    def __call__(
        self,
        multi_task_batch_inputs: MultiTaskBatchInputs,
        is_training: bool,
    ) -> MultiTaskBatchInputs:
        """Build :class:`PointsData` from the batch's point cloud.

        Args:
            multi_task_batch_inputs: Batch holding the collated point cloud.
            is_training: Unused; the quantization here is deterministic, and the
                train/eval difference lives in the ``GridSample`` transform.

        Returns:
            The batch with ``points_data`` filled in.

        Raises:
            ValueError: If the batch carries no point cloud, or too few point columns.
        """
        del is_training

        point_cloud_gt_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        if point_cloud_gt_batch is None:
            raise ValueError(
                "MultiTaskGTBatch must contain point cloud data for point-grid preprocessing."
            )

        points = point_cloud_gt_batch.points
        required_columns = max(self.feature_indices) + 1
        if points.shape[-1] < required_columns:
            raise ValueError(
                f"Point cloud has {points.shape[-1]} column(s) but feature_indices needs "
                f"{required_columns}."
            )

        device = points.device
        coord = points[:, PointFieldIndex.X : PointFieldIndex.Z + 1].contiguous().float()
        grid_coord = torch.floor(
            (coord - self.origin.to(device)) / self.grid_size.to(device)
        ).to(torch.int32)
        feat = points[:, self.feature_indices].contiguous().float()
        if self.intensity_scale != 1.0 and PointFieldIndex.INTENSITY in self.feature_indices:
            column = self.feature_indices.index(PointFieldIndex.INTENSITY)
            feat[:, column] *= self.intensity_scale

        batch_indices = point_cloud_gt_batch.batch_indices.to(torch.int64)
        batch_size = int(multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size())
        counts = torch.bincount(batch_indices, minlength=batch_size)
        offset = torch.cumsum(counts, dim=0).to(device)

        return multi_task_batch_inputs.model_copy(
            update={
                "points_data": PointsData(
                    coord=coord, feat=feat, grid_coord=grid_coord, offset=offset
                )
            }
        )
