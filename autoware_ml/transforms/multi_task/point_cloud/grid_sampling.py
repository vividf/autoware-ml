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

"""Grid sampling (voxel-grid subsampling) for serialization-based point models (PTv3)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.datamodule.multi_task.dataclasses.segmentation3d import Segmentation3DGTSample
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.types.geometry import PointFieldIndex


class GridSample(MultiTaskBaseTransform):
    """Keep one representative point per grid cell.

    Serialization-based backbones consume one point per occupied cell, so the cloud is
    quantized before the model sees it. Two properties make this a transform rather than
    a runtime preprocessing step:

    - it *drops* points, and the point-wise targets have to be dropped with them, which
      only the sample level can do consistently;
    - which point survives is a training-time choice (a random member of the cell in
      train mode, the first in eval), so it belongs where the other augmentations live.

    The pre-quantization points and labels are recorded on the segmentation GT together
    with ``inverse``, because metrics score the original cloud, not the quantized one.

    Cells are indexed from a fixed origin derived from ``point_cloud_range``, not from
    the batch's own minimum, so the same point always lands in the same cell — the
    deployed graph recomputes ``grid_coord`` from coordinates alone and has no per-sample
    minimum to reproduce.

    Required keys: ``point_cloud_data``.
    """

    _required_keys = ["point_cloud_data"]

    def __init__(
        self,
        grid_size: float | Sequence[float],
        point_cloud_range: Sequence[float],
        mode: str = "train",
    ) -> None:
        """Initialize the transform.

        Args:
            grid_size: Cell size, scalar or per-axis.
            point_cloud_range: ``(x_min, y_min, z_min, x_max, y_max, z_max)``; its minimum
                corner fixes the cell origin.
            mode: ``"train"`` keeps a random member of each cell, ``"eval"`` the first.

        Raises:
            ValueError: On an unknown mode.
        """
        super().__init__(probability=None)
        if mode not in ("train", "eval"):
            raise ValueError(f"GridSample mode must be 'train' or 'eval', got {mode!r}.")
        self.grid_size = np.broadcast_to(np.asarray(grid_size, dtype=np.float32), (3,)).copy()
        self.origin = np.asarray(point_cloud_range[:3], dtype=np.float32)
        self.mode = mode

    def grid_coord(self, coord: np.ndarray) -> np.ndarray:
        """Cell index of each point, from the fixed origin."""
        return np.floor((coord - self.origin) / self.grid_size).astype(np.int32)

    def _keep_indices(self, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pick one point per occupied cell.

        Returns:
            The kept point indices, and for every input point the position of its cell's
            representative within those kept indices.
        """
        _, cell_of_point, counts = np.unique(cell, axis=0, return_inverse=True, return_counts=True)
        cell_of_point = cell_of_point.reshape(-1)
        order = np.argsort(cell_of_point, kind="stable")
        first_of_cell = np.cumsum(counts) - counts
        if self.mode == "train":
            within_cell = np.floor(np.random.random(counts.size) * counts).astype(np.int64)
        else:
            within_cell = np.zeros(counts.size, dtype=np.int64)
        keep = order[first_of_cell + within_cell]
        # Kept points stay in their original order, so `inverse` indexes that order.
        keep_sorted = np.sort(keep)
        rank_of_cell = np.argsort(np.argsort(keep, kind="stable"), kind="stable")
        return keep_sorted, rank_of_cell[cell_of_point]

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Quantize the sample's cloud, carrying its point-wise targets along."""
        point_cloud_data = multi_task_gt_sample.point_cloud_data
        if point_cloud_data is None or not len(point_cloud_data):
            return multi_task_gt_sample

        coord = (
            point_cloud_data.points[:, PointFieldIndex.X : PointFieldIndex.Z + 1]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        keep, inverse = self._keep_indices(self.grid_coord(coord))

        segmentation = multi_task_gt_sample.segmentation3d_gt_sample
        if segmentation is not None:
            labels = np.asarray(segmentation.gt_semantic_mask).reshape(-1)
            if labels.shape[0] != coord.shape[0]:
                raise ValueError(
                    f"GridSample got {labels.shape[0]} label(s) for {coord.shape[0]} point(s); "
                    "the point-wise targets are no longer aligned with the cloud."
                )
            segmentation = Segmentation3DGTSample(
                gt_semantic_mask=labels[keep].astype(np.int32, copy=False),
                origin_semantic_mask=labels.astype(np.int32, copy=False),
                origin_coord=coord,
                inverse=inverse.astype(np.int64, copy=False),
            )

        kept_mask = torch.zeros(coord.shape[0], dtype=torch.bool)
        kept_mask[torch.from_numpy(keep)] = True
        point_cloud_data.remove_points(kept_mask)

        return MultiTaskGTSample(
            lidar_point_cloud_samples=multi_task_gt_sample.lidar_point_cloud_samples,
            point_cloud_data=point_cloud_data,
            detection3d_gt_bboxes_3d=multi_task_gt_sample.detection3d_gt_bboxes_3d,
            segmentation3d_gt_sample=segmentation,
        )
