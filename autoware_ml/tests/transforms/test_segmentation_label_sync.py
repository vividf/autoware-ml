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

"""Point-wise labels must follow every transform that drops or reorders points.

A desync here does not raise: training simply optimizes against labels belonging to
other points, which is why each of these transforms is pinned by a test that checks the
labels moved with the cloud rather than merely that the transform ran.
"""

from __future__ import annotations

import numpy as np
import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.datamodule.multi_task.dataclasses.segmentation3d import Segmentation3DGTSample
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.types.geometry import PointFeatureName
from autoware_ml.transforms.multi_task.point_cloud.geometry import (
    PointsRandomShuffle,
    PointsRangeFilter,
)
from autoware_ml.transforms.multi_task.point_cloud.grid_sampling import GridSample


def _sample(coords: list[list[float]], labels: list[int]) -> MultiTaskGTSample:
    points = torch.tensor([[*coord, 0.0] for coord in coords], dtype=torch.float32)
    return MultiTaskGTSample(
        lidar_point_cloud_samples=None,
        point_cloud_data=LiDARPoints(
            points=points,
            point_feature_names=[
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
            timestamp=0.0,
        ),
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=Segmentation3DGTSample(
            gt_semantic_mask=np.array(labels, dtype=np.int32)
        ),
    )


def _labels(sample: MultiTaskGTSample) -> list[int]:
    return np.asarray(sample.segmentation3d_gt_sample.gt_semantic_mask).reshape(-1).tolist()


def test_range_filter_drops_the_labels_of_dropped_points() -> None:
    sample = _sample([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [1.0, 1.0, 0.0]], [7, 8, 9])

    filtered = PointsRangeFilter(points_range=(-10.0, -10.0, -5.0, 10.0, 10.0, 5.0))(sample)

    assert len(filtered.point_cloud_data) == 2
    assert _labels(filtered) == [7, 9]


def test_shuffle_reorders_the_labels_with_the_points() -> None:
    coords = [[float(i), 0.0, 0.0] for i in range(6)]
    sample = _sample(coords, list(range(6)))

    torch.manual_seed(0)
    shuffled = PointsRandomShuffle()(sample)

    # The label of each point still matches that point's x coordinate.
    xs = shuffled.point_cloud_data.points[:, 0].tolist()
    assert _labels(shuffled) == [int(x) for x in xs]


def test_grid_sample_keeps_one_label_per_cell_and_records_the_original() -> None:
    # Two points share a cell, the third is far away.
    sample = _sample([[0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [5.0, 5.0, 0.0]], [3, 3, 4])

    quantized = GridSample(
        grid_size=0.5, point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0], mode="eval"
    )(sample)

    segmentation = quantized.segmentation3d_gt_sample
    assert len(quantized.point_cloud_data) == 2
    assert np.asarray(segmentation.gt_semantic_mask).tolist() == [3, 4]
    # Metrics score the original cloud, so it is recorded alongside the mapping back.
    assert np.asarray(segmentation.origin_semantic_mask).tolist() == [3, 3, 4]
    assert segmentation.origin_coord.shape == (3, 3)
    assert np.asarray(segmentation.inverse).tolist() == [0, 0, 1]


def test_grid_sample_rejects_labels_that_no_longer_match_the_cloud() -> None:
    sample = _sample([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]], [1])

    try:
        GridSample(grid_size=0.5, point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0])(sample)
    except ValueError as error:
        assert "no longer aligned" in str(error)
    else:
        raise AssertionError("expected a ValueError for misaligned labels")
