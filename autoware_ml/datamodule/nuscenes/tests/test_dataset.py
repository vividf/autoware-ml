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

"""Unit tests for the frame metadata of the nuScenes dataset."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.nuscenes.dataset import NuScenesDataset


def build_records_dataframe(scene_names: list[str]) -> pl.DataFrame:
    """One record per scene name, every record holding the same nuScenes lidar path layout."""
    lidar_frames = [
        [
            {
                "lidar_frame_id": f"frame_{index}",
                "lidar_keyframe": True,
                "lidar_sensor_id": "lidar",
                "lidar_sensor_channel_name": "LIDAR_TOP",
                "lidar_timestamp_seconds": float(index),
                "lidar_pointcloud_path": f"/nuscenes/samples/LIDAR_TOP/{index}.pcd.bin",
                "lidar_pointcloud_source_path": None,
                "lidar_pointcloud_num_features": 5,
                "lidar_sensor_to_ego_pose_matrix": np.eye(4).tolist(),
                "lidar_frame_ego_pose_to_global_matrix": np.eye(4).tolist(),
                "lidar_sensor_to_lidar_sweep_matrix": np.eye(4).tolist(),
                "lidar_pointcloud_semantic_mask_path": None,
            }
        ]
        for index in range(len(scene_names))
    ]
    return pl.DataFrame(
        {
            DatasetTableSchema.LIDAR_FRAMES.name: lidar_frames,
            DatasetTableSchema.SCENARIO_ID.name: scene_names,
            DatasetTableSchema.TIMESTAMP_SECONDS.name: [float(i) for i in range(len(scene_names))],
        },
        schema={
            DatasetTableSchema.LIDAR_FRAMES.name: DatasetTableSchema.LIDAR_FRAMES.dtype,
            DatasetTableSchema.SCENARIO_ID.name: DatasetTableSchema.SCENARIO_ID.dtype,
            DatasetTableSchema.TIMESTAMP_SECONDS.name: DatasetTableSchema.TIMESTAMP_SECONDS.dtype,
        },
    )


class TestNuScenesFrameMeta(unittest.TestCase):
    """The scene identity of a nuScenes record is its scene name, not a path fragment."""

    def test_scene_token_is_the_scene_name(self) -> None:
        """
        Input: two records of different nuScenes scenes whose lidar paths share the
        ``samples/LIDAR_TOP`` layout under a one-directory root.
        Expected: a path fragment would be the same for both (or fail on the shallow root);
        the scene token is the record's scenario id, so the frames of different scenes
        differ and a streaming model resets its memory between them.
        Check: the scene tokens and the sample times of the frame metadata.
        """
        dataset = NuScenesDataset(
            "/nuscenes",
            dataset_records_dataframe=build_records_dataframe(["scene-0001", "scene-0002"]),
            transforms=None,
            dataset_tasks={},
            lidar_intensity_scale=1.0,
            det3d_supervised=False,
            seg3d_supervised=False,
        )

        metas = [dataset.get_frame_meta_sample(index, None) for index in range(2)]

        self.assertEqual([meta.scene_token for meta in metas], ["scene-0001", "scene-0002"])
        self.assertEqual([meta.timestamp for meta in metas], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
