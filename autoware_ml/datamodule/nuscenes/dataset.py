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

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDataModel
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset


class NuScenesDataset(T4Dataset):
    """
    Dataset serving every task from the record table of a nuScenes database.

    The nuScenes record generator stores every frame path as a complete path, so frame paths
    are used as stored instead of being resolved against the database root.
    """

    def _update_frame_path(self, frame_path: str) -> str:
        """
        Return the frame path unchanged, since NuScenesRecordsGenerator already stores
        full, directly-usable absolute paths.

        Args:
          frame_path: Frame path as stored in the parquet record.

        Returns:
          str: The same path, unmodified.
        """

        return frame_path

    def get_scene_token(self, idx: int, lidar_frame: LidarFrameDataModel) -> str:
        """
        Return the nuScenes scene name of the record.

        The T4 scene directory cannot be read off a nuScenes path (``samples/LIDAR_TOP/...``
        is shared by every scene, and a shallow root has too few components), so the record's
        ``scenario_id`` names the scene; nuScenes has no lanelet map to resolve from it.

        Args:
          idx: Index of the record.
          lidar_frame: Keyframe lidar frame of the record, unused.

        Returns:
          str: The scene name of the record.
        """
        del lidar_frame
        return str(self.dataset_records_dataframe.item(idx, DatasetTableSchema.SCENARIO_ID.name))
