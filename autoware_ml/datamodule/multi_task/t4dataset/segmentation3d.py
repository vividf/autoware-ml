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

"""Point-wise semantic segmentation GT from the T4 dataset records."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.datamodule.multi_task.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.datamodule.multi_task.dataclasses.segmentation3d import Segmentation3DGTSample

logger = logging.getLogger(__name__)

#: Trailing components of a recorded absolute point-cloud path that locate its scene inside
#: the database root: ``<dataset>/<scenario>/<version>/data/<sensor>/<frame>.bin``. The first
#: three name the scene directory, which is what the mask path is recorded relative to.
_POINTCLOUD_RELATIVE_PATH_DEPTH = 6
_SCENE_PATH_DEPTH = 3


class T4Segmentation3DTask(BaseDatasetTask):
    """Dataset task producing point-wise semantic labels for the T4 dataset.

    The labels on disk are raw *category indices* of the dataset that recorded them, so
    they mean nothing without that dataset's ``category_mapping`` (index -> category
    name). Training labels are a separate, model-facing space, and the two are bridged
    by ``class_mapping`` (category name -> training label): a category the model does
    not learn maps to ``ignore_label_index`` rather than silently becoming class 0.
    """

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame | None,
        class_mapping: Mapping[str, int],
        ignore_label_index: int = -1,
        mask_dtype: str = "uint8",
    ) -> None:
        """Initialize the task.

        Args:
            database_root_path: Root the recorded mask paths are resolved against.
            dataset_records_dataframe: Records to read, or None until assigned.
            class_mapping: Category name -> training label.
            ignore_label_index: Training label for categories outside ``class_mapping``.
            mask_dtype: Element type of the stored mask files.
        """
        super().__init__(
            database_root_path=database_root_path,
            dataset_records_dataframe=dataset_records_dataframe,
        )
        if not class_mapping:
            raise ValueError(
                "T4Segmentation3DTask needs a non-empty class_mapping (category name -> "
                "training label); without it every point would be ignored."
            )
        self.class_mapping = dict(class_mapping)
        self.ignore_label_index = ignore_label_index
        self.mask_dtype = np.dtype(mask_dtype)

    def pre_filter_dataset_records(
        self, dataset_records_dataframe: pl.DataFrame | None
    ) -> pl.DataFrame | None:
        """Keep only the columns segmentation reads: the lidar frames and the categories."""
        if dataset_records_dataframe is None:
            return None
        return dataset_records_dataframe.select(
            [
                DatasetTableSchema.LIDAR_FRAMES.name,
                DatasetTableSchema.CATEGORY_MAPPING.name,
            ]
        )

    def __str__(self) -> str:
        """String representation of the dataset task."""
        return "T4Segmentation3DTask"

    def _mask_path(self, idx: int) -> Path:
        """Resolve the keyframe's mask file for one record.

        The records generator writes the mask path *relative to its scene directory*
        (``lidarseg/annotation/<token>.bin``) while point-cloud paths are absolute, so the
        scene is taken from the same frame's point-cloud path and re-rooted under
        ``database_root_path`` — the convention the dataset uses for point clouds too, and
        what makes a database built on one machine readable on another.

        Args:
            idx: Record index.

        Returns:
            Absolute path of the mask file.

        Raises:
            ValueError: When no lidar frame of the record carries a mask path.
        """
        lidar_frames: Sequence[Mapping[str, Any]] = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.LIDAR_FRAMES.name
        )
        mask_column = LidarFrameDatasetSchema.lidar_pointcloud_semantic_mask_path.name
        pointcloud_column = LidarFrameDatasetSchema.lidar_pointcloud_path.name
        frame = next(
            (frame for frame in lidar_frames if frame.get(mask_column) is not None), None
        )
        if frame is None:
            raise ValueError(
                f"Record {idx} has no lidar frame carrying {mask_column!r}; the scenario was "
                "scanned without lidarseg annotations."
            )
        scene = Path(
            *Path(str(frame[pointcloud_column])).parts[
                -_POINTCLOUD_RELATIVE_PATH_DEPTH : -_POINTCLOUD_RELATIVE_PATH_DEPTH
                + _SCENE_PATH_DEPTH
            ]
        )
        return self.database_root_path / scene / str(frame[mask_column])

    def _raw_to_training_label(self, idx: int) -> npt.NDArray[np.int64]:
        """Build the lookup turning this record's raw category indices into training labels."""
        category_mapping = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.CATEGORY_MAPPING.name
        )
        names: Sequence[str] = category_mapping["category_names"]
        indices: Sequence[int] = category_mapping["category_indices"]
        lookup = np.full(int(max(indices)) + 1 if len(indices) else 0, self.ignore_label_index,
                         dtype=np.int64)
        for name, raw_index in zip(names, indices):
            lookup[int(raw_index)] = self.class_mapping.get(str(name), self.ignore_label_index)
        return lookup

    def get_data_sample(self, idx: int) -> MultiTaskGTSample:
        """Load one record's point-wise labels, remapped into the training label space.

        Args:
            idx: Record index.

        Returns:
            A sample carrying only the segmentation slot; the dataset merges the tasks.

        Raises:
            ValueError: If no records are assigned, or the record has no mask path.
        """
        if self.dataset_records_dataframe is None:
            raise ValueError("Dataset records dataframe is not available.")

        raw_labels = np.fromfile(self._mask_path(idx), dtype=self.mask_dtype).astype(np.int64)
        lookup = self._raw_to_training_label(idx)
        # A raw index beyond this dataset's category table is corrupt metadata, not an
        # unlearned category, so it is reported rather than folded into "ignore".
        out_of_table = raw_labels >= lookup.shape[0]
        if bool(out_of_table.any()):
            raise ValueError(
                f"Record {idx} has raw label(s) {sorted(set(raw_labels[out_of_table].tolist()))} "
                f"outside its category table of size {lookup.shape[0]}."
            )
        labels = lookup[raw_labels].astype(np.int32, copy=False)

        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            point_cloud_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=Segmentation3DGTSample(gt_semantic_mask=labels),
        )

    def log_dataset_info(self) -> None:
        """Log how many records carry segmentation annotations."""
        if self.dataset_records_dataframe is None:
            return
        logger.info(
            "%s: %d record(s), %d training class(es), ignore label %d",
            self,
            self.dataset_records_dataframe.height,
            len(set(self.class_mapping.values())),
            self.ignore_label_index,
        )
