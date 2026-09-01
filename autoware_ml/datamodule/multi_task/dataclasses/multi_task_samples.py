from __future__ import annotations

from typing import Sequence, NamedTuple

from jaxtyping import Float32, Int32
import torch
from torch import Tensor

from autoware_ml.datamodule.multi_task.dataclasses.detection3d import (
    Detection3DGTBatch,
)
from autoware_ml.datamodule.multi_task.dataclasses.segmentation3d import Segmentation3DGTSample
from autoware_ml.datamodule.multi_task.dataclasses.transformation import LiDARTransformationSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.points.base_points import BasePoints


class PointCloudGTBatch(NamedTuple):
    """Named tuple to represent pointcloud features in a batch size with their batch indices."""

    points: Float32[
        Tensor, "batch_size*num_points num_features"
    ]  # (B*P, number of features for each point)
    batch_indices: Int32[Tensor, " batch_size*num_points"]  # (B*P, ), batch indices for each point

    @staticmethod
    def collate_gt_samples(
        point_gt_samples: Sequence[BasePoints],
    ) -> PointCloudGTBatch | None:
        """
        Collate a sequence of points (BasePoints) into a single PointCloudGTBatch.

        Args:
          point_gt_samples: Sequence of points (BasePoints) to be collated.

        Returns:
          PointCloudGTBatch: Collated point cloud GT batch.
        """
        if len(point_gt_samples) == 0:
            return None

        # Concatenate all points from the sequence of point_gt_samples
        points = torch.cat([sample.points for sample in point_gt_samples], dim=0)

        # Convert it to (0, 0, 0, 1, 1, 1, 2, 2, 2, ...) for each point in the batch
        batch_indices = torch.cat(
            [
                torch.full(
                    (point.points.shape[0],), i, dtype=torch.int32, device=point.points.device
                )
                for i, point in enumerate(point_gt_samples)
            ],
            dim=0,
        )

        if points.shape[0] != batch_indices.shape[0]:
            raise ValueError(
                "Mismatch between number of points and batch indices. "
                f"Points shape: {points.shape}, Batch indices shape: {batch_indices.shape}"
            )

        return PointCloudGTBatch(
            points=points,
            batch_indices=batch_indices,
        )

    def to_device(self, device: torch.device) -> PointCloudGTBatch:
        """
        Move the PointCloudGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          PointCloudGTBatch: The batch moved to the specified device.
        """
        return PointCloudGTBatch(
            points=self.points.to(device),
            batch_indices=self.batch_indices.to(device),
        )


class LiDARPointCloudSample(NamedTuple):
    """
    Named tuple to represent a single row of LiDAR point cloud data,
    which contains the dataset record for the LiDAR point cloud task.
    """

    point_cloud_path: str
    timestamp: float
    # Transformation matrix from LiDAR sensor frame to ego pose of this LiDAR sensor frame
    sensor_to_ego_pose_matrix: Float32[Tensor, "4 4"]  # (4, 4)
    # Transformation matrix from ego pose of this LiDAR sensor frame to global frame
    lidar_to_ego_pose_to_global_matrix: Float32[Tensor, "4 4"]  # (4, 4)
    # Transformation matrix from the main lidar sensor to other lidar sweeps at this frame
    lidar_sensor_to_lidar_sweep_matrix: Float32[Tensor, "4 4"]  # (4, 4)


class MultiTaskGTSample(NamedTuple):
    """
    Named tuple to represent a single row/sample of multi-task data when inputting to the
    multi-task model.
    """

    # Can be multi-sweep LiDAR point cloud data, which is a list of LiDAR point cloud data rows for each sweep.
    lidar_point_cloud_samples: Sequence[LiDARPointCloudSample] | None

    # (number of point clouds, number of features for each point), can be None
    # if it doesn't need to be loaded
    point_cloud_data: BasePoints | None

    detection3d_gt_bboxes_3d: BaseBBoxes3D | None
    segmentation3d_gt_sample: Segmentation3DGTSample | None

    # Information about lidar transformation
    lidar_transformation_sample: LiDARTransformationSample | None = None

    # Seconds spent loading this sample and running it through the transform pipeline.
    # Assigned by the dataset once the pipeline has finished.
    io_processing_time: float = 0.0


class MultiTaskGTBatch(NamedTuple):
    """
    Named tuple to represent a batch of multi-task data after collating from sequence of
    MultiTaskGTSample when inputting to the multi-task model.
    """

    # 3D branch
    point_cloud_gt_batch: PointCloudGTBatch | None
    detection3d_gt_batch: Detection3DGTBatch | None
    # TODO (Kok Seang): 3D segmentation

    # Summed io_processing_time of every sample collated into this batch.
    io_processing_time: float = 0.0

    def to_device(self, device: torch.device) -> MultiTaskGTBatch:
        """
        Move the MultiTaskGTBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          MultiTaskGTBatch: The batch moved to the specified device.
        """
        return MultiTaskGTBatch(
            point_cloud_gt_batch=self.point_cloud_gt_batch.to_device(device)
            if self.point_cloud_gt_batch is not None
            else None,
            detection3d_gt_batch=self.detection3d_gt_batch.to_device(device)
            if self.detection3d_gt_batch is not None
            else None,
            io_processing_time=self.io_processing_time,
        )

    def infer_batch_size(self) -> Int32:
        """
        Infer the batch size from the collated multi-task GT batch.

        Returns:
            Batch size if it can be inferred, otherwise raises ValueError.
        """
        if self.point_cloud_gt_batch is not None:
            return torch.max(self.point_cloud_gt_batch.batch_indices) + 1
        elif self.detection3d_gt_batch is not None:
            return self.detection3d_gt_batch.gt_bboxes_3d.shape[0]
        else:
            raise ValueError("Cannot infer batch size from an empty MultiTaskGTBatch.")

    @staticmethod
    def collate_pointcloud_gt_samples(
        gt_samples: Sequence[MultiTaskGTSample],
    ) -> PointCloudGTBatch | None:
        """
        Collate a sequence of point cloud GT samples into a PointCloudGTBatch.

        Args:
          gt_samples: Sequence of MultiTaskGTSample to be collated.

        Returns:
          PointCloudGTBatch: Collated point cloud GT batch.
        """
        if len(gt_samples) == 0:
            return None

        available_pointcloud = gt_samples[0].point_cloud_data is not None
        if not available_pointcloud:
            return None

        pointcloud_samples = []
        for sample in gt_samples:
            if sample.point_cloud_data is None:
                raise ValueError("All samples must have point_cloud_data for collating.")
            pointcloud_samples.append(sample.point_cloud_data)

        point_cloud_gt_batch = PointCloudGTBatch.collate_gt_samples(pointcloud_samples)
        return point_cloud_gt_batch

    @staticmethod
    def collate_detection3d_gt_samples(
        gt_samples: Sequence[MultiTaskGTSample], max_num_3d_gt_bboxes: int
    ) -> Detection3DGTBatch | None:
        """
        Collate a sequence of detection3d GT samples into a Detection3DGTBatch.

        Args:
          gt_samples: Sequence of MultiTaskGTSample to be collated.
          max_num_3d_gt_bboxes: The maximum number of 3D ground truth bounding boxes
            for each sample in the batch.

        Returns:
          Detection3DGTBatch: Collated detection3d GT batch.
        """
        if len(gt_samples) == 0:
            return None

        # Check if detection3d_gt_bboxes_3d are available in the samples
        available_detection3d_gt_bboxes_3d = gt_samples[0].detection3d_gt_bboxes_3d is not None
        if not available_detection3d_gt_bboxes_3d:
            return None

        detection3d_gt_bboxes_3d = []
        for sample in gt_samples:
            if sample.detection3d_gt_bboxes_3d is None:
                raise ValueError("All samples must have detection3d_gt_bboxes_3d for collating.")

            detection3d_gt_bboxes_3d.append(sample.detection3d_gt_bboxes_3d)

        detection3d_gt_batch = Detection3DGTBatch.collate_gt_samples(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            max_num_3d_gt_bboxes=max_num_3d_gt_bboxes,
        )
        return detection3d_gt_batch

    @staticmethod
    def collate_gt_samples(
        gt_samples: Sequence[MultiTaskGTSample], max_num_3d_gt_bboxes: int
    ) -> MultiTaskGTBatch:
        """
        Collate a sequence of MultiTaskGTSample into a MultiTaskGTBatch.

        Args:
          gt_samples: Sequence of MultiTaskGTSample to be collated.

        Returns:
          MultiTaskGTBatch: Collated multi-task GT batch.
        """
        # Collate point cloud GT batch
        point_cloud_gt_batch = MultiTaskGTBatch.collate_pointcloud_gt_samples(gt_samples)

        # Collate detection3d GT batch
        detection3d_gt_batch = MultiTaskGTBatch.collate_detection3d_gt_samples(
            gt_samples=gt_samples, max_num_3d_gt_bboxes=max_num_3d_gt_bboxes
        )

        return MultiTaskGTBatch(
            point_cloud_gt_batch=point_cloud_gt_batch,
            detection3d_gt_batch=detection3d_gt_batch,
            io_processing_time=sum(sample.io_processing_time for sample in gt_samples),
        )
