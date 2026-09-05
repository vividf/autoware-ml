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

"""Unit tests for point pillar preprocessing."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.preprocessing.detection3d.point_pillar_preprocessor import PointPillarPreprocessor
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTBatch,
    PointCloudGTBatch,
    Detection3DGTBatch,
)
from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs


class TestPointPillarPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the same PointPillarPreprocessor instance for all tests. Note that this class will
        be called in each test case.
        """
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.point_pillar_preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=2,
            max_voxels=8,
            voxelization_z_order_first=False,  # This is used for backward-compatible, and will be removed very soon.
        )

    def test_builds_padded_pillars(self) -> None:
        """Test that the __call__ builds padded pillars from a batch of point clouds."""
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.1, 0.1, 0.0, 1.0],
                        [0.2, 0.2, 0.0, 2.0],
                        [1.1, 1.1, 0.0, 3.0],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 0, 0], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch, voxels_data=None
        )

        outputs = self.point_pillar_preprocessor(multi_task_batch_inputs, is_training=True)
        self.assertIsNotNone(outputs.voxels_data)
        self.assertEqual(outputs.voxels_data.voxels.shape, (2, 2, 4))
        self.assertEqual(outputs.voxels_data.num_points.tolist(), [2, 1])
        self.assertEqual(outputs.voxels_data.coords.shape, (2, 3))
        self.assertEqual(outputs.voxels_data.coords[:, 0].tolist(), [0, 1])

    def test_builds_padded_pillars_z_order_first(self) -> None:
        """Test that the __call__ builds padded pillars from a batch of point clouds with z-order first."""
        self.point_pillar_preprocessor.voxelization_z_order_first = True
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.1, 0.1, 0.0, 1.0],
                        [0.2, 0.2, 0.0, 2.0],
                        [1.1, 1.1, 0.0, 3.0],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 0, 0], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch, voxels_data=None
        )

        outputs = self.point_pillar_preprocessor(multi_task_batch_inputs, is_training=True)
        self.assertIsNotNone(outputs.voxels_data)
        self.assertEqual(outputs.voxels_data.voxels.shape, (2, 2, 4))
        self.assertEqual(outputs.voxels_data.num_points.tolist(), [2, 1])
        self.assertEqual(outputs.voxels_data.coords.shape, (2, 3))
        self.assertEqual(outputs.voxels_data.coords[:, 0].tolist(), [0, 0])

    def test_per_sample_coords(self) -> None:
        """Test that the voxel coordinates are correctly computed for each sample in the batch."""
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 1, 2], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch, voxels_data=None
        )
        outputs = self.point_pillar_preprocessor(multi_task_batch_inputs, is_training=True)
        self.assertIsNotNone(outputs.voxels_data)
        self.assertTrue(
            torch.allclose(
                outputs.voxels_data.coords,
                torch.tensor(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    device=self.device,
                    dtype=torch.int32,
                ),
            )
        )

    def test_per_sample_batch_indices(self) -> None:
        """Test that the batch indices are correctly computed for each sample in the batch."""
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 2, 4], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch, voxels_data=None
        )
        outputs = self.point_pillar_preprocessor(multi_task_batch_inputs, is_training=True)
        self.assertIsNotNone(outputs.voxels_data)
        self.assertTrue(
            torch.allclose(
                outputs.voxels_data.batch_indices,
                torch.tensor([0, 2, 4], device=self.device, dtype=torch.int32),
            )
        )

    def test_empty_sample_in_batch(self) -> None:
        """
        Test that the PointPillarPreprocessor correctly handles a batch containing an empty
        sample.
        """
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], device=self.device, dtype=torch.float32)
        empty = torch.zeros((0, 4), device=self.device, dtype=torch.float32)
        points = torch.cat([point, empty, point], dim=0).to(self.device)
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=points,
                batch_indices=torch.tensor([0, 0, 1], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch, voxels_data=None
        )
        # Raise a ValueError because the length of points list does not match the length of
        # batch indices in MultiTaskGTBatch.
        with self.assertRaises(ValueError):
            self.point_pillar_preprocessor(multi_task_batch_inputs, is_training=True)

    def test_empty_batch_returns_empty_pillar_tensors(self) -> None:
        """
        Test that the PointPillarPreprocessor returns empty pillar tensors when given an
        empty batch.
        """
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.zeros((0, 4), device=self.device, dtype=torch.float32),
                batch_indices=torch.tensor([], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch, voxels_data=None
        )
        outputs = self.point_pillar_preprocessor(multi_task_batch_inputs, is_training=True)
        self.assertIsNotNone(outputs.voxels_data)
        self.assertEqual(outputs.voxels_data.voxels.shape, (0, 2, 4))
        self.assertEqual(outputs.voxels_data.num_points.shape, (0,))
        self.assertEqual(outputs.voxels_data.coords.shape, (0, 3))
        self.assertEqual(outputs.voxels_data.batch_indices.shape, (0,))

    def _single_sample_inputs(self, points: torch.Tensor) -> MultiTaskBatchInputs:
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=points.to(self.device),
                batch_indices=torch.zeros(points.shape[0], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        return MultiTaskBatchInputs(multi_task_gt_batch=multi_task_gt_batch, voxels_data=None)

    def test_eval_mode_uses_eval_max_voxels_budget(self) -> None:
        """
        Test that the voxel budget switches with ``is_training``: training truncates at
        ``max_voxels`` while evaluation keeps pillars up to ``eval_max_voxels``.
        """
        preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=2,
            max_voxels=1,
            eval_max_voxels=8,
            voxelization_z_order_first=True,
        )
        # Three points in three distinct pillars
        points = torch.tensor(
            [
                [0.1, 0.1, 0.0, 1.0],
                [1.1, 1.1, 0.0, 2.0],
                [2.1, 2.1, 0.0, 3.0],
            ],
            dtype=torch.float32,
        )

        train_outputs = preprocessor(self._single_sample_inputs(points), is_training=True)
        self.assertEqual(train_outputs.voxels_data.voxels.shape[0], 1)

        eval_outputs = preprocessor(self._single_sample_inputs(points), is_training=False)
        self.assertEqual(eval_outputs.voxels_data.voxels.shape[0], 3)

    def test_eval_mode_without_eval_max_voxels_raises(self) -> None:
        """
        Test that running in evaluation mode without an explicit ``eval_max_voxels`` raises
        instead of silently reusing the training budget.
        """
        inputs = self._single_sample_inputs(
            torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        )

        with self.assertRaises(ValueError):
            self.point_pillar_preprocessor(inputs, is_training=False)

    def test_train_mode_does_not_require_eval_max_voxels(self) -> None:
        """
        Test that training-mode forward keeps working when ``eval_max_voxels`` is not set,
        so existing training configs stay valid.
        """
        inputs = self._single_sample_inputs(
            torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        )

        outputs = self.point_pillar_preprocessor(inputs, is_training=True)

        self.assertEqual(outputs.voxels_data.voxels.shape[0], 1)

    def test_passthrough_of_existing_keys(self) -> None:
        """
        Test that the PointPillarPreprocessor correctly passes through existing
        keys in the input batch dictionary.
        """
        gt_bboxes_3d = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.57, 0.0, 1.0, 1.0]],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        gt_labels_3d = torch.tensor([[1], [2]], device=self.device, dtype=torch.int32)
        gt_valid_bboxes = torch.tensor([1, 1], device=self.device, dtype=torch.int32)
        gt_bboxes_num_points = torch.tensor([[10], [20]], device=self.device, dtype=torch.int32)

        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 2, 4], device=self.device, dtype=torch.int32),
            ),
            detection3d_gt_batch=Detection3DGTBatch(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_valid_bboxes=gt_valid_bboxes,
                gt_bboxes_num_points=gt_bboxes_num_points,
            ),
        )
        multi_task_batch_inputs = MultiTaskBatchInputs(
            multi_task_gt_batch=multi_task_gt_batch, voxels_data=None
        )
        outputs = self.point_pillar_preprocessor(multi_task_batch_inputs, is_training=True)
        self.assertIsNotNone(outputs.voxels_data)
        self.assertIsNotNone(outputs.multi_task_gt_batch.point_cloud_gt_batch)
        self.assertIsNotNone(outputs.multi_task_gt_batch.detection3d_gt_batch)

        self.assertTrue(
            torch.allclose(
                outputs.multi_task_gt_batch.detection3d_gt_batch.gt_bboxes_3d, gt_bboxes_3d
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs.multi_task_gt_batch.detection3d_gt_batch.gt_labels_3d, gt_labels_3d
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs.multi_task_gt_batch.detection3d_gt_batch.gt_valid_bboxes, gt_valid_bboxes
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs.multi_task_gt_batch.detection3d_gt_batch.gt_bboxes_num_points,
                gt_bboxes_num_points,
            )
        )


if __name__ == "__main__":
    unittest.main()
