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

"""Unit tests for CenterPoint detection model."""

import unittest

import torch

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.datamodule.multi_task.dataclasses.detection3d import (
    Detection3DGTBatch,
)
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTBatch
from autoware_ml.models.detection3d.main_modules.centerpoint import CenterPointDetectionModel
from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import PillarFeatureNet
from autoware_ml.models.detection3d.encoders.pillars.point_pillar_scatter import PointPillarsScatter
from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.models.multi_task_base_model import LogDictConfigs
from autoware_ml.preprocessing.detection3d.point_pillar_preprocessor import PointPillarPreprocessor
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class TestCenterPointDetectionModel(unittest.TestCase):
    """Unit tests for the CenterPointDetectionModel class."""

    def setUp(self) -> None:
        """Set up the common inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.class_names = ["car", "pedestrian", "cyclist"]
        self.num_classes = len(self.class_names)
        torch.manual_seed(0)
        self.data_preprocessor = DataPreprocessor(
            preprocessor_modules=[
                PointPillarPreprocessor(
                    voxel_size=[0.5, 0.5, 4.0],
                    point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
                    max_num_points=16,
                    max_voxels=16,
                    voxelization_z_order_first=False,
                )
            ]
        )
        self.pillar_feature_net = PillarFeatureNet(
            in_channels=5,
            feat_channels=[32, 32],
            voxel_size=[0.5, 0.5, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
        )
        self.middle_encoder = PointPillarsScatter(in_channels=32, output_shape=[16, 16])
        self.backbone = SECONDBackbone(
            in_channels=32,
            out_channels=[64, 128, 256],
            layer_nums=[1, 1, 1],
            layer_strides=[2, 2, 2],
        )
        self.neck = SECONDFPN(
            in_channels=[64, 128, 256],
            out_channels=[128, 128, 128],
            upsample_strides=[1, 2, 4],
        )
        self.bbox_head = CenterHead(
            in_channels=384,
            class_names=self.class_names,
            shared_channels=64,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=True,
        )
        self.log_dict_configs = LogDictConfigs(
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.centerpoint = CenterPointDetectionModel(
            data_preprocessor=self.data_preprocessor,
            pts_voxel_encoder=self.pillar_feature_net,
            pts_middle_encoder=self.middle_encoder,
            pts_backbone=self.backbone,
            pts_neck=self.neck,
            bbox_head=self.bbox_head,
            log_dict_configs=self.log_dict_configs,
        ).to(self.device)

    def test_centerpoint_weights_mean_std(self) -> None:
        """
        Test that the CenterPointDetectionModel weights are initialized
        with means of 0.0 and std < 0.1.
        """
        weights = []
        biases = []
        for name, param in self.centerpoint.named_parameters():
            if "weight" in name:
                weights.append(param.data)
            if "bias" in name:
                biases.append(param.data)

        weights = torch.cat([w.flatten() for w in weights])
        biases = torch.cat([b.flatten() for b in biases])
        weight_mean = weights.mean().item()
        weight_std = weights.std().item()
        bias_mean = biases.mean().item()
        bias_std = biases.std().item()

        expected_weight_mean = 0.0
        # Biases should be almost similar since the bias for the heatmap head is initialized
        # to a negative value and the rest are initialized to zero.
        expected_bias_mean = -0.008480
        expected_bias_std = 0.08971667289733887
        self.assertAlmostEqual(weight_mean, expected_weight_mean, places=2)
        self.assertLess(weight_std, 0.1)
        self.assertAlmostEqual(bias_mean, expected_bias_mean, places=2)
        self.assertAlmostEqual(bias_std, expected_bias_std, places=2)

    def _build_multi_task_batch_inputs(self) -> MultiTaskBatchInputs:
        """
        Build a MultiTaskBatchInputs batch with voxelized lidar inputs and detection ground truth
        that the CenterPointDetectionModel can be run end to end on.
        """
        num_pillars = 12
        voxels_data = VoxelsData(
            voxels=torch.randn((num_pillars, 5, 5), dtype=torch.float32, device=self.device),
            # (x, y, z) voxel coordinates inside the 16x16 scatter canvas
            coords=torch.randint(0, 8, (num_pillars, 3), dtype=torch.int32, device=self.device),
            num_points=torch.randint(1, 5, (num_pillars,), dtype=torch.int32, device=self.device),
            batch_indices=torch.tensor(
                [0] * (num_pillars // 2) + [1] * (num_pillars // 2),
                dtype=torch.int32,
                device=self.device,
            ),
        )
        # (batch_size, max_num_bboxes, num_Box3DFieldIndex)
        gt_bboxes_3d = torch.tensor(
            [
                [[2.0, 3.0, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, -0.2]],
                [[5.0, 6.0, 0.1, 2.0, 1.0, 1.2, -0.5, 0.2, 0.3, 0.0]],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        detection3d_gt_batch = Detection3DGTBatch(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=torch.tensor([[0], [1]], dtype=torch.int32, device=self.device),
            gt_valid_bboxes=torch.tensor([1, 1], dtype=torch.int32, device=self.device),
            gt_bboxes_num_points=torch.tensor(
                [[100], [200]], dtype=torch.int32, device=self.device
            ),
        )
        return MultiTaskBatchInputs(
            multi_task_gt_batch=MultiTaskGTBatch(
                point_cloud_gt_batch=None, detection3d_gt_batch=detection3d_gt_batch
            ),
            voxels_data=voxels_data,
        )

    def test_centerpoint_forward_compute_metrics_and_decode_run(self) -> None:
        """
        Test that the model runs end to end over voxelized inputs, producing head outputs of the
        expected shape, a differentiable loss, and one decoded prediction entry per sample.
        """
        multi_task_batch_inputs = self._build_multi_task_batch_inputs()

        multi_task_outputs = self.centerpoint(multi_task_batch_inputs)
        metrics = self.centerpoint.compute_metrics(multi_task_batch_inputs, multi_task_outputs)
        multi_task_predictions = self.centerpoint.decode_outputs(multi_task_outputs)

        self.assertIsNotNone(multi_task_outputs.detection3d_head_outputs)
        assert multi_task_outputs.detection3d_head_outputs is not None
        center_head_outputs = multi_task_outputs.detection3d_head_outputs.center_head_outputs
        assert center_head_outputs is not None
        # The 16x16 canvas is downsampled by the backbone and fused back to 8x8 by the neck
        self.assertEqual(
            center_head_outputs.heatmaps.shape,
            (self.batch_size, self.num_classes, 8, 8),
        )

        self.assertIn("loss", metrics)
        self.assertIn("loss_heatmap", metrics)
        self.assertIn("loss_bbox", metrics)
        self.assertTrue(torch.isfinite(metrics["loss"]).all())

        predictions = multi_task_predictions.detection3d_predictions
        assert predictions is not None
        self.assertEqual(len(predictions), self.batch_size)
        for sample_predictions in predictions:
            # use_velocity=True on the shared head, so decoded boxes carry 9 parameters
            self.assertEqual(sample_predictions.bboxes_3d.shape[1], 9)
            self.assertEqual(
                sample_predictions.scores_3d.shape[0], sample_predictions.bboxes_3d.shape[0]
            )
            self.assertEqual(
                sample_predictions.labels_3d.shape[0], sample_predictions.bboxes_3d.shape[0]
            )


if __name__ == "__main__":
    unittest.main()
