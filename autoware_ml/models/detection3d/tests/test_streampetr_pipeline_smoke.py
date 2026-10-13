"""End to end smoke test of StreamPETR, from the record table to streamed training steps."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.detection3d import Detection3DTask
from autoware_ml.datamodule.samplers import GroupStreamingSampler
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.corpus import write_corpus
from autoware_ml.datamodule.tests.test_camera_pipeline_smoke import CAMERAS, write_images
from autoware_ml.models.common.necks.lss_fpn import GeneralizedLSSFPN
from autoware_ml.models.detection3d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.streampetr import StreamPETRDetectionModel
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.preprocessing.base import DataPreprocessing
from autoware_ml.preprocessing.camera.grid_mask import BatchGridMask
from autoware_ml.preprocessing.camera.normalize import BatchNormalizeImages
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.filters import BBoxesLabelNameFilter, BBoxesRangeFilter
from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.transforms.camera.geometry import CameraGlobalRotScaleTrans, ImageAug3D
from autoware_ml.transforms.camera.loading import LoadImagesFromFile

POINT_CLOUD_RANGE = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]
IMAGE_SIZE = [32, 48]


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = torch.nn.Conv2d(3, 16, kernel_size=8, stride=8)
        self.stage2 = torch.nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.stage3 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=2)

    def forward(self, x):
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        return c3, c4, self.stage3(c4)


def build_model() -> StreamPETRDetectionModel:
    """A small StreamPETR with the auxiliary 2D head, both on stride-8 tokens."""
    return StreamPETRDetectionModel(
        img_backbone=_TinyBackbone(),
        img_neck=GeneralizedLSSFPN(in_channels=[16, 32, 64], out_channels=32),
        bbox_head=StreamPETRHead(
            num_classes=1,
            in_channels=32,
            hidden_dim=32,
            num_queries=8,
            num_decoder_layers=2,
            num_heads=4,
            feedforward_channels=64,
            memory_len=8,
            topk_proposals=4,
            num_propagated=4,
            depth_num=8,
            position_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
            scalar=2,
            noise_scale=0.5,
            split=0.5,
            bbox_coder=NMSFreeBBoxCoder3D(pc_range=POINT_CLOUD_RANGE, max_num=8),
            assigner=HungarianAssigner3D(
                cls_cost=ClassificationCost(weight=2.0),
                reg_cost=BBox3DL1Cost(weight=0.25),
                iou_cost=IoU3DCost(weight=0.0),
            ),
            point_cloud_range=POINT_CLOUD_RANGE,
            code_weights=[2.0, 2.0] + [1.0] * 8,
        ),
        img_roi_head=FocalHead2D(num_classes=1, in_channels=32, embed_dims=32, stride=8),
    )


def build_transforms() -> TransformsCompose:
    """The training pipeline of the T4 configs, shrunk to the synthetic images."""
    return TransformsCompose(
        pipeline=[
            BBoxesLabelNameFilter(label_names_to_keep=["car"], class_names=["car"]),
            LoadImagesFromFile(shuffle_order=True),
            ImageAug3D(
                final_dim=IMAGE_SIZE,
                resize_lim=[-0.02, 0.02],
                resize_relative=True,
                bot_pct_lim=[0.0, 0.0],
                rand_flip=True,
                training=True,
            ),
            CameraGlobalRotScaleTrans(yaw_rot_range=[-0.3, 0.3], scale_ratio_range=[0.95, 1.05]),
            LoadAnnotations2DFromBoxes3D(),
            BBoxesRangeFilter(point_cloud_range=POINT_CLOUD_RANGE),
        ]
    )


class TestStreamPETRPipelineSmoke(unittest.TestCase):
    """The ported data path carries two streamed scenes through StreamPETR training steps."""

    def setUp(self) -> None:
        """Write two scenes of two frames, stored out of scene and time order."""
        torch.manual_seed(0)
        np.random.seed(0)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        records = write_images(root, write_corpus(root, num_records=4)).with_columns(
            pl.Series(DatasetTableSchema.SCENARIO_ID.name, ["b", "a", "b", "a"]),
            pl.Series(DatasetTableSchema.TIMESTAMP_SECONDS.name, [21.0, 10.5, 20.0, 10.0]),
        )
        self.dataset = T4Dataset(
            lidar_intensity_scale=255.0,
            database_root_path=str(root),
            max_num_3d_gt_bboxes=8,
            dataset_records_dataframe=records,
            transforms=build_transforms(),
            dataset_tasks={"Detection3D": Detection3DTask},
            det3d_supervised=True,
            seg3d_supervised=True,
            camera_names=CAMERAS,
        )
        self.preprocessing = DataPreprocessing(
            [
                BatchNormalizeImages(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                BatchGridMask(prob=1.0),
            ]
        )

    def test_a_lane_streams_every_scene_in_time_order(self) -> None:
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False)

        self.assertEqual(list(sampler), [2, 0, 3, 1])

    def test_streamed_frames_train_the_detector_and_carry_its_memory(self) -> None:
        model = build_model().train()
        model.log_dict = lambda *args, **kwargs: None
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False)
        continuity = []
        stream_continuity = model.bbox_head.stream_continuity

        def record_continuity(scenario_ids, timestamp):
            prev_exists = stream_continuity(scenario_ids, timestamp)
            continuity.append((list(scenario_ids), timestamp.tolist(), prev_exists.tolist()))
            return prev_exists

        model.bbox_head.stream_continuity = record_continuity

        for index in sampler:
            batch = self.dataset.collate_fn([self.dataset[index]])
            batch_inputs = self.preprocessing(batch, is_training=True)
            frame_meta = batch_inputs.multi_task_gt_batch.frame_meta_batch
            assert frame_meta is not None
            # The augmentation of the frame is folded into its pose for the memory warp.
            self.assertFalse(torch.equal(frame_meta.ego2globals[0], torch.eye(4)))
            assert batch_inputs.multi_task_gt_batch.detection2d_gt_batch is not None

            loss = model.training_step(batch_inputs, batch_idx=0)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

        # Scene b starts the lane, its second frame continues the memory, scene a resets it.
        self.assertEqual(
            continuity,
            [
                (["b"], [20.0], [0.0]),
                (["b"], [21.0], [1.0]),
                (["a"], [10.0], [0.0]),
                (["a"], [10.5], [1.0]),
            ],
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.img_roi_head.parameters())
        )
