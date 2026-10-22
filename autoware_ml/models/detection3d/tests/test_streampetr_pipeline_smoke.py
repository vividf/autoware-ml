"""End to end smoke test of StreamPETR, from the record table to streamed training steps."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.common.detection3d import Detection3DTask
from autoware_ml.datamodule.samplers import GroupStreamingSampler
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.corpus import SCENE, write_corpus
from autoware_ml.datamodule.tests.test_camera_pipeline_smoke import CAMERAS, write_images
from autoware_ml.models.common.necks.lss_fpn import GeneralizedLSSFPN
from autoware_ml.models.detection3d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.streampetr import StreamPETRDetection3DModel
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.preprocessing.camera.grid_mask import BatchGridMask
from autoware_ml.preprocessing.camera.normalize import BatchNormalizeImages
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.filters import (
    BBoxesLabelNameFilter,
    BBoxesPhysicalFilter,
    BBoxesRangeFilter,
)
from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.transforms.camera.loading import LoadImagesFromFile
from autoware_ml.transforms.camera.resize import ResizeCropFlipRotImage
from autoware_ml.transforms.camera_lidar.geometry import GlobalRotScaleTrans

POINT_CLOUD_RANGE = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]
IMAGE_SIZE = [32, 48]
CLASS_NAMES = ["car"]


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


def build_model() -> StreamPETRDetection3DModel:
    """A small StreamPETR with the auxiliary 2D head, both on stride-8 tokens.

    The preprocessor and the partial annotations are the T4 recipe's, with the single class
    of the corpus standing in for the traffic cones and barriers.
    """
    return StreamPETRDetection3DModel(
        data_preprocessor=DataPreprocessor(
            preprocessor_modules=[
                BatchNormalizeImages(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                BatchGridMask(prob=1.0),
            ]
        ),
        img_backbone=_TinyBackbone(),
        img_neck=GeneralizedLSSFPN(in_channels=[16, 32, 64], out_channels=32),
        bbox_head=StreamPETRHead(
            class_names=CLASS_NAMES,
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
                point_cloud_range=POINT_CLOUD_RANGE,
            ),
            point_cloud_range=POINT_CLOUD_RANGE,
            code_weights=[2.0, 2.0] + [1.0] * 8,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        img_roi_head=FocalHead2D(class_names=CLASS_NAMES, in_channels=32, embed_dims=32, stride=8),
        partial_annotations=PartialAnnotations(CLASS_NAMES, CLASS_NAMES),
    )


def build_transforms() -> TransformsCompose:
    """The training pipeline of the T4 configs, shrunk to the synthetic images."""
    return TransformsCompose(
        pipeline=[
            BBoxesPhysicalFilter(),
            BBoxesLabelNameFilter(label_names_to_keep=CLASS_NAMES, class_names=CLASS_NAMES),
            LoadImagesFromFile(shuffle_order=True),
            ResizeCropFlipRotImage(
                target_size=IMAGE_SIZE,
                resize_range=0.02,
                bottom_crop_ratio_range=[0.0, 0.0],
                random_horizontal_flip=True,
                training=True,
            ),
            GlobalRotScaleTrans(yaw_rot_range=[-0.3, 0.3], scale_ratio_range=[0.95, 1.05]),
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
            # Scene a was annotated without the partially annotated classes.
            pl.Series(
                DatasetTableSchema.TRAFFIC_CONE_BARRIER_BBOX_STATUS.name,
                [True, False, True, False],
            ),
        )
        self.dataset = T4Dataset(
            lidar_intensity_scale=255.0,
            database_root_path=str(root),
            max_num_3d_gt_bboxes=8,
            dataset_records_dataframe=records,
            transforms=build_transforms(),
            dataset_tasks={"detection3d": Detection3DTask},
            det3d_supervised=True,
            seg3d_supervised=True,
            camera_names=CAMERAS,
        )

    def test_a_lane_streams_every_scene_in_time_order(self) -> None:
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False, seed=0)

        self.assertEqual(list(sampler), [2, 0, 3, 1])

    def test_streamed_frames_train_the_detector_and_carry_its_memory(self) -> None:
        model = build_model().train()
        model.log_dict = lambda *args, **kwargs: None
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False, seed=0)
        continuity = []
        statuses = []
        stream_continuity = model.bbox_head.stream_continuity

        def record_continuity(scene_tokens, timestamp):
            prev_exists = stream_continuity(scene_tokens, timestamp)
            continuity.append((list(scene_tokens), timestamp.tolist(), prev_exists.tolist()))
            return prev_exists

        model.bbox_head.stream_continuity = record_continuity

        for index in sampler:
            batch = self.dataset.collate_fn([self.dataset[index]])
            # The preprocessor runs where Lightning hands the batch to the model.
            batch_inputs = model.on_after_batch_transfer(batch, dataloader_idx=0)
            gt_batch = batch_inputs.multi_task_gt_batch
            frame_meta = gt_batch.frame_meta_batch
            assert frame_meta is not None
            # The augmentation of the frame is folded into its pose for the memory warp.
            self.assertFalse(torch.equal(frame_meta.ego2globals[0], torch.eye(4)))
            assert gt_batch.detection2d_gt_batch is not None
            assert gt_batch.detection3d_gt_batch is not None
            statuses.append(gt_batch.detection3d_gt_batch.gt_traffic_cone_barrier_bbox_status)

            loss = model.training_step(batch_inputs, batch_idx=0)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

        # Scenario b starts the lane and its second frame continues the memory; the corpus
        # stores every record under one scene directory, so scenario a, whose frames come
        # earlier in time, resets the memory by its time.
        self.assertEqual(
            continuity,
            [
                ([SCENE], [20.0], [0.0]),
                ([SCENE], [21.0], [1.0]),
                ([SCENE], [10.0], [0.0]),
                ([SCENE], [10.5], [1.0]),
            ],
        )
        # The annotation flag of the record reaches the batch, where the partial annotations
        # read it.
        self.assertEqual(
            [status.tolist() for status in statuses], [[True], [True], [False], [False]]
        )
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.img_roi_head.parameters())
        )

    def test_evaluation_decodes_the_streamed_frames(self) -> None:
        model = build_model().eval()
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False, seed=0)
        for index in sampler:
            batch = self.dataset.collate_fn([self.dataset[index]])
            with torch.no_grad():
                batch_inputs = model.on_after_batch_transfer(batch, dataloader_idx=0)
                outputs = model(batch_inputs)
                predictions = model.decode_outputs(batch_inputs, outputs)
            self.assertIsNone(outputs.detection3d().streampetr_head().auxiliary_2d)
            self.assertEqual(len(predictions.detection3d_predictions), 1)
