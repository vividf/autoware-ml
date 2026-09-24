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

"""BEVFusion lidar stage graph: declaration validity, ABI names, the ONNX fallback, the
rulebook-precompute opt-in, and the packed-output decode's agreement with the head."""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from autoware_ml.deployment.stages import GraphStage, TorchStage, validate_stages
from autoware_ml.models.detection3d.bevfusion import (
    BEVFusionDetectionModel,
    decode_packed_detections,
)
from autoware_ml.models.detection3d.encoders.sparse import SparseEncoder
from autoware_ml.models.detection3d.heads.transfusion import TransFusionHead
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.ops.spconv.onnx_fusion import fuse_sparse_graph
from autoware_ml.ops.spconv.rulebook import rulebook_input_names
from autoware_ml.types.backend import Backend


class _FakeVoxelEncoder(nn.Module):
    def forward(self, voxels, num_points, voxel_coords):
        return voxels.mean(dim=1)


def _lidar_model(middle_encoder: nn.Module) -> BEVFusionDetectionModel:
    return BEVFusionDetectionModel(
        pts_voxel_encoder=_FakeVoxelEncoder(),
        pts_middle_encoder=middle_encoder,
        pts_backbone=nn.Identity(),
        pts_neck=nn.Identity(),
        bbox_head=nn.Identity(),
    )


def _sparse_encoder(**kwargs) -> SparseEncoder:
    return SparseEncoder(
        in_channels=4,
        sparse_shape=(16, 16, 8),
        base_channels=4,
        encoder_channels=((4, 4, 8), (8, 8)),
        encoder_paddings=((1, 1, 1), (1, 1)),
        output_channels=8,
        dense_output_shapes=(16, 16, 1),
        **kwargs,
    ).eval()


@pytest.mark.skipif(not IS_SPCONV_AVAILABLE, reason="BEVFusion sparse encoder requires spconv")
def test_lidar_declaration_keeps_the_runtime_module_and_falls_back_on_onnx() -> None:
    model = _lidar_model(_sparse_encoder())
    stages = validate_stages(model.build_stages())
    assert [type(stage) for stage in stages] == [TorchStage, GraphStage]
    graph = stages[1]
    assert graph.name == "bevfusion_lidar"
    assert graph.inputs == ("voxels", "coors", "num_points_per_voxel")
    assert graph.outputs == ("bbox_pred", "score", "label_pred")
    assert graph.output_fields == tuple((n, n) for n in graph.outputs)
    assert set(graph.onnx_dynamic_axes) == set(graph.inputs)
    # TensorRT executes the plugin ops (deploy.tensorrt.plugin_libraries); ONNX Runtime has
    # no implementation for them, so only that backend falls back to PyTorch.
    assert graph.torch_fallback_backends == (Backend.ONNX,)
    # The bias/ReLU fold into the plugin nodes is part of the declaration.
    assert fuse_sparse_graph in graph.onnx_transforms
    assert model.verification_caveat


@pytest.mark.skipif(not IS_SPCONV_AVAILABLE, reason="BEVFusion sparse encoder requires spconv")
def test_rulebook_precompute_is_an_opt_in_that_adds_a_glue_stage_and_graph_inputs() -> None:
    encoder = _sparse_encoder(export_precompute_rulebooks=True)
    model = _lidar_model(encoder)
    stages = validate_stages(model.build_stages())
    assert [stage.name for stage in stages] == [
        "fetch_voxels",
        "precompute_rulebooks",
        "bevfusion_lidar",
    ]
    graph = stages[2]
    expected = rulebook_input_names(encoder.downsample_stages())
    assert expected and graph.inputs[3:] == expected
    assert set(graph.onnx_dynamic_axes) == set(graph.inputs)


def test_camera_lidar_models_keep_their_hand_written_export_specs() -> None:
    model = _lidar_model(nn.Identity())
    model.view_transform = nn.Identity()  # any non-None image branch
    assert model.build_stages() is None


def _coder() -> TransFusionBBoxCoder:
    return TransFusionBBoxCoder(
        pc_range=[-10.0, -10.0],
        out_size_factor=2,
        voxel_size=[0.5, 0.5],
        post_center_range=[-100.0, -100.0, -100.0, 100.0, 100.0, 100.0],
        score_threshold=0.1,
        code_size=10,
    )


def _packed_channels() -> torch.Tensor:
    """Two proposals in the runtime's packed channel layout; the second scores low."""
    return torch.tensor(
        [
            [4.0, 8.0],  # center x (grid)
            [6.0, 2.0],  # center y (grid)
            [1.0, 0.0],  # height
            [0.0, 0.0],  # dim log l
            [0.0, 0.0],  # dim log w
            [0.0, 0.0],  # dim log h
            [1.0, 0.0],  # rot sin
            [0.0, 1.0],  # rot cos
            [0.5, 0.0],  # vel x
            [0.25, 0.0],  # vel y
        ]
    )


def _head_stub() -> SimpleNamespace:
    head = SimpleNamespace(num_classes=3, bbox_coder=_coder(), nms_type=None)
    # Borrow the head's real post-processing rather than restating it here.
    head.decode_detections = types.MethodType(TransFusionHead.decode_detections, head)
    return head


def test_packed_decode_applies_coder_math_and_score_filter() -> None:
    outputs = {
        "bbox_pred": _packed_channels(),
        "score": torch.tensor([0.9, 0.05]),
        "label_pred": torch.tensor([1.0, 2.0]),
    }
    detections = decode_packed_detections(_head_stub(), outputs)
    assert len(detections) == 1
    sample = detections[0]
    assert sample["scores_3d"].tolist() == [torch.tensor(0.9).item()]
    assert sample["labels_3d"].tolist() == [1]
    box = sample["bboxes_3d"][0]
    assert box[0].item() == 4.0 * 2 * 0.5 - 10.0  # metric x
    assert box[1].item() == 6.0 * 2 * 0.5 - 10.0  # metric y
    assert abs(box[2].item() - 0.5) < 1e-6  # height - h/2 (dim exp(0)=1)
    assert abs(box[7].item() - 0.5) < 1e-6 and abs(box[8].item() - 0.25) < 1e-6


def test_packed_decode_matches_the_head_on_the_same_proposals() -> None:
    """The deployed path and the PyTorch path must produce the same detections."""
    head = _head_stub()
    channels = _packed_channels()
    scores = torch.tensor([0.9, 0.4])
    labels = torch.tensor([1, 2])
    score_matrix = torch.zeros((1, 3, 2))
    score_matrix[0, labels, torch.arange(2)] = scores
    from_head = head.decode_detections(
        score_matrix,
        channels[6:8].unsqueeze(0),
        channels[3:6].unsqueeze(0),
        channels[0:2].unsqueeze(0),
        channels[2:3].unsqueeze(0),
        channels[8:10].unsqueeze(0),
    )[0]
    from_packed = decode_packed_detections(
        head, {"bbox_pred": channels, "score": scores, "label_pred": labels.float()}
    )[0]
    for key in ("bboxes_3d", "scores_3d", "labels_3d"):
        assert torch.equal(from_head[key], from_packed[key]), key
