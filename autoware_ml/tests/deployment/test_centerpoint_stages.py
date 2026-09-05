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

"""CenterPoint stage graph: the hand-written ``forward`` and the staged pytorch run agree.

``forward`` is what trains; the stage graph is what deploys. This pins the two to each
other so neither can drift silently, and checks the derived export contract (stage /
artifact names, ONNX I/O names, output-field table) against the frozen ABI.
"""

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTBatch,
    PointCloudGTBatch,
)
from autoware_ml.deployment.pipeline import StagedPipeline
from autoware_ml.deployment.stages import graph_stages, validate_stages
from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.encoders.pillars.pillar_feature_net import PillarFeatureNet
from autoware_ml.models.detection3d.encoders.pillars.point_pillar_scatter import (
    PointPillarsScatter,
)
from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.models.detection3d.main_modules.centerpoint import CenterPointDetectionModel
from autoware_ml.models.detection3d.main_modules.centerpoint.stages import (
    BACKBONE_NECK_HEAD_STAGE,
    VOXEL_ENCODER_STAGE,
)
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.models.multi_task_base_model import LogDictConfigs
from autoware_ml.ops.voxelization.voxelization import VoxelsData
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

_POINT_CLOUD_RANGE = [-10.0, -10.0, -3.0, 10.0, 10.0, 5.0]
_VOXEL_SIZE = [0.5, 0.5, 8.0]
_GRID = 40  # (10 - -10) / 0.5


def _tiny_centerpoint(use_velocity: bool = True) -> CenterPointDetectionModel:
    torch.manual_seed(0)
    return CenterPointDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        pts_voxel_encoder=PillarFeatureNet(
            in_channels=5,
            feat_channels=[8, 8],
            voxel_size=_VOXEL_SIZE,
            point_cloud_range=_POINT_CLOUD_RANGE,
        ),
        pts_middle_encoder=PointPillarsScatter(in_channels=8, output_shape=[_GRID, _GRID]),
        pts_backbone=SECONDBackbone(
            in_channels=8,
            out_channels=[16, 32],
            layer_nums=[1, 1],
            layer_strides=[1, 2],
            activation_checkpointing=False,
        ),
        pts_neck=SECONDFPN(
            in_channels=[16, 32],
            out_channels=[16, 16],
            upsample_strides=[1, 2],
            activation_checkpointing=False,
        ),
        bbox_head=CenterHead(
            in_channels=32,
            class_names=["car", "pedestrian"],
            shared_channels=8,
            point_cloud_range=_POINT_CLOUD_RANGE,
            voxel_size=_VOXEL_SIZE,
            out_size_factor=2,
            min_radius=2,
            score_threshold=0.0,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=use_velocity,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
    ).eval()


def _batch(
    batch_size: int = 2, pillars_per_sample: int = 12, max_points: int = 6
) -> MultiTaskBatchInputs:
    torch.manual_seed(1)
    num_pillars = batch_size * pillars_per_sample
    batch_indices = torch.arange(batch_size, dtype=torch.int32).repeat_interleave(
        pillars_per_sample
    )
    coords = torch.stack(
        [
            torch.randint(0, _GRID, (num_pillars,), dtype=torch.int32),
            torch.randint(0, _GRID, (num_pillars,), dtype=torch.int32),
            torch.zeros(num_pillars, dtype=torch.int32),
        ],
        dim=1,
    )
    # Points inside their pillar: (x, y, z, intensity, time)
    centers_xy = (coords[:, :2].float() + 0.5) * torch.tensor(_VOXEL_SIZE[:2]) + torch.tensor(
        _POINT_CLOUD_RANGE[:2]
    )
    xy = centers_xy[:, None, :] + (torch.rand(num_pillars, max_points, 2) - 0.5) * 0.4
    z = torch.rand(num_pillars, max_points, 1) * 2.0 - 1.0
    rest = torch.rand(num_pillars, max_points, 2)
    voxels = torch.cat([xy, z, rest], dim=-1)
    num_points = torch.randint(1, max_points + 1, (num_pillars,), dtype=torch.int32)
    mask = torch.arange(max_points)[None, :] < num_points[:, None]
    voxels = voxels * mask[..., None]
    voxels_data = VoxelsData(
        voxels=voxels, coords=coords, num_points=num_points, batch_indices=batch_indices
    )
    points = voxels.reshape(-1, 5)
    point_batch = batch_indices.repeat_interleave(max_points)
    gt_batch = MultiTaskGTBatch(
        point_cloud_gt_batch=PointCloudGTBatch(points=points, batch_indices=point_batch),
        detection3d_gt_batch=None,
    )
    return MultiTaskBatchInputs(multi_task_gt_batch=gt_batch, voxels_data=voxels_data)


class TestCenterPointStages:
    def test_declaration_matches_frozen_abi(self):
        model = _tiny_centerpoint()
        stages = validate_stages(model.build_stages())
        graph = graph_stages(stages)
        assert [s.name for s in graph] == [VOXEL_ENCODER_STAGE, BACKBONE_NECK_HEAD_STAGE]
        assert graph[0].inputs == ("input_features",) and graph[0].outputs == ("pillar_features",)
        assert graph[1].inputs == ("spatial_features",)
        assert graph[1].outputs == ("heatmap", "reg", "height", "dim", "rot", "vel")
        assert dict(graph[1].output_fields)["heatmap"] == "heatmaps"

    def test_velocity_follows_the_head(self):
        graph = graph_stages(_tiny_centerpoint(use_velocity=False).build_stages())
        assert graph[1].outputs == ("heatmap", "reg", "height", "dim", "rot")

    def test_forward_matches_staged_pytorch_run(self):
        model = _tiny_centerpoint()
        batch = _batch()
        with torch.no_grad():
            reference = model(batch).detection3d_head_outputs.center_head_outputs

        pipeline = StagedPipeline(
            model.build_stages(),
            backend="pytorch",
            device=torch.device("cpu"),
            assemble=model.assemble_outputs,
        )
        result = pipeline.infer(batch)
        staged = pipeline.assemble(result).detection3d_head_outputs.center_head_outputs

        for field in ("heatmaps", "centers", "heights", "dims", "rots", "vels"):
            assert torch.allclose(getattr(reference, field), getattr(staged, field), atol=1e-5), (
                field
            )
        assert result.graph_stage_names == (VOXEL_ENCODER_STAGE, BACKBONE_NECK_HEAD_STAGE)

    def test_export_trace_inputs_come_from_the_context(self):
        model = _tiny_centerpoint()
        batch = _batch()
        pipeline = StagedPipeline(
            model.build_stages(), backend="pytorch", device=torch.device("cpu")
        )
        _, context = pipeline.run(batch)
        # Voxel encoder is traced from decorated pillar features (N, P, 11);
        # the head from the dense BEV canvas (B, C, H, W).
        assert context["input_features"].shape == (24, 6, 11)
        assert context["spatial_features"].shape == (2, 8, _GRID, _GRID)
        for stage in graph_stages(pipeline.stages):
            outputs = stage.module(*(context[name] for name in stage.inputs))
            outputs = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
            assert len(outputs) == len(stage.outputs)

    def test_batch_size_is_derived_from_voxels_not_ground_truth(self):
        model = _tiny_centerpoint()
        batch = _batch(batch_size=3)
        no_gt = MultiTaskBatchInputs(
            multi_task_gt_batch=MultiTaskGTBatch(
                point_cloud_gt_batch=None, detection3d_gt_batch=None
            ),
            voxels_data=batch.voxels_data,
        )
        pipeline = StagedPipeline(
            model.build_stages(), backend="pytorch", device=torch.device("cpu")
        )
        result = pipeline.infer(no_gt)
        assert result.outputs["heatmap"].shape[0] == 3

    def test_no_model_name_leaks_into_the_generic_packages(self):
        """Identifiers and string literals in deployment/evaluation/quantization never name a model.

        Comments and docstrings may point at the reference implementation; code may not.
        """
        import io
        import pathlib
        import tokenize

        root = pathlib.Path(__file__).resolve().parents[2]
        triple_quotes = ('"' * 3, "'" * 3)
        offenders = []
        for package in ("deployment", "evaluation", "quantization"):
            for path in (root / package).rglob("*.py"):
                tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
                for token in tokens:
                    if token.type == tokenize.COMMENT:
                        continue
                    if token.type == tokenize.STRING and token.string.lstrip("rbuRBU").startswith(
                        triple_quotes
                    ):
                        continue  # docstring / block string
                    if "centerpoint" in token.string.lower():
                        offenders.append(
                            f"{path.relative_to(root)}:{token.start[0]}: {token.string!r}"
                        )
        assert offenders == [], "\n".join(offenders)


@pytest.mark.parametrize("field", ["heatmaps", "dims"])
def test_assemble_outputs_are_float_contiguous(field):
    model = _tiny_centerpoint()
    pipeline = StagedPipeline(
        model.build_stages(),
        backend="pytorch",
        device=torch.device("cpu"),
        assemble=model.assemble_outputs,
    )
    outputs = pipeline.assemble(
        pipeline.infer(_batch())
    ).detection3d_head_outputs.center_head_outputs
    tensor = getattr(outputs, field)
    assert tensor.dtype == torch.float32 and tensor.is_contiguous()
