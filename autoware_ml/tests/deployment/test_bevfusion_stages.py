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

"""BEVFusion split stage graph: declaration validity, ABI names, fallback flags,
and the per-stage fallback / external-onnx framework semantics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import torch
from torch import nn

from autoware_ml.deployment.config import DeployConfig
from autoware_ml.deployment.export import available_backends
from autoware_ml.deployment.pipeline import StagedPipeline, _ModuleRunner
from autoware_ml.deployment.stages import GraphStage, TorchStage, validate_stages
from autoware_ml.models.detection3d.main_modules.bevfusion.stages import (
    DENSE_STAGE,
    LIDAR_BEV,
    SPARSE_STAGE,
    assemble_bevfusion_outputs,
    build_bevfusion_lidar_stages,
    decode_packed_detections,
)
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.types.backend import Backend


def _stub_model() -> SimpleNamespace:
    return SimpleNamespace(
        pts_voxel_encoder=nn.Identity(),
        pts_middle_encoder=nn.Identity(),
        pts_backbone=nn.Identity(),
        pts_neck=nn.Identity(),
        bbox_head=nn.Identity(),
    )


def test_declaration_is_valid_with_the_awml_split_abi() -> None:
    stages = validate_stages(build_bevfusion_lidar_stages(_stub_model()))
    sparse, dense = stages[1], stages[2]
    assert sparse.name == SPARSE_STAGE and dense.name == DENSE_STAGE
    assert sparse.inputs == ("voxels", "coors", "num_points_per_voxel")
    assert sparse.outputs == (LIDAR_BEV,) and dense.inputs == (LIDAR_BEV,)
    assert dense.outputs == ("bbox_pred", "score", "label_pred")
    # TensorRT executes the sparse graph's plugin ops (deploy.tensorrt.plugin_libraries);
    # ONNX Runtime has no implementation for them, so only that backend falls back.
    assert sparse.torch_fallback_backends == (Backend.ONNX,)
    # Both graphs are exported by the framework; the sparse one needs the runtime's
    # spconv plugin to execute, not a separate exporter.
    assert sparse.external_onnx is False and dense.external_onnx is False


def _fallback_test_stages() -> tuple:
    """A minimal two-graph declaration exercising the fallback fields."""

    def seed(context):
        return {"x": torch.ones(1, 2)}

    sparse_like = GraphStage(
        "sparse_like",
        module=nn.Identity(),
        inputs=("x",),
        outputs=("mid",),
        torch_fallback_backends=(Backend.ONNX,),
        external_onnx=True,
    )
    dense_like = GraphStage(
        "dense_like",
        module=nn.Identity(),
        inputs=("mid",),
        outputs=("y",),
        output_fields=(("y", "y"),),
    )
    return (TorchStage("seed", run=seed), sparse_like, dense_like)


def test_fallback_stage_uses_the_torch_module_on_its_fallback_backend(tmp_path) -> None:
    import onnx
    from onnx import TensorProto, helper

    # Only the dense-like stage needs an ONNX artifact on the onnx backend.
    x = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["mid"], ["y"])], "dense_like", [x], [y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(tmp_path / "dense_like.onnx"))

    pipeline = StagedPipeline(
        _fallback_test_stages(),
        backend=Backend.ONNX,
        device=torch.device("cpu"),
        artifacts_dir=tmp_path,
    )
    assert isinstance(pipeline._runners["sparse_like"], _ModuleRunner)
    assert not isinstance(pipeline._runners["dense_like"], _ModuleRunner)


def test_available_backends_exempts_fallback_stages_from_artifacts(tmp_path) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["mid"], ["y"])], "dense_like", [x], [y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(tmp_path / "dense_like.onnx"))

    available = available_backends(_fallback_test_stages(), tmp_path)
    # onnx is available without sparse_like.onnx (fallback); tensorrt is not
    # (no engines, and sparse_like has no tensorrt fallback).
    assert Backend.ONNX in available and Backend.TENSORRT not in available


def test_export_skips_engines_for_tensorrt_fallback_stages(tmp_path, monkeypatch) -> None:
    """A stage TensorRT cannot execute must not have an engine built for it.

    The sparse stage exports an ONNX full of runtime plugin ops; building an engine
    from it fails (the plugin is not registered) and the pipeline would never use it,
    because the stage runs in PyTorch on the tensorrt backend.
    """
    from autoware_ml.deployment import export as export_module

    built: list[str] = []
    monkeypatch.setattr(
        export_module,
        "build_engine",
        lambda onnx_path, engine_path, **kwargs: built.append(Path(onnx_path).stem),
    )

    plugin_stage = GraphStage(
        "plugin_like",
        module=nn.Identity(),
        inputs=("x",),
        outputs=("y",),
        torch_fallback_backends=(Backend.ONNX, Backend.TENSORRT),
    )
    plain_stage = GraphStage(
        "plain_like",
        module=nn.Identity(),
        inputs=("y",),
        outputs=("z",),
        output_fields=(("z", "z"),),
    )

    def seed(context):
        return {"x": torch.ones(1, 2)}

    deploy_cfg = DeployConfig.from_dict(
        OmegaConf.create(
            {
                "onnx": {"enabled": True, "dynamo": False, "opset_version": 17},
                "tensorrt": {"enabled": True},
                "stages": {},
            }
        )
    )
    export_module.export_stages(
        (TorchStage("seed", run=seed), plugin_stage, plain_stage),
        batch_inputs=None,
        deploy_cfg=deploy_cfg,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert built == ["plain_like"]
    # The ONNX is still written for the fallback stage: it is the deployed artifact.
    assert (tmp_path / "plugin_like.onnx").exists()


def test_assemble_wraps_packed_runtime_tensors() -> None:
    outputs = {
        "bbox_pred": torch.zeros(10, 4),
        "score": torch.zeros(4),
        "label_pred": torch.zeros(4),
    }
    assembled = assemble_bevfusion_outputs(outputs)
    packed = assembled.detection3d_head_outputs.transfusion_packed_detections
    assert packed is not None and packed.bbox_pred.shape == (10, 4)
    assert assembled.detection3d_head_outputs.transfusion_head_outputs is None


def test_packed_decode_applies_coder_math_and_score_filter() -> None:
    coder = TransFusionBBoxCoder(
        pc_range=[-10.0, -10.0],
        out_size_factor=2,
        voxel_size=[0.5, 0.5],
        post_center_range=[-100.0, -100.0, -100.0, 100.0, 100.0, 100.0],
        score_threshold=0.1,
        code_size=10,
    )
    head = SimpleNamespace(num_classes=3, bbox_coder=coder, nms_type=None)
    # Two proposals: the second falls below the score threshold.
    bbox_pred = torch.tensor(
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
    packed = SimpleNamespace(
        bbox_pred=bbox_pred,
        score=torch.tensor([0.9, 0.05]),
        label_pred=torch.tensor([1.0, 2.0]),
    )
    boxes, scores, labels = decode_packed_detections(head, packed)
    assert scores.tolist() == [torch.tensor(0.9).item()]
    assert labels.tolist() == [1]
    box = boxes[0]
    assert box[0].item() == 4.0 * 2 * 0.5 - 10.0  # metric x
    assert box[1].item() == 6.0 * 2 * 0.5 - 10.0  # metric y
    assert abs(box[2].item() - 0.5) < 1e-6  # height - h/2 (dim exp(0)=1)
    assert abs(box[6].item() - torch.atan2(torch.tensor(1.0), torch.tensor(0.0)).item()) < 1e-6
    assert abs(box[7].item() - 0.5) < 1e-6 and abs(box[8].item() - 0.25) < 1e-6
