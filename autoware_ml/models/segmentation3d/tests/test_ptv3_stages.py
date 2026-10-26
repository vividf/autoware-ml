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

"""Unit tests for the PTv3 stage graphs."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch
from omegaconf import OmegaConf

from autoware_ml.dataclasses.models.detection3d.head_outputs import TransFusionHeadOutputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.deployment.stages import (
    TorchStage,
    model_output_names,
    run_stages_in_torch,
    validate_stages,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_seg3d_model,
    build_trans_model,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET3D_HEAD_STAGE,
    ENCODER_EXPORT_INPUT_NAMES,
    ENCODER_STAGE,
    SEG3D_HEAD_OUTPUT_NAMES,
    SEG3D_HEAD_STAGE,
    SERIALIZE_STAGE,
    build_seg3d_head_stage,
    encoder_stage_input_names,
    serialize_stage_output_names,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.types.backend import Backend
from autoware_ml.utils.deploy import export_to_onnx

REQUIRES_SPARSE_CUDA = pytest.mark.skipif(
    not (IS_SPCONV_AVAILABLE and torch.cuda.is_available()),
    reason="PTv3 export needs spconv and CUDA",
)


@REQUIRES_SPARSE_CUDA
def test_seg3d_declaration_names_the_runtime_modules_and_is_name_covered() -> None:
    model = build_seg3d_model().cuda().eval()
    stages = validate_stages(model.build_stages())
    assert [s.name for s in stages] == [SERIALIZE_STAGE, ENCODER_STAGE, SEG3D_HEAD_STAGE]
    assert isinstance(stages[0], TorchStage)
    # validate_stages() has checked that an earlier stage writes every graph input.
    assert stages[0].outputs == serialize_stage_output_names(len(model.encoder.stride))
    encoder = stages[1]
    assert encoder.inputs[:5] == ENCODER_EXPORT_INPUT_NAMES
    assert not any(name.endswith("_cluster") for name in encoder.inputs)
    assert encoder.torch_fallback_backends == (Backend.ONNX,)
    assert stages[2].outputs == ("pred_labels", "pred_probs")


@REQUIRES_SPARSE_CUDA
def test_seg3d_export_specs_are_derived_and_keep_the_split_contract() -> None:
    model = build_seg3d_model().cuda().eval()
    batch = build_inputs(device=torch.device("cuda"))
    specs = model.build_export_specs(batch)
    assert list(specs) == ["ptv3_encoder", "ptv3_seg3d_head"]
    encoder_spec = specs["ptv3_encoder"]
    assert encoder_spec.input_names == encoder_stage_input_names(len(model.encoder.stride))
    assert "serialized_pooling_0_patch_order" in encoder_spec.input_names
    assert not any("_cluster" in name for name in encoder_spec.input_names)
    assert encoder_spec.dynamic_axes["voxels"] == {0: "num_voxels"}
    for spec in specs.values():
        assert spec.supported_stages == frozenset({"onnx", "tensorrt"})
    with torch.no_grad():
        stage_feats = encoder_spec.module(*encoder_spec.args)
        pred_labels, pred_probs = specs["ptv3_seg3d_head"].module(*specs["ptv3_seg3d_head"].args)
    num_voxels = batch.voxels_data.voxels.shape[0]
    assert len(stage_feats) == len(model.encoder.stride) + 1
    assert pred_labels.shape == (num_voxels,)
    assert pred_probs.shape[0] == num_voxels


@REQUIRES_SPARSE_CUDA
def test_derived_specs_export_onnx_with_the_declared_io_contract(tmp_path: Path) -> None:
    # forward() is no numeric reference: it reshuffles the serialization orders and adapts the
    # attention windows, which the export copy pins.
    model = build_seg3d_model().cuda().eval()
    batch = build_inputs(device=torch.device("cuda"))
    stages = {stage.name: stage for stage in validate_stages(model.build_stages())}
    onnx_cfg = OmegaConf.create(
        {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
    )
    for name, spec in model.build_export_specs(batch).items():
        path = tmp_path / f"{name}.onnx"
        export_to_onnx(spec, onnx_cfg, path)
        graph = onnx.load(path).graph
        assert [tensor.name for tensor in graph.input] == list(stages[name].inputs)
        assert [tensor.name for tensor in graph.output] == list(stages[name].outputs)
        for tensor in graph.input:
            axes = stages[name].onnx_dynamic_axes.get(tensor.name, {})
            dims = tensor.type.tensor_type.shape.dim
            assert {i: dims[i].dim_param for i in axes} == dict(axes), tensor.name


@REQUIRES_SPARSE_CUDA
def test_seg3d_head_stage_emits_the_head_graph_names() -> None:
    model = build_seg3d_model().cuda().eval()
    stage = build_seg3d_head_stage(model, model.seg3d_head)
    assert stage.outputs == SEG3D_HEAD_OUTPUT_NAMES == ("pred_labels", "pred_probs")
    assert set(stage.onnx_dynamic_axes) >= set(SEG3D_HEAD_OUTPUT_NAMES)


def test_deployed_eval_output_scores_the_graph_outputs_like_the_logits() -> None:
    torch.manual_seed(0)
    model = build_seg3d_model().eval()
    batch = build_inputs()
    logits = torch.randn(batch.voxels_data.voxels.shape[0], 3)
    reference = model.build_eval_output(
        batch, ModelOutputs(segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=logits))
    )
    staged = model.build_deployed_eval_output(
        batch, {"pred_labels": logits.argmax(dim=1), "pred_probs": logits.softmax(dim=1)}
    )
    assert len(reference["seg3d_frames"]) == len(staged["seg3d_frames"]) == 1
    for ref_frame, got_frame in zip(reference["seg3d_frames"], staged["seg3d_frames"], strict=True):
        assert set(ref_frame) == set(got_frame)
        for key, ref_value in ref_frame.items():
            got_value = got_frame[key]
            if isinstance(ref_value, torch.Tensor):
                torch.testing.assert_close(got_value, ref_value)
            else:
                assert got_value == ref_value
    # The sweep point (time lag != 0) is not scored: 7 of the 8 fixture points.
    assert staged["seg3d_frames"][0]["pred"].shape[0] == 7


@REQUIRES_SPARSE_CUDA
def test_det3d_declaration_specs_and_deployed_eval_output() -> None:
    model = build_trans_model().cuda().eval()
    stages = validate_stages(model.build_stages())
    assert [s.name for s in stages] == [SERIALIZE_STAGE, ENCODER_STAGE, DET3D_HEAD_STAGE]
    batch = build_inputs(device=torch.device("cuda"))
    specs = model.build_export_specs(batch)
    assert list(specs) == ["ptv3_encoder", "ptv3_det3d_head"]
    assert specs["ptv3_det3d_head"].output_names == list(model.export_output_names)

    state = run_stages_in_torch(stages, batch, torch.device("cuda"))
    outputs = {name: state[name] for name in model_output_names(stages)}
    rebuilt = TransFusionHeadOutputs.from_export_tensors(outputs)
    for name, tensor in zip(outputs, rebuilt.export_tensors(list(outputs))):
        assert torch.equal(tensor, outputs[name]), name
    deployed = model.build_deployed_eval_output(batch, outputs)
    assert {"predictions", "gt_boxes", "gt_labels"} <= deployed.keys()
    assert len(deployed["predictions"]) == 1
