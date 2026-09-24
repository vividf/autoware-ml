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

"""PTv3 stage graphs: declaration validity, the export contract they derive, and parity of
the staged pipeline with ``forward()``."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.deployment.stages import (
    GraphStage,
    TorchStage,
    run_stages_in_torch,
    validate_stages,
)
from autoware_ml.models.detection3d.tests.ptv3_detection_fixtures import (
    build_inputs,
    build_seg_model,
    build_trans_model,
    move_batch_to_device,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET_HEAD_STAGE,
    ENCODER_STAGE,
    SEG_HEAD_STAGE,
    SERIALIZE_STAGE,
    build_ptv3_export_context,
    build_seg_head_export_spec,
    encoder_stage_input_names,
    serialize_stage_output_names,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.types.backend import Backend

REQUIRES_SPARSE_CUDA = pytest.mark.skipif(
    not (IS_SPCONV_AVAILABLE and torch.cuda.is_available()),
    reason="PTv3 export needs spconv and CUDA",
)


def _assert_every_graph_input_is_produced(stages, num_poolings: int) -> None:
    available = serialize_stage_output_names(num_poolings)
    for stage in stages:
        if isinstance(stage, GraphStage):
            missing = set(stage.inputs) - available
            assert not missing, f"stage {stage.name!r} reads unproduced tensors: {sorted(missing)}"
            available |= set(stage.outputs)


@REQUIRES_SPARSE_CUDA
def test_seg_declaration_names_the_runtime_modules_and_is_name_covered() -> None:
    model = build_seg_model().cuda().eval()
    stages = validate_stages(model.build_stages())
    assert [s.name for s in stages] == [SERIALIZE_STAGE, ENCODER_STAGE, SEG_HEAD_STAGE]
    assert isinstance(stages[0], TorchStage)
    num_poolings = len(model.encoder.stride)
    _assert_every_graph_input_is_produced(stages, num_poolings)
    encoder = stages[1]
    assert encoder.inputs[:4] == ("grid_coord", "feat", "serialized_order", "serialized_inverse")
    assert not any(name.endswith("_cluster") for name in encoder.inputs)
    assert encoder.torch_fallback_backends == (Backend.ONNX,)
    assert stages[2].outputs == ("pred_labels", "pred_probs")


@REQUIRES_SPARSE_CUDA
def test_seg_export_specs_are_derived_and_keep_the_split_contract() -> None:
    model = build_seg_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))
    specs = model.build_export_specs(batch)
    assert list(specs) == ["ptv3_encoder", "ptv3_seg3d_head"]
    encoder_spec = specs["ptv3_encoder"]
    assert encoder_spec.input_param_names == encoder_stage_input_names(len(model.encoder.stride))
    assert "serialized_pooling_0_serialized_order" in encoder_spec.input_param_names
    assert not any("_cluster" in name for name in encoder_spec.input_param_names)
    assert "grid_coord" in encoder_spec.dynamic_axes
    with torch.no_grad():
        stage_feats = encoder_spec.module(*encoder_spec.args)
        pred_labels, pred_probs = specs["ptv3_seg3d_head"].module(*specs["ptv3_seg3d_head"].args)
    assert len(stage_feats) == len(model.encoder.stride) + 1
    assert pred_labels.shape == (batch["coord"].shape[0],)
    assert pred_probs.shape[0] == batch["coord"].shape[0]


@REQUIRES_SPARSE_CUDA
def test_staged_pipeline_reproduces_the_hand_built_export_path() -> None:
    """The stage graph runs the same export modules on the same tensors as the export
    context does, so the staged pipeline and the hand-built split export agree exactly.

    (``forward()`` itself is not the reference: it re-shuffles serialization orders and
    uses adaptive attention windows, which the export copy deliberately pins.)
    """
    model = build_seg_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))
    context = build_ptv3_export_context(model, batch)
    head_spec = build_seg_head_export_spec(
        context,
        model.seg3d_head.prepare_for_export(model.EXPORT_ORDER),
        model.get_export_output_names(),
    )
    with torch.no_grad():
        pred_labels, pred_probs = head_spec.module(*head_spec.args)
    staged = run_stages_in_torch(model.build_stages(), batch, torch.device("cuda"))
    torch.testing.assert_close(staged["pred_probs"], pred_probs)
    assert torch.equal(staged["pred_labels"], pred_labels)
    for index, feat in enumerate(context.stage_feats):
        torch.testing.assert_close(staged[f"point_feat_{index}"], feat)


@REQUIRES_SPARSE_CUDA
def test_det_declaration_and_specs() -> None:
    model = build_trans_model().cuda().eval()
    stages = validate_stages(model.build_stages())
    assert [s.name for s in stages] == [SERIALIZE_STAGE, ENCODER_STAGE, DET_HEAD_STAGE]
    _assert_every_graph_input_is_produced(stages, len(model.encoder.stride))
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))
    specs = model.build_export_specs(batch)
    assert list(specs) == ["ptv3_encoder", "ptv3_det3d_head"]
    assert specs["ptv3_det3d_head"].output_names == list(model.export_output_names)
