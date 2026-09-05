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

"""PTv3 stage-graph declarations: name coverage between glue, encoder, and heads.

The declarations are checked with stub submodules — the load-bearing risk is the
tensor-name contract (every graph input must be produced by an earlier stage),
which is independent of the real modules.
"""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from autoware_ml.deployment.stages import graph_stages, validate_stages
from autoware_ml.models.segmentation3d.main_modules.ptv3.stages import (
    DET_HEAD_STAGE,
    ENCODER_STAGE,
    SEG_HEAD_STAGE,
    SERIALIZE_STAGE,
    build_ptv3_det_stages,
    build_ptv3_seg_stages,
    encoder_input_names,
    serialize_output_names,
)

_NUM_POOLINGS = 3  # -> stage_count 4


class _StubHead(nn.Module):
    def __init__(self, dec_depths: tuple[int, ...]) -> None:
        super().__init__()
        self.dec_depths = dec_depths

    def prepare_for_export(self, order=None):  # noqa: ANN001 - mirrors the model API
        return self


class _StubBBoxHead(nn.Module):
    def prepare_for_export(self):
        return self


def _stub_model(task: str) -> SimpleNamespace:
    model = SimpleNamespace(
        encoder=SimpleNamespace(stride=(2,) * _NUM_POOLINGS),
        point_cloud_range=[-10.0, -10.0, -3.0, 10.0, 10.0, 5.0],
        grid_size=0.5,
        EXPORT_ORDER=("z", "z-trans"),
        _prepare_encoder_export=lambda: nn.Identity(),
    )
    if task == "seg":
        model.seg3d_head = _StubHead(dec_depths=(1, 0, 1))
        model.get_export_output_names = lambda: ["pred_labels", "pred_probs"]
    else:
        model.bev_neck = nn.Identity()
        model.bbox_head = _StubBBoxHead()
        model.get_export_output_names = lambda: ["heatmap", "reg", "height", "dim", "rot"]
    return model


def _assert_every_graph_input_is_produced(stages) -> None:
    available = serialize_output_names(_NUM_POOLINGS)
    for graph in graph_stages(stages):
        missing = set(graph.inputs) - available
        assert not missing, f"stage {graph.name!r} reads unproduced tensors: {sorted(missing)}"
        available |= set(graph.outputs)


def test_seg_declaration_is_valid_and_name_covered() -> None:
    stages = validate_stages(build_ptv3_seg_stages(_stub_model("seg")))
    assert [s.name for s in stages] == [SERIALIZE_STAGE, ENCODER_STAGE, SEG_HEAD_STAGE]
    _assert_every_graph_input_is_produced(stages)


def test_det_declaration_is_valid_and_name_covered() -> None:
    stages = validate_stages(build_ptv3_det_stages(_stub_model("det")))
    assert [s.name for s in stages] == [SERIALIZE_STAGE, ENCODER_STAGE, DET_HEAD_STAGE]
    _assert_every_graph_input_is_produced(stages)


def test_encoder_inputs_follow_the_legacy_name_rule() -> None:
    names = encoder_input_names(_NUM_POOLINGS)
    assert names[:3] == ["grid_coord", "feat", "serialized_code"]
    assert all(name.startswith("serialized_pooling_") for name in names[3:])
    # cluster is head-side only and must NOT be an encoder input.
    assert not any(name.endswith("_cluster") for name in names)
