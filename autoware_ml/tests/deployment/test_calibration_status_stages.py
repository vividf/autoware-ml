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

"""Calibration-status stage graph: declaration validity, wrapper math, ONNX export."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from autoware_ml.deployment.stages import StageContext, validate_stages
from autoware_ml.models.calibration_status.main_modules.calibration_status.stages import (
    CLASSIFIER_STAGE,
    FUSED_IMAGE,
    PROBABILITIES,
    build_calibration_status_stages,
)


class _TinyHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)


def _tiny_model() -> SimpleNamespace:
    torch.manual_seed(0)
    return SimpleNamespace(
        backbone=nn.Sequential(nn.Conv2d(5, 8, 3, stride=2, padding=1), nn.ReLU()),
        neck=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()),
        head=_TinyHead(8, 2),
    )


def test_stage_declaration_is_valid_and_named_for_the_runtime_abi() -> None:
    stages = validate_stages(build_calibration_status_stages(_tiny_model()))
    graph = stages[-1]
    assert graph.name == CLASSIFIER_STAGE
    assert graph.inputs == (FUSED_IMAGE,) and graph.outputs == (PROBABILITIES,)
    assert graph.output_fields == ((PROBABILITIES, "calibration_probabilities"),)


def test_wrapper_matches_submodule_composition_and_returns_probabilities() -> None:
    model = _tiny_model()
    stages = build_calibration_status_stages(model)
    wrapper = stages[-1].module.eval()

    x = torch.randn(2, 5, 16, 16)
    with torch.no_grad():
        out = wrapper(x)
        expected = model.head.predict(model.head(model.neck(model.backbone(x))))
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(2))  # probabilities, not logits


def test_fetch_glue_reads_fused_img_from_batch_inputs() -> None:
    stages = build_calibration_status_stages(_tiny_model())
    fetch = stages[0]
    context = StageContext(
        batch_inputs=SimpleNamespace(fused_img=torch.randn(1, 5, 8, 8)),
        device=torch.device("cpu"),
    )
    produced = fetch.run(context)
    assert set(produced) == {FUSED_IMAGE}
    assert produced[FUSED_IMAGE].shape == (1, 5, 8, 8)


def test_graph_exports_to_onnx_with_the_abi_names(tmp_path) -> None:
    import onnx

    from autoware_ml.deployment.onnx.export import export_to_onnx

    stages = build_calibration_status_stages(_tiny_model())
    graph = stages[-1]
    path = tmp_path / f"{graph.name}.onnx"
    export_to_onnx(
        graph.module.eval(),
        (torch.randn(1, 5, 16, 16),),
        path,
        input_names=list(graph.inputs),
        output_names=list(graph.outputs),
        opset_version=16,
        dynamo=False,
        dynamic_axes={
            FUSED_IMAGE: {0: "batch_size", 2: "height", 3: "width"},
            PROBABILITIES: {0: "batch_size"},
        },
    )
    model = onnx.load(str(path))
    assert [i.name for i in model.graph.input] == [FUSED_IMAGE]
    assert [o.name for o in model.graph.output] == [PROBABILITIES]
