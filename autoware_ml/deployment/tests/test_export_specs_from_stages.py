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

"""``build_stages()`` derives ``build_export_specs()``; models without stages are untouched."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import nn

from autoware_ml.deployment.export_specs import derive_export_specs
from autoware_ml.deployment.stages import GraphStage, Stage, TorchStage
from autoware_ml.models.base import BaseModel
from autoware_ml.types.backend import Backend


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(2) * 2)

    def forward(self, features):
        return self.linear(features)


class _Head(nn.Module):
    def forward(self, bev):
        return bev + 1, bev - 1


class _StagedModel(BaseModel):
    """Two exported graphs with glue between them, declared once as a stage graph."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.head = _Head()

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        bev = self.encoder(points.float()).sum(dim=0, keepdim=True)
        plus, minus = self.head(bev)
        return {"plus": plus, "minus": minus}

    def compute_metrics(self, batch_inputs_dict: Mapping[str, Any], outputs: Any):
        del batch_inputs_dict
        return {"loss": outputs["plus"].sum()}

    def build_stages(self) -> Sequence[Stage]:
        def prep(context):
            return {"features": context.batch["points"].float()}

        def scatter(context):
            return {"bev": context["encoded"].sum(dim=0, keepdim=True)}

        return (
            TorchStage("prep", run=prep),
            GraphStage("encoder", module=self.encoder, inputs=("features",), outputs=("encoded",)),
            TorchStage("scatter", run=scatter),
            GraphStage(
                "head",
                module=self.head,
                inputs=("bev",),
                outputs=("plus", "minus"),
                output_fields=(("plus", "plus"), ("minus", "minus")),
            ),
        )


class _PlainModel(BaseModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0

    def compute_metrics(self, batch_inputs_dict: Mapping[str, Any], outputs: Any):
        del batch_inputs_dict
        return {"loss": outputs.sum()}


def _batch() -> dict[str, torch.Tensor]:
    return {"points": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "unused": torch.zeros(1)}


def test_specs_are_derived_one_per_graph_stage_in_order() -> None:
    model = _StagedModel().eval()
    specs = model.build_export_specs(_batch())

    assert list(specs) == ["encoder", "head"]
    encoder, head = specs["encoder"], specs["head"]
    assert encoder.input_param_names == ["features"]
    assert encoder.output_names == ["encoded"]
    assert head.input_param_names == ["bev"]
    assert head.output_names == ["plus", "minus"]
    assert encoder.module is model.encoder
    assert head.module is model.head
    assert encoder.dynamic_axes is None
    assert encoder.supported_stages == frozenset({"onnx", "tensorrt"})


def test_trace_inputs_are_the_context_tensors_the_glue_produced() -> None:
    model = _StagedModel().eval()
    specs = model.build_export_specs(_batch())

    (features,) = specs["encoder"].args
    assert torch.equal(features, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    (bev,) = specs["head"].args
    # encoder doubles, scatter sums the rows: [[2,4],[6,8]] -> [[8,12]]
    assert torch.equal(bev, torch.tensor([[8.0, 12.0]]))
    # Running the exported modules on their own trace inputs reproduces forward().
    plus, minus = specs["head"].module(*specs["head"].args)
    forward = model(_batch()["points"])
    assert torch.equal(plus, forward["plus"]) and torch.equal(minus, forward["minus"])


def test_stage_declared_axes_and_tensorrt_fallback_land_in_the_spec() -> None:
    stages = (
        TorchStage("seed", run=lambda ctx: {"x": ctx.batch["x"]}),
        GraphStage(
            "points",
            module=nn.Identity(),
            inputs=("x",),
            outputs=("y",),
            output_fields=(("y", "y"),),
            onnx_dynamic_axes={"x": {0: "num_points"}, "y": {0: "num_points"}},
            torch_fallback_backends=(Backend.TENSORRT,),
        ),
    )
    specs = derive_export_specs(stages, {"x": torch.ones(3, 2)}, torch.device("cpu"))
    spec = specs["points"]
    assert spec.dynamic_axes == {"x": {0: "num_points"}, "y": {0: "num_points"}}
    assert spec.supported_stages == frozenset({"onnx"})


def test_stage_onnx_transforms_land_in_the_spec_in_order() -> None:
    def fuse(path):
        return path

    def stamp_scales(path):
        return path

    stages = (
        TorchStage("seed", run=lambda ctx: {"x": ctx.batch["x"]}),
        GraphStage(
            "sparse",
            module=nn.Identity(),
            inputs=("x",),
            outputs=("y",),
            output_fields=(("y", "y"),),
            onnx_transforms=(fuse, stamp_scales),
        ),
    )
    specs = derive_export_specs(stages, {"x": torch.ones(3, 2)}, torch.device("cpu"))
    assert specs["sparse"].onnx_transforms == (fuse, stamp_scales)


def test_models_without_a_stage_graph_keep_the_end_to_end_default() -> None:
    model = _PlainModel()
    assert model.build_stages() is None
    specs = model.build_export_specs({"x": torch.tensor([1.0])})
    assert list(specs) == ["end_to_end"]
    assert specs["end_to_end"].input_param_names == ["x"]


def test_a_stage_reading_an_unproduced_name_fails_with_the_available_names() -> None:
    stages = (
        TorchStage("seed", run=lambda ctx: {"x": ctx.batch["x"]}),
        GraphStage(
            "g", module=nn.Identity(), inputs=("typo",), outputs=("y",), output_fields=(("y", "y"),)
        ),
    )
    with pytest.raises(KeyError, match="available: \\['x'\\]"):
        derive_export_specs(stages, {"x": torch.ones(1)}, torch.device("cpu"))
