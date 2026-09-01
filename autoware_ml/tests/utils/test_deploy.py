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

"""Unit tests for the legacy ExportSpec deploy helpers (delete with utils/deploy.py at Q5)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from autoware_ml.utils.deploy import (
    ExportSpec,
    export_to_onnx,
    get_export_parameter_names,
    supports_export_stage,
)


class _DummyModel(torch.nn.Module):
    def forward(
        self, voxels: torch.Tensor, num_points: torch.Tensor, **kwargs: object
    ) -> torch.Tensor:
        return voxels + num_points.unsqueeze(-1)


def test_get_export_parameter_names_ignores_variadic_parameters() -> None:
    model = _DummyModel()

    assert get_export_parameter_names(model) == ["voxels", "num_points"]


def test_export_to_onnx_prefers_export_spec_output_names(tmp_path: Path) -> None:
    class _SingleOutput(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    output_path = tmp_path / "model.onnx"
    onnx_cfg = OmegaConf.create(
        {
            "opset_version": 17,
            "dynamo": False,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["configured_output"],
        }
    )

    export_to_onnx(
        model=_SingleOutput(),
        input_sample=(torch.ones(2, 3),),
        onnx_cfg=onnx_cfg,
        input_param_names=["input"],
        output_names_override=["exported_output"],
        dynamic_axes_override=None,
        output_path=output_path,
    )

    assert output_path.exists()


def test_export_to_onnx_prefers_export_spec_dynamic_axes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        torch.onnx,
        "export",
        lambda **kwargs: captured_kwargs.update(kwargs),
    )

    dynamic_axes_override = {"input": {0: "export_spec_batch"}}
    onnx_cfg = OmegaConf.create(
        {
            "opset_version": 17,
            "dynamo": False,
            "dynamic_axes": {"input": {0: "configured_batch"}},
        }
    )

    export_to_onnx(
        model=torch.nn.Identity(),
        input_sample=(torch.ones(2, 3),),
        onnx_cfg=onnx_cfg,
        input_param_names=["input"],
        output_names_override=None,
        dynamic_axes_override=dynamic_axes_override,
        output_path=tmp_path / "model.onnx",
    )

    assert captured_kwargs["dynamic_axes"] == dynamic_axes_override


def test_supports_export_stage_uses_export_spec_capabilities() -> None:
    spec = ExportSpec(
        module=_DummyModel(),
        args=(torch.ones(1, 1), torch.ones(1)),
        input_param_names=["voxels", "num_points"],
        supported_stages=frozenset({"onnx"}),
    )

    assert supports_export_stage(spec, "onnx") is True
    assert supports_export_stage(spec, "tensorrt") is False
