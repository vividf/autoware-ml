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

"""Post-export verification end to end on a toy stage-graph model (pytorch vs onnx, CPU)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from autoware_ml.deployment.post_export import run_post_export
from autoware_ml.deployment.stages import GraphStage, Stage, TorchStage
from autoware_ml.models.base import BaseModel


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, features):
        return self.linear(features)


class _Head(nn.Module):
    def forward(self, bev):
        return bev + 1, bev - 1


class _StagedModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.head = _Head()

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        plus, minus = self.head(self.encoder(points.float()))
        return {"plus": plus, "minus": minus}

    def compute_metrics(self, batch_inputs_dict: Mapping[str, Any], outputs: Any):
        del batch_inputs_dict
        return {"loss": outputs["plus"].sum()}

    def build_stages(self) -> Sequence[Stage]:
        return (
            TorchStage("prep", run=lambda ctx: {"features": ctx.batch["points"].float()}),
            GraphStage("encoder", module=self.encoder, inputs=("features",), outputs=("encoded",)),
            GraphStage(
                "head",
                module=self.head,
                inputs=("encoded",),
                outputs=("plus", "minus"),
                output_fields=(("plus", "plus"), ("minus", "minus")),
            ),
        )


class _PlainModel(BaseModel):
    def forward(self, x):
        return x

    def compute_metrics(self, batch_inputs_dict, outputs):
        return {"loss": outputs.sum()}


class _DataModule(L.LightningDataModule):
    def __init__(self, num_batches: int = 3) -> None:
        super().__init__()
        self.points = [torch.randn(4, 2) for _ in range(num_batches)]
        self.setup_calls: list[str] = []

    def setup(self, stage: str) -> None:
        self.setup_calls.append(stage)

    def predict_dataloader(self):
        return DataLoader([{"points": p} for p in self.points], batch_size=None)


def _export_stage_artifacts(model: _StagedModel, tmp_path) -> None:
    specs = model.build_export_specs({"points": torch.randn(4, 2)})
    for name, spec in specs.items():
        torch.onnx.export(
            spec.module,
            spec.args,
            str(tmp_path / f"{name}.onnx"),
            input_names=spec.input_param_names,
            output_names=spec.output_names,
            opset_version=17,
            dynamo=False,
        )


def _cfg(enabled: bool = True, tolerance: float | None = None) -> dict:
    scenario = {
        "ref": {"backend": "pytorch", "device": "cpu"},
        "test": {"backend": "onnx", "device": "cpu"},
    }
    if tolerance is not None:
        scenario["tolerance"] = tolerance
    return {
        "onnx": {"enabled": True},
        "tensorrt": {"enabled": False},
        "verification": {"enabled": enabled, "num_verify_batches": 2, "scenarios": [scenario]},
    }


def test_disabled_sections_do_nothing_even_without_stages(tmp_path) -> None:
    datamodule = _DataModule()
    run_post_export(
        {"onnx": {}, "tensorrt": {}}, _PlainModel(), datamodule, tmp_path, torch.device("cpu")
    )
    assert datamodule.setup_calls == []


def test_enabling_verification_on_a_model_without_stages_is_an_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="build_stages"):
        run_post_export(_cfg(), _PlainModel(), _DataModule(), tmp_path, torch.device("cpu"))


def test_onnx_artifacts_verify_against_the_pytorch_reference(tmp_path, caplog) -> None:
    torch.manual_seed(0)
    model = _StagedModel().eval()
    _export_stage_artifacts(model, tmp_path)
    datamodule = _DataModule()
    with caplog.at_level(logging.INFO):
        run_post_export(_cfg(), model, datamodule, tmp_path, torch.device("cpu"))
    assert "Backend verification passed" in caplog.text
    assert datamodule.setup_calls == ["predict"]
    assert "2 batch(es)" in caplog.text


def test_a_diverging_artifact_fails_the_deploy(tmp_path) -> None:
    torch.manual_seed(0)
    model = _StagedModel().eval()
    _export_stage_artifacts(model, tmp_path)
    # Perturb the weights after export: the artifacts no longer match the reference.
    with torch.no_grad():
        model.encoder.linear.weight.add_(10.0)
    with pytest.raises(RuntimeError, match="verification FAILED"):
        run_post_export(_cfg(tolerance=0.5), model, _DataModule(), tmp_path, torch.device("cpu"))


def test_a_declared_caveat_skips_verification_before_touching_data(tmp_path, caplog) -> None:
    model = _StagedModel().eval()
    model.verification_caveat = "raw outputs are stochastic by construction"

    class _Exploding(_DataModule):
        def predict_dataloader(self):
            raise AssertionError("verification should have been skipped before loading data")

    with caplog.at_level(logging.WARNING):
        run_post_export(_cfg(), model, _Exploding(), tmp_path, torch.device("cpu"))
    assert "Verification SKIPPED" in caplog.text and "stochastic" in caplog.text
    assert BaseModel.verification_caveat is None
