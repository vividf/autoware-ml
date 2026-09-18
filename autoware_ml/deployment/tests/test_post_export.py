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

from autoware_ml.datamodule.data_module import DataLoaderConfig
from autoware_ml.deployment.post_export import run_post_export
from autoware_ml.deployment.stages import GraphStage, Stage, TorchStage
from autoware_ml.deployment.tests.toy_batch import ToyBatch
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.types.dataset import SplitType


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, features):
        return self.linear(features)


class _Head(nn.Module):
    def forward(self, bev):
        return bev + 1, bev - 1


class _ToyModel(ModuleBaseModel):
    """The toy models take a plain batch; the base class only needs the two ctor args."""

    def __init__(self) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch  # the toy batch is its own model input

    def decode_outputs(self, multi_task_batch_inputs: Any, outputs: Any) -> Any:
        del multi_task_batch_inputs
        return outputs


class _StagedModel(_ToyModel):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.head = _Head()

    def forward(self, multi_task_batch_inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        plus, minus = self.head(self.encoder(multi_task_batch_inputs["points"].float()))
        return {"plus": plus, "minus": minus}

    def compute_metrics(self, multi_task_batch_inputs: Mapping[str, Any], outputs: Any):
        del multi_task_batch_inputs
        return {"loss": outputs["plus"].sum()}

    def build_deployed_eval_output(
        self, batch_inputs: Mapping[str, Any], outputs: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        return {"plus": outputs["plus"], "target": batch_inputs["target"]}

    def get_log_batch_size(self, batch_inputs):
        return batch_inputs["points"].shape[0]

    def clone_metrics(self, stage):
        return [_ErrorSuite()]

    def build_stages(self) -> Sequence[Stage]:
        return (
            TorchStage(
                "prep",
                run=lambda state: {"features": state.batch["points"].float()},
                inputs=(),
                outputs=("features",),
            ),
            GraphStage("encoder", module=self.encoder, inputs=("features",), outputs=("encoded",)),
            GraphStage(
                "head",
                module=self.head,
                inputs=("encoded",),
                outputs=("plus", "minus"),
            ),
        )


class _PlainModel(_ToyModel):
    def forward(self, multi_task_batch_inputs):
        return multi_task_batch_inputs["points"]

    def compute_metrics(self, multi_task_batch_inputs, outputs):
        return {"loss": outputs.sum()}


class _DataModule(L.LightningDataModule):
    def __init__(self, num_batches: int = 3) -> None:
        super().__init__()
        self.points = [torch.randn(4, 2) for _ in range(num_batches)]
        self.setup_calls: list[str] = []
        self.built_batch_sizes: list[tuple[SplitType, int]] = []

    def prepare_data(self) -> None:
        self.setup_calls.append("prepare_data")

    def setup(self, stage: str) -> None:
        self.setup_calls.append(stage)

    def predict_dataloader(self):
        return DataLoader([ToyBatch(points=p) for p in self.points], batch_size=None)

    # The split's own setting is the training-time one; evaluation builds the loader at
    # its own batch size (deploy.evaluation.batch_size) and this records what it asked for.
    test_dataloader_config = DataLoaderConfig(batch_size=16)

    def build_dataloader(self, split: SplitType, config: DataLoaderConfig) -> DataLoader:
        self.built_batch_sizes.append((split, config.batch_size))
        return DataLoader(
            [ToyBatch(points=p, target=p.sum(dim=1, keepdim=True)) for p in self.points],
            batch_size=None,
        )


class _ErrorSuite:
    """Duck-typed suite: mean absolute error of ``plus`` against the batch target."""

    prefix = "toy"
    headline_metrics = ("error",)

    def __init__(self) -> None:
        self.total, self.count = 0.0, 0

    def to(self, device):
        return self

    def reset(self) -> None:
        self.total, self.count = 0.0, 0

    def required_keys(self):
        return ("plus", "target")

    def update(self, eval_out) -> None:
        self.total += float((eval_out["plus"] - eval_out["target"]).abs().mean())
        self.count += 1

    def result(self, stage) -> dict[str, float]:
        return {"error": self.total / max(self.count, 1)}


def _export_stage_artifacts(model: _StagedModel, tmp_path) -> None:
    specs = model.build_export_specs({"points": torch.randn(4, 2)})
    for name, spec in specs.items():
        torch.onnx.export(
            spec.module,
            spec.args,
            str(tmp_path / f"{name}.onnx"),
            input_names=spec.input_names,
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
    assert datamodule.setup_calls == ["prepare_data", "predict"]
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


def test_evaluation_scores_every_available_backend_and_logs_metrics(tmp_path) -> None:
    torch.manual_seed(0)
    model = _StagedModel().eval()
    _export_stage_artifacts(model, tmp_path)
    logged: dict[str, float] = {}
    cfg = {
        "onnx": {"enabled": True},
        "tensorrt": {"enabled": False},
        "evaluation": {
            "enabled": True,
            "num_samples": -1,
            "num_warmup": 1,
            "backends": {
                "pytorch": {"enabled": True, "device": "cpu"},
                "onnx": {"enabled": True, "device": "cpu"},
                "tensorrt": {"enabled": True, "device": "cuda"},
            },
        },
    }
    datamodule = _DataModule()
    results = run_post_export(
        cfg, model, datamodule, tmp_path, torch.device("cpu"), log_metric=logged.__setitem__
    )
    # tensorrt has no artifacts in this run and is skipped; the other two are scored.
    assert [r.backend.value for r in results] == ["pytorch", "onnx"]
    # One frame per batch regardless of the split's training-time setting (16 here).
    assert datamodule.built_batch_sizes == [(SplitType.TEST, 1), (SplitType.TEST, 1)]
    assert results[0].num_samples == 12
    assert results[0].metrics["test/pytorch/toy/error"] == pytest.approx(
        results[1].metrics["test/onnx/toy/error"], abs=1e-5
    )
    assert "latency/onnx/model_graphs_mean_ms" in logged
    assert "test/pytorch/toy/error" in logged
