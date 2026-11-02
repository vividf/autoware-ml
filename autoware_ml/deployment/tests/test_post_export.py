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

"""Unit tests for the post-export steps on a toy stage-graph model, on the CPU."""

from __future__ import annotations

import json
import logging
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from autoware_ml.datamodule.data_module import DataLoaderConfig
from autoware_ml.deployment.config import PostExportConfig
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
    def __init__(self) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

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

    test_dataloader_config = DataLoaderConfig(batch_size=16)

    def build_dataloader(self, split: SplitType, config: DataLoaderConfig) -> DataLoader:
        self.built_batch_sizes.append((split, config.batch_size))
        return DataLoader(
            [ToyBatch(points=p, target=p.sum(dim=1, keepdim=True)) for p in self.points],
            batch_size=None,
        )


class _ErrorSuite:
    """Duck-typed metric suite."""

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


def _cfg(enabled: bool = True, tolerance: float | None = None) -> PostExportConfig:
    scenario = {
        "ref": {"backend": "pytorch", "device": "cpu"},
        "test": {"backend": "onnx", "device": "cpu"},
    }
    if tolerance is not None:
        scenario["tolerance"] = tolerance
    return PostExportConfig.from_deploy_cfg(
        {
            "onnx": {"enabled": True},
            "tensorrt": {"enabled": False},
            "verification": {"enabled": enabled, "num_verify_batches": 2, "scenarios": [scenario]},
        }
    )


class TestPostExportVerification(unittest.TestCase):
    """Unit tests for verifying the exported artifacts against the PyTorch reference."""

    def setUp(self) -> None:
        """Give each test its own directory for the exported artifacts."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_disabled_sections_do_nothing_even_without_stages(self) -> None:
        datamodule = _DataModule()
        run_post_export(
            PostExportConfig.from_deploy_cfg({"onnx": {}, "tensorrt": {}}),
            _PlainModel(),
            datamodule,
            self.tmp_path,
            torch.device("cpu"),
        )
        self.assertEqual(datamodule.setup_calls, [])

    def test_enabling_verification_on_a_model_without_stages_is_an_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "build_stages"):
            run_post_export(
                _cfg(), _PlainModel(), _DataModule(), self.tmp_path, torch.device("cpu")
            )

    def test_onnx_artifacts_verify_against_the_pytorch_reference(self) -> None:
        torch.manual_seed(0)
        model = _StagedModel().eval()
        _export_stage_artifacts(model, self.tmp_path)
        datamodule = _DataModule()
        with self.assertLogs(level=logging.INFO) as logs:
            run_post_export(_cfg(), model, datamodule, self.tmp_path, torch.device("cpu"))
        text = "\n".join(logs.output)
        self.assertIn("Backend verification passed", text)
        self.assertEqual(datamodule.setup_calls, ["prepare_data", "predict"])
        self.assertIn("2 batch(es)", text)

    def test_a_diverging_artifact_fails_the_deploy(self) -> None:
        torch.manual_seed(0)
        model = _StagedModel().eval()
        _export_stage_artifacts(model, self.tmp_path)
        # The PyTorch reference now differs from the exported artifacts.
        with torch.no_grad():
            model.encoder.linear.weight.add_(10.0)
        with self.assertRaisesRegex(RuntimeError, "verification FAILED"):
            run_post_export(
                _cfg(tolerance=0.5), model, _DataModule(), self.tmp_path, torch.device("cpu")
            )


def _evaluation_cfg() -> PostExportConfig:
    return PostExportConfig.from_deploy_cfg(
        {
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
    )


class TestPostExportEvaluation(unittest.TestCase):
    """Unit tests for scoring every available backend and writing the results."""

    def setUp(self) -> None:
        """Give each test its own directory for the exported artifacts and the results."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_evaluation_scores_every_available_backend_and_logs_metrics(self) -> None:
        torch.manual_seed(0)
        model = _StagedModel().eval()
        _export_stage_artifacts(model, self.tmp_path)
        logged: dict[str, float] = {}
        datamodule = _DataModule()
        results = run_post_export(
            _evaluation_cfg(),
            model,
            datamodule,
            self.tmp_path,
            torch.device("cpu"),
            log_metric=logged.__setitem__,
        )
        # tensorrt has no artifacts and is skipped.
        self.assertEqual([r.backend.value for r in results], ["pytorch", "onnx"])
        # deploy.evaluation.batch_size, not the split's training-time 16.
        self.assertEqual(datamodule.built_batch_sizes, [(SplitType.TEST, 1), (SplitType.TEST, 1)])
        self.assertEqual(results[0].num_samples, 12)
        self.assertAlmostEqual(
            results[0].metrics["test/pytorch/toy/error"],
            results[1].metrics["test/onnx/toy/error"],
            delta=1e-5,
        )
        self.assertIn("latency/onnx/model_graphs_mean_ms", logged)
        self.assertIn("test/pytorch/toy/error", logged)

    def test_evaluation_writes_every_metric_without_a_tracker(self) -> None:
        torch.manual_seed(0)
        model = _StagedModel().eval()
        _export_stage_artifacts(model, self.tmp_path)
        results = run_post_export(
            _evaluation_cfg(), model, _DataModule(), self.tmp_path, torch.device("cpu")
        )
        entries = json.loads((self.tmp_path / "evaluation.json").read_text())["results"]
        self.assertEqual([entry["backend"] for entry in entries], ["pytorch", "onnx"])
        for entry, result in zip(entries, results):
            self.assertEqual(entry["metrics"].keys(), result.metrics.keys())
            for key, value in result.metrics.items():
                np.testing.assert_allclose(entry["metrics"][key], value, rtol=1e-6, atol=1e-12)
            np.testing.assert_allclose(
                entry["latency_ms"]["model_graphs"]["mean"],
                result.latency["model_graphs"].mean,
                rtol=1e-6,
                atol=1e-12,
            )


if __name__ == "__main__":
    unittest.main()
