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

"""Unit tests for the per-backend evaluation loop."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from autoware_ml.deployment.evaluation.evaluator import (
    MODEL_STAGE,
    RESULTS_FILE_NAME,
    EvaluationResult,
    evaluate_backend,
    flatten_results,
    write_results_json,
)
from autoware_ml.deployment.evaluation.latency import LatencyStats
from autoware_ml.deployment.pipeline import StagedPipeline
from autoware_ml.deployment.stages import GraphStage, Stage, TorchStage
from autoware_ml.deployment.tests.toy_batch import ToyBatch
from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.report import MetricKey, MetricReport
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.types.backend import Backend


class _SumSuite:
    """Duck-typed metric suite."""

    prefix = "toy"
    headline_metrics = ("error",)

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def to(self, device):
        return self

    def reset(self) -> None:
        self.total, self.count = 0.0, 0

    def required_keys(self):
        return ("prediction", "target")

    def update(self, eval_out: Mapping[str, Any]) -> None:
        self.total += float((eval_out["prediction"] - eval_out["target"]).abs().sum())
        self.count += 1

    def result(self, stage: EvalStage) -> MetricReport:
        report = MetricReport()
        report.record("error", self.total)
        report.add(MetricKey(name="error", range_suffix="0m_50m"), self.total / max(self.count, 1))
        return report


class _Model(ModuleBaseModel):
    def __init__(self) -> None:
        super().__init__(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )
        self.scale = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.scale.weight.fill_(2.0)

    def forward(self, multi_task_batch_inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"y": self.scale(multi_task_batch_inputs["x"])}

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return batch

    def compute_metrics(self, multi_task_batch_inputs, outputs):
        return {"loss": outputs["y"].sum()}

    def decode_outputs(self, multi_task_batch_inputs: Any, outputs: Any) -> Any:
        del multi_task_batch_inputs
        return outputs

    def build_deployed_eval_output(
        self, batch_inputs: Mapping[str, Any], outputs: Mapping[str, torch.Tensor]
    ) -> dict[str, Any]:
        return {"prediction": outputs["y"], "target": batch_inputs["target"]}

    def get_log_batch_size(self, batch_inputs):
        return batch_inputs["x"].shape[0]

    def clone_metrics(self, stage: EvalStage):
        return [_SumSuite()]

    def build_stages(self) -> Sequence[Stage]:
        return (
            TorchStage(
                "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
            ),
            GraphStage(
                "scale",
                module=self.scale,
                inputs=("x",),
                outputs=("y",),
            ),
        )


def _loader(num_batches: int = 4, batch_size: int = 3) -> DataLoader:
    items = [
        ToyBatch(
            x=torch.full((batch_size, 1), float(i)), target=torch.full((batch_size, 1), 2.0 * i)
        )
        for i in range(num_batches)
    ]
    return DataLoader(items, batch_size=None)


class TestEvaluateBackend(unittest.TestCase):
    """Unit tests for scoring and timing one backend over a split."""

    def test_pytorch_backend_scores_the_split_and_times_every_stage(self) -> None:
        model = _Model().eval()
        pipeline = StagedPipeline(
            model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        result = evaluate_backend(model, _loader(), pipeline, torch.device("cpu"), num_warmup=1)

        self.assertIs(result.backend, Backend.PYTORCH)
        self.assertEqual(result.split, "test")
        self.assertEqual(result.num_samples, 12)
        self.assertEqual(
            result.metrics, {"test/pytorch/toy/error": 0.0, "test/pytorch/toy/error_0m_50m": 0.0}
        )
        self.assertEqual(
            set(result.latency), {"prep", "scale", MODEL_STAGE, "preprocess", "decode_and_metrics"}
        )
        self.assertEqual(result.headline_metrics, ("error",))
        flat = flatten_results([result])
        self.assertIn("latency/pytorch/model_graphs_mean_ms", flat)
        self.assertEqual(flat["test/pytorch/toy/error"], 0.0)

    def test_num_samples_limits_at_batch_granularity_and_val_reports_under_val(self) -> None:
        model = _Model().eval()
        pipeline = StagedPipeline(
            model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        result = evaluate_backend(
            model, _loader(), pipeline, torch.device("cpu"), num_samples=4, stage=EvalStage.VAL
        )
        self.assertEqual(result.num_samples, 6)  # the limit is checked per batch of three
        self.assertTrue(next(iter(result.metrics)).startswith("val/pytorch/"))

    def test_zero_samples_is_an_error(self) -> None:
        model = _Model().eval()
        pipeline = StagedPipeline(
            model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "zero samples"):
            evaluate_backend(model, _loader(num_batches=0), pipeline, torch.device("cpu"))

    def test_a_missing_eval_key_names_the_model(self) -> None:
        class _Broken(_Model):
            def build_deployed_eval_output(self, batch_inputs, outputs):
                return {"prediction": outputs["y"]}

        model = _Broken().eval()
        pipeline = StagedPipeline(
            model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "_Broken.build_deployed_eval_output"):
            evaluate_backend(model, _loader(), pipeline, torch.device("cpu"))


class TestWriteResultsJson(unittest.TestCase):
    """Unit tests for the results file the evaluation writes."""

    def setUp(self) -> None:
        """Give each test its own directory for the results file."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_results_json_keeps_every_metric_and_nulls_non_finite_values(self) -> None:
        result = EvaluationResult(
            backend=Backend.TENSORRT,
            device="cuda",
            split="test",
            metrics={
                "test/tensorrt/det3d/mAP_car": 0.5,
                "test/tensorrt/det3d/mAP_bus": float("nan"),
            },
            latency={MODEL_STAGE: LatencyStats.from_samples([2.0, 4.0])},
            num_samples=2,
        )
        path = write_results_json([result], self.tmp_path / RESULTS_FILE_NAME)

        def reject(constant: str) -> None:
            self.fail(f"non-strict JSON constant {constant}")

        (entry,) = json.loads(path.read_text(), parse_constant=reject)["results"]
        self.assertEqual(
            (entry["backend"], entry["split"], entry["num_samples"]), ("tensorrt", "test", 2)
        )
        self.assertEqual(
            entry["metrics"],
            {
                "test/tensorrt/det3d/mAP_bus": None,
                "test/tensorrt/det3d/mAP_car": 0.5,
            },
        )
        np.testing.assert_allclose(
            entry["latency_ms"][MODEL_STAGE]["mean"], 3.0, rtol=1e-6, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
