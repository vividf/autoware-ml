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

"""The evaluate loop on a toy stage-graph model with duck-typed metric suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from autoware_ml.deployment.pipeline import StagedPipeline
from autoware_ml.deployment.stages import GraphStage, Stage, TorchStage
from autoware_ml.deployment.tests.toy_batch import ToyBatch
from autoware_ml.evaluation.evaluator import MODEL_STAGE, evaluate_backend, flatten_results
from autoware_ml.metrics.base import EvalStage
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.types.backend import Backend


class _SumSuite:
    """Duck-typed suite: accumulates ``sum(prediction - target)``; reports it once."""

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

    def result(self, stage: EvalStage) -> dict[str, float]:
        return {"error": self.total, "error_0m_50m": self.total / max(self.count, 1)}


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
        return batch  # the toy batch is its own model input

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


def test_pytorch_backend_scores_the_split_and_times_every_stage() -> None:
    model = _Model().eval()
    pipeline = StagedPipeline(
        model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
    )
    result = evaluate_backend(model, _loader(), pipeline, torch.device("cpu"), num_warmup=1)

    assert result.backend is Backend.PYTORCH and result.split == "test"
    assert result.num_samples == 12
    assert result.metrics == {"test/pytorch/toy/error": 0.0, "test/pytorch/toy/error_0m_50m": 0.0}
    assert set(result.latency) == {"prep", "scale", MODEL_STAGE, "preprocess", "decode_and_metrics"}
    assert result.headline_metrics == ("error",)
    flat = flatten_results([result])
    assert "latency/pytorch/model_graphs_mean_ms" in flat and flat["test/pytorch/toy/error"] == 0.0


def test_num_samples_limits_at_batch_granularity_and_val_reports_under_val() -> None:
    model = _Model().eval()
    pipeline = StagedPipeline(
        model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
    )
    result = evaluate_backend(
        model, _loader(), pipeline, torch.device("cpu"), num_samples=4, stage=EvalStage.VAL
    )
    assert result.num_samples == 6  # two batches of three: the limit is checked per batch
    assert next(iter(result.metrics)).startswith("val/pytorch/")


def test_zero_samples_is_an_error() -> None:
    model = _Model().eval()
    pipeline = StagedPipeline(
        model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="zero samples"):
        evaluate_backend(model, _loader(num_batches=0), pipeline, torch.device("cpu"))


def test_a_missing_eval_key_names_the_model(tmp_path) -> None:
    class _Broken(_Model):
        def build_deployed_eval_output(self, batch_inputs, outputs):
            return {"prediction": outputs["y"]}

    model = _Broken().eval()
    pipeline = StagedPipeline(
        model.build_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
    )
    with pytest.raises(ValueError, match="_Broken"):
        evaluate_backend(model, _loader(), pipeline, torch.device("cpu"))
