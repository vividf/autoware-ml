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
from autoware_ml.evaluation.evaluator import MODEL_STAGE, evaluate_backend, flatten_results
from autoware_ml.metrics.base import EvalStage
from autoware_ml.models.base import BaseModel
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


class _Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.scale.weight.fill_(2.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"y": self.scale(x)}

    def compute_metrics(self, batch_inputs_dict, outputs):
        return {"loss": outputs["y"].sum()}

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        return {"prediction": outputs["y"], "target": batch["target"]}

    def get_log_batch_size(self, batch_inputs_dict):
        return batch_inputs_dict["x"].shape[0]

    def clone_metrics(self, stage: EvalStage):
        return [_SumSuite()]

    def build_stages(self) -> Sequence[Stage]:
        return (
            TorchStage("prep", run=lambda ctx: {"x": ctx.batch["x"].float()}),
            GraphStage(
                "scale",
                module=self.scale,
                inputs=("x",),
                outputs=("y",),
                output_fields=(("y", "y"),),
            ),
        )


def _loader(num_batches: int = 4, batch_size: int = 3) -> DataLoader:
    items = [
        {"x": torch.full((batch_size, 1), float(i)), "target": torch.full((batch_size, 1), 2.0 * i)}
        for i in range(num_batches)
    ]
    return DataLoader(items, batch_size=None)


def test_pytorch_backend_scores_the_split_and_times_every_stage() -> None:
    model = _Model().eval()
    pipeline = StagedPipeline(model.build_stages(), backend="pytorch", device=torch.device("cpu"))
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
    pipeline = StagedPipeline(model.build_stages(), backend="pytorch", device=torch.device("cpu"))
    result = evaluate_backend(
        model, _loader(), pipeline, torch.device("cpu"), num_samples=4, stage=EvalStage.VAL
    )
    assert result.num_samples == 6  # two batches of three: the limit is checked per batch
    assert next(iter(result.metrics)).startswith("val/pytorch/")


def test_zero_samples_is_an_error() -> None:
    model = _Model().eval()
    pipeline = StagedPipeline(model.build_stages(), backend="pytorch", device=torch.device("cpu"))
    with pytest.raises(ValueError, match="zero samples"):
        evaluate_backend(model, _loader(num_batches=0), pipeline, torch.device("cpu"))


def test_a_missing_eval_key_names_the_model(tmp_path) -> None:
    class _Broken(_Model):
        def build_eval_output(self, batch, outputs):
            return {"prediction": outputs["y"]}

    model = _Broken().eval()
    pipeline = StagedPipeline(model.build_stages(), backend="pytorch", device=torch.device("cpu"))
    with pytest.raises(ValueError, match="_Broken.build_eval_output"):
        evaluate_backend(model, _loader(), pipeline, torch.device("cpu"))
