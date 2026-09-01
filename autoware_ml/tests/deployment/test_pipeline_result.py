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

"""Unit tests for PipelineResult timing rollup and output ordering."""

from __future__ import annotations

import pytest
import torch

pipeline = pytest.importorskip(
    "autoware_ml.deployment.pipeline",
    reason="deployment.pipeline transitively needs the project batch dataclasses",
)
PipelineResult = pipeline.PipelineResult


class TestPipelineResult:
    def test_model_ms_sums_only_graph_stages(self):
        result = PipelineResult(
            outputs={},
            output_names=[],
            stage_times_ms={"pillar_decorate": 1.0, "graph_a": 2.0, "graph_b": 3.5},
            graph_stage_names=("graph_a", "graph_b"),
        )
        assert result.model_ms == pytest.approx(5.5)

    def test_model_ms_treats_missing_graph_stage_as_zero(self):
        result = PipelineResult(
            outputs={},
            output_names=[],
            stage_times_ms={"graph_a": 2.0},
            graph_stage_names=("graph_a", "graph_b"),
        )
        assert result.model_ms == pytest.approx(2.0)

    def test_model_ms_is_zero_without_graph_stages(self):
        result = PipelineResult(
            outputs={},
            output_names=[],
            stage_times_ms={"anything": 4.0},
        )
        assert result.model_ms == 0.0

    def test_ordered_outputs_respects_output_names_order(self):
        heatmap = torch.zeros(1)
        reg = torch.ones(1)
        # Insertion order deliberately differs from the frozen ABI order.
        result = PipelineResult(
            outputs={"reg": reg, "heatmap": heatmap},
            output_names=["heatmap", "reg"],
        )
        ordered = result.ordered_outputs()
        assert ordered[0] is heatmap
        assert ordered[1] is reg

    def test_ordered_outputs_raises_on_missing_name(self):
        result = PipelineResult(outputs={"reg": torch.ones(1)}, output_names=["heatmap", "reg"])
        with pytest.raises(KeyError):
            result.ordered_outputs()
