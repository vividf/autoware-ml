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

"""The generic pipeline and artifact discovery — on toy stages, pytorch and onnx backends."""

from __future__ import annotations

import logging
import os

import onnx
import pytest
import torch
from torch import nn

from autoware_ml.deployment.pipeline import (
    PipelineCache,
    PipelineResult,
    StagedPipeline,
    available_backends,
)
from autoware_ml.deployment.stages import GraphStage, TorchStage
from autoware_ml.types.backend import Backend


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _SplitHead(nn.Module):
    def forward(self, x):
        return x + 1, x - 1


def _toy_stages():
    return (
        TorchStage("prep", run=lambda ctx: {"x": ctx.batch["x"].float()}),
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        TorchStage("glue", run=lambda ctx: {"z": ctx["y"].to(ctx.device) + 0.5}),
        GraphStage(
            "head",
            module=_SplitHead(),
            inputs=("z",),
            outputs=("plus", "minus"),
            output_fields=(("plus", "a"), ("minus", "b")),
        ),
    )


class TestStagedPipeline:
    def test_pytorch_backend_runs_all_stages_in_order(self):
        pipeline = StagedPipeline(_toy_stages(), backend="pytorch", device=torch.device("cpu"))
        result, context = pipeline.run({"x": torch.tensor([1, 2, 3])})

        # prep: x=[1,2,3]; encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        assert torch.equal(result.outputs["plus"], torch.tensor([3.5, 5.5, 7.5]))
        assert torch.equal(result.outputs["minus"], torch.tensor([1.5, 3.5, 5.5]))
        assert result.output_names == ["plus", "minus"]
        assert result.graph_stage_names == ("encoder", "head")
        assert set(result.stage_times_ms) == {"prep", "encoder", "glue", "head"}
        assert set(context.tensors) == {"x", "y", "z", "plus", "minus"}

    def test_assemble_maps_onnx_names_to_forward_output_keys(self):
        pipeline = StagedPipeline(
            _toy_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        result = pipeline.infer({"x": torch.tensor([1.0])})
        assembled = pipeline.assemble(result)
        assert set(assembled) == {"a", "b"}
        assert assembled["a"].item() == pytest.approx(3.5)

    def test_assemble_hook_builds_the_forward_output_type(self):
        pipeline = StagedPipeline(
            _toy_stages(),
            backend="pytorch",
            device=torch.device("cpu"),
            assemble=lambda fields: (fields["a"], fields["b"]),
        )
        result = pipeline.infer({"x": torch.tensor([1.0])})
        a, b = pipeline.assemble(result)
        assert a.item() == pytest.approx(3.5) and b.item() == pytest.approx(1.5)

    def test_non_pytorch_backend_requires_artifacts_dir(self):
        with pytest.raises(ValueError, match="artifacts_dir"):
            StagedPipeline(_toy_stages(), backend="onnx", device=torch.device("cpu"))

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            StagedPipeline(_toy_stages(), backend="tflite", device=torch.device("cpu"))

    def test_graph_stage_output_arity_is_checked(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage(
            "head",
            module=_Double(),
            inputs=("z",),
            outputs=("plus", "minus"),
            output_fields=(("plus", "a"),),
        )
        pipeline = StagedPipeline(stages, backend="pytorch", device=torch.device("cpu"))
        with pytest.raises(ValueError, match="returned 1 tensor"):
            pipeline.infer({"x": torch.tensor([1.0])})

    def test_cache_builds_each_backend_device_once(self, tmp_path):
        cache = PipelineCache(_toy_stages(), tmp_path)
        first = cache.get("pytorch", "cpu")
        assert cache.get(Backend.PYTORCH, torch.device("cpu")) is first

    def test_onnx_backend_matches_pytorch_and_reports_fallbacks(self, tmp_path):
        """The onnx backend runs exported artifacts; a declared fallback runs torch instead."""
        stages = _toy_stages()
        for stage in (stages[1], stages[3]):
            torch.onnx.export(
                stage.module,
                (torch.ones(3),),
                str(tmp_path / f"{stage.name}.onnx"),
                input_names=list(stage.inputs),
                output_names=list(stage.outputs),
                dynamic_axes={name: {0: "n"} for name in (*stage.inputs, *stage.outputs)},
                opset_version=17,
                dynamo=False,
            )
        reference = StagedPipeline(stages, backend="pytorch", device=torch.device("cpu"))
        onnx_pipeline = StagedPipeline(
            stages, backend="onnx", device=torch.device("cpu"), artifacts_dir=tmp_path
        )
        batch = {"x": torch.tensor([1, 2, 3])}
        ref, test = reference.infer(batch), onnx_pipeline.infer(batch)
        for name in ref.output_names:
            torch.testing.assert_close(test.outputs[name], ref.outputs[name])
        assert onnx_pipeline.fallback_stage_names == ()

        fallback = list(stages)
        fallback[1] = GraphStage(
            "encoder",
            module=_Double(),
            inputs=("x",),
            outputs=("y",),
            torch_fallback_backends=(Backend.ONNX,),
        )
        os.remove(tmp_path / "encoder.onnx")
        mixed = StagedPipeline(
            fallback, backend="onnx", device=torch.device("cpu"), artifacts_dir=tmp_path
        )
        assert mixed.fallback_stage_names == ("encoder",)
        torch.testing.assert_close(mixed.infer(batch).outputs["plus"], ref.outputs["plus"])
        assert available_backends(fallback, tmp_path) == {Backend.PYTORCH, Backend.ONNX}


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
            graph_stage_names=("graph_a", "b"),
        )
        assert result.model_ms == pytest.approx(2.0)

    def test_ordered_outputs_respects_output_names_order(self):
        heatmap, reg = torch.zeros(1), torch.ones(1)
        result = PipelineResult(
            outputs={"reg": reg, "heatmap": heatmap}, output_names=["heatmap", "reg"]
        )
        ordered = result.ordered_outputs()
        assert ordered[0] is heatmap and ordered[1] is reg

    def test_ordered_outputs_raises_on_missing_name(self):
        result = PipelineResult(outputs={"reg": torch.ones(1)}, output_names=["heatmap", "reg"])
        with pytest.raises(KeyError):
            result.ordered_outputs()


class TestAvailableBackends:
    def _touch(self, path, mtime=None):
        path.write_bytes(b"stub")
        if mtime is not None:
            os.utime(path, (mtime, mtime))

    def test_pytorch_always_available(self, tmp_path):
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH}

    def test_onnx_requires_every_graph_stage_file(self, tmp_path):
        self._touch(tmp_path / "encoder.onnx")
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH}
        self._touch(tmp_path / "head.onnx")
        assert available_backends(_toy_stages(), tmp_path) == {Backend.PYTORCH, Backend.ONNX}

    def test_stale_engine_stays_available_but_warns(self, tmp_path, caplog):
        for name in ("encoder", "head"):
            self._touch(tmp_path / f"{name}.engine", mtime=1_000)
            self._touch(tmp_path / f"{name}.onnx", mtime=2_000)
        with caplog.at_level(logging.WARNING):
            available = available_backends(_toy_stages(), tmp_path)
        assert Backend.TENSORRT in available
        assert "STALE TENSORRT ENGINE" in caplog.text


def test_onnx_helper_is_importable_without_a_gpu() -> None:
    assert onnx.__version__
