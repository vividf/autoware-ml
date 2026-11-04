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

"""Unit tests for the staged pipeline and artifact discovery."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest

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
        TorchStage(
            "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
        ),
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        TorchStage(
            "glue",
            run=lambda state: {"z": state["y"].to(state.device) + 0.5},
            inputs=("y",),
            outputs=("z",),
        ),
        GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "minus")),
    )


class TestStagedPipeline(unittest.TestCase):
    """The staged pipeline runs its stages in order on PyTorch and ONNX Runtime."""

    def setUp(self) -> None:
        """Give each test its own directory for the stage artifacts."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_pytorch_backend_runs_all_stages_in_order(self):
        pipeline = StagedPipeline(
            _toy_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        result = pipeline.infer({"x": torch.tensor([1, 2, 3])})

        # prep: x=[1,2,3]; encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        self.assertTrue(torch.equal(result.outputs["plus"], torch.tensor([3.5, 5.5, 7.5])))
        self.assertTrue(torch.equal(result.outputs["minus"], torch.tensor([1.5, 3.5, 5.5])))
        self.assertEqual(result.output_names, ("plus", "minus"))
        self.assertEqual(result.graph_stage_names, ("encoder", "head"))
        self.assertEqual(set(result.stage_times_ms), {"prep", "encoder", "glue", "head"})

    def test_trace_records_every_tensor_and_the_graph_inputs(self):
        pipeline = StagedPipeline(
            _toy_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        state = pipeline.trace({"x": torch.tensor([1, 2, 3])})
        self.assertEqual(set(state.tensors), {"x", "y", "z", "plus", "minus"})
        self.assertEqual(set(state.graph_inputs), {"encoder", "head"})
        self.assertTrue(torch.equal(state.graph_inputs["head"][0], torch.tensor([2.5, 4.5, 6.5])))

    def test_infer_records_no_graph_inputs(self):
        pipeline = StagedPipeline(
            _toy_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        state, _ = pipeline._run({"x": torch.tensor([1.0])}, trace=False)
        self.assertEqual(state.graph_inputs, {})

    def test_assemble_keys_by_the_model_output_names(self):
        pipeline = StagedPipeline(
            _toy_stages(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        result = pipeline.infer({"x": torch.tensor([1.0])})
        assembled = pipeline.assemble(result)
        self.assertEqual(set(assembled), {"plus", "minus"})
        self.assertTrue(math.isclose(assembled["plus"].item(), 3.5, rel_tol=1e-6))

    def test_non_pytorch_backend_requires_artifacts_dir(self):
        with self.assertRaisesRegex(ValueError, "artifacts_dir"):
            StagedPipeline(_toy_stages(), backend=Backend.ONNX, device=torch.device("cpu"))

    def test_graph_stage_output_arity_is_checked(self):
        stages = list(_toy_stages())
        stages[3] = GraphStage("head", module=_Double(), inputs=("z",), outputs=("plus", "minus"))
        pipeline = StagedPipeline(stages, backend=Backend.PYTORCH, device=torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "returned 1 tensor"):
            pipeline.infer({"x": torch.tensor([1.0])})

    def test_cache_builds_each_backend_device_once(self):
        cache = PipelineCache(_toy_stages(), self.tmp_path)
        first = cache.get(Backend.PYTORCH, "cpu")
        self.assertIs(cache.get(Backend.PYTORCH, torch.device("cpu")), first)

    def test_onnx_backend_matches_pytorch_and_reports_fallbacks(self):
        stages = _toy_stages()
        for stage in (stages[1], stages[3]):
            torch.onnx.export(
                stage.module,
                (torch.ones(3),),
                str(self.tmp_path / f"{stage.name}.onnx"),
                input_names=list(stage.inputs),
                output_names=list(stage.outputs),
                dynamic_axes={name: {0: "n"} for name in (*stage.inputs, *stage.outputs)},
                opset_version=17,
                dynamo=False,
            )
        reference = StagedPipeline(stages, backend=Backend.PYTORCH, device=torch.device("cpu"))
        onnx_pipeline = StagedPipeline(
            stages, backend=Backend.ONNX, device=torch.device("cpu"), artifacts_dir=self.tmp_path
        )
        batch = {"x": torch.tensor([1, 2, 3])}
        ref, test = reference.infer(batch), onnx_pipeline.infer(batch)
        for name in ref.output_names:
            torch.testing.assert_close(test.outputs[name], ref.outputs[name])
        self.assertEqual(onnx_pipeline.fallback_stage_names, ())

        fallback = list(stages)
        fallback[1] = GraphStage(
            "encoder",
            module=_Double(),
            inputs=("x",),
            outputs=("y",),
            torch_fallback_backends=(Backend.ONNX,),
        )
        os.remove(self.tmp_path / "encoder.onnx")
        mixed = StagedPipeline(
            fallback, backend=Backend.ONNX, device=torch.device("cpu"), artifacts_dir=self.tmp_path
        )
        self.assertEqual(mixed.fallback_stage_names, ("encoder",))
        torch.testing.assert_close(mixed.infer(batch).outputs["plus"], ref.outputs["plus"])
        self.assertEqual(
            available_backends(fallback, self.tmp_path), {Backend.PYTORCH, Backend.ONNX}
        )


class TestPipelineResult(unittest.TestCase):
    """A pipeline result sums the graph-stage times and orders its outputs by name."""

    def test_model_ms_sums_only_graph_stages(self):
        result = PipelineResult(
            outputs={},
            output_names=(),
            stage_times_ms={"pillar_decorate": 1.0, "graph_a": 2.0, "graph_b": 3.5},
            graph_stage_names=("graph_a", "graph_b"),
        )
        self.assertTrue(math.isclose(result.model_ms, 5.5, rel_tol=1e-6))

    def test_model_ms_treats_missing_graph_stage_as_zero(self):
        result = PipelineResult(
            outputs={},
            output_names=(),
            stage_times_ms={"graph_a": 2.0},
            graph_stage_names=("graph_a", "b"),
        )
        self.assertTrue(math.isclose(result.model_ms, 2.0, rel_tol=1e-6))

    def test_ordered_outputs_respects_output_names_order(self):
        heatmap, reg = torch.zeros(1), torch.ones(1)
        result = PipelineResult(
            outputs={"reg": reg, "heatmap": heatmap}, output_names=("heatmap", "reg")
        )
        ordered = result.ordered_outputs()
        self.assertIs(ordered[0], heatmap)
        self.assertIs(ordered[1], reg)

    def test_ordered_outputs_raises_on_missing_name(self):
        result = PipelineResult(outputs={"reg": torch.ones(1)}, output_names=("heatmap", "reg"))
        with self.assertRaises(KeyError):
            result.ordered_outputs()


class TestAvailableBackends(unittest.TestCase):
    """The available backends are those whose artifacts exist for every graph stage."""

    def setUp(self) -> None:
        """Give each test its own directory for the stage artifacts."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def _touch(self, path, mtime=None):
        path.write_bytes(b"stub")
        if mtime is not None:
            os.utime(path, (mtime, mtime))

    def test_pytorch_always_available(self):
        self.assertEqual(available_backends(_toy_stages(), self.tmp_path), {Backend.PYTORCH})

    def test_onnx_requires_every_graph_stage_file(self):
        self._touch(self.tmp_path / "encoder.onnx")
        self.assertEqual(available_backends(_toy_stages(), self.tmp_path), {Backend.PYTORCH})
        self._touch(self.tmp_path / "head.onnx")
        self.assertEqual(
            available_backends(_toy_stages(), self.tmp_path), {Backend.PYTORCH, Backend.ONNX}
        )

    def test_stale_engine_stays_available_but_warns(self):
        for name in ("encoder", "head"):
            self._touch(self.tmp_path / f"{name}.engine", mtime=1_000)
            self._touch(self.tmp_path / f"{name}.onnx", mtime=2_000)
        with self.assertLogs("autoware_ml.deployment.pipeline", level="WARNING") as logs:
            available = available_backends(_toy_stages(), self.tmp_path)
        self.assertIn(Backend.TENSORRT, available)
        self.assertIn("STALE TENSORRT ENGINE", "\n".join(logs.output))
