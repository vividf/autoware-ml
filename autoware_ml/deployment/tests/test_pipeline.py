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

"""Unit tests for the deployed pipeline and artifact discovery."""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from autoware_ml.deployment.pipeline import (
    GLUE_STAGE,
    DeployedPipeline,
    PipelineCache,
    PipelineResult,
    available_backends,
)
from autoware_ml.deployment.stages import DeployedModel, GraphStage
from autoware_ml.types.backend import Backend


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _SplitHead(nn.Module):
    def forward(self, x):
        return x + 1, x - 1


def _toy_graphs(head: nn.Module | None = None, encoder_fallback=()):
    return (
        GraphStage(
            "encoder",
            module=_Double(),
            inputs=("x",),
            outputs=("y",),
            torch_fallback_backends=encoder_fallback,
        ),
        GraphStage("head", module=head or _SplitHead(), inputs=("z",), outputs=("plus", "minus")),
    )


def _toy_forward(batch, graphs):
    y = graphs["encoder"](x=batch["x"].float())["y"]
    return graphs["head"](z=y.to(graphs.device) + 0.5)


def _toy_deployment(head: nn.Module | None = None, encoder_fallback=()) -> DeployedModel:
    return DeployedModel(_toy_graphs(head, encoder_fallback), _toy_forward)


class TestDeployedPipeline(unittest.TestCase):
    """The deployed pipeline runs the deployed function on PyTorch and ONNX Runtime."""

    def setUp(self) -> None:
        """Give each test its own directory for the graph artifacts."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_pytorch_backend_runs_the_function_and_times_graphs_and_glue(self):
        pipeline = DeployedPipeline(
            _toy_deployment(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        result = pipeline.infer({"x": torch.tensor([1, 2, 3])})

        # encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        self.assertTrue(torch.equal(result.outputs["plus"], torch.tensor([3.5, 5.5, 7.5])))
        self.assertTrue(torch.equal(result.outputs["minus"], torch.tensor([1.5, 3.5, 5.5])))
        self.assertEqual(result.output_names, ("plus", "minus"))
        self.assertEqual(result.graph_stage_names, ("encoder", "head"))
        self.assertEqual(set(result.stage_times_ms), {"encoder", "head", GLUE_STAGE})

    def test_trace_records_the_outputs_and_the_graph_inputs(self):
        pipeline = DeployedPipeline(
            _toy_deployment(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        run = pipeline.trace({"x": torch.tensor([1, 2, 3])})
        self.assertEqual(list(run.outputs), ["plus", "minus"])
        self.assertEqual(set(run.graph_inputs), {"encoder", "head"})
        self.assertTrue(torch.equal(run.graph_inputs["head"][0], torch.tensor([2.5, 4.5, 6.5])))

    def test_trace_refuses_a_graph_input_modified_after_the_call(self):
        def forward(batch, graphs):
            x = batch["x"].float()
            y = graphs["encoder"](x=x)["y"]
            x.add_(1.0)
            return graphs["head"](z=y)

        pipeline = DeployedPipeline(
            DeployedModel(_toy_graphs(), forward), Backend.PYTORCH, torch.device("cpu")
        )
        pipeline.infer({"x": torch.tensor([1.0])})
        with self.assertRaisesRegex(ValueError, r"\{'encoder': \['x'\]\}"):
            pipeline.trace({"x": torch.tensor([1.0])})

    def test_assemble_keys_by_the_model_output_names(self):
        pipeline = DeployedPipeline(
            _toy_deployment(), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        result = pipeline.infer({"x": torch.tensor([1.0])})
        assembled = pipeline.assemble(result)
        self.assertEqual(set(assembled), {"plus", "minus"})
        self.assertTrue(math.isclose(assembled["plus"].item(), 3.5, rel_tol=1e-6))

    def test_non_pytorch_backend_requires_artifacts_dir(self):
        with self.assertRaisesRegex(ValueError, "artifacts_dir"):
            DeployedPipeline(_toy_deployment(), backend=Backend.ONNX, device=torch.device("cpu"))

    def test_graph_output_arity_is_checked(self):
        pipeline = DeployedPipeline(
            _toy_deployment(head=_Double()), backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "returned 1 tensor"):
            pipeline.infer({"x": torch.tensor([1.0])})

    def test_cache_builds_each_backend_device_once(self):
        cache = PipelineCache(_toy_deployment(), self.tmp_path)
        first = cache.get(Backend.PYTORCH, "cpu")
        self.assertIs(cache.get(Backend.PYTORCH, torch.device("cpu")), first)

    def test_onnx_backend_matches_pytorch_and_reports_fallbacks(self):
        deployment = _toy_deployment()
        for graph in deployment.graphs:
            torch.onnx.export(
                graph.module,
                (torch.ones(3),),
                str(self.tmp_path / f"{graph.name}.onnx"),
                input_names=list(graph.inputs),
                output_names=list(graph.outputs),
                dynamic_axes={name: {0: "n"} for name in (*graph.inputs, *graph.outputs)},
                opset_version=17,
                dynamo=False,
            )
        reference = DeployedPipeline(
            deployment, backend=Backend.PYTORCH, device=torch.device("cpu")
        )
        onnx_pipeline = DeployedPipeline(
            deployment,
            backend=Backend.ONNX,
            device=torch.device("cpu"),
            artifacts_dir=self.tmp_path,
        )
        batch = {"x": torch.tensor([1, 2, 3])}
        ref, test = reference.infer(batch), onnx_pipeline.infer(batch)
        for name in ref.output_names:
            torch.testing.assert_close(test.outputs[name], ref.outputs[name])
        self.assertEqual(onnx_pipeline.fallback_stage_names, ())

        fallback = _toy_deployment(encoder_fallback=(Backend.ONNX,))
        os.remove(self.tmp_path / "encoder.onnx")
        mixed = DeployedPipeline(
            fallback, backend=Backend.ONNX, device=torch.device("cpu"), artifacts_dir=self.tmp_path
        )
        self.assertEqual(mixed.fallback_stage_names, ("encoder",))
        torch.testing.assert_close(mixed.infer(batch).outputs["plus"], ref.outputs["plus"])
        self.assertEqual(
            available_backends(fallback, self.tmp_path), {Backend.PYTORCH, Backend.ONNX}
        )


class TestPipelineResult(unittest.TestCase):
    """A pipeline result sums the graph times and orders its outputs by name."""

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
    """The available backends are those whose artifacts exist for every graph."""

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
        self.assertEqual(available_backends(_toy_deployment(), self.tmp_path), {Backend.PYTORCH})

    def test_onnx_requires_every_graph_file(self):
        self._touch(self.tmp_path / "encoder.onnx")
        self.assertEqual(available_backends(_toy_deployment(), self.tmp_path), {Backend.PYTORCH})
        self._touch(self.tmp_path / "head.onnx")
        self.assertEqual(
            available_backends(_toy_deployment(), self.tmp_path), {Backend.PYTORCH, Backend.ONNX}
        )

    def test_stale_engine_stays_available_but_warns(self):
        for name in ("encoder", "head"):
            self._touch(self.tmp_path / f"{name}.engine", mtime=1_000)
            self._touch(self.tmp_path / f"{name}.onnx", mtime=2_000)
        with self.assertLogs("autoware_ml.deployment.pipeline", level="WARNING") as logs:
            available = available_backends(_toy_deployment(), self.tmp_path)
        self.assertIn(Backend.TENSORRT, available)
        self.assertIn("STALE TENSORRT ENGINE", "\n".join(logs.output))
