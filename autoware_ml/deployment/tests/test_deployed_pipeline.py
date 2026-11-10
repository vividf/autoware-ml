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

"""Tests for the function form of a deployment (PROTOTYPE)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from autoware_ml.deployment.deployed import (
    GLUE_STAGE,
    DeployedModel,
    DeployedPipeline,
    Graphs,
)
from autoware_ml.deployment.export_specs import (
    build_export_specs_from_deployed,
    build_export_specs_from_stages,
)
from autoware_ml.deployment.pipeline import PipelineCache, StagedPipeline, available_backends
from autoware_ml.deployment.stages import GraphStage, TorchStage
from autoware_ml.types.backend import Backend

CPU = torch.device("cpu")


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _SplitHead(nn.Module):
    def forward(self, z):
        return z + 1, z - 1


def _graphs() -> tuple[GraphStage, GraphStage]:
    return (
        GraphStage("encoder", module=_Double(), inputs=("x",), outputs=("y",)),
        GraphStage("head", module=_SplitHead(), inputs=("z",), outputs=("plus", "minus")),
    )


def _forward(batch, graphs: Graphs):
    y = graphs["encoder"](x=batch["x"].float())["y"]
    return graphs["head"](z=y.to(graphs.device) + 0.5)


def _stages():
    """The same model as a stage list."""
    encoder, head = _graphs()
    return (
        TorchStage(
            "prep", run=lambda state: {"x": state.batch["x"].float()}, inputs=(), outputs=("x",)
        ),
        encoder,
        TorchStage(
            "glue",
            run=lambda state: {"z": state["y"].to(state.device) + 0.5},
            inputs=("y",),
            outputs=("z",),
        ),
        head,
    )


def _batch():
    return {"x": torch.arange(6).reshape(2, 3)}


class TestDeployedPipeline(unittest.TestCase):
    """The deployed forward runs the declared graphs, on PyTorch and on the artifacts."""

    def test_pytorch_run_returns_the_function_outputs_and_times_graphs_and_glue(self):
        pipeline = DeployedPipeline(DeployedModel(_graphs(), _forward), Backend.PYTORCH, CPU)
        result = pipeline.infer(_batch())

        expected = _batch()["x"].float() * 2 + 0.5
        self.assertEqual(result.output_names, ("plus", "minus"))
        torch.testing.assert_close(result.outputs["plus"], expected + 1)
        torch.testing.assert_close(result.outputs["minus"], expected - 1)
        self.assertEqual(set(result.stage_times_ms), {"encoder", "head", GLUE_STAGE})
        self.assertEqual(result.graph_stage_names, ("encoder", "head"))

    def test_matches_the_stage_list_of_the_same_model(self):
        staged = StagedPipeline(_stages(), Backend.PYTORCH, CPU).infer(_batch())
        deployed = DeployedPipeline(DeployedModel(_graphs(), _forward), Backend.PYTORCH, CPU).infer(
            _batch()
        )

        self.assertEqual(deployed.output_names, staged.output_names)
        for name in staged.output_names:
            self.assertTrue(torch.equal(deployed.outputs[name], staged.outputs[name]), name)

    def test_export_specs_equal_the_stage_list_specs(self):
        from_stages = build_export_specs_from_stages(_stages(), _batch(), CPU)
        from_function = build_export_specs_from_deployed(
            DeployedModel(_graphs(), _forward), _batch(), CPU
        )

        self.assertEqual(list(from_function), list(from_stages))
        for name, spec in from_stages.items():
            other = from_function[name]
            self.assertEqual(other.input_names, spec.input_names)
            self.assertEqual(other.output_names, spec.output_names)
            self.assertEqual(len(other.args), len(spec.args))
            for ours, theirs in zip(other.args, spec.args):
                self.assertTrue(torch.equal(ours, theirs), name)

    def test_onnx_backend_runs_the_exported_artifacts(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        artifacts = Path(directory.name)
        declaration = DeployedModel(_graphs(), _forward)
        for name, spec in build_export_specs_from_deployed(declaration, _batch(), CPU).items():
            torch.onnx.export(
                spec.module,
                spec.args,
                artifacts / f"{name}.onnx",
                input_names=spec.input_names,
                output_names=spec.output_names,
                dynamo=False,
            )
        self.assertIn(Backend.ONNX, available_backends(declaration, artifacts))

        cache = PipelineCache(declaration, artifacts)
        reference = cache.get(Backend.PYTORCH, CPU).infer(_batch())
        onnx_pipeline = cache.get(Backend.ONNX, CPU)
        self.assertIsInstance(onnx_pipeline, DeployedPipeline)
        result = onnx_pipeline.infer(_batch())
        for name in reference.output_names:
            torch.testing.assert_close(
                onnx_pipeline.assemble(result)[name], reference.outputs[name], msg=name
            )

    def test_torch_fallback_graph_runs_its_module_and_is_reported(self):
        encoder, head = _graphs()
        fallback = GraphStage(
            "encoder",
            module=encoder.module,
            inputs=encoder.inputs,
            outputs=encoder.outputs,
            torch_fallback_backends=(Backend.ONNX,),
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        artifacts = Path(directory.name)
        torch.onnx.export(
            head.module,
            (torch.zeros(2, 3),),
            artifacts / "head.onnx",
            input_names=["z"],
            output_names=["plus", "minus"],
            dynamo=False,
        )
        pipeline = DeployedPipeline(
            DeployedModel((fallback, head), _forward), Backend.ONNX, CPU, artifacts
        )
        self.assertEqual(pipeline.fallback_stage_names, ("encoder",))
        torch.testing.assert_close(
            pipeline.infer(_batch()).outputs["plus"], _batch()["x"].float() * 2 + 1.5
        )


class TestDeployedForwardContract(unittest.TestCase):
    """What a run refuses: the mistakes a stage list rejects from its declaration."""

    def _run(self, forward):
        return DeployedPipeline(DeployedModel(_graphs(), forward), Backend.PYTORCH, CPU)

    def test_unknown_or_missing_graph_inputs_are_named(self):
        def forward(batch, graphs):
            return graphs["encoder"](features=batch["x"].float())

        with self.assertRaisesRegex(TypeError, r"takes \['x'\].*missing \['x'\].*'features'"):
            self._run(forward).infer(_batch())

    def test_a_graph_that_is_not_declared_is_named(self):
        def forward(batch, graphs):
            return graphs["backbone"](x=batch["x"])

        with self.assertRaisesRegex(KeyError, "No graph 'backbone'.*'encoder', 'head'"):
            self._run(forward).infer(_batch())

    def test_a_declared_graph_that_is_not_called_is_an_error(self):
        def forward(batch, graphs):
            return graphs["encoder"](x=batch["x"].float())

        with self.assertRaisesRegex(RuntimeError, r"did not call the declared graphs \['head'\]"):
            self._run(forward).infer(_batch())

    def test_a_graph_called_twice_is_an_error(self):
        def forward(batch, graphs):
            graphs["encoder"](x=batch["x"].float())
            return graphs["encoder"](x=batch["x"].float())

        with self.assertRaisesRegex(RuntimeError, "'encoder' was called twice"):
            self._run(forward).infer(_batch())

    def test_non_tensor_inputs_and_outputs_are_refused(self):
        def list_input(batch, graphs):
            return graphs["encoder"](x=[1.0, 2.0])

        with self.assertRaisesRegex(TypeError, "binds tensors only"):
            self._run(list_input).infer(_batch())

        def tuple_output(batch, graphs):
            y = graphs["encoder"](x=batch["x"].float())["y"]
            return tuple(graphs["head"](z=y).values())

        with self.assertRaisesRegex(TypeError, "non-empty mapping"):
            self._run(tuple_output).infer(_batch())

    def test_trace_refuses_a_graph_input_modified_after_the_graph_ran(self):
        def forward(batch, graphs):
            x = batch["x"].float()
            y = graphs["encoder"](x=x)["y"]
            x.add_(1.0)
            return graphs["head"](z=y)

        with self.assertRaisesRegex(ValueError, r"modified graph inputs.*'encoder': \['x'\]"):
            self._run(forward).trace(_batch())

    def test_declaration_refuses_duplicate_or_missing_graphs(self):
        encoder, _ = _graphs()
        with self.assertRaisesRegex(ValueError, "Duplicate graph names"):
            DeployedModel((encoder, encoder), _forward)
        with self.assertRaisesRegex(ValueError, "at least one graph"):
            DeployedModel((), _forward)


if __name__ == "__main__":
    unittest.main()
