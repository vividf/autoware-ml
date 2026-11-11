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

"""Unit tests for the deployment declaration in ``stages.py`` and ``types/backend.py``."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch
from torch import nn

from autoware_ml.deployment.stages import (
    DeployedModel,
    GraphStage,
    artifact_path,
    run_deployment_in_torch,
    validate_graphs,
)
from autoware_ml.types.backend import Backend


class _Double(nn.Module):
    def forward(self, x):
        return x * 2


class _DoubleInPlace(nn.Module):
    def forward(self, x):
        return x.mul_(2)


class _SplitHead(nn.Module):
    def forward(self, x):
        return x + 1, x - 1


class _DictHead(nn.Module):
    def forward(self, x):
        return {"plus": x + 1, "minus": x - 1}


class _Subtract(nn.Module):
    def forward(self, a, b):
        return a - b


class _NestedHead(nn.Module):
    def forward(self, x):
        return x + 1, (x - 1, x * 2)


class _ScalarHead(nn.Module):
    def forward(self, x):
        return x + 1, 3


def _toy_graphs(encoder: nn.Module | None = None, head: nn.Module | None = None):
    return (
        GraphStage("encoder", module=encoder or _Double(), inputs=("x",), outputs=("y",)),
        GraphStage("head", module=head or _SplitHead(), inputs=("z",), outputs=("plus", "minus")),
    )


def _toy_forward(batch, graphs):
    y = graphs["encoder"](x=batch["x"].float())["y"]
    return graphs["head"](z=y.to(graphs.device) + 0.5)


def _run(forward=_toy_forward, graphs=None, x=None):
    return run_deployment_in_torch(
        DeployedModel(_toy_graphs() if graphs is None else graphs, forward),
        {"x": torch.tensor([1.0]) if x is None else x},
        torch.device("cpu"),
    )


class TestGraphDeclaration(unittest.TestCase):
    """Unit tests for the graph declaration and its checks."""

    def test_validate_returns_a_tuple_in_declaration_order(self) -> None:
        """Test that validation returns the graphs as a tuple, in order."""
        graphs = validate_graphs(list(_toy_graphs()))
        self.assertIsInstance(graphs, tuple)
        self.assertEqual([graph.name for graph in graphs], ["encoder", "head"])

    def test_duplicate_graph_names_rejected(self) -> None:
        """Test that two graphs with one name are rejected: the name is the artifact name."""
        encoder, _ = _toy_graphs()
        with self.assertRaisesRegex(ValueError, r"Duplicate graph names: \['encoder'\]"):
            validate_graphs((encoder, encoder))

    def test_graph_stage_needs_inputs_and_outputs(self) -> None:
        """Test that a graph without inputs or outputs is rejected."""
        with self.assertRaisesRegex(ValueError, "must declare inputs and outputs"):
            GraphStage("g", module=nn.Identity(), inputs=(), outputs=("y",))

    def test_graph_stage_names_each_onnx_tensor_once(self) -> None:
        """Test that a graph naming one ONNX tensor twice is rejected."""
        with self.assertRaisesRegex(ValueError, r"lists \['x'\] more than once in its inputs"):
            GraphStage("g", module=_Subtract(), inputs=("x", "x"), outputs=("y",))
        with self.assertRaisesRegex(ValueError, r"lists \['y'\] more than once in its outputs"):
            GraphStage("g", module=_SplitHead(), inputs=("x",), outputs=("y", "y"))

    def test_dynamic_axes_name_the_graph_tensors_only(self) -> None:
        """Test that dynamic axes of a tensor the graph does not have are rejected."""
        with self.assertRaisesRegex(ValueError, r"onnx_dynamic_axes for \['z'\]"):
            GraphStage(
                "g",
                module=nn.Identity(),
                inputs=("x",),
                outputs=("y",),
                onnx_dynamic_axes={"x": {0: "n"}, "z": {0: "n"}},
            )

    def test_entries_other_than_graph_stages_rejected(self) -> None:
        """Test that a declaration holding something other than a GraphStage is rejected."""
        with self.assertRaisesRegex(TypeError, r"GraphStage entries, got \['Identity'\]"):
            validate_graphs((_toy_graphs()[0], nn.Identity()))

    def test_names_are_declared_as_tuples_not_strings(self) -> None:
        """Test that a bare string for a tuple of names is rejected."""
        with self.assertRaisesRegex(TypeError, r"inputs as the string 'x'; declare a tuple"):
            GraphStage("g", module=nn.Identity(), inputs="x", outputs=("y",))

    def test_no_graph_rejected(self) -> None:
        """Test that a deployment without a graph is rejected."""
        with self.assertRaisesRegex(ValueError, "at least one exportable GraphStage"):
            DeployedModel((), _toy_forward)

    def test_deployed_model_checks_its_graphs(self) -> None:
        """Test that the deployment checks and keeps its graphs as a tuple."""
        deployment = DeployedModel(list(_toy_graphs()), _toy_forward)
        self.assertIsInstance(deployment.graphs, tuple)
        encoder, _ = _toy_graphs()
        with self.assertRaisesRegex(ValueError, "Duplicate graph names"):
            DeployedModel((encoder, encoder), _toy_forward)


class TestRunDeploymentInTorch(unittest.TestCase):
    """Unit tests for running a deployment in PyTorch and its runtime checks."""

    def test_runs_the_function_and_returns_its_outputs(self) -> None:
        """Test that the function runs the graphs and its return value is the model outputs."""
        run = _run(x=torch.tensor([1, 2, 3]))
        # encoder: y=2x; glue: z=y+0.5; head: plus=z+1, minus=z-1
        self.assertEqual(list(run.outputs), ["plus", "minus"])
        self.assertTrue(torch.equal(run.outputs["plus"], torch.tensor([3.5, 5.5, 7.5])))
        self.assertTrue(torch.equal(run.outputs["minus"], torch.tensor([1.5, 3.5, 5.5])))
        self.assertEqual(list(run.graph_inputs), ["encoder", "head"])

    def test_keyword_inputs_bind_in_the_declared_order(self) -> None:
        """Test that a graph's arguments follow its declared ``inputs``, not the call's order."""
        graphs = (
            GraphStage("ab", module=_Subtract(), inputs=("a", "b"), outputs=("a_minus_b",)),
            GraphStage("ba", module=_Subtract(), inputs=("b", "a"), outputs=("b_minus_a",)),
        )

        def forward(batch, graphs):
            a, b = torch.tensor([10.0]), torch.tensor([1.0])
            return {**graphs["ab"](b=b, a=a), **graphs["ba"](a=a, b=b)}

        run = _run(forward, graphs)
        self.assertTrue(torch.equal(run.outputs["a_minus_b"], torch.tensor([9.0])))
        self.assertTrue(torch.equal(run.outputs["b_minus_a"], torch.tensor([-9.0])))

    def test_graph_mapping_return_is_rejected(self) -> None:
        """Test that a graph module returning a mapping is rejected."""
        with self.assertRaisesRegex(TypeError, "'head' returned a mapping"):
            _run(graphs=_toy_graphs(head=_DictHead()))

    def test_graph_output_arity_is_checked(self) -> None:
        """Test that a graph returning fewer tensors than it declares is rejected."""
        with self.assertRaisesRegex(ValueError, "returned 1 tensor"):
            _run(graphs=_toy_graphs(head=_Double()))

    def test_graph_outputs_are_tensors(self) -> None:
        """Test that a nested or non-tensor graph output is rejected."""
        for module, other in ((_NestedHead(), "tuple"), (_ScalarHead(), "int")):
            with self.subTest(other=other):
                with self.assertRaisesRegex(TypeError, rf"'head' returned \['{other}'\] among"):
                    _run(graphs=_toy_graphs(head=module))

    def test_graph_inputs_are_tensors(self) -> None:
        """Test that a graph input that is not a tensor is rejected."""

        def forward(batch, graphs):
            return graphs["head"](z=[graphs["encoder"](x=batch["x"])["y"]])

        with self.assertRaisesRegex(TypeError, r"'head' binds tensors only; got \{'z': 'list'\}"):
            _run(forward)

    def test_unknown_or_missing_input_names_are_named(self) -> None:
        """Test that a call with other keyword names than declared names both sides."""

        def forward(batch, graphs):
            return graphs["encoder"](features=batch["x"])

        with self.assertRaisesRegex(TypeError, r"missing \['x'\], unknown \['features'\]"):
            _run(forward)

    def test_an_undeclared_graph_is_named(self) -> None:
        """Test that calling a graph the model does not declare lists the declared ones."""

        def forward(batch, graphs):
            return graphs["backbone"](x=batch["x"])

        with self.assertRaisesRegex(KeyError, r"No graph 'backbone'.*\['encoder', 'head'\]"):
            _run(forward)

    def test_a_declared_graph_left_uncalled_is_rejected(self) -> None:
        """Test that a function skipping a declared graph is rejected: every graph is exported."""

        def forward(batch, graphs):
            return graphs["encoder"](x=batch["x"])

        with self.assertRaisesRegex(RuntimeError, r"did not call the declared graphs \['head'\]"):
            _run(forward)

    def test_a_graph_called_twice_is_rejected(self) -> None:
        """Test that a second call of one graph in a run is rejected."""

        def forward(batch, graphs):
            graphs["encoder"](x=batch["x"])
            return graphs["encoder"](x=batch["x"])

        with self.assertRaisesRegex(RuntimeError, "'encoder' was called twice"):
            _run(forward)

    def test_the_function_returns_named_tensors(self) -> None:
        """Test that a function returning no mapping, or a non-tensor output, is rejected."""

        def as_tuple(batch, graphs):
            y = graphs["encoder"](x=batch["x"])["y"]
            return tuple(graphs["head"](z=y).values())

        def with_a_list(batch, graphs):
            y = graphs["encoder"](x=batch["x"])["y"]
            return {**graphs["head"](z=y), "extra": [y]}

        with self.assertRaisesRegex(TypeError, "non-empty mapping of name -> tensor, got tuple"):
            _run(as_tuple)
        with self.assertRaisesRegex(TypeError, r"not tensors: \['extra'\]"):
            _run(with_a_list)

    def test_graph_module_modifying_its_input_in_place_is_rejected(self) -> None:
        """Test that a graph module modifying its input in place is rejected."""
        with self.assertRaisesRegex(ValueError, "GraphStage 'encoder' modified its inputs"):
            _run(graphs=_toy_graphs(encoder=_DoubleInPlace()))

    def test_a_graph_input_modified_after_the_call_is_rejected(self) -> None:
        """Test that the function modifying a graph's input after the call is rejected."""

        def forward(batch, graphs):
            x = batch["x"].float()
            y = graphs["encoder"](x=x)["y"]
            x.add_(100.0)
            return graphs["head"](z=y)

        with self.assertRaisesRegex(ValueError, r"after the graphs ran: \{'encoder': \['x'\]\}"):
            _run(forward)

    def test_graph_inputs_are_the_tensors_the_graph_ran_on(self) -> None:
        """Test that the recorded graph inputs are the tensors the function passed."""
        seen = {}

        def forward(batch, graphs):
            y = graphs["encoder"](x=batch["x"])["y"]
            seen["z"] = y + 0.5
            return graphs["head"](z=seen["z"])

        run = _run(forward)
        (head_input,) = run.graph_inputs["head"]
        self.assertIs(head_input, seen["z"])


class TestBackend(unittest.TestCase):
    """Unit tests for the backend names and their artifact paths."""

    def test_artifact_path_per_backend(self) -> None:
        """Test that each exporting backend has its artifact path and pytorch has none."""
        self.assertEqual(artifact_path("/tmp", "s", Backend.ONNX), Path("/tmp/s.onnx"))
        self.assertEqual(artifact_path("/tmp", "s", Backend.TENSORRT), Path("/tmp/s.engine"))
        with self.assertRaisesRegex(ValueError, "no exported artifact"):
            artifact_path("/tmp", "s", Backend.PYTORCH)

    def test_backend_name_formats_as_its_config_spelling(self) -> None:
        """Test that a backend formats as its config spelling and an unknown one is rejected."""
        self.assertEqual(f"{Backend.TENSORRT}", "tensorrt")
        self.assertIs(Backend("onnx"), Backend.ONNX)
        with self.assertRaisesRegex(ValueError, "Unknown backend 'tensorflow'"):
            Backend("tensorflow")


if __name__ == "__main__":
    unittest.main()
