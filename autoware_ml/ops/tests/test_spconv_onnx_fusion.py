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

"""Unit tests for folding the bias add and ReLU into the ImplicitGemm plugin node."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from autoware_ml.ops.spconv.contract import (
    IMPLICIT_GEMM_BIAS_SLOT,
    IMPLICIT_GEMM_INPUTS,
    IMPLICIT_GEMM_WEIGHT_SCALE_SLOT,
)
from autoware_ml.ops.spconv.onnx_fusion import (
    ACT_RELU,
    fuse_implicit_gemm_bias_activation,
    fuse_sparse_graph,
)


def _implicit_gemm(inputs: list[str], output: str, name: str) -> onnx.NodeProto:
    node = helper.make_node("ImplicitGemm", inputs, [output], name=name, domain="autoware")
    node.attribute.append(helper.make_attribute("act_type", 0))
    return node


def _graph(*nodes: onnx.NodeProto, outputs: list[str], initializers: list[str]) -> onnx.ModelProto:
    features = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", 4])
    graph = helper.make_graph(
        list(nodes),
        "sparse",
        [features],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["n", 4]) for name in outputs],
        [
            helper.make_tensor(name, TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32))
            for name in initializers
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)],
    )


def _attribute(node: onnx.NodeProto, name: str) -> int:
    return next(attribute.i for attribute in node.attribute if attribute.name == name)


def _initializer(model: onnx.ModelProto, name: str) -> onnx.TensorProto:
    return next(tensor for tensor in model.graph.initializer if tensor.name == name)


class TestFuseImplicitGemmBiasActivation(unittest.TestCase):
    """Folding the bias Add and ReLU after an ImplicitGemm into the seven-input node."""

    def test_bias_and_relu_fold_into_the_plugin_node(self) -> None:
        """ImplicitGemm -> Add(bias) -> Relu becomes one node with the bias and act_type."""
        model = _graph(
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
            helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
            helper.make_node("Relu", ["biased"], ["activated"], name="relu"),
            outputs=["activated"],
            initializers=["bias"],
        )

        fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)

        self.assertEqual((fused_biases, fused_activations), (1, 1))
        self.assertEqual([node.op_type for node in model.graph.node], ["ImplicitGemm"])
        gemm = model.graph.node[0]
        self.assertEqual(
            list(gemm.input), ["features", "w", "p", "m", "a", "w.weight_scale", "bias"]
        )
        self.assertEqual(list(gemm.output), ["activated"])
        self.assertEqual(_attribute(gemm, "act_type"), ACT_RELU)

    def test_a_folded_bias_takes_the_seven_input_form(self) -> None:
        """An empty FP32 weight scale named after the module goes in slot 5, the bias in 6."""
        model = _graph(
            _implicit_gemm(["features", "cpe.0.weight", "p", "m", "a"], "gemm_out", "gemm"),
            helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
            outputs=["biased"],
            initializers=["bias"],
        )

        fuse_implicit_gemm_bias_activation(model)

        gemm = model.graph.node[0]
        self.assertEqual(len(gemm.input), IMPLICIT_GEMM_INPUTS)
        weight_scale = _initializer(model, gemm.input[IMPLICIT_GEMM_WEIGHT_SCALE_SLOT])
        self.assertEqual(weight_scale.name, "cpe.0.weight_scale")
        self.assertEqual(
            (weight_scale.data_type, list(weight_scale.dims)), (TensorProto.FLOAT, [0])
        )
        self.assertEqual(gemm.input[IMPLICIT_GEMM_BIAS_SLOT], "bias")

    def test_a_half_precision_bias_is_folded_as_fp32(self) -> None:
        """Both plugins read slot 6 as FP32, so an fp16 bias is folded as an FP32 copy."""
        model = _graph(
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
            helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
            outputs=["biased"],
            initializers=[],
        )
        bias = np.array([0.5, -1.25, 2.0, 3.0], dtype=np.float16)
        model.graph.initializer.append(numpy_helper.from_array(bias, "bias"))

        fuse_implicit_gemm_bias_activation(model)

        folded = _initializer(model, model.graph.node[0].input[IMPLICIT_GEMM_BIAS_SLOT])
        self.assertEqual(folded.data_type, TensorProto.FLOAT)
        np.testing.assert_array_equal(numpy_helper.to_array(folded), bias.astype(np.float32))

    def test_bias_folds_without_an_activation(self) -> None:
        """A bias Add alone folds and leaves act_type at none."""
        model = _graph(
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
            helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
            outputs=["biased"],
            initializers=["bias"],
        )

        fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)

        self.assertEqual((fused_biases, fused_activations), (1, 0))
        gemm = model.graph.node[0]
        self.assertEqual(list(gemm.output), ["biased"])
        self.assertEqual(_attribute(gemm, "act_type"), 0)

    def test_a_relu_is_not_folded_without_the_bias_add(self) -> None:
        """The plugin computes relu(gemm + b): a ReLU before the Add stays in the graph."""
        model = _graph(
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
            helper.make_node("Relu", ["gemm_out"], ["activated"], name="relu"),
            helper.make_node("Add", ["activated", "bias"], ["biased"], name="add"),
            outputs=["biased"],
            initializers=["bias"],
        )

        fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)

        self.assertEqual((fused_biases, fused_activations), (0, 0))
        self.assertEqual(
            [node.op_type for node in model.graph.node], ["ImplicitGemm", "Relu", "Add"]
        )

    def test_a_shared_intermediate_is_left_alone(self) -> None:
        """A biased tensor another node also reads keeps its ReLU outside the plugin."""
        model = _graph(
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
            helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
            helper.make_node("Relu", ["biased"], ["activated"], name="relu"),
            helper.make_node("Identity", ["biased"], ["kept"], name="identity"),
            outputs=["activated", "kept"],
            initializers=["bias"],
        )

        _, fused_activations = fuse_implicit_gemm_bias_activation(model)

        self.assertEqual(fused_activations, 0)
        self.assertIn("Relu", [node.op_type for node in model.graph.node])


class TestFuseSparseGraph(unittest.TestCase):
    """The file-level pass: saving the folded graph and reporting what did not fold."""

    def setUp(self) -> None:
        """Give each test its own directory for the ONNX file."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_an_unfoldable_plugin_node_is_reported(self) -> None:
        """No initializer feeds the Add, so there is no bias to fold, and the pass warns."""
        model = _graph(
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
            helper.make_node("Add", ["gemm_out", "features"], ["biased"], name="add"),
            outputs=["biased"],
            initializers=["w", "p", "m", "a"],
        )
        path = self.tmp_path / "unfused.onnx"
        onnx.save(model, str(path))

        with self.assertLogs("autoware_ml.ops.spconv.onnx_fusion", level="WARNING") as logs:
            fuse_sparse_graph(path)

        self.assertIn("kept their bias Add outside the plugin", "\n".join(logs.output))

    def test_structurally_identical_nodes_are_removed_by_identity(self) -> None:
        """Two equal Adds are removed by identity, not by value, and both nodes fold."""
        model = _graph(
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_a_out", "gemm_a"),
            _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_b_out", "gemm_b"),
            helper.make_node("Add", ["gemm_a_out", "bias"], ["a_biased"]),
            helper.make_node("Add", ["gemm_b_out", "bias"], ["b_biased"]),
            helper.make_node("Mul", ["a_biased", "b_biased"], ["out"], name="join"),
            outputs=["out"],
            initializers=["w", "p", "m", "a", "bias"],
        )
        path = self.tmp_path / "twin_adds.onnx"
        onnx.save(model, str(path))

        fuse_sparse_graph(path)

        fused = onnx.load(str(path))
        self.assertFalse([node for node in fused.graph.node if node.op_type == "Add"])
        gemms = [node for node in fused.graph.node if node.op_type == "ImplicitGemm"]
        self.assertEqual(len(gemms), 2)
        self.assertTrue(all(len(node.input) == IMPLICIT_GEMM_INPUTS for node in gemms))
        # The twins share their weight, so they share the one empty weight scale too.
        names = [tensor.name for tensor in fused.graph.initializer]
        self.assertEqual(names.count("w.weight_scale"), 1)
        onnx.checker.check_model(fused)
