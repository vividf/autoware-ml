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

"""Folding sparse bias adds and ReLUs into the ImplicitGemm plugin node."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from autoware_ml.ops.spconv.onnx_fusion import (
    ACT_RELU,
    fuse_implicit_gemm_bias_activation,
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


def test_bias_and_relu_fold_into_the_plugin_node() -> None:
    model = _graph(
        _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
        helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
        helper.make_node("Relu", ["biased"], ["activated"], name="relu"),
        outputs=["activated"],
        initializers=["bias"],
    )

    fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)

    assert (fused_biases, fused_activations) == (1, 1)
    assert [node.op_type for node in model.graph.node] == ["ImplicitGemm"]
    gemm = model.graph.node[0]
    # The bias arrives as the plugin's optional sixth input, and the node takes over the
    # chain's output name so downstream readers are unaffected.
    assert list(gemm.input) == ["features", "w", "p", "m", "a", "bias"]
    assert list(gemm.output) == ["activated"]
    assert _attribute(gemm, "act_type") == ACT_RELU


def test_bias_folds_without_an_activation() -> None:
    model = _graph(
        _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
        helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
        outputs=["biased"],
        initializers=["bias"],
    )

    fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)

    assert (fused_biases, fused_activations) == (1, 0)
    gemm = model.graph.node[0]
    assert list(gemm.output) == ["biased"]
    assert _attribute(gemm, "act_type") == 0


def test_a_relu_is_not_folded_without_the_bias_add() -> None:
    """The plugin activates after its bias add, so folding a ReLU across a separate
    bias add would change the result: relu(x) + b != relu(x + b)."""
    model = _graph(
        _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
        helper.make_node("Relu", ["gemm_out"], ["activated"], name="relu"),
        helper.make_node("Add", ["activated", "bias"], ["biased"], name="add"),
        outputs=["biased"],
        initializers=["bias"],
    )

    fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)

    assert (fused_biases, fused_activations) == (0, 0)
    assert [node.op_type for node in model.graph.node] == ["ImplicitGemm", "Relu", "Add"]


def test_a_shared_intermediate_is_left_alone() -> None:
    """A bias add read by a second consumer cannot be folded away."""
    model = _graph(
        _implicit_gemm(["features", "w", "p", "m", "a"], "gemm_out", "gemm"),
        helper.make_node("Add", ["gemm_out", "bias"], ["biased"], name="add"),
        helper.make_node("Relu", ["biased"], ["activated"], name="relu"),
        helper.make_node("Identity", ["biased"], ["kept"], name="identity"),
        outputs=["activated", "kept"],
        initializers=["bias"],
    )

    fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)

    assert fused_activations == 0
    assert "Relu" in [node.op_type for node in model.graph.node]
