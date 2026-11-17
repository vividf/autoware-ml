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

"""Unit tests for writing calibrated sparse scales into ImplicitGemm nodes."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from autoware_ml.ops.spconv.onnx_fusion import fuse_implicit_gemm_bias_activation
from autoware_ml.ops.spconv.onnx_int8 import (
    SparseLayerScales,
    quantize_implicit_gemm_nodes,
)

_C_OUT = 3
_C_IN = 2


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _weights(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(_C_OUT, 3, 3, 3, _C_IN)).astype(np.float32)


def _graph(*, stems: list[str], with_bias: bool = True) -> onnx.ModelProto:
    """One exported ImplicitGemm per stem (plus a bias Add), run through the bias fusion."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    for index, stem in enumerate(stems):
        weight_name = f"extractor.pts_middle_encoder.{stem}.weight"
        initializers.append(numpy_helper.from_array(_weights(index), name=weight_name))
        node = helper.make_node(
            "ImplicitGemm",
            [f"features_{index}", weight_name, "pair", "mask", "argsort"],
            [f"gemm_{index}" if with_bias else f"out_{index}"],
            name=f"/{stem}/ImplicitGemm",
            domain="autoware",
        )
        node.attribute.extend(
            [helper.make_attribute("act_type", 0), helper.make_attribute("output_scale", 1.0)]
        )
        nodes.append(node)
        if with_bias:
            bias_name = f"extractor.pts_middle_encoder.{stem}.bias"
            initializers.append(
                numpy_helper.from_array(np.arange(_C_OUT, dtype=np.float32) + 1, name=bias_name)
            )
            nodes.append(helper.make_node("Add", [f"gemm_{index}", bias_name], [f"out_{index}"]))
    graph = helper.make_graph(
        nodes,
        "sparse",
        [
            helper.make_tensor_value_info(f"features_{i}", TensorProto.FLOAT, ["n", _C_IN])
            for i in range(len(stems))
        ]
        + [
            helper.make_tensor_value_info(name, TensorProto.INT32, ["k"])
            for name in ("pair", "mask", "argsort")
        ],
        [
            helper.make_tensor_value_info(f"out_{i}", TensorProto.FLOAT, ["n", _C_OUT])
            for i in range(len(stems))
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)]
    )
    fuse_implicit_gemm_bias_activation(model)
    return model


def _scales(input_scale: float = 0.04) -> SparseLayerScales:
    return SparseLayerScales(
        input_scale=input_scale,
        weight_scale=np.abs(_weights()).reshape(_C_OUT, -1).max(axis=1) / 127.0,
    )


def _gemm(model: onnx.ModelProto) -> onnx.NodeProto:
    return next(node for node in model.graph.node if node.op_type == "ImplicitGemm")


class TestQuantizeImplicitGemmNodes(unittest.TestCase):
    """A calibrated ImplicitGemm node takes the quantized row of the shared plugin contract."""

    def test_calibrated_node_takes_the_contract_int8_row(self) -> None:
        """
        Input: one exported node with a folded bias and its calibrated scales.
        Expected: a QuantizeLinear feeds it INT8 features; INT8 filters, weight_scale in
        input 5, the bias in input 6, input_scale and output_type "f16", no output_scale.
        Check: the node, its Quantize and its initializers.
        """
        model = _graph(stems=["conv_input.0"])
        layer = _scales()

        converted, kept_fp = quantize_implicit_gemm_nodes(model, {"conv_input.0": layer})

        self.assertEqual((converted, kept_fp), (1, 0))
        node = _gemm(model)
        self.assertEqual(len(node.input), 7)
        attributes = _attributes(node)
        input_scale = float(np.float16(layer.input_scale))
        self.assertAlmostEqual(attributes["input_scale"], input_scale, places=7)
        self.assertEqual(attributes["output_type"], b"f16")
        self.assertNotIn("output_scale", attributes)
        self.assertNotIn("precision", attributes)
        self.assertEqual(attributes["act_type"], 0)

        quantize = model.graph.node[0]
        self.assertEqual(quantize.op_type, "QuantizeLinear")
        self.assertEqual(list(quantize.output), [node.input[0]])
        self.assertEqual(quantize.input[0], "features_0")
        initializers = _initializers(model)
        self.assertAlmostEqual(float(initializers[quantize.input[1]]), input_scale, places=7)
        self.assertEqual(initializers[quantize.input[2]].dtype, np.int8)

        filters = initializers[node.input[1]]
        self.assertEqual(filters.dtype, np.int8)
        self.assertEqual(np.abs(filters).max(), 127)
        weight_scale = initializers[node.input[5]]
        self.assertEqual(weight_scale.dtype, np.float32)
        np.testing.assert_array_equal(weight_scale, layer.weight_scale.astype(np.float32))
        # The folded bias stays where the fusion put it, in real units.
        self.assertEqual(node.input[6], "extractor.pts_middle_encoder.conv_input.0.bias")
        np.testing.assert_array_equal(initializers[node.input[6]], [1.0, 2.0, 3.0])
        onnx.checker.check_model(model)

    def test_the_plugin_epilogue_gives_back_the_convolution(self) -> None:
        """
        Input: the written node, and features quantized with its input_scale.
        Expected: ``input_scale * weight_scale[n] * acc + bias[n]`` over the written codes is
        the float convolution, up to quantization error.
        Check: the codes give back the weights to within half a step, and the epilogue
        result matches the float reference.
        """
        model = _graph(stems=["conv_input.0"])
        quantize_implicit_gemm_nodes(model, {"conv_input.0": _scales()})
        node = _gemm(model)
        initializers = _initializers(model)
        input_scale = _attributes(node)["input_scale"]
        filter_codes = initializers[node.input[1]].astype(np.float32)
        weight_scale = initializers[node.input[5]]

        per_channel = weight_scale.reshape(-1, 1, 1, 1, 1)
        self.assertTrue(
            np.all(np.abs(filter_codes * per_channel - _weights()) <= per_channel / 2 + 1e-7)
        )

        # One output point that sees every tap: the convolution is a dot product over
        # k^3 * C_in.
        features = np.random.default_rng(1).uniform(-2.0, 2.0, size=(27, _C_IN))
        features = features.astype(np.float32)
        feature_codes = np.clip(np.rint(features / input_scale), -128, 127)
        accumulator = np.einsum("nkc,kc->n", filter_codes.reshape(_C_OUT, 27, _C_IN), feature_codes)
        plugin = input_scale * weight_scale * accumulator + initializers[node.input[6]]
        reference = np.einsum("nkc,kc->n", _weights().reshape(_C_OUT, 27, _C_IN), features)
        np.testing.assert_allclose(plugin, reference + [1, 2, 3], rtol=0.03)

    def test_uncalibrated_nodes_stay_floating_point(self) -> None:
        """
        Input: two nodes, only one of them calibrated.
        Expected: the other keeps the exported seven-input form.
        Check: an empty weight scale, no input_scale and float filters.
        """
        model = _graph(stems=["conv_input.0", "encoder_layers.encoder_layer4.0.conv1"])

        converted, kept_fp = quantize_implicit_gemm_nodes(
            model, {"encoder_layers.encoder_layer4.0.conv1": _scales()}
        )

        self.assertEqual((converted, kept_fp), (1, 1))
        kept = next(node for node in model.graph.node if node.name == "/conv_input.0/ImplicitGemm")
        self.assertEqual(len(kept.input), 7)
        self.assertEqual(_initializers(model)[kept.input[5]].shape, (0,))
        self.assertNotIn("input_scale", _attributes(kept))
        self.assertEqual(_initializers(model)[kept.input[1]].dtype, np.float32)

    def test_node_without_a_folded_bias_gets_an_empty_one(self) -> None:
        """
        Input: a calibrated five-input node.
        Expected: it gains input 5 and an empty FP32 bias in input 6.
        Check: seven inputs and a zero-length FP32 bias.
        """
        model = _graph(stems=["conv_out.0"], with_bias=False)

        quantize_implicit_gemm_nodes(model, {"conv_out.0": _scales()})

        node = _gemm(model)
        self.assertEqual(len(node.input), 7)
        bias = _initializers(model)[node.input[6]]
        self.assertEqual(bias.dtype, np.float32)
        self.assertEqual(bias.shape, (0,))


class TestQuantizeImplicitGemmNodesRefuses(unittest.TestCase):
    """Graphs and scales that do not fit the contract are refused, naming the problem."""

    def test_six_inputs_are_not_a_contract_form(self) -> None:
        """A six-input node (a bias in slot 5) has no place in the contract."""
        model = _graph(stems=["conv_input.0"])
        del _gemm(model).input[5]

        with self.assertRaisesRegex(ValueError, "has 6 inputs"):
            quantize_implicit_gemm_nodes(model, {"conv_input.0": _scales()})

    def test_a_calibrated_layer_with_no_node_is_an_error(self) -> None:
        """A calibrated layer the graph does not contain would deploy in FP16 unnoticed."""
        model = _graph(stems=["conv_input.0"])

        with self.assertRaisesRegex(ValueError, "no ImplicitGemm node"):
            quantize_implicit_gemm_nodes(model, {"conv_input.0": _scales(), "missing.0": _scales()})

    def test_weight_scale_must_match_the_filter(self) -> None:
        """One weight scale per output channel of the filter."""
        model = _graph(stems=["conv_input.0"])
        wrong = SparseLayerScales(input_scale=0.04, weight_scale=np.array([0.01], np.float32))

        with self.assertRaisesRegex(ValueError, "weight scales"):
            quantize_implicit_gemm_nodes(model, {"conv_input.0": wrong})
