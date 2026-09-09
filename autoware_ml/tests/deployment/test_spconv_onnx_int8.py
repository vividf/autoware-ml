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

"""Writing calibrated sparse scales into the exported ImplicitGemm nodes."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from autoware_ml.ops.spconv.onnx_int8 import (
    PRECISION_INT8,
    SparseLayerScales,
    quantize_implicit_gemm_nodes,
)

_C_OUT = 3
_C_IN = 2


def _attributes(node: onnx.NodeProto) -> dict[str, object]:
    return {
        attribute.name: (attribute.i if attribute.type == attribute.INT else attribute.f)
        for attribute in node.attribute
    }


def _graph(*, stems: list[str], with_bias: bool = True) -> onnx.ModelProto:
    """A graph with one ImplicitGemm per stem, shaped like the exported sparse tower."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    for index, stem in enumerate(stems):
        weight_name = f"extractor.pts_middle_encoder.{stem}.weight"
        inputs = [f"features_{index}", weight_name, "pair", "mask", "argsort"]
        initializers.append(
            numpy_helper.from_array(
                np.zeros((_C_OUT, 3, 3, 3, _C_IN), dtype=np.float16), name=weight_name
            )
        )
        if with_bias:
            bias_name = f"extractor.pts_middle_encoder.{stem}.bias"
            inputs.append(bias_name)
            initializers.append(
                numpy_helper.from_array(np.arange(_C_OUT, dtype=np.float16) + 1, name=bias_name)
            )
        node = helper.make_node(
            "ImplicitGemm",
            inputs,
            [f"out_{index}"],
            name=f"/{stem}/ImplicitGemm",
            domain="autoware",
        )
        node.attribute.append(helper.make_attribute("act_type", 1))
        nodes.append(node)
    graph = helper.make_graph(
        nodes,
        "sparse",
        [
            helper.make_tensor_value_info(f"features_{i}", TensorProto.FLOAT16, ["n", _C_IN])
            for i in range(len(stems))
        ],
        [
            helper.make_tensor_value_info(f"out_{i}", TensorProto.FLOAT16, ["n", _C_OUT])
            for i in range(len(stems))
        ],
        initializer=initializers,
    )
    return helper.make_model(graph)


def _scales(input_scale: float = 0.04) -> SparseLayerScales:
    return SparseLayerScales(
        input_scale=input_scale,
        weight_scale=np.array([0.01, 0.02, 0.03], dtype=np.float32),
    )


def test_calibrated_node_takes_the_plugin_int8_contract():
    model = _graph(stems=["conv_input.0"])
    layer = _scales()

    converted, kept_fp = quantize_implicit_gemm_nodes(model, {"conv_input.0": layer})

    assert (converted, kept_fp) == (1, 0)
    node = model.graph.node[0]
    assert len(node.input) == 7
    attributes = _attributes(node)
    assert attributes["precision"] == PRECISION_INT8
    assert attributes["input_scale"] == pytest.approx(layer.input_scale)
    # The activation the export fusion folded in must survive the rewrite.
    assert attributes["act_type"] == 1
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    channel_scale = initializers[node.input[5]]
    bias_scaled = initializers[node.input[6]]
    assert channel_scale.dtype == np.float32 and bias_scaled.dtype == np.float32
    np.testing.assert_allclose(channel_scale, layer.input_scale * layer.weight_scale, rtol=1e-6)
    np.testing.assert_allclose(bias_scaled, [1.0, 2.0, 3.0])


def test_output_scale_cancels_out_of_the_plugin_arithmetic():
    """``output_scale`` is inert, which is why the graph pins it to 1.

    The plugin recovers the weight scale as ``channel_scale * output_scale / input_scale``
    and then folds ``output_scale`` back into the GEMM's scale and bias. Whatever value it
    carries, the dequantized result is the same — so the exported graph does not need an
    activation-chain output scale, and this test is what says so.
    """
    layer = _scales()
    accumulator = np.array([100.0, -50.0, 25.0], dtype=np.float32)  # INT32 GEMM output
    bias = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def plugin_output(output_scale: float) -> np.ndarray:
        channel_scale = layer.channel_scale / output_scale
        bias_scaled = bias / output_scale
        # ImplicitGemmPlugin::enqueueInt8 (quantize_features.cu).
        gemm_scale = channel_scale * output_scale
        gemm_bias = bias_scaled * output_scale
        weight_scale = channel_scale * output_scale / layer.input_scale
        np.testing.assert_allclose(weight_scale, layer.weight_scale, rtol=1e-6)
        return accumulator * gemm_scale + gemm_bias

    np.testing.assert_allclose(plugin_output(1.0), plugin_output(0.017), rtol=1e-6)


def test_uncalibrated_nodes_stay_floating_point():
    model = _graph(stems=["conv_input.0", "encoder_layers.encoder_layer4.0.conv1"])

    converted, kept_fp = quantize_implicit_gemm_nodes(
        model, {"encoder_layers.encoder_layer4.0.conv1": _scales()}
    )

    assert (converted, kept_fp) == (1, 1)
    kept = model.graph.node[0]
    assert len(kept.input) == 6
    assert "precision" not in _attributes(kept)


def test_node_without_a_folded_bias_gets_zeros():
    model = _graph(stems=["conv_out.0"], with_bias=False)

    quantize_implicit_gemm_nodes(model, {"conv_out.0": _scales()})

    node = model.graph.node[0]
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    np.testing.assert_array_equal(initializers[node.input[6]], np.zeros(_C_OUT, dtype=np.float32))


def test_a_calibrated_layer_with_no_node_is_an_error():
    model = _graph(stems=["conv_input.0"])

    with pytest.raises(ValueError, match="no ImplicitGemm node"):
        quantize_implicit_gemm_nodes(model, {"conv_input.0": _scales(), "missing.0": _scales()})


def test_channel_scale_must_match_the_filter():
    model = _graph(stems=["conv_input.0"])
    wrong = SparseLayerScales(input_scale=0.04, weight_scale=np.array([0.01], dtype=np.float32))

    with pytest.raises(ValueError, match="channel scales"):
        quantize_implicit_gemm_nodes(model, {"conv_input.0": wrong})
