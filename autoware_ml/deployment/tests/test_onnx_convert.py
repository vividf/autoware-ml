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


"""Unit tests for the fp16 conversion dispatch."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from autoware_ml.deployment.onnx.convert import convert_onnx_precision
from autoware_ml.utils.onnx_precision import OnnxPrecision


def _qdq_gemm_model() -> onnx.ModelProto:
    scale = helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(1e-4)])
    zero_point = helper.make_tensor("zp", TensorProto.INT8, [], [0])
    weight = numpy_helper.from_array(np.eye(4, dtype=np.float32), name="w")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["q"], name="q"),
        helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"], name="dq"),
        helper.make_node("Gemm", ["dq", "w"], ["out"], name="gemm"),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq_gemm",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4])],
        initializer=[scale, zero_point, weight],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])


def test_fp16_conversion_routes_quantized_graphs_to_the_island_cast(tmp_path) -> None:
    onnx_path = tmp_path / "qdq.onnx"
    onnx.save(_qdq_gemm_model(), onnx_path.as_posix())

    convert_onnx_precision(onnx_path, OnnxPrecision.FP16)

    model = onnx.load(onnx_path.as_posix())
    inits = {i.name: i for i in model.graph.initializer}
    assert inits["s"].data_type == TensorProto.FLOAT
    assert numpy_helper.to_array(inits["s"]) == np.float32(1e-4)
    assert inits["w"].data_type == TensorProto.FLOAT
    nodes = {n.name: n for n in model.graph.node}
    assert nodes["gemm"].input[0] == nodes["dq"].output[0]
    for value_info in list(model.graph.input) + list(model.graph.output):
        assert value_info.type.tensor_type.elem_type == TensorProto.FLOAT
    onnx.checker.check_model(model, full_check=True)


def test_fp16_conversion_routes_plugin_graphs_to_the_island_cast(tmp_path) -> None:
    weight = numpy_helper.from_array(np.ones(4, dtype=np.float32), name="w")
    graph = helper.make_graph(
        [
            helper.make_node("PluginOp", ["x", "w"], ["mid"], domain="autoware", name="plugin"),
            helper.make_node("Relu", ["mid"], ["out"], name="relu"),
        ],
        "plugin",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, [2, 4])],
        initializer=[weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)]
    )
    onnx_path = tmp_path / "plugin.onnx"
    onnx.save(model, onnx_path.as_posix())

    convert_onnx_precision(onnx_path, OnnxPrecision.FP16)

    converted = onnx.load(onnx_path.as_posix())
    assert {i.name: i.data_type for i in converted.graph.initializer}["w"] == TensorProto.FLOAT16
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
