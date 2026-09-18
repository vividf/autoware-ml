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


"""ONNX Runtime runner: session options follow the graph's quantization."""

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from autoware_ml.deployment.backends.onnx_runner import OnnxModuleRunner


def _gemm_graph(tmp_path, quantized: bool):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    inits = [weight]
    if quantized:
        inits += [
            helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(0.1)]),
            helper.make_tensor("zp", TensorProto.INT8, [], [0]),
        ]
        nodes = [
            helper.make_node("QuantizeLinear", ["x", "s", "zp"], ["q"]),
            helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"]),
            helper.make_node("Gemm", ["dq", "w"], ["y"]),
        ]
    else:
        nodes = [helper.make_node("Gemm", ["x", "w"], ["y"])]
    model = helper.make_model(
        helper.make_graph(nodes, "g", [x], [y], inits), opset_imports=[helper.make_opsetid("", 19)]
    )
    path = tmp_path / ("qdq.onnx" if quantized else "plain.onnx")
    onnx.save(model, str(path))
    return path


def test_runner_disables_graph_optimizations_for_qdq_graphs(tmp_path) -> None:
    """A quantized graph runs exactly as exported: ORT's optimizer mis-executes fp16 Q/DQ."""

    runner = OnnxModuleRunner(_gemm_graph(tmp_path, quantized=True), torch.device("cpu"))
    level = runner.session.get_session_options().graph_optimization_level
    assert level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    outputs, _ = runner.run({"x": torch.arange(8, dtype=torch.float32).reshape(2, 4)})
    assert torch.allclose(
        outputs["y"], torch.arange(8, dtype=torch.float32).reshape(2, 4), atol=0.06
    )


def test_runner_keeps_default_optimizations_for_plain_graphs(tmp_path) -> None:
    runner = OnnxModuleRunner(_gemm_graph(tmp_path, quantized=False), torch.device("cpu"))
    level = runner.session.get_session_options().graph_optimization_level
    assert level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
