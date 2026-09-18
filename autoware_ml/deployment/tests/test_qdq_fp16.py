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

"""The island-aware fp16 cast and the Q/DQ parameter fold, on hand-built ONNX graphs."""

from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16
from autoware_ml.deployment.onnx.qdq import fold_qdq_params


def producer_named(model, tensor_name):
    return next(node for node in model.graph.node if tensor_name in node.output)


def test_cast_graph_to_fp16_converts_internals_and_keeps_io(tmp_path) -> None:
    """Plugin graphs go FP16 wholesale: initializers and internal casts become FP16,
    while the graph I/O (the artifact ABI) stays FP32."""

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    n = helper.make_tensor_value_info("n", TensorProto.INT32, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    weight = helper.make_tensor("w", TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32))
    graph = helper.make_graph(
        [
            helper.make_node("PluginOp", ["x", "w"], ["mid"], domain="autoware"),
            # A pre-existing int->float cast: the converter leaves its target FLOAT,
            # which would meet FP16 tensors downstream.
            helper.make_node("Cast", ["n"], ["n_float"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["n_float", "axes"], ["n_col"]),
            helper.make_node("Div", ["mid", "n_col"], ["y"]),
        ],
        "plugin_graph",
        [x, n],
        [y],
        [weight, helper.make_tensor("axes", TensorProto.INT64, [1], [1])],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)],
    )
    path = tmp_path / "plugin_graph.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    weights = {i.name: i.data_type for i in converted.graph.initializer}
    assert weights["w"] == TensorProto.FLOAT16
    casts = {
        node.output[0]: next(a.i for a in node.attribute if a.name == "to")
        for node in converted.graph.node
        if node.op_type == "Cast"
    }
    assert casts["n_float"] == TensorProto.FLOAT16
    # Boundary casts keep the ABI: the output-feeding cast stays FLOAT.
    assert TensorProto.FLOAT in casts.values()
    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_cast_graph_to_fp16_rewires_internal_consumers_of_kept_fp32_outputs(tmp_path) -> None:
    """A graph output that is also consumed internally must not feed the FP32 copy.

    ``keep_io_types`` inserts the boundary Cast under the output's own name (PTv3's
    encoder emits its per-stage point features and keeps pooling them), so an internal
    consumer would read FP32 and meet FP16 weights.
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    feat = helper.make_tensor_value_info("feat", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    weight = helper.make_tensor("w", TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32))
    graph = helper.make_graph(
        [
            # A plugin op keeps AutoCast out and forces the whole-graph cast.
            helper.make_node("PluginOp", ["x", "w"], ["feat"], domain="autoware"),
            # `feat` is both a graph output and an internal input.
            helper.make_node("Mul", ["feat", "w"], ["y"]),
        ],
        "reused_output_graph",
        [x],
        [feat, y],
        [weight],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)],
    )
    path = tmp_path / "reused_output_graph.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    boundary_casts = [
        node for node in converted.graph.node if node.op_type == "Cast" and node.output[0] == "feat"
    ]
    assert len(boundary_casts) == 1, "the kept-FP32 output should come from one boundary cast"
    mul = next(node for node in converted.graph.node if node.op_type == "Mul")
    # The internal consumer reads the FP16 tensor the boundary cast came from, not "feat".
    assert mul.input[0] == boundary_casts[0].input[0] != "feat"
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT


def test_cast_graph_to_fp16_keeps_qdq_islands_fp32_and_castless(tmp_path) -> None:
    """A Q/DQ graph converts to FP16 *around* intact FP32 linear islands.

    The island — the Gemm's Q/DQ, their scale, and the Gemm itself — must come through
    byte-identical and with no Cast on its internal edges: an FP16-rounded scale on an
    INT8 Gemm hits TensorRT's subnormal-combined-scale NaN, and a Cast between DQ and
    its consumer defeats the Q/DQ fusion. A Q/DQ pair that feeds anything else (here a
    Mul) is sea: retyped to fp16 with its own fp16 scale copy.
    """

    # 0.0001 is not representable in fp16 (rounds to ~0.00010002); a round trip shows.
    scale_value = np.float32(1e-4)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    bias = helper.make_tensor("b", TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32))
    gain = helper.make_tensor("g", TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32))
    scale = helper.make_tensor("s", TensorProto.FLOAT, [], [scale_value])
    zero_point = helper.make_tensor("zp", TensorProto.INT8, [], [0])
    graph = helper.make_graph(
        [
            # A plugin op routes the graph into the whole-graph cast in the first place.
            helper.make_node("PluginOp", ["x", "g"], ["mid"], domain="autoware", name="plugin"),
            helper.make_node("QuantizeLinear", ["mid", "s", "zp"], ["q"], name="q"),
            helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"], name="dq"),
            helper.make_node("QuantizeLinear", ["w", "s", "zp"], ["wq"], name="wq"),
            helper.make_node("DequantizeLinear", ["wq", "s", "zp"], ["wdq"], name="wdq"),
            helper.make_node("Gemm", ["dq", "wdq", "b"], ["gemm_out"], name="gemm"),
            # A pointwise chain into a re-quantization that feeds a non-linear op: sea.
            helper.make_node("Relu", ["gemm_out"], ["relu_out"], name="relu"),
            helper.make_node("QuantizeLinear", ["relu_out", "s", "zp"], ["q2"], name="q2"),
            helper.make_node("DequantizeLinear", ["q2", "s", "zp"], ["dq2"], name="dq2"),
            helper.make_node("Mul", ["dq2", "g2"], ["y"], name="mul"),
        ],
        "qdq_island_graph",
        [x],
        [y],
        [
            weight,
            bias,
            gain,
            helper.make_tensor("g2", TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32)),
            scale,
            zero_point,
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 19), helper.make_opsetid("autoware", 1)],
    )
    path = tmp_path / "qdq_island_graph.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    inits = {i.name: i for i in converted.graph.initializer}
    # The island keeps its exact FP32 tensors: the scale above all, but also the
    # quantized weight (it feeds Q) and the GEMM bias.
    assert inits["s"].data_type == TensorProto.FLOAT
    assert numpy_helper.to_array(inits["s"]) == scale_value
    assert inits["w"].data_type == TensorProto.FLOAT
    assert inits["b"].data_type == TensorProto.FLOAT
    # Everything outside the linear island converts, the sea Mul's gain included.
    assert inits["g2"].data_type == TensorProto.FLOAT16
    assert inits["g"].data_type == TensorProto.FLOAT16

    nodes = {n.name: n for n in converted.graph.node}
    # Island edges are direct: the GEMM's inputs are produced by the DQ nodes themselves
    # (the converter renames the tensors; what matters is that no node sits between).
    assert nodes["gemm"].input[0] == nodes["dq"].output[0]
    assert nodes["gemm"].input[1] == nodes["wdq"].output[0]
    # The island's Q/DQ read the FP32 scale directly.
    for name in ("q", "dq", "wq", "wdq"):
        assert nodes[name].input[1] == "s"
    # The Gemm output crosses into the sea through one FP16 cast; the sea Q/DQ pair is
    # fp16-typed end to end, reading the split fp16 copy of the shared scale.
    relu_source = producer_named(converted, nodes["relu"].input[0])
    assert relu_source.op_type == "Cast" and relu_source.input[0] == nodes["gemm"].output[0]
    assert nodes["q2"].input[0] == nodes["relu"].output[0]
    for name in ("q2", "dq2"):
        assert nodes[name].input[1] == "s__fp16"
    assert inits["s__fp16"].data_type == TensorProto.FLOAT16

    # The island's boundaries are single casts: no fp16 round-trip pairs anywhere.
    def cast_to(node):
        return next((a.i for a in node.attribute if a.name == "to"), None)

    producer = {o: n for n in converted.graph.node for o in n.output}
    for node in converted.graph.node:
        if node.op_type == "Cast" and cast_to(node) == TensorProto.FLOAT:
            upstream = producer.get(node.input[0])
            assert not (
                upstream is not None
                and upstream.op_type == "Cast"
                and cast_to(upstream) == TensorProto.FLOAT16
            ), f"fp16 round trip at {node.name}"
    onnx.checker.check_model(converted, full_check=True)


def test_cast_graph_to_fp16_keeps_fp8_qdq_islands_fp32_and_castless(tmp_path) -> None:
    """FP8 quantization islands survive the FP16 cast like INT8 ones do.

    modelopt exports FP8 as TRT-domain custom ops (``TRT_FP8QuantizeLinear`` /
    ``TRT_FP8DequantizeLinear``) with the scale coming from a Constant *node*, not an
    initializer — the graph shape its TorchScript symbolic actually emits. The island
    pass must recognize these spellings, or the whole-graph cast rounds the FP8 scales
    through fp16 (the INT8 version of that mistake measured mIoU 0.545 -> 0.067).
    """

    scale_value = np.float32(1e-4)  # not representable in fp16; a round trip shows
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    gain = helper.make_tensor("g", TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32))
    scale_tensor = helper.make_tensor("s_value", TensorProto.FLOAT, [], [scale_value])
    graph = helper.make_graph(
        [
            # A plugin op routes the graph into the whole-graph cast in the first place.
            helper.make_node("PluginOp", ["x", "g"], ["mid"], domain="autoware", name="plugin"),
            helper.make_node("Constant", [], ["s"], name="s_const", value=scale_tensor),
            helper.make_node("TRT_FP8QuantizeLinear", ["mid", "s"], ["q"], domain="trt", name="q"),
            helper.make_node(
                "TRT_FP8DequantizeLinear", ["q", "s"], ["dq"], domain="trt", name="dq"
            ),
            helper.make_node("TRT_FP8QuantizeLinear", ["w", "s"], ["wq"], domain="trt", name="wq"),
            helper.make_node(
                "TRT_FP8DequantizeLinear", ["wq", "s"], ["wdq"], domain="trt", name="wdq"
            ),
            helper.make_node("MatMul", ["dq", "wdq"], ["mm_out"], name="matmul"),
            helper.make_node("Mul", ["mm_out", "g"], ["y"], name="mul"),
        ],
        "fp8_island_graph",
        [x],
        [y],
        [weight, gain],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("autoware", 1),
            helper.make_opsetid("trt", 1),
        ],
    )
    path = tmp_path / "fp8_island_graph.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    nodes = {n.name: n for n in converted.graph.node}
    # The scale constant keeps its exact FP32 value.
    scale_attr = next(a for a in nodes["s_const"].attribute if a.name == "value")
    assert scale_attr.t.data_type == TensorProto.FLOAT
    assert numpy_helper.to_array(scale_attr.t) == scale_value
    # The quantized weight (feeding Q) stays FP32; outside the island conversion happened.
    inits = {i.name: i for i in converted.graph.initializer}
    assert inits["w"].data_type == TensorProto.FLOAT
    assert inits["g"].data_type == TensorProto.FLOAT16
    # Island edges are direct (the converter renames tensors; what matters is that no
    # node sits between): DQ feeds the MatMul, and every Q/DQ reads the scale straight
    # from the Constant — the converter's fp32->fp16->fp32 boundary pairs around the
    # blocked scale, which would round it, must have been collapsed.
    assert nodes["matmul"].input[0] == nodes["dq"].output[0]
    assert nodes["matmul"].input[1] == nodes["wdq"].output[0]
    for name in ("q", "dq", "wq", "wdq"):
        assert nodes[name].input[1] == nodes["s_const"].output[0]


def _conv_qdq_graph(opset: int):
    """Plugin + INT8 conv: Q/DQ feeding a Conv, the sea rule's own case."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 4, 4])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [2, 2, 1, 1], np.eye(2, dtype=np.float32).flatten()
    )
    scale = helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(1e-4)])
    zero_point = helper.make_tensor("zp", TensorProto.INT8, [], [0])
    graph = helper.make_graph(
        [
            helper.make_node("PluginOp", ["x"], ["mid"], domain="autoware", name="plugin"),
            helper.make_node("QuantizeLinear", ["mid", "s", "zp"], ["q"], name="q"),
            helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"], name="dq"),
            helper.make_node("QuantizeLinear", ["w", "s", "zp"], ["wq"], name="wq"),
            helper.make_node("DequantizeLinear", ["wq", "s", "zp"], ["wdq"], name="wdq"),
            helper.make_node("Conv", ["dq", "wdq"], ["conv_out"], name="conv"),
            helper.make_node("Relu", ["conv_out"], ["y"], name="relu"),
        ],
        "conv_qdq_graph",
        [x],
        [y],
        [weight, scale, zero_point],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset), helper.make_opsetid("autoware", 1)],
    )


def test_cast_graph_to_fp16_retypes_conv_qdq_to_fp16_with_no_island(tmp_path) -> None:
    """Conv-side Q/DQ is sea: fp16-typed end to end, scale included, with no casts.

    TensorRT's INT8 conv kernels handle fp16 scales correctly (CenterPoint and BEVFusion
    evaluate identically with 26 subnormal combined scales among them), and a uniformly
    typed graph is the simplest one to build — so nothing is protected here.
    """

    path = tmp_path / "conv_qdq_graph.onnx"
    onnx.save(_conv_qdq_graph(opset=19), str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    inits = {i.name: i for i in converted.graph.initializer}
    assert inits["s"].data_type == TensorProto.FLOAT16
    assert inits["w"].data_type == TensorProto.FLOAT16
    assert inits["zp"].data_type == TensorProto.INT8
    nodes = {n.name: n for n in converted.graph.node}
    # Direct edges everywhere: the only casts are the graph's FP32 I/O boundary.
    assert nodes["conv"].input[0] == "dq" and nodes["conv"].input[1] == "wdq"
    assert nodes["q"].input[0] == "mid"
    casts = [n for n in converted.graph.node if n.op_type == "Cast"]
    assert sorted(c.output[0] for c in casts) == ["x__fp16", "y"]
    onnx.checker.check_model(converted, full_check=True)


def test_cast_graph_to_fp16_refuses_fp16_qdq_below_opset_19(tmp_path) -> None:
    """Retyping standard Q/DQ to fp16 needs opset 19; an older graph is refused, not bent."""

    path = tmp_path / "conv_qdq_graph_17.onnx"
    onnx.save(_conv_qdq_graph(opset=17), str(path))
    with pytest.raises(ValueError, match="opset 19"):
        cast_graph_to_fp16(path)


def test_cast_graph_to_fp16_islands_a_layout_hop_between_dq_and_matmul(tmp_path) -> None:
    """DQ -> Transpose -> MatMul (modelopt's Linear spelling) is one castless island.

    The hop joins the island so no Cast sits between the DQ and the MatMul; its own
    non-float inputs (the Reshape's int64 shape here) are never cast — the reason every
    island-eligible op has a float-slot row.
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    scale = helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(1e-4)])
    zero_point = helper.make_tensor("zp", TensorProto.INT8, [], [0])
    shape = helper.make_tensor("new_shape", TensorProto.INT64, [2], [4, 4])
    graph = helper.make_graph(
        [
            helper.make_node("PluginOp", ["x"], ["mid"], domain="autoware", name="plugin"),
            helper.make_node("QuantizeLinear", ["mid", "s", "zp"], ["q"], name="q"),
            helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"], name="dq"),
            helper.make_node("QuantizeLinear", ["w", "s", "zp"], ["wq"], name="wq"),
            helper.make_node("DequantizeLinear", ["wq", "s", "zp"], ["wdq"], name="wdq"),
            helper.make_node("Reshape", ["wdq", "new_shape"], ["w_r"], name="reshape"),
            helper.make_node("Transpose", ["w_r"], ["w_t"], name="transpose", perm=[1, 0]),
            helper.make_node("MatMul", ["dq", "w_t"], ["mm"], name="matmul"),
            helper.make_node("Relu", ["mm"], ["y"], name="relu"),
        ],
        "layout_hop_island",
        [x],
        [y],
        [weight, scale, zero_point, shape],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 19), helper.make_opsetid("autoware", 1)]
    )
    path = tmp_path / "layout_hop_island.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    nodes = {n.name: n for n in converted.graph.node}
    inits = {i.name: i for i in converted.graph.initializer}
    assert nodes["reshape"].input == ["wdq", "new_shape"]
    assert nodes["transpose"].input == ["w_r"]
    assert nodes["matmul"].input == ["dq", "w_t"]
    assert inits["new_shape"].data_type == TensorProto.INT64
    assert inits["s"].data_type == TensorProto.FLOAT
    assert inits["w"].data_type == TensorProto.FLOAT
    relu_source = producer_named(converted, nodes["relu"].input[0])
    assert relu_source.op_type == "Cast" and relu_source.input[0] == "mm"
    onnx.checker.check_model(converted, full_check=True)


def test_cast_graph_to_fp16_rejects_control_flow_subgraphs(tmp_path) -> None:
    """If/Loop/Scan bodies are not converted; the pass must refuse loudly, not corrupt."""

    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    const = helper.make_tensor("c", TensorProto.FLOAT, [1], np.ones(1, dtype=np.float32))
    branch = helper.make_graph(
        [helper.make_node("Identity", ["c"], ["branch_out"])],
        "branch",
        [],
        [helper.make_tensor_value_info("branch_out", TensorProto.FLOAT, [1])],
        [const],
    )
    graph = helper.make_graph(
        [helper.make_node("If", ["cond"], ["y"], then_branch=branch, else_branch=branch)],
        "control_flow_graph",
        [cond],
        [y],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "control_flow_graph.onnx"
    onnx.save(model, str(path))

    with pytest.raises(NotImplementedError, match="control-flow"):
        cast_graph_to_fp16(path)


def test_cast_graph_to_fp16_splits_a_cast_feeding_both_the_island_and_the_sea(tmp_path) -> None:
    """An int->float glue cast read by both worlds has to serve both.

    Keeping the cast FP32 for the island's sake hands FP32 to an FP16 sea consumer,
    which a strongly-typed engine rejects; retargeting it to FP16 rounds what the island
    calibrated against. Both sides get their own cast, the way an amphibious initializer
    gets its own copy.
    """

    scale = helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(1e-4)])
    zero_point = helper.make_tensor("zp", TensorProto.INT8, [], [0])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, [4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])
    graph = helper.make_graph(
        [
            # The glue cast: int64 indices lifted to float.
            helper.make_node("Cast", ["idx"], ["lifted"], to=TensorProto.FLOAT, name="lift"),
            helper.make_node("QuantizeLinear", ["w", "s", "zp"], ["wq"], name="wq"),
            helper.make_node("DequantizeLinear", ["wq", "s", "zp"], ["wdq"], name="wdq"),
            # Island consumer of the lifted tensor (the MatMul's second operand).
            helper.make_node("MatMul", ["wdq", "lifted"], ["island_out"], name="island_mul"),
            # Sea consumer of the same tensor.
            helper.make_node("PluginOp", ["lifted"], ["sea_out"], domain="autoware", name="plugin"),
            helper.make_node("Add", ["island_out", "sea_out"], ["y"], name="join"),
        ],
        "amphibious_cast",
        [idx],
        [y],
        [weight, scale, zero_point],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)]
    )
    path = tmp_path / "amphibious_cast.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    casts = {
        node.name: next(a.i for a in node.attribute if a.name == "to")
        for node in converted.graph.node
        if node.op_type == "Cast"
    }
    # The original cast still serves the island in FP32; the sea reads an FP16 twin cast
    # made from the same int64 source (not chained off the FP32 one).
    assert casts["lift"] == TensorProto.FLOAT
    twin = next(node for node in converted.graph.node if node.name == "lifted__fp16")
    assert twin.input == ["idx"]
    assert casts["lifted__fp16"] == TensorProto.FLOAT16
    plugin = next(node for node in converted.graph.node if node.name == "plugin")
    assert plugin.input == ["lifted__fp16"]
    island_mul = next(node for node in converted.graph.node if node.name == "island_mul")
    assert island_mul.input[1] == "lifted"
    onnx.checker.check_model(converted, full_check=True)


def test_cast_graph_to_fp16_keeps_node_order_when_an_island_reads_a_sea_graph_output(
    tmp_path,
) -> None:
    """A sea-produced FP32 graph output that an island also consumes must stay loadable.

    The output name ends up owned by a boundary cast spliced after the producer; an
    island cast reading that name would be ordered *before* the node that produces it,
    and the ONNX loader rejects a non-topological graph.
    """

    scale = helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(1e-4)])
    zero_point = helper.make_tensor("zp", TensorProto.INT8, [], [0])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 4])
    shared = helper.make_tensor_value_info("shared", TensorProto.FLOAT, [4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])
    graph = helper.make_graph(
        [
            helper.make_node("PluginOp", ["x"], ["shared"], domain="autoware", name="plugin"),
            helper.make_node("QuantizeLinear", ["w", "s", "zp"], ["wq"], name="wq"),
            helper.make_node("DequantizeLinear", ["wq", "s", "zp"], ["wdq"], name="wdq"),
            # Island consumer of the tensor that is also a graph output.
            helper.make_node("MatMul", ["wdq", "shared"], ["y"], name="island_mul"),
        ],
        "sea_output_read_by_island",
        [x],
        [shared, y],
        [weight, scale, zero_point],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)]
    )
    path = tmp_path / "sea_output_read_by_island.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    onnx.checker.check_model(converted, full_check=True)  # rejects a non-topological node order
    produced: set[str] = {i.name for i in converted.graph.initializer}
    produced |= {i.name for i in converted.graph.input}
    for node in converted.graph.node:
        assert not (set(node.input) - produced), f"{node.name} reads before it is produced"
        produced |= set(node.output)
    # The public output name is still declared FP32 and still produced.
    assert {output.name for output in converted.graph.output} == {"shared", "y"}


def _qdq(x_name, out_name, scale="s", zero_point="zp", tag=""):
    return [
        helper.make_node(
            "QuantizeLinear", [x_name, scale, zero_point], [f"q{tag}"], name=f"q{tag}"
        ),
        helper.make_node(
            "DequantizeLinear", [f"q{tag}", scale, zero_point], [out_name], name=f"dq{tag}"
        ),
    ]


def _qdq_initializers():
    return [
        helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(0.1)]),
        helper.make_tensor("zp", TensorProto.INT8, [], [0]),
    ]


def _cast_edges(graph):
    """``{source tensor: target dtype}`` for every Cast in the graph."""
    return {
        node.input[0]: next(a.i for a in node.attribute if a.name == "to")
        for node in graph.node
        if node.op_type == "Cast"
    }


def test_cast_graph_to_fp16_never_casts_shape_edges_even_without_value_info(tmp_path) -> None:
    """Island membership says nothing about an edge's dtype; the edge does.

    Two standard-op graphs with no ``value_info`` at all, both valid before conversion:
    a Shape reading a (sea) DQ output — its int64 result must not become FP16 for the
    ConstantOfShape reading it — and a Reshape hop inside a linear island whose shape
    input comes from the sea (the int64 edge must not be "lifted" to FLOAT). Both were
    corrupted by a pass that assumed undeclared tensors are FLOAT.
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])

    # (1) Q -> DQ -> Shape -> ConstantOfShape.
    graph = helper.make_graph(
        _qdq("x", "dq")
        + [
            helper.make_node("Shape", ["dq"], ["shp"], name="shape"),
            helper.make_node(
                "ConstantOfShape",
                ["shp"],
                ["y"],
                name="cos",
                value=helper.make_tensor("v", TensorProto.FLOAT, [1], [1.0]),
            ),
        ],
        "shape_from_island",
        [x],
        [y],
        _qdq_initializers(),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / "shape_from_island.onnx"
    onnx.save(model, str(path))
    cast_graph_to_fp16(path)
    converted = onnx.load(str(path))
    onnx.checker.check_model(converted, full_check=True)
    assert "shp" not in _cast_edges(converted.graph)
    cos = next(node for node in converted.graph.node if node.name == "cos")
    assert cos.input[0] == "shp"

    # (2) Shape(x) -> Reshape(DQ(x), shape) -> MatMul: the hop is in the island.
    y33 = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    graph = helper.make_graph(
        [helper.make_node("Shape", ["x"], ["shp"], name="shape")]
        + _qdq("x", "dq")
        + [
            helper.make_node("Reshape", ["dq", "shp"], ["r"], name="reshape"),
            helper.make_node("MatMul", ["r", "w33"], ["mm"], name="matmul"),
            helper.make_node("Relu", ["mm"], ["y"], name="relu_sea"),
        ],
        "shape_into_island",
        [x],
        [y33],
        _qdq_initializers()
        + [
            helper.make_tensor(
                "w33", TensorProto.FLOAT, [3, 3], np.eye(3, dtype=np.float32).flatten()
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / "shape_into_island.onnx"
    onnx.save(model, str(path))
    cast_graph_to_fp16(path)
    converted = onnx.load(str(path))
    onnx.checker.check_model(converted, full_check=True)
    edges = _cast_edges(converted.graph)
    assert "shp" not in edges
    reshape = next(node for node in converted.graph.node if node.name == "reshape")
    assert reshape.input == ["dq", "shp"]
    # The float island output still crosses into the sea through one FP16 cast.
    assert edges.get("mm") == TensorProto.FLOAT16


def test_cast_graph_to_fp16_leaves_integer_island_outputs_alone(tmp_path) -> None:
    """A sea op with a float and an integer output: the integer one is never cast.

    MaxPool's ``Indices`` are int64; ``onnx.checker`` accepts an int64 -> FP16 Cast, but
    FP16 cannot hold an index above 2048 exactly, so the sea must read the indices as
    they are (the conv-side Q/DQ around the pool are sea too, fp16-typed).
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    idx_f = helper.make_tensor_value_info("idx_f", TensorProto.FLOAT, [1, 1, 2, 2])
    graph = helper.make_graph(
        _qdq("x", "dq")
        + [
            helper.make_node(
                "MaxPool",
                ["dq"],
                ["pooled", "indices"],
                name="pool",
                kernel_shape=[2, 2],
                strides=[2, 2],
            ),
        ]
        + _qdq("pooled", "y", tag="2")
        + [helper.make_node("Cast", ["indices"], ["idx_f"], to=TensorProto.FLOAT, name="idx_cast")],
        "maxpool_indices",
        [x],
        [y, idx_f],
        _qdq_initializers(),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    onnx.checker.check_model(model, full_check=True)
    path = tmp_path / "maxpool_indices.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    onnx.checker.check_model(converted, full_check=True)
    idx_cast = next(node for node in converted.graph.node if node.name == "idx_cast")
    assert idx_cast.input[0] == "indices", "the sea reads the int64 indices directly"
    assert not any(
        node.op_type == "Cast" and node.input[0] == "indices" and node.name != "idx_cast"
        for node in converted.graph.node
    )


def test_cast_graph_to_fp16_refuses_an_island_edge_the_graph_does_not_type(tmp_path) -> None:
    """An island MatMul reading a plugin's untyped output is an error, not a guess.

    Exported graphs declare plugin outputs (the exporter records the traced dtype); a
    graph that does not gets a refusal naming the tensor, because assuming FLOAT is
    exactly how an index edge gets cast. Declaring the type makes the same graph pass,
    with the integer edge untouched.
    """

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    nodes = (
        [
            helper.make_node("IndexOp", ["x"], ["pairs"], domain="autoware", name="index_plugin"),
        ]
        + _qdq("x", "dq")
        + [
            helper.make_node("MatMul", ["dq", "pairs"], ["y"], name="gemm_plugin"),
        ]
    )
    graph = helper.make_graph(nodes, "untyped_plugin_edge", [x], [y], _qdq_initializers())
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 19), helper.make_opsetid("autoware", 1)]
    )
    path = tmp_path / "untyped_plugin_edge.onnx"
    onnx.save(model, str(path))
    with pytest.raises(ValueError, match="cannot type tensor 'pairs'"):
        cast_graph_to_fp16(path)

    model.graph.value_info.append(helper.make_tensor_value_info("pairs", TensorProto.FLOAT, None))
    onnx.save(model, str(path))
    cast_graph_to_fp16(path)
    converted = onnx.load(str(path))
    gemm = next(node for node in converted.graph.node if node.name == "gemm_plugin")
    assert gemm.input[1] == "pairs__fp32", "the declared FP32 plugin edge gets its boundary cast"
    assert gemm.input[0] == "dq", "the DQ edge stays castless"


def _qdq_graph_with_helper_params(tmp_path, name="qdq_helpers"):
    """A quantized graph spelled the way modelopt's symbolics spell one.

    The scale is a ``Constant`` shared by the Q and its DQ; the zero point is a
    ``Constant`` in the quantizer's compute dtype behind a ``Cast`` to int8 -- so neither
    value sits on the Q/DQ node, which is what :func:`fold_qdq_params` repairs.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["scale"],
            name="scale_const",
            value=numpy_helper.from_array(np.float32(0.25), "scale_value"),
        ),
        helper.make_node(
            "Constant",
            [],
            ["zp_float"],
            name="zp_const",
            value=numpy_helper.from_array(np.float32(0.0), "zp_value"),
        ),
        helper.make_node("Cast", ["zp_float"], ["zp"], name="zp_cast", to=TensorProto.INT8),
        helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["q"], name="q"),
        helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["y"], name="dq"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, name, [x], [y]),
        opset_imports=[helper.make_operatorsetid("", 19)],
    )
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))
    return path


def test_fold_qdq_params_moves_scale_and_zero_point_onto_the_nodes(tmp_path) -> None:
    """Q/DQ parameters become initializers, exactly valued, and the helper nodes go.

    Netron inlines a ``Constant`` only when its output feeds a single node input, so a
    scale shared by a Q/DQ pair -- and a zero point one ``Cast`` away -- render as
    unreadable edges. As initializers both print on the node itself.
    """
    path = _qdq_graph_with_helper_params(tmp_path)
    fold_qdq_params(path)

    model = onnx.load(str(path))
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    assert set(initializers) == {"scale", "zp"}
    assert initializers["scale"].dtype == np.float32 and initializers["scale"] == np.float32(0.25)
    # The Cast is applied, not dropped: the zero point keeps its int8 spelling.
    assert initializers["zp"].dtype == np.int8 and initializers["zp"] == 0
    # Only the Q/DQ pair is left; the Constant/Cast helpers are gone.
    assert [node.op_type for node in model.graph.node] == ["QuantizeLinear", "DequantizeLinear"]
    assert [node.input for node in model.graph.node] == [["x", "scale", "zp"], ["q", "scale", "zp"]]
    onnx.checker.check_model(model, full_check=True)


def test_fold_qdq_params_is_idempotent_and_leaves_plain_graphs_alone(tmp_path) -> None:
    path = _qdq_graph_with_helper_params(tmp_path)
    fold_qdq_params(path)
    once = onnx.load(str(path)).SerializeToString()
    fold_qdq_params(path)
    assert onnx.load(str(path)).SerializeToString() == once

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    plain = helper.make_model(
        helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "plain", [x], [y])
    )
    plain_path = tmp_path / "plain.onnx"
    onnx.save(plain, str(plain_path))
    before = plain_path.read_bytes()
    fold_qdq_params(plain_path)
    assert plain_path.read_bytes() == before


def test_fold_qdq_params_keeps_computed_params_and_shared_helper_nodes(tmp_path) -> None:
    """Only exactly-evaluable constant chains fold, and a helper still read stays.

    A scale that a graph computes (here from a graph input) is not a constant, so it is
    left connected; a ``Constant`` a non-Q/DQ node also consumes survives the cleanup
    even though the Q/DQ side of it folded.
    """
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    dynamic_scale = helper.make_tensor_value_info("dynamic_scale", TensorProto.FLOAT, [])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    shared = helper.make_tensor_value_info("shared_out", TensorProto.INT8, [])
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["zp_float"],
            name="zp_const",
            value=numpy_helper.from_array(np.float32(0.0), "zp_value"),
        ),
        helper.make_node("Cast", ["zp_float"], ["zp"], name="zp_cast", to=TensorProto.INT8),
        # A second consumer of the folded zero point, outside the Q/DQ pair.
        helper.make_node("Identity", ["zp"], ["shared_out"], name="shared"),
        helper.make_node("QuantizeLinear", ["x", "dynamic_scale", "zp"], ["q"], name="q"),
        helper.make_node("DequantizeLinear", ["q", "dynamic_scale", "zp"], ["y"], name="dq"),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "mixed", [x, dynamic_scale], [y, shared]),
        opset_imports=[helper.make_operatorsetid("", 19)],
    )
    path = tmp_path / "mixed.onnx"
    onnx.save(model, str(path))
    fold_qdq_params(path)

    folded = onnx.load(str(path))
    # The graph input stays the scale; only the constant zero point became a tensor.
    assert [init.name for init in folded.graph.initializer] == ["zp"]
    assert folded.graph.node[0].input == ["zp"] and folded.graph.node[0].name == "shared"
    assert [node.name for node in folded.graph.node] == ["shared", "q", "dq"]
    onnx.checker.check_model(folded, full_check=True)
