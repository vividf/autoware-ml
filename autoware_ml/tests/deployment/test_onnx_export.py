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

"""Unit tests for the ONNX export primitive (dynamic shape/axes builders, modifier gate)."""

from __future__ import annotations

from omegaconf import OmegaConf
import torch
from onnx import TensorProto, helper
from onnx import numpy_helper
import numpy as np
import onnx
import pytest

from autoware_ml.deployment.onnx.export import (
    build_dynamic_axes,
    build_dynamic_shapes,
    export_to_onnx,
    normalize_dynamic_shapes_for_model,
)
from autoware_ml.deployment.onnx.modify import should_modify_graph
from autoware_ml.deployment.onnx.precision import onnx_has_qdq
from autoware_ml.deployment.onnx.autocast import keep_topk_in_fp16
from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16


def producer_named(model, tensor_name):
    return next(node for node in model.graph.node if tensor_name in node.output)


def test_build_dynamic_axes_from_axes_spec() -> None:
    spec = {
        "feat": {0: "voxels_num"},
        "pred_probs": {0: "voxels_num"},
    }

    assert build_dynamic_axes(spec) == {
        "feat": {0: "voxels_num"},
        "pred_probs": {0: "voxels_num"},
    }


def test_build_dynamic_axes_down_converts_dynamic_shapes_spec() -> None:
    spec = {
        "points": {0: {"name": "num_points", "min": 2}},
        "inverse_map": {0: {"name": "num_points", "min": 2}},
    }

    assert build_dynamic_axes(spec) == {
        "points": {0: "num_points"},
        "inverse_map": {0: "num_points"},
    }


def test_build_dynamic_shapes_matches_positional_export_inputs() -> None:
    spec = {
        "points": {0: {"name": "num_points", "min": 2}},
        "coors": {0: {"name": "num_points", "min": 2}},
        "inverse_map": {0: {"name": "num_points", "min": 2}},
    }

    dynamic_shapes = build_dynamic_shapes(
        spec,
        ["points", "coors", "voxel_coors", "inverse_map"],
    )

    assert dynamic_shapes is not None
    assert len(dynamic_shapes) == 4
    assert dynamic_shapes[0] is not None
    assert dynamic_shapes[1] is not None
    assert dynamic_shapes[2] is None
    assert dynamic_shapes[3] is not None


def test_build_dynamic_shapes_accepts_omegaconf_nodes() -> None:
    cfg = OmegaConf.create({"input": {0: "batch"}})

    dynamic_shapes = build_dynamic_shapes(cfg, ["input"])

    assert dynamic_shapes is not None and dynamic_shapes[0] is not None


def test_normalize_dynamic_shapes_wraps_varargs_forward() -> None:
    class _VarArgsModel(torch.nn.Module):
        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return args[0]

    dynamic_shapes = ({0: "dim0"}, {0: "dim1"})

    assert normalize_dynamic_shapes_for_model(_VarArgsModel(), dynamic_shapes) == (dynamic_shapes,)


def test_should_modify_graph_handles_none_and_config() -> None:
    assert should_modify_graph(None) is False
    assert should_modify_graph(OmegaConf.create({"_target_": "pkg.Modifier"})) is True
    assert should_modify_graph({"_target_": "pkg.Modifier"}) is True


def test_onnx_has_qdq_detects_quantize_nodes(tmp_path) -> None:
    def graph(nodes, name):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
        model = helper.make_model(helper.make_graph(nodes, name, [x], [y]))
        path = tmp_path / f"{name}.onnx"
        onnx.save(model, str(path))
        return path

    plain = graph([helper.make_node("Relu", ["x"], ["y"])], "plain")
    assert onnx_has_qdq(plain) is False

    qdq_nodes = [
        onnx.helper.make_node("QuantizeLinear", ["x", "s"], ["q"]),
        onnx.helper.make_node("DequantizeLinear", ["q", "s"], ["y"]),
    ]
    qdq = graph(qdq_nodes, "qdq")
    # initializers must be attached for a valid graph; has_qdq only reads node types.
    assert onnx_has_qdq(qdq) is True

    # FP8 exports as modelopt's TRT-domain custom ops, not standard QuantizeLinear.
    fp8_nodes = [
        onnx.helper.make_node("TRT_FP8QuantizeLinear", ["x", "s"], ["q"], domain="trt"),
        onnx.helper.make_node("TRT_FP8DequantizeLinear", ["q", "s"], ["y"], domain="trt"),
    ]
    assert onnx_has_qdq(graph(fp8_nodes, "fp8_qdq")) is True


def test_export_to_onnx_writes_named_graph(tmp_path) -> None:
    class _AddOne(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    path = tmp_path / "model.onnx"
    export_to_onnx(
        _AddOne(),
        (torch.ones(2, 3),),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={"input": {0: "batch"}},
    )

    model = onnx.load(str(path))
    assert [i.name for i in model.graph.input] == ["input"]
    assert [o.name for o in model.graph.output] == ["output"]
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"


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


def _topk_graph(path, *, values: str, source_type=TensorProto.FLOAT16):
    """Write a ``x -> Cast(fp32) -> TopK`` graph.

    ``values`` says what happens to the values output: ``"internal"`` feeds an FP32 Mul,
    ``"output"`` leaves the graph as a declared-FLOAT output, ``"unused"`` goes nowhere
    (indices-only use). ``source_type`` is the dtype of ``x`` the Cast lifts.
    """
    x = helper.make_tensor_value_info("x", source_type, [1, 8])
    values_info = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 2])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2])
    scaled = helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [1, 2])
    k = helper.make_tensor("k", TensorProto.INT64, [1], np.array([2], dtype=np.int64))
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], np.array([1.0], dtype=np.float32))
    nodes = [
        helper.make_node("Cast", ["x"], ["x32"], to=TensorProto.FLOAT, name="lift"),
        helper.make_node("TopK", ["x32", "k"], ["values", "indices"], name="topk"),
    ]
    initializers = [k]
    if values == "internal":
        nodes.append(helper.make_node("Mul", ["values", "one"], ["scaled"], name="scale"))
        outputs = [scaled, indices]
        initializers.append(one)
    elif values == "output":
        outputs = [values_info, indices]
    else:
        outputs = [indices]
    graph = helper.make_graph(nodes, "topk_graph", [x], outputs, initializers)
    graph.value_info.append(helper.make_tensor_value_info("x32", TensorProto.FLOAT, [1, 8]))
    if values == "internal":
        graph.value_info.append(values_info)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(path))
    return path


def _topk_and_values_cast(graph):
    topk = next(node for node in graph.node if node.op_type == "TopK")
    cast_back = next(
        (node for node in graph.node if node.op_type == "Cast" and node.output[0] == "values"),
        None,
    )
    return topk, cast_back


def test_keep_topk_in_fp16_selects_in_fp16_and_casts_the_values_back_for_consumers(
    tmp_path,
) -> None:
    """TopK ranks the FP16 tensor directly; its FP32 consumer still gets FP32 values.

    The point is to skip casting the whole heatmap, not to retype the consumers: the
    *selected* values (k elements) are cast back under their original name, so the Mul
    and its FP32 operand are exactly as exported and the graph stays type-valid.
    """

    path = _topk_graph(tmp_path / "topk_graph.onnx", values="internal")
    keep_topk_in_fp16(path)

    converted = onnx.load(str(path))
    onnx.checker.check_model(converted, full_check=True)
    topk, cast_back = _topk_and_values_cast(converted.graph)
    assert topk.input[0] == "x", "TopK must read the FP16 tensor directly"
    assert cast_back is not None and cast_back.input == [topk.output[0]]
    assert next(a.i for a in cast_back.attribute if a.name == "to") == TensorProto.FLOAT
    mul = next(node for node in converted.graph.node if node.name == "scale")
    assert list(mul.input) == ["values", "one"], "the consumer is untouched"
    # The heatmap cast nobody reads any more is gone, with its value_info.
    assert not any(node.name == "lift" for node in converted.graph.node)
    assert not any(info.name == "x32" for info in converted.graph.value_info)

    # Idempotent: the second run finds no FP32 cast feeding a TopK.
    before = converted.SerializeToString()
    keep_topk_in_fp16(path)
    assert onnx.load(str(path)).SerializeToString() == before


def test_keep_topk_in_fp16_keeps_an_exported_values_output_fp32(tmp_path) -> None:
    """The values output's declared type is the artifact's interface (keep_io_types).

    The optimization still applies — the cast-back produces the public FLOAT tensor
    under its own name, so the file promises exactly what it did before.
    """

    path = _topk_graph(tmp_path / "topk_output.onnx", values="output")
    declared_before = [
        (o.name, o.type.tensor_type.elem_type) for o in onnx.load(str(path)).graph.output
    ]

    keep_topk_in_fp16(path)

    after = onnx.load(str(path))
    onnx.checker.check_model(after, full_check=True)
    assert [(o.name, o.type.tensor_type.elem_type) for o in after.graph.output] == declared_before
    topk, cast_back = _topk_and_values_cast(after.graph)
    assert topk.input[0] == "x"
    assert cast_back is not None and cast_back.input == [topk.output[0]]


def test_keep_topk_in_fp16_leaves_unread_values_in_fp16(tmp_path) -> None:
    """Indices-only use: nothing reads the values, so nothing needs them cast back."""

    path = _topk_graph(tmp_path / "topk_indices_only.onnx", values="unused")
    keep_topk_in_fp16(path)

    after = onnx.load(str(path))
    onnx.checker.check_model(after, full_check=True)
    topk, cast_back = _topk_and_values_cast(after.graph)
    assert topk.input[0] == "x"
    assert cast_back is None
    assert topk.output[0] == "values"


def test_keep_topk_in_fp16_ignores_a_cast_that_lifts_integers(tmp_path) -> None:
    """A Cast-to-FLOAT feeding TopK is only an FP16 round-trip if its source is FP16.

    Bypassing an int64 -> FLOAT lift would make TopK rank integers and hand an int64
    values tensor to FP32 consumers; the transform leaves such a graph alone.
    """

    path = _topk_graph(
        tmp_path / "topk_int_source.onnx", values="internal", source_type=TensorProto.INT64
    )
    before = onnx.load(str(path)).SerializeToString()

    keep_topk_in_fp16(path)

    assert onnx.load(str(path)).SerializeToString() == before


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
