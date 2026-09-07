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

from autoware_ml.deployment.onnx.export import (
    build_dynamic_axes,
    build_dynamic_shapes,
    export_to_onnx,
    normalize_dynamic_shapes_for_model,
)
from autoware_ml.deployment.onnx.modify import should_modify_graph
from autoware_ml.deployment.onnx.precision import onnx_has_qdq


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
    import onnx
    from onnx import TensorProto, helper

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
    import onnx

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
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

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
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

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
    """A Q/DQ graph converts to FP16 *around* intact FP32 quantization islands.

    The island — Q/DQ, their scale constants, and the GEMM consuming the dequantized
    tensors — must come through byte-identical and with no Cast on its internal edges:
    an FP16-rounded scale changes the quantization itself, and a Cast between DQ and its
    consumer defeats TensorRT's INT8 fusion (both failure modes measured on PTv3).
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

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
            # A commuting pointwise chain into a re-quantization: the island must grow
            # through the Relu so the quantized chain stays castless (a Cast here blocks
            # TensorRT's Q propagation into the Gemm, which then materializes FP32).
            helper.make_node("Relu", ["gemm_out"], ["relu_out"], name="relu"),
            helper.make_node("QuantizeLinear", ["relu_out", "s", "zp"], ["q2"], name="q2"),
            helper.make_node("DequantizeLinear", ["q2", "s", "zp"], ["dq2"], name="dq2"),
            # mul consumes a DQ output, so it belongs to the island; its gain therefore
            # stays FP32, and the plugin-only "g" shows the outside conversion instead.
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
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)],
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
    # The island mul's gain stays FP32; outside the island (plugin-only g) converts.
    assert inits["g2"].data_type == TensorProto.FLOAT
    assert inits["g"].data_type == TensorProto.FLOAT16

    nodes = {n.name: n for n in converted.graph.node}
    # Island edges are direct: the GEMM's inputs are produced by the DQ nodes themselves
    # (the converter renames the tensors; what matters is that no node sits between).
    assert nodes["gemm"].input[0] == nodes["dq"].output[0]
    assert nodes["gemm"].input[1] == nodes["wdq"].output[0]
    # The quantized chain into the re-quantization is castless: the Relu joined the
    # island (Gemm -> Relu -> Q2 with no Cast on either edge).
    assert nodes["relu"].input[0] == nodes["gemm"].output[0]
    assert nodes["q2"].input[0] == nodes["relu"].output[0]
    # Q/DQ still read the scale directly (no Cast between the constant and the island).
    for name in ("q", "dq", "wq", "wdq", "q2", "dq2"):
        assert nodes[name].input[1] == "s"

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
    onnx.checker.check_model(converted)


def test_cast_graph_to_fp16_keeps_fp8_qdq_islands_fp32_and_castless(tmp_path) -> None:
    """FP8 quantization islands survive the FP16 cast like INT8 ones do.

    modelopt exports FP8 as TRT-domain custom ops (``TRT_FP8QuantizeLinear`` /
    ``TRT_FP8DequantizeLinear``) with the scale coming from a Constant *node*, not an
    initializer — the graph shape its TorchScript symbolic actually emits. The island
    pass must recognize these spellings, or the whole-graph cast rounds the FP8 scales
    through fp16 (the INT8 version of that mistake measured mIoU 0.545 -> 0.067).
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

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


def test_cast_graph_to_fp16_grows_islands_through_shape_ops_without_casting_int_edges(
    tmp_path,
) -> None:
    """A Reshape between the quantized Gemm and the next Q joins the island castless.

    Shape ops are pure data movement, so TensorRT's Q/DQ propagation crosses them; the
    island must include them (castless chain) while never casting their integer inputs
    (the int64 shape here — the reason the whitelist and the float-slot table are kept
    in lockstep by the import-time check).
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten()
    )
    scale = helper.make_tensor("s", TensorProto.FLOAT, [], [np.float32(0.1)])
    zero_point = helper.make_tensor("zp", TensorProto.INT8, [], [0])
    shape = helper.make_tensor("new_shape", TensorProto.INT64, [2], [1, 8])
    gain = helper.make_tensor("g", TensorProto.FLOAT, [8], np.ones(8, dtype=np.float32))
    graph = helper.make_graph(
        [
            helper.make_node("PluginOp", ["x", "g"], ["mid"], domain="autoware", name="plugin"),
            helper.make_node("QuantizeLinear", ["mid", "s", "zp"], ["q"], name="q"),
            helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"], name="dq"),
            helper.make_node("Gemm", ["dq", "w"], ["gemm_out"], name="gemm"),
            helper.make_node("Reshape", ["gemm_out", "new_shape"], ["reshaped"], name="reshape"),
            helper.make_node("QuantizeLinear", ["reshaped", "s", "zp"], ["q2"], name="q2"),
            helper.make_node("DequantizeLinear", ["q2", "s", "zp"], ["y"], name="dq2"),
        ],
        "shape_op_island_graph",
        [x],
        [y],
        [weight, scale, zero_point, shape, gain],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("autoware", 1)],
    )
    path = tmp_path / "shape_op_island_graph.onnx"
    onnx.save(model, str(path))

    cast_graph_to_fp16(path)

    converted = onnx.load(str(path))
    nodes = {n.name: n for n in converted.graph.node}
    inits = {i.name: i for i in converted.graph.initializer}
    # Castless chain through the Reshape: Gemm -> Reshape -> Q2 direct edges.
    assert nodes["reshape"].input[0] == nodes["gemm"].output[0]
    assert nodes["q2"].input[0] == nodes["reshape"].output[0]
    # The int64 shape input is untouched — no cast, same initializer.
    assert nodes["reshape"].input[1] == "new_shape"
    assert inits["new_shape"].data_type == TensorProto.INT64
    # Island tensors stay FP32; the sea (plugin gain) converted.
    assert inits["s"].data_type == TensorProto.FLOAT
    assert inits["w"].data_type == TensorProto.FLOAT
    assert inits["g"].data_type == TensorProto.FLOAT16
    onnx.checker.check_model(converted)


def test_cast_graph_to_fp16_rejects_control_flow_subgraphs(tmp_path) -> None:
    """If/Loop/Scan bodies are not converted; the pass must refuse loudly, not corrupt."""
    import numpy as np
    import onnx
    import pytest
    from onnx import TensorProto, helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

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


def _topk_graph(path, *, values_is_graph_output: bool):
    """Write a ``x(fp16) -> Cast(fp32) -> TopK`` graph, with values internal or exported."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 8])
    values = helper.make_tensor_value_info("values", TensorProto.FLOAT, [1, 2])
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 2])
    scaled = helper.make_tensor_value_info("scaled", TensorProto.FLOAT, [1, 2])
    k = helper.make_tensor("k", TensorProto.INT64, [1], np.array([2], dtype=np.int64))
    one = helper.make_tensor("one", TensorProto.FLOAT, [1], np.array([1.0], dtype=np.float32))
    nodes = [
        helper.make_node("Cast", ["x"], ["x32"], to=TensorProto.FLOAT, name="lift"),
        helper.make_node("TopK", ["x32", "k"], ["values", "indices"], name="topk"),
    ]
    outputs = [values, indices]
    initializers = [k]
    if not values_is_graph_output:
        # values feeds an internal consumer instead of leaving the graph.
        nodes.append(helper.make_node("Mul", ["values", "one"], ["scaled"], name="scale"))
        outputs = [scaled, indices]
        initializers.append(one)
    graph = helper.make_graph(nodes, "topk_graph", [x], outputs, initializers)
    graph.value_info.append(helper.make_tensor_value_info("x32", TensorProto.FLOAT, [1, 8]))
    if not values_is_graph_output:
        graph.value_info.append(values)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))
    return path


def test_keep_topk_in_fp16_bypasses_the_cast_and_is_a_noop_otherwise(tmp_path) -> None:
    """The transform lets TopK rank the FP16 tensor directly; untouched graphs pass through."""
    import onnx

    from autoware_ml.deployment.onnx.autocast import keep_topk_in_fp16

    path = _topk_graph(tmp_path / "topk_graph.onnx", values_is_graph_output=False)
    keep_topk_in_fp16(path)

    converted = onnx.load(str(path))
    topk = next(node for node in converted.graph.node if node.op_type == "TopK")
    assert topk.input[0] == "x", "TopK must read the FP16 tensor directly"
    values_info = next(info for info in converted.graph.value_info if info.name == "values")
    assert values_info.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16

    # A graph whose TopK already reads fp16 (or fp32 exports) is untouched.
    before = converted.SerializeToString()
    keep_topk_in_fp16(path)
    assert onnx.load(str(path)).SerializeToString() == before


def test_keep_topk_in_fp16_leaves_an_exported_values_output_alone(tmp_path) -> None:
    """The values output's declared type is the artifact's interface (keep_io_types).

    Bypassing the cast would make the tensor FP16 while the graph still promises FLOAT to
    every consumer of the file, so the transform declines and the round-trip stays.
    """
    import onnx

    from autoware_ml.deployment.onnx.autocast import keep_topk_in_fp16

    path = _topk_graph(tmp_path / "topk_output.onnx", values_is_graph_output=True)
    before = onnx.load(str(path)).SerializeToString()

    keep_topk_in_fp16(path)

    after = onnx.load(str(path))
    assert after.SerializeToString() == before
    topk = next(node for node in after.graph.node if node.op_type == "TopK")
    assert topk.input[0] == "x32"
    onnx.checker.check_model(after)


def test_cast_graph_to_fp16_splits_a_cast_feeding_both_the_island_and_the_sea(tmp_path) -> None:
    """An int->float glue cast read by both worlds has to serve both.

    Keeping the cast FP32 for the island's sake hands FP32 to an FP16 sea consumer,
    which a strongly-typed engine rejects; retargeting it to FP16 rounds what the island
    calibrated against. Both sides get their own cast, the way an amphibious initializer
    gets its own copy.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

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
            # Island consumer of the lifted tensor (Mul's data slot).
            helper.make_node("Mul", ["wdq", "lifted"], ["island_out"], name="island_mul"),
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
    onnx.checker.check_model(converted)


def test_cast_graph_to_fp16_keeps_node_order_when_an_island_reads_a_sea_graph_output(
    tmp_path,
) -> None:
    """A sea-produced FP32 graph output that an island also consumes must stay loadable.

    The output name ends up owned by a boundary cast spliced after the producer; an
    island cast reading that name would be ordered *before* the node that produces it,
    and the ONNX loader rejects a non-topological graph.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16

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
            helper.make_node("Mul", ["wdq", "shared"], ["y"], name="island_mul"),
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
    onnx.checker.check_model(converted)  # rejects a non-topological node order
    produced: set[str] = {i.name for i in converted.graph.initializer}
    produced |= {i.name for i in converted.graph.input}
    for node in converted.graph.node:
        assert not (set(node.input) - produced), f"{node.name} reads before it is produced"
        produced |= set(node.output)
    # The public output name is still declared FP32 and still produced.
    assert {output.name for output in converted.graph.output} == {"shared", "y"}
