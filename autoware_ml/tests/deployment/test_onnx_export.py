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

from autoware_ml.deployment.onnx_export import (
    build_dynamic_axes,
    build_dynamic_shapes,
    export_to_onnx,
    normalize_dynamic_shapes_for_model,
    onnx_has_qdq,
    should_modify_graph,
)


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

    from autoware_ml.deployment.onnx_export import cast_graph_to_fp16

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

    from autoware_ml.deployment.onnx_export import cast_graph_to_fp16

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
        node
        for node in converted.graph.node
        if node.op_type == "Cast" and node.output[0] == "feat"
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

    from autoware_ml.deployment.onnx_export import cast_graph_to_fp16

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
            helper.make_node("Mul", ["gemm_out", "g"], ["y"], name="mul"),
        ],
        "qdq_island_graph",
        [x],
        [y],
        [weight, bias, gain, scale, zero_point],
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
    # Outside the island the conversion happened.
    assert inits["g"].data_type == TensorProto.FLOAT16

    nodes = {n.name: n for n in converted.graph.node}
    # Island edges are direct: the GEMM's inputs are produced by the DQ nodes themselves
    # (the converter renames the tensors; what matters is that no node sits between).
    assert nodes["gemm"].input[0] == nodes["dq"].output[0]
    assert nodes["gemm"].input[1] == nodes["wdq"].output[0]
    # Q/DQ still read the scale directly (no Cast between the constant and the island).
    for name in ("q", "dq", "wq", "wdq"):
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
