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
