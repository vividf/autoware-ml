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


"""Element types of an ONNX graph's tensors, for passes that rewrite dtypes.

A precision pass reasons per tensor edge — ONNX legality is defined there, not per
node — so before it inserts a single Cast it needs to know what every boundary tensor
holds. :func:`tensor_types` answers that from the graph's own declarations plus ONNX
shape inference, seeded through the quantize/dequantize ops (whose custom-domain FP8
spellings have no schema inference could use).
"""

from __future__ import annotations

import onnx
from onnx import TensorProto, helper, shape_inference

#: Quantize/dequantize node spellings. INT8 exports as standard ONNX Q/DQ; FP8 exports
#: as modelopt's TRT-domain custom ops (its E4M3 symbolic bypasses standard
#: ``QuantizeLinear``, whose float8 form it never emits).
QUANTIZE_OPS = ("QuantizeLinear", "TRT_FP8QuantizeLinear")
DEQUANTIZE_OPS = ("DequantizeLinear", "TRT_FP8DequantizeLinear")
QDQ_OPS = QUANTIZE_OPS + DEQUANTIZE_OPS

_FLOAT_TYPES = (TensorProto.FLOAT, TensorProto.FLOAT16)


def is_float(elem_type: int | None) -> bool:
    """Whether an element type is one of the two precisions the FP16 passes trade between."""
    return elem_type in _FLOAT_TYPES


def tensor_types(model: onnx.ModelProto) -> dict[str, int]:
    """Element type of every tensor the graph settles, by name.

    Sources, in order: graph inputs/outputs and existing ``value_info``; initializers;
    ``Constant`` node values; the quantize/dequantize outputs (a Q produces its
    zero-point's type, a DQ its scale's — the same rule for the standard and the
    TRT-domain FP8 spellings); then ONNX shape inference carries those through every
    standard op. Custom-domain ops other than Q/DQ contribute nothing themselves — an
    exported graph declares their outputs (the exporter records the traced dtype), a
    hand-built one has to say so in ``value_info``. Absent from the result means "the
    graph does not settle this tensor's type"; callers decide whether that is an error.
    """
    graph = model.graph
    types: dict[str, int] = {}
    for info in list(graph.input) + list(graph.output) + list(graph.value_info):
        elem_type = info.type.tensor_type.elem_type
        if elem_type:
            types[info.name] = elem_type
    for init in graph.initializer:
        types[init.name] = init.data_type
    for node in graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                types[node.output[0]] = attribute.t.data_type
            elif attribute.name in ("value_float", "value_floats"):
                types[node.output[0]] = TensorProto.FLOAT
            elif attribute.name in ("value_int", "value_ints"):
                types[node.output[0]] = TensorProto.INT64

    seeds: list[onnx.ValueInfoProto] = []
    for node in graph.node:
        if not node.output or node.output[0] in types:
            continue
        produced: int | None = None
        if node.op_type in QUANTIZE_OPS:
            declared = next((a.i for a in node.attribute if a.name == "output_dtype"), 0)
            zero_point = node.input[2] if len(node.input) > 2 else ""
            if declared:
                produced = declared
            elif zero_point:
                produced = types.get(zero_point)
            elif node.op_type.startswith("TRT_FP8"):
                produced = TensorProto.FLOAT8E4M3FN
            else:
                produced = TensorProto.UINT8  # QuantizeLinear's default without a zero point
        elif node.op_type in DEQUANTIZE_OPS and len(node.input) > 1:
            produced = types.get(node.input[1])
        if produced:
            types[node.output[0]] = produced
            seeds.append(helper.make_tensor_value_info(node.output[0], produced, None))

    seeded = onnx.ModelProto()
    seeded.CopyFrom(model)
    seeded.graph.value_info.extend(seeds)
    inferred = shape_inference.infer_shapes(seeded, strict_mode=False)
    for info in inferred.graph.value_info:
        elem_type = info.type.tensor_type.elem_type
        if elem_type and info.name not in types:
            types[info.name] = elem_type
    return types
