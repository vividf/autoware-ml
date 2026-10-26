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


"""FP16 cast for exported graphs that carry Q/DQ nodes or runtime plugin ops.

The graph goes fp16 (the *sea*) except for fp32 *islands*: the Q/DQ pairs feeding a
Gemm/MatMul and the ops TensorRT only parses in fp32. Conv-side Q/DQ go fp16 with the sea.
Each float edge between the two gets one Cast, and the graph I/O stays fp32.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

from autoware_ml.deployment.onnx.dtypes import (
    DEQUANTIZE_OPS,
    QDQ_OPS,
    QUANTIZE_OPS,
    is_float,
    tensor_types,
)
from autoware_ml.deployment.onnx.inspect import onnx_custom_op_domains, onnx_has_qdq
from autoware_ml.ops.spconv.contract import (
    AUTOWARE_DOMAIN,
    IMPLICIT_GEMM_FP32_INPUT_SLOTS,
    IMPLICIT_GEMM_OP,
)

__all__ = ["cast_graph_to_fp16", "onnx_custom_op_domains", "onnx_has_qdq"]


logger = logging.getLogger(__name__)


def _assign_missing_node_names(graph) -> None:
    """Name the unnamed nodes; the island bookkeeping is keyed by node name."""
    taken = {node.name for node in graph.node if node.name}
    for index, node in enumerate(graph.node):
        if not node.name:
            candidate = f"{node.op_type}_{index}"
            while candidate in taken:
                candidate += "_"
            node.name = candidate
            taken.add(candidate)


#: Ops whose Q/DQ stay fp32-typed. TensorRT 10.8/10.16 emits NaN from a strongly typed
#: fp16 Q/DQ -> INT8 Gemm when the fp16 combined scale ``s_x * s_w[c]`` is subnormal (the
#: fused kernel folds the dequant scales into an fp16 intermediate); fp32-typed Q/DQ take
#: another kernel and are exact. Conv kernels are immune: the 26 conv modules of CenterPoint
#: and BEVFusion with subnormal combined scales evaluate identically with fp16 Q/DQ.
#: Splitting by node name or scale threshold was tried and is unreliable: a name list
#: missed two failing Gemms, and only 6 of the 15 PTv3 head Gemms with a subnormal
#: combined scale fail.
_LINEAR_OPS = frozenset({"Gemm", "MatMul"})

#: Ops TensorRT's parser rejects in fp16 (``Range``: "only supports int32, int64, fp32");
#: they stay fp32 islands of their own.
_FP32_ONLY_OPS = frozenset({"Range"})

#: Layout ops between a DQ and its linear op (modelopt's FP8 export spells a Linear as
#: weight DQ -> Transpose -> MatMul). They join the island because a Cast between a DQ and
#: its consumer defeats TensorRT's Q/DQ fusion ("Per-tensor quantization/dequantization
#: layer should have 1 scale factor element").
_LAYOUT_OPS = frozenset({"Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity"})

#: Float input slots of every island op per its ONNX spec, so that boundary casts never
#: touch an integer edge (a zero point, a Reshape shape); ``None`` means every input.
_ALL_FLOAT = None
_ISLAND_FLOAT_INPUT_SLOTS: dict = {
    # A DQ's data input is int8 / fp8.
    "QuantizeLinear": (0, 1),
    "TRT_FP8QuantizeLinear": (0, 1),
    "DequantizeLinear": (1,),
    "TRT_FP8DequantizeLinear": (1,),
    # Inputs after the data are int64 shapes / axes.
    "Transpose": (0,),
    "Reshape": (0,),
    "Flatten": (0,),
    "Squeeze": (0,),
    "Unsqueeze": (0,),
    "Identity": (0,),
    "Gemm": _ALL_FLOAT,
    "MatMul": _ALL_FLOAT,
    # The per-edge dtype check skips integer Ranges.
    "Range": _ALL_FLOAT,
}
_MISSING_SLOT_ENTRIES = (_LINEAR_OPS | _LAYOUT_OPS | _FP32_ONLY_OPS | set(QDQ_OPS)) - set(
    _ISLAND_FLOAT_INPUT_SLOTS
)
assert not _MISSING_SLOT_ENTRIES, (
    f"Island-eligible ops missing a float-slot row: {sorted(_MISSING_SLOT_ENTRIES)}. "
    "Every one needs a row so island boundary casts never touch integer edges."
)

#: Plugin input slots that stay fp32 in the sea, keyed by ``(domain, op_type)`` and then by
#: input count (one op can have several input forms). Casting them to fp16 fails the engine
#: build with "could not find any supported formats consistent with input types".
_PLUGIN_FP32_INPUT_SLOTS: dict[tuple[str, str], dict[int, tuple[int, ...]]] = {
    (AUTOWARE_DOMAIN, IMPLICIT_GEMM_OP): IMPLICIT_GEMM_FP32_INPUT_SLOTS,
}


def _plugin_fp32_slots(node) -> tuple[int, ...]:
    by_arity = _PLUGIN_FP32_INPUT_SLOTS.get((node.domain, node.op_type))
    if not by_arity:
        return ()
    return by_arity.get(len(node.input), ())


#: QuantizeLinear / DequantizeLinear accept an fp16 input and scale from opset 19.
_FP16_QDQ_MIN_OPSET = 19


def _quantized_island_names(graph) -> list[str]:
    """Return the names of the island nodes.

    A linear island is a DQ whose output reaches a Gemm/MatMul, directly or through layout
    ops, with its Q, their scale and zero-point producers, the layout ops and the linear
    op: the pattern TensorRT fuses into one INT8 kernel. ``_FP32_ONLY_OPS`` nodes are
    islands of their own.
    """
    producer_of = {out: node for node in graph.node for out in node.output}
    consumers_of: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers_of.setdefault(name, []).append(node)

    island: dict[str, None] = {}

    def admit(node) -> None:
        island[node.name] = None
        if node.op_type in QDQ_OPS:
            for name in node.input[1:]:  # scale / zero_point
                producer = producer_of.get(name)
                if producer is not None:
                    island[producer.name] = None

    for node in graph.node:
        if node.op_type not in DEQUANTIZE_OPS or not node.output:
            continue
        hops: list = []
        linear: list = []
        frontier = [(node.output[0], [])]
        while frontier:
            name, path = frontier.pop()
            for consumer in consumers_of.get(name, []):
                if consumer.op_type in _LINEAR_OPS:
                    linear.append(consumer)
                    hops.extend(path)
                elif consumer.op_type in _LAYOUT_OPS and consumer.input[0] == name:
                    frontier.append((consumer.output[0], path + [consumer]))
        if not linear:
            continue
        admit(node)
        quantize = producer_of.get(node.input[0])
        if quantize is not None and quantize.op_type in QUANTIZE_OPS:
            admit(quantize)
        for member in hops + linear:
            island[member.name] = None
    for node in graph.node:
        if node.op_type in _FP32_ONLY_OPS and not node.domain:
            island[node.name] = None
    return list(island)


def cast_graph_to_fp16(onnx_path: Path) -> None:
    """Cast a graph to fp16 in place around its fp32 islands, keeping the graph I/O fp32.

    Casts are decided per tensor edge, so integer and boolean edges (a Shape result,
    MaxPool indices) are never cast, and an island edge of unknown type is refused rather
    than guessed as FLOAT.

    Not built on onnxconverter-common: blocking the island nodes there puts round-trip
    casts around each of them and leaves stale ``value_info`` that onnxruntime's loader
    rejects.

    Raises:
        NotImplementedError: For a graph with control flow.
        ValueError: For conv-side Q/DQ below opset 19, a Q/DQ scale that underflows in
            fp16, or an island edge of unknown type.
    """

    model = onnx.load(str(onnx_path))
    graph = model.graph
    control_flow = sorted({n.op_type for n in graph.node if n.op_type in ("If", "Loop", "Scan")})
    if control_flow:
        raise NotImplementedError(
            f"cast_graph_to_fp16 does not handle control-flow subgraphs ({', '.join(control_flow)} "
            f"in {onnx_path.name}): their bodies would keep FP32 tensors against the converted "
            "FP16 sea. Export the stage without in-graph control flow, or extend the pass to "
            "recurse into subgraph bodies first."
        )
    _assign_missing_node_names(graph)
    # Before any rewrite: the boundary decisions need the original types.
    types = tensor_types(model)
    island = set(_quantized_island_names(graph))
    sea_standard_qdq = [
        n for n in graph.node if n.op_type in QDQ_OPS and not n.domain and n.name not in island
    ]
    default_opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)
    if sea_standard_qdq and default_opset < _FP16_QDQ_MIN_OPSET:
        raise ValueError(
            f"cast_graph_to_fp16 retypes {len(sea_standard_qdq)} conv-side Q/DQ node(s) of "
            f"{onnx_path.name} to fp16, which QuantizeLinear/DequantizeLinear only allow from "
            f"opset {_FP16_QDQ_MIN_OPSET}; the graph is opset {default_opset}. Export with "
            f"deploy.onnx.opset_version >= {_FP16_QDQ_MIN_OPSET}."
        )

    def settled_type(name: str, node, role: str) -> int:
        dtype = types.get(name)
        if dtype is None:
            raise ValueError(
                f"cast_graph_to_fp16 cannot type tensor {name!r}, {role} of island node "
                f"{node.name!r} ({node.op_type}): it is not a graph input, initializer or "
                "Constant, and ONNX shape inference does not reach it — a custom-domain "
                "producer without value_info, typically. Declare it (the exporter records "
                "the traced dtype; a hand-built graph needs a value_info entry) or give "
                f"{node.op_type!r} a float-slot row in _ISLAND_FLOAT_INPUT_SLOTS. Guessing "
                "FLOAT is not an option: that is how a shape edge gets cast."
            )
        return dtype

    node_by_name = {node.name: node for node in graph.node}
    producer_of = {out: node for node in graph.node for out in node.output}
    consumers_of: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers_of.setdefault(name, []).append(node)

    def in_island(node) -> bool:
        return node.name in island

    pinned_fp32_slots: set[tuple[str, int]] = set()
    pinned_fp32_names: set[str] = set()
    for node in graph.node:
        for index in _plugin_fp32_slots(node):
            if index < len(node.input) and node.input[index]:
                pinned_fp32_slots.add((node.name, index))
                pinned_fp32_names.add(node.input[index])

    def reads_fp32_pinned(node, name: str) -> bool:
        return any(
            (node.name, index) in pinned_fp32_slots
            for index, value in enumerate(node.input)
            if value == name
        )

    # --- 1. Initializers: fp16 for the sea, fp32 for islands and pinned plugin slots, and a
    # copy of each when both read one.
    fp16_twin: dict[str, str] = {}
    new_initializers = []
    qdq_scale_names = {
        node.input[1] for node in graph.node if node.op_type in QDQ_OPS and len(node.input) > 1
    }
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        users = consumers_of.get(init.name, [])
        sea_users = [n for n in users if not in_island(n) and not reads_fp32_pinned(n, init.name)]
        island_users = [n for n in users if in_island(n) or reads_fp32_pinned(n, init.name)]
        if not sea_users:
            continue
        values = numpy_helper.to_array(init)
        half_values = values.astype("float16")
        if init.name in qdq_scale_names and bool(((values > 0) & (half_values == 0)).any()):
            # A scale under fp16's smallest subnormal (~6e-8: amax < ~7.5e-6, a channel dead
            # after the BN fold) flushes to zero; that is for the calibration to fix.
            raise ValueError(
                f"Q/DQ scale {init.name!r} has {int(((values > 0) & (half_values == 0)).sum())} "
                "positive entries that underflow to zero in fp16 (min "
                f"{float(values[values > 0].min()):.3e}). Keep this module fp32 "
                "(skip_quantize) or recalibrate; an fp16-typed zero scale is not a Q/DQ."
            )
        half = numpy_helper.from_array(
            half_values,
            init.name + "__fp16" if island_users else init.name,
        )
        if island_users:
            new_initializers.append(half)
            fp16_twin[init.name] = half.name
            for node in sea_users:
                for index, name in enumerate(node.input):
                    if name == init.name:
                        node.input[index] = half.name
        else:
            init.CopyFrom(half)
    graph.initializer.extend(new_initializers)

    # --- 2. Float tensor attributes of sea nodes (Constant, ConstantOfShape) go fp16.
    for node in graph.node:
        if in_island(node):
            continue
        for attribute in node.attribute:
            if attribute.type == attribute.TENSOR and attribute.t.data_type == TensorProto.FLOAT:
                attribute.t.CopyFrom(
                    numpy_helper.from_array(
                        numpy_helper.to_array(attribute.t).astype("float16"),
                        attribute.t.name,
                    )
                )

    # (anchor, cast): step 7 splices each cast after its anchor node, or first if None.
    inserted: list[tuple] = []

    def make_cast(source: str, target: str, to, anchor_name) -> None:
        inserted.append(
            (anchor_name, helper.make_node("Cast", [source], [target], to=to, name=target))
        )

    # --- 3. Sea casts to FLOAT now target fp16, except those feeding an island float slot.
    island_float_inputs = set()
    for node in graph.node:
        if not in_island(node):
            continue
        slots = _ISLAND_FLOAT_INPUT_SLOTS.get(node.op_type)
        for index, name in enumerate(node.input):
            if slots is None or index in slots:
                island_float_inputs.add(name)
    island_float_inputs |= pinned_fp32_names
    # Outputs of the sea casts kept fp32 for an island; step 5 must not cast them again.
    island_fp32_sources: set[str] = set()
    for node in list(graph.node):
        if in_island(node) or node.op_type != "Cast":
            continue
        to_float = [a for a in node.attribute if a.name == "to" and a.i == TensorProto.FLOAT]
        if not to_float:
            continue
        produced = node.output[0]
        if produced not in island_float_inputs:
            for attribute in to_float:
                attribute.i = TensorProto.FLOAT16
            continue
        island_fp32_sources.add(produced)
        sea_users = [n for n in consumers_of.get(produced, []) if not in_island(n)]
        if not sea_users:
            continue
        # Read by both: the sea gets an fp16 twin cast of the same source, not one chained
        # off the fp32 cast, which would round twice.
        twin_name = produced + "__fp16"
        if twin_name in node_by_name:
            raise ValueError(
                f"Cannot split amphibious cast {node.name!r}: {twin_name!r} already exists."
            )
        make_cast(node.input[0], twin_name, TensorProto.FLOAT16, node.name)
        node_by_name[twin_name] = True
        for user in sea_users:
            for index, name in enumerate(user.input):
                if name == produced:
                    user.input[index] = twin_name

    # --- 4. fp32 graph inputs reach the sea through one fp16 cast.
    for graph_input in graph.input:
        if graph_input.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        sea_users = [n for n in consumers_of.get(graph_input.name, []) if not in_island(n)]
        if not sea_users:
            continue
        cast_name = graph_input.name + "__fp16"
        make_cast(graph_input.name, cast_name, TensorProto.FLOAT16, None)
        for node in sea_users:
            for index, name in enumerate(node.input):
                if name == graph_input.name:
                    node.input[index] = cast_name

    graph_input_names = {i.name for i in graph.input}
    # fp32 graph outputs of sea nodes. Step 6 hands their names to casts spliced after the
    # casts of step 5, so an island cast reads the internal fp16 tensor instead, keeping the
    # node order topological.
    resited_sea_outputs = {
        output.name
        for output in graph.output
        if output.type.tensor_type.elem_type == TensorProto.FLOAT
        and producer_of.get(output.name) is not None
        and not in_island(producer_of[output.name])
    }

    # --- 5. Island boundaries: one fp32 cast per float edge in, one fp16 cast per float
    # edge out.
    for node in list(graph.node):
        if not in_island(node):
            continue
        slots = _ISLAND_FLOAT_INPUT_SLOTS.get(node.op_type)
        for index, name in enumerate(node.input):
            if not name:
                continue  # omitted optional input
            if slots is not None and index not in slots:
                continue
            dtype = types.get(name)
            if dtype is None and slots is None:
                settled_type(name, node, f"input {index}")
            if dtype is not None and not is_float(dtype):
                continue
            if name in island_fp32_sources:
                continue
            source = producer_of.get(name)
            if source is not None and in_island(source):
                continue
            if source is None and name not in graph_input_names:
                continue  # initializer, kept fp32 in step 1
            if source is None and name in graph_input_names:
                continue
            cast_source = name + "__fp16" if name in resited_sea_outputs else name
            cast_name = name + "__fp32"
            if cast_name not in node_by_name:
                make_cast(cast_source, cast_name, TensorProto.FLOAT, source.name)
                node_by_name[cast_name] = True
            node.input[index] = cast_name
        for out in node.output:
            if not out:
                continue
            sea_users = [n for n in consumers_of.get(out, []) if not in_island(n)]
            if not sea_users:
                continue
            if settled_type(out, node, "output") != TensorProto.FLOAT:
                continue
            cast_name = out + "__fp16"
            make_cast(out, cast_name, TensorProto.FLOAT16, node.name)
            for user in sea_users:
                for index, name in enumerate(user.input):
                    if name == out:
                        user.input[index] = cast_name

    # --- 5b. Pinned plugin slots fed by the sea get one fp32 cast, like island inputs.
    for node in list(graph.node):
        for index in _plugin_fp32_slots(node):
            name = node.input[index]
            source = producer_of.get(name)
            if source is None or in_island(source) or name in island_fp32_sources:
                continue  # already fp32: an initializer, island output or glue cast
            dtype = types.get(name)
            if dtype is not None and not is_float(dtype):
                continue
            cast_source = name + "__fp16" if name in resited_sea_outputs else name
            cast_name = name + "__fp32"
            if cast_name not in node_by_name:
                make_cast(cast_source, cast_name, TensorProto.FLOAT, source.name)
                node_by_name[cast_name] = True
            node.input[index] = cast_name

    # --- 6. A sea node's fp32 graph output: the node writes an internal fp16 tensor, which
    # internal consumers read (PTv3's encoder re-reads its per-stage outputs), and a cast
    # owns the output name.
    for graph_output in graph.output:
        if graph_output.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        producer = producer_of.get(graph_output.name)
        if producer is None or in_island(producer):
            continue
        internal = graph_output.name + "__fp16"
        for index, name in enumerate(producer.output):
            if name == graph_output.name:
                producer.output[index] = internal
        for node in consumers_of.get(graph_output.name, []):
            for index, name in enumerate(node.input):
                if name == graph_output.name:
                    node.input[index] = internal
        make_cast(internal, graph_output.name, TensorProto.FLOAT, producer.name)

    # --- 7. Splice in the casts and drop value_info: a stale FLOAT entry is a type error in
    # onnxruntime's loader.
    front = [cast for anchor, cast in inserted if anchor is None]
    after: dict[str, list] = {}
    for anchor, cast in inserted:
        if anchor is not None:
            after.setdefault(anchor, []).append(cast)
    rebuilt = list(front)
    for node in graph.node:
        rebuilt.append(node)
        rebuilt.extend(after.get(node.name, ()))
    del graph.node[:]
    graph.node.extend(rebuilt)
    del graph.value_info[:]

    onnx.save(model, str(onnx_path))
    logger.info(
        "Cast %s to FP16 around %d linear-island node(s); graph I/O kept FP32.",
        onnx_path.name,
        len(island),
    )
