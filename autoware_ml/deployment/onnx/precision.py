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
from dataclasses import dataclass, field
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
from autoware_ml.ops.spconv.contract import (
    AUTOWARE_DOMAIN,
    IMPLICIT_GEMM_FP32_INPUT_SLOTS,
    IMPLICIT_GEMM_OP,
)

logger = logging.getLogger(__name__)


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
_ISLAND_FLOAT_INPUT_SLOTS: dict[str, tuple[int, ...] | None] = {
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

#: QuantizeLinear / DequantizeLinear accept an fp16 input and scale from opset 19.
_FP16_QDQ_MIN_OPSET = 19

_CONTROL_FLOW_OPS = frozenset({"If", "Loop", "Scan"})


def _plugin_fp32_slots(node: onnx.NodeProto) -> tuple[int, ...]:
    by_arity = _PLUGIN_FP32_INPUT_SLOTS.get((node.domain, node.op_type))
    if not by_arity:
        return ()
    return by_arity.get(len(node.input), ())


def _assign_missing_node_names(graph: onnx.GraphProto) -> None:
    """Name the unnamed nodes; the island bookkeeping is keyed by node name."""
    taken = {node.name for node in graph.node if node.name}
    for index, node in enumerate(graph.node):
        if not node.name:
            candidate = f"{node.op_type}_{index}"
            while candidate in taken:
                candidate += "_"
            node.name = candidate
            taken.add(candidate)


def _quantized_island_names(graph: onnx.GraphProto) -> list[str]:
    """Return the names of the island nodes.

    A linear island is a DQ whose output reaches a Gemm/MatMul, directly or through layout
    ops, with its Q, their scale and zero-point producers, the layout ops and the linear
    op: the pattern TensorRT fuses into one INT8 kernel. ``_FP32_ONLY_OPS`` nodes are
    islands of their own.
    """
    producer_of = {out: node for node in graph.node for out in node.output}
    consumers_of: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers_of.setdefault(name, []).append(node)

    island: dict[str, None] = {}

    def admit(node: onnx.NodeProto) -> None:
        island[node.name] = None
        if node.op_type in QDQ_OPS:
            for name in node.input[1:]:  # scale / zero_point
                producer = producer_of.get(name)
                if producer is not None:
                    island[producer.name] = None

    for node in graph.node:
        if node.op_type not in DEQUANTIZE_OPS or not node.output:
            continue
        hops: list[onnx.NodeProto] = []
        linear: list[onnx.NodeProto] = []
        frontier: list[tuple[str, list[onnx.NodeProto]]] = [(node.output[0], [])]
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


def _replace_input(node: onnx.NodeProto, old: str, new: str) -> None:
    for index, name in enumerate(node.input):
        if name == old:
            node.input[index] = new


@dataclass
class _Cast:
    """One graph of the cast: its indexes and the casts to splice in.

    ``types`` are the element types before any rewrite: the boundary decisions need the
    original ones. The island, pinned plugin slots and tensor producers / consumers do not
    change during the rewrite, since every step only retypes tensors or redirects edges to
    the casts it adds.
    """

    model: onnx.ModelProto
    types: dict[str, int]
    island: set[str]
    producer_of: dict[str, onnx.NodeProto]
    consumers_of: dict[str, list[onnx.NodeProto]]
    node_by_name: dict[str, onnx.NodeProto]
    pinned_fp32_slots: set[tuple[str, int]]
    pinned_fp32_names: set[str]
    graph_input_names: set[str]
    #: ``(anchor node name, cast)``: spliced after the anchor, or first for ``None``.
    inserted: list[tuple[str | None, onnx.NodeProto]] = field(default_factory=list)
    #: Names the casts added so far own; a second cast of one tensor reuses the first.
    cast_names: set[str] = field(default_factory=set)

    @classmethod
    def index(cls, model: onnx.ModelProto) -> _Cast:
        graph = model.graph
        _assign_missing_node_names(graph)
        consumers_of: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for name in node.input:
                consumers_of.setdefault(name, []).append(node)
        pinned_fp32_slots: set[tuple[str, int]] = set()
        pinned_fp32_names: set[str] = set()
        for node in graph.node:
            for slot in _plugin_fp32_slots(node):
                if slot < len(node.input) and node.input[slot]:
                    pinned_fp32_slots.add((node.name, slot))
                    pinned_fp32_names.add(node.input[slot])
        return cls(
            model=model,
            types=tensor_types(model),
            island=set(_quantized_island_names(graph)),
            producer_of={out: node for node in graph.node for out in node.output},
            consumers_of=consumers_of,
            node_by_name={node.name: node for node in graph.node},
            pinned_fp32_slots=pinned_fp32_slots,
            pinned_fp32_names=pinned_fp32_names,
            graph_input_names={i.name for i in graph.input},
        )

    @property
    def graph(self) -> onnx.GraphProto:
        return self.model.graph

    def in_island(self, node: onnx.NodeProto) -> bool:
        return node.name in self.island

    def sea_users(self, name: str) -> list[onnx.NodeProto]:
        return [n for n in self.consumers_of.get(name, []) if not self.in_island(n)]

    def reads_fp32_pinned(self, node: onnx.NodeProto, name: str) -> bool:
        return any(
            (node.name, slot) in self.pinned_fp32_slots
            for slot, value in enumerate(node.input)
            if value == name
        )

    def settled_type(self, name: str, node: onnx.NodeProto, role: str) -> int:
        """The original element type of ``name``; an island edge the graph does not type fails."""
        dtype = self.types.get(name)
        if dtype is None:
            raise ValueError(
                f"cast_graph_to_fp16 cannot type tensor {name!r}, {role} of island node "
                f"{node.name!r} ({node.op_type}): it is not a graph input, initializer or "
                "Constant, and ONNX shape inference does not reach it, typically a "
                "custom-domain producer without value_info. Declare it (the exporter records "
                "the traced dtype; a hand-built graph needs a value_info entry) or give "
                f"{node.op_type!r} a float-slot row in _ISLAND_FLOAT_INPUT_SLOTS. Guessing "
                "FLOAT is not an option: that is how a shape edge gets cast."
            )
        return dtype

    def add_cast(self, source: str, target: str, to: int, anchor: str | None) -> bool:
        """Add ``target = Cast(source)`` after ``anchor``; ``False`` when ``target`` exists."""
        if target in self.cast_names or target in self.node_by_name:
            return False
        self.inserted.append(
            (anchor, helper.make_node("Cast", [source], [target], to=to, name=target))
        )
        self.cast_names.add(target)
        return True


def _check_supported(cast: _Cast, onnx_path: Path) -> None:
    control_flow = sorted({n.op_type for n in cast.graph.node if n.op_type in _CONTROL_FLOW_OPS})
    if control_flow:
        raise NotImplementedError(
            f"cast_graph_to_fp16 does not handle control-flow subgraphs ({', '.join(control_flow)} "
            f"in {onnx_path.name}): their bodies would keep FP32 tensors against the converted "
            "FP16 sea. Export the stage without in-graph control flow, or extend the pass to "
            "recurse into subgraph bodies first."
        )
    sea_standard_qdq = [
        n
        for n in cast.graph.node
        if n.op_type in QDQ_OPS and not n.domain and not cast.in_island(n)
    ]
    default_opset = next(
        (o.version for o in cast.model.opset_import if o.domain in ("", "ai.onnx")), 0
    )
    if sea_standard_qdq and default_opset < _FP16_QDQ_MIN_OPSET:
        raise ValueError(
            f"cast_graph_to_fp16 retypes {len(sea_standard_qdq)} conv-side Q/DQ node(s) of "
            f"{onnx_path.name} to fp16, which QuantizeLinear/DequantizeLinear only allow from "
            f"opset {_FP16_QDQ_MIN_OPSET}; the graph is opset {default_opset}. Export with "
            f"deploy.onnx.opset_version >= {_FP16_QDQ_MIN_OPSET}."
        )


def _retype_initializers(cast: _Cast) -> None:
    """Initializers go fp16 for the sea and stay fp32 for islands and pinned plugin slots.

    One read by both sides gets an fp16 twin for the sea. A Q/DQ scale that underflows in
    fp16 is refused: the fold of :mod:`autoware_ml.deployment.onnx.qdq` made every scale an
    initializer, so this check sees all of them.
    """
    graph = cast.graph
    new_initializers = []
    qdq_scale_names = {
        node.input[1] for node in graph.node if node.op_type in QDQ_OPS and len(node.input) > 1
    }
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        users = cast.consumers_of.get(init.name, [])
        sea_users = [
            n for n in users if not cast.in_island(n) and not cast.reads_fp32_pinned(n, init.name)
        ]
        island_users = [
            n for n in users if cast.in_island(n) or cast.reads_fp32_pinned(n, init.name)
        ]
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
            half_values, init.name + "__fp16" if island_users else init.name
        )
        if island_users:
            new_initializers.append(half)
            for node in sea_users:
                _replace_input(node, init.name, half.name)
        else:
            init.CopyFrom(half)
    graph.initializer.extend(new_initializers)


def _retype_constant_attributes(cast: _Cast) -> None:
    """Float tensor attributes of sea nodes (Constant, ConstantOfShape) go fp16."""
    for node in cast.graph.node:
        if cast.in_island(node):
            continue
        for attribute in node.attribute:
            if attribute.type == attribute.TENSOR and attribute.t.data_type == TensorProto.FLOAT:
                attribute.t.CopyFrom(
                    numpy_helper.from_array(
                        numpy_helper.to_array(attribute.t).astype("float16"), attribute.t.name
                    )
                )


def _island_float_inputs(cast: _Cast) -> set[str]:
    """Names an island node reads in a float slot, plus the pinned plugin slots."""
    names: set[str] = set()
    for node in cast.graph.node:
        if not cast.in_island(node):
            continue
        slots = _ISLAND_FLOAT_INPUT_SLOTS.get(node.op_type)
        for index, name in enumerate(node.input):
            if slots is None or index in slots:
                names.add(name)
    return names | cast.pinned_fp32_names


def _retarget_sea_casts(cast: _Cast) -> set[str]:
    """Sea casts to FLOAT now target fp16, except those feeding an island float slot.

    Returns:
        The outputs of the sea casts kept fp32 for an island; the boundary casts must not
        cast them again.
    """
    island_float_inputs = _island_float_inputs(cast)
    island_fp32_sources: set[str] = set()
    for node in list(cast.graph.node):
        if cast.in_island(node) or node.op_type != "Cast":
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
        sea_users = cast.sea_users(produced)
        if not sea_users:
            continue
        # Read by both: the sea gets an fp16 twin cast of the same source, not one chained
        # off the fp32 cast, which would round twice.
        twin_name = produced + "__fp16"
        if not cast.add_cast(node.input[0], twin_name, TensorProto.FLOAT16, node.name):
            raise ValueError(
                f"Cannot split amphibious cast {node.name!r}: {twin_name!r} already exists."
            )
        for user in sea_users:
            _replace_input(user, produced, twin_name)
    return island_fp32_sources


def _cast_graph_inputs(cast: _Cast) -> None:
    """fp32 graph inputs reach the sea through one fp16 cast."""
    for graph_input in cast.graph.input:
        if graph_input.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        sea_users = cast.sea_users(graph_input.name)
        if not sea_users:
            continue
        cast_name = graph_input.name + "__fp16"
        cast.add_cast(graph_input.name, cast_name, TensorProto.FLOAT16, None)
        for node in sea_users:
            _replace_input(node, graph_input.name, cast_name)


def _resited_sea_outputs(cast: _Cast) -> set[str]:
    """fp32 graph outputs produced by sea nodes.

    :func:`_resite_sea_graph_outputs` hands their names to casts spliced after the casts of
    :func:`_cast_island_boundaries`, so an island cast of one reads the internal fp16 tensor
    instead, keeping the node order topological.
    """
    return {
        output.name
        for output in cast.graph.output
        if output.type.tensor_type.elem_type == TensorProto.FLOAT
        and cast.producer_of.get(output.name) is not None
        and not cast.in_island(cast.producer_of[output.name])
    }


def _cast_island_boundaries(
    cast: _Cast, island_fp32_sources: set[str], resited_sea_outputs: set[str]
) -> None:
    """One fp32 cast per float edge into an island, one fp16 cast per float edge out of it.

    A pinned plugin slot fed by the sea gets its fp32 cast like an island input.
    """

    def cast_edge_to_fp32(node: onnx.NodeProto, index: int) -> None:
        name = node.input[index]
        source = cast.producer_of.get(name)
        cast_source = name + "__fp16" if name in resited_sea_outputs else name
        cast_name = name + "__fp32"
        cast.add_cast(cast_source, cast_name, TensorProto.FLOAT, source.name)
        node.input[index] = cast_name

    for node in list(cast.graph.node):
        if not cast.in_island(node):
            continue
        slots = _ISLAND_FLOAT_INPUT_SLOTS.get(node.op_type)
        for index, name in enumerate(node.input):
            if not name:
                continue  # omitted optional input
            if slots is not None and index not in slots:
                continue
            dtype = cast.types.get(name)
            if dtype is None and slots is None:
                cast.settled_type(name, node, f"input {index}")
            if dtype is not None and not is_float(dtype):
                continue
            if name in island_fp32_sources:
                continue
            source = cast.producer_of.get(name)
            if source is None or cast.in_island(source):
                continue  # an initializer or graph input (kept fp32), or an island edge
            cast_edge_to_fp32(node, index)
        for out in node.output:
            if not out:
                continue
            sea_users = cast.sea_users(out)
            if not sea_users:
                continue
            if cast.settled_type(out, node, "output") != TensorProto.FLOAT:
                continue
            cast_name = out + "__fp16"
            cast.add_cast(out, cast_name, TensorProto.FLOAT16, node.name)
            for user in sea_users:
                _replace_input(user, out, cast_name)

    for node in list(cast.graph.node):
        for index in _plugin_fp32_slots(node):
            name = node.input[index]
            source = cast.producer_of.get(name)
            if source is None or cast.in_island(source) or name in island_fp32_sources:
                continue  # already fp32: an initializer, island output or glue cast
            dtype = cast.types.get(name)
            if dtype is not None and not is_float(dtype):
                continue
            cast_edge_to_fp32(node, index)


def _resite_sea_graph_outputs(cast: _Cast) -> None:
    """A sea node's fp32 graph output: the node writes an internal fp16 tensor and a cast
    owns the output name, so internal consumers (PTv3's encoder re-reads its per-stage
    outputs) read the fp16 tensor."""
    for graph_output in cast.graph.output:
        if graph_output.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        producer = cast.producer_of.get(graph_output.name)
        if producer is None or cast.in_island(producer):
            continue
        internal = graph_output.name + "__fp16"
        for index, name in enumerate(producer.output):
            if name == graph_output.name:
                producer.output[index] = internal
        for node in cast.consumers_of.get(graph_output.name, []):
            _replace_input(node, graph_output.name, internal)
        cast.add_cast(internal, graph_output.name, TensorProto.FLOAT, producer.name)


def _splice_casts(cast: _Cast) -> None:
    """Insert the casts after their anchors and drop ``value_info``: a stale FLOAT entry is a
    type error in onnxruntime's loader."""
    graph = cast.graph
    front = [node for anchor, node in cast.inserted if anchor is None]
    after: dict[str, list[onnx.NodeProto]] = {}
    for anchor, node in cast.inserted:
        if anchor is not None:
            after.setdefault(anchor, []).append(node)
    rebuilt = list(front)
    for node in graph.node:
        rebuilt.append(node)
        rebuilt.extend(after.get(node.name, ()))
    del graph.node[:]
    graph.node.extend(rebuilt)
    del graph.value_info[:]


def cast_graph_to_fp16(onnx_path: Path) -> None:
    """Cast a graph to fp16 in place around its fp32 islands, keeping the graph I/O fp32.

    Casts are decided per tensor edge, so integer and boolean edges (a Shape result,
    MaxPool indices) are never cast, and an island edge of unknown type is refused rather
    than guessed as FLOAT. The Q/DQ scales must already be initializers
    (:func:`autoware_ml.deployment.onnx.qdq.fold_qdq_params`).

    Not built on onnxconverter-common: blocking the island nodes there puts round-trip
    casts around each of them and leaves stale ``value_info`` that onnxruntime's loader
    rejects.

    Raises:
        NotImplementedError: For a graph with control flow.
        ValueError: For conv-side Q/DQ below opset 19, a Q/DQ scale that underflows in
            fp16, or an island edge of unknown type.
    """
    cast = _Cast.index(onnx.load(str(onnx_path)))
    _check_supported(cast, onnx_path)
    _retype_initializers(cast)
    _retype_constant_attributes(cast)
    island_fp32_sources = _retarget_sea_casts(cast)
    _cast_graph_inputs(cast)
    resited_sea_outputs = _resited_sea_outputs(cast)
    _cast_island_boundaries(cast, island_fp32_sources, resited_sea_outputs)
    _resite_sea_graph_outputs(cast)
    _splice_casts(cast)
    onnx.save(cast.model, str(onnx_path))
    logger.info(
        "Cast %s to FP16 around %d linear-island node(s); graph I/O kept FP32.",
        onnx_path.name,
        len(cast.island),
    )
