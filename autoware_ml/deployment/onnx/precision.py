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


"""Precision passes over exported ONNX graphs, and the graph facts that select them.

Three graph kinds, three treatments (driven by :mod:`autoware_ml.deployment.export`):

- Q/DQ graphs without plugin ops keep the precision their checkpoint bakes in.
- Plugin graphs (``autoware::`` domains) take :func:`cast_graph_to_fp16` — whole-graph
  FP16; when the graph is also quantized, its linear (Gemm/MatMul) Q/DQ stay fp32-typed.
- Everything else takes :func:`autocast_to_fp16` (ModelOpt AutoCast, per-node).
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


logger = logging.getLogger(__name__)


def onnx_has_qdq(onnx_path: Path) -> bool:
    """Whether the ONNX graph contains quantize/dequantize nodes (INT8 or FP8)."""

    model = onnx.load(str(onnx_path), load_external_data=False)
    return any(node.op_type in QDQ_OPS for node in model.graph.node)


def onnx_custom_op_domains(onnx_path: Path) -> tuple[str, ...]:
    """Non-standard operator domains used by the graph's nodes.

    Nodes outside the default ONNX domain (and ``ai.onnx.*``) are runtime plugins —
    ``autoware::ImplicitGemm`` and friends. AutoCast cannot process such a graph:
    it infers types with TensorRT's ONNX parser, which rejects an op whose plugin
    is not registered in the exporting process.
    """

    model = onnx.load(str(onnx_path), load_external_data=False)
    domains = {
        node.domain
        for node in model.graph.node
        if node.domain and not node.domain.startswith("ai.onnx")
    }
    return tuple(sorted(domains))


def _assign_missing_node_names(graph) -> None:
    """Give every node a unique name (the converter's node_block_list matches by name)."""
    taken = {node.name for node in graph.node if node.name}
    for index, node in enumerate(graph.node):
        if not node.name:
            candidate = f"{node.op_type}_{index}"
            while candidate in taken:
                candidate += "_"
            node.name = candidate
            taken.add(candidate)


#: The quantized compute ops whose Q/DQ stay fp32-typed. TensorRT 10.8/10.16 emits NaN
#: from a strongly-typed fp16 Q/DQ -> INT8 **Gemm** when the fp16 combined scale
#: ``s_x * s_w[c]`` goes subnormal (the fused kernel folds the dequant scales into an fp16
#: intermediate); fp32-typed Q/DQ take a different kernel path and are exact. Conv kernels
#: are immune: CenterPoint and BEVFusion carry 26 conv modules with subnormal combined
#: scales and evaluate identically under fp16-typed Q/DQ. Hence the split is by op type,
#: not by node name or scale threshold — both were tried and are unreliable (a name list
#: missed two culprits; 15 of PTv3's 19 head Gemms are subnormal yet only 6 misbehave).
#: Evidence: work_dirs/reviews/fp16-typed-qdq-nogo.md, uniform-fp16-exception-rule.md.
_LINEAR_OPS = frozenset({"Gemm", "MatMul"})

#: Pure layout ops a DQ output may pass through on its way to the linear op (a weight
#: DQ -> Transpose -> MatMul is how modelopt's FP8 export spells a Linear). They join the
#: island so the DQ -> op edge stays castless (a Cast between DQ and its consumer defeats
#: TensorRT's Q/DQ fusion — measured as "Per-tensor quantization/dequantization layer
#: should have 1 scale factor element").
_LAYOUT_OPS = frozenset({"Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity"})

#: Positions of the FLOAT-typed inputs, per island op, as the op's ONNX spec fixes them.
#: The boundary-cast logic must never cast an integer edge (a Q/DQ zero-point, a Reshape
#: shape...), so every op that can be an island member has a row; ``None`` means "every
#: input is float".
_ALL_FLOAT = None
_ISLAND_FLOAT_INPUT_SLOTS: dict = {
    # Q/DQ: data + scale are float for Q; only the scale for DQ (its data is int8/fp8).
    "QuantizeLinear": (0, 1),
    "TRT_FP8QuantizeLinear": (0, 1),
    "DequantizeLinear": (1,),
    "TRT_FP8DequantizeLinear": (1,),
    # Layout hops: data input only — trailing inputs are int64 shape/axes.
    "Transpose": (0,),
    "Reshape": (0,),
    "Flatten": (0,),
    "Squeeze": (0,),
    "Unsqueeze": (0,),
    "Identity": (0,),
    # The quantized linear op itself: all-float by spec.
    "Gemm": _ALL_FLOAT,
    "MatMul": _ALL_FLOAT,
}
_MISSING_SLOT_ENTRIES = (_LINEAR_OPS | _LAYOUT_OPS | set(QDQ_OPS)) - set(_ISLAND_FLOAT_INPUT_SLOTS)
assert not _MISSING_SLOT_ENTRIES, (
    f"Island-eligible ops missing a float-slot row: {sorted(_MISSING_SLOT_ENTRIES)}. "
    "Every one needs a row so island boundary casts never touch integer edges."
)

#: Minimum default-domain opset for fp16-typed Q/DQ (fp16 ``x`` and ``y_scale`` on
#: QuantizeLinear/DequantizeLinear are legal from opset 19).
_FP16_QDQ_MIN_OPSET = 19


def _quantized_island_names(graph) -> list[str]:
    """The nodes that stay FP32-typed when the graph is cast to FP16: the linear islands.

    An island is a DQ whose output reaches a Gemm/MatMul (directly or through layout
    ops), plus that DQ's Q, both nodes' scale/zero-point producers (their FP32 values
    *are* the quantization), the layout hops and the linear op itself — the exact
    pattern TensorRT fuses into one INT8 kernel, kept castless. Every other Q/DQ (the
    conv family) is sea: it is retyped to fp16 with the rest of the graph, scale
    included. See ``_LINEAR_OPS`` for why the split runs along op type.
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
        # Follow the DQ output forward through layout ops; collect the linear consumers
        # and the hops that lead to them.
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
    return list(island)


def cast_graph_to_fp16(onnx_path: Path) -> None:
    """Convert a graph to FP16 in place, around FP32 linear Q/DQ islands, keeping the I/O FP32.

    The FP16 path for graphs AutoCast cannot process: plugin graphs (AutoCast types the
    graph with TensorRT's parser, which rejects unregistered plugin ops) and quantized
    graphs (AutoCast rejects Q/DQ models). The whole graph becomes FP16 — conv-family
    Q/DQ included, scale and all (legal from opset 19; plugins run FP16 when their
    tensors are) — except the linear islands (:func:`_quantized_island_names`): a Q/DQ
    pair feeding a Gemm/MatMul stays fp32-typed exactly as the checkpoint calibrated it,
    with single Casts on the float edges where it meets the FP16 sea. ``keep_io_types``
    semantics hold the artifact ABI at FP32.

    Casts are decided per tensor edge, not per node: an island member's integer or
    boolean edges (a Shape result, MaxPool's indices, an Expand shape) are never touched,
    whatever the node's membership says. Every boundary edge must have a settled type —
    from the op's float-slot row, or from the graph itself via
    :func:`~autoware_ml.deployment.onnx.dtypes.tensor_types`; one that has neither is a
    :class:`ValueError`, because guessing FLOAT is how a shape edge gets cast.

    Implemented in-house rather than via onnxconverter-common: the library inserted
    boundary casts around every blocked node (round-trip pairs inside islands), left
    stale value_info entries that hard-fail onnxruntime's loader, and needed the island
    list protected from recomputation — three patch layers this pass makes unnecessary
    by only ever creating casts at true island/IO boundaries.

    Raises:
        NotImplementedError: For graphs with control-flow subgraphs.
        ValueError: For a graph below opset 19 that has conv-side Q/DQ to retype.
        ValueError: For an island boundary tensor whose element type the graph does not
            settle (typically a custom-domain producer without ``value_info``).
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
    # Element types before any rewrite: the boundary decisions below are per edge.
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

    # --- 1. Initializers: island consumers keep FP32; sea consumers get FP16 (a split
    # copy when an initializer feeds both worlds).
    fp16_twin: dict[str, str] = {}
    new_initializers = []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        users = consumers_of.get(init.name, [])
        sea_users = [n for n in users if not in_island(n)]
        island_users = [n for n in users if in_island(n)]
        if not sea_users:
            continue  # island-only (or unused): keep FP32
        half = numpy_helper.from_array(
            numpy_helper.to_array(init).astype("float16"),
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

    # --- 2. Sea nodes' float tensor attributes (Constant, ConstantOfShape, ...) go FP16;
    # island Constants (Q/DQ scales) keep their exact FP32 bytes.
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

    # New casts are appended with an anchor to splice after (None = graph front).
    inserted: list[tuple] = []

    def make_cast(source: str, target: str, to, anchor_name) -> None:
        inserted.append(
            (anchor_name, helper.make_node("Cast", [source], [target], to=to, name=target))
        )

    # --- 3. Pre-existing sea casts to FLOAT (int64 -> float glue) now target FLOAT16;
    # ones feeding an island float slot keep producing FP32 for it.
    island_float_inputs = set()
    for node in graph.node:
        if not in_island(node):
            continue
        slots = _ISLAND_FLOAT_INPUT_SLOTS.get(node.op_type)
        for index, name in enumerate(node.input):
            if slots is None or index in slots:
                island_float_inputs.add(name)
    # Sea tensors that are already FP32 *for the island's sake*: step 5 must not cast them
    # again (it would insert an FP32 -> FP32 no-op on the island's own input edge).
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
            continue  # island-only consumer: the cast keeps producing FP32
        # Amphibious glue cast: the island slot needs its FP32 and the sea needs FP16, so
        # the cast splits the way an amphibious initializer does in step 1. Casting the
        # int64 source twice (rather than chaining int64 -> FP32 -> FP16) keeps the sea
        # copy exact for the same reason the FP32 copy is exact.
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

    # --- 4. FP32 graph inputs feed sea consumers through one FP16 cast (island
    # consumers keep reading the FP32 input directly).
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
    # Names step 6 re-homes: an FP32 graph output produced by a sea node. Its public name
    # will belong to a boundary cast spliced *after* the casts made here, so an island
    # reading it must read the producer's internal FP16 tensor instead — otherwise the
    # node order stops being topological and the ONNX loader rejects the graph.
    resited_sea_outputs = {
        output.name
        for output in graph.output
        if output.type.tensor_type.elem_type == TensorProto.FLOAT
        and producer_of.get(output.name) is not None
        and not in_island(producer_of[output.name])
    }

    # --- 5. Island boundaries: a float edge entering an island from the sea gets one
    # FP32 cast; a float island output consumed by the sea gets one FP16 cast. Integer
    # and boolean edges pass through untouched in both directions.
    for node in list(graph.node):
        if not in_island(node):
            continue
        slots = _ISLAND_FLOAT_INPUT_SLOTS.get(node.op_type)
        for index, name in enumerate(node.input):
            if not name:
                continue  # omitted optional input
            if slots is not None and index not in slots:
                continue  # integer slot by spec (a Q/DQ zero-point, a Reshape shape...)
            dtype = types.get(name)
            if dtype is None and slots is None:
                settled_type(name, node, f"input {index}")
            if dtype is not None and not is_float(dtype):
                continue  # integer / bool edge: casting it to FLOAT would break the graph
            if name in island_fp32_sources:
                continue  # a sea cast kept FP32 precisely to feed this slot
            source = producer_of.get(name)
            if source is not None and in_island(source):
                continue  # island-internal edge: castless by construction
            if source is None and name not in graph_input_names:
                continue  # initializer: island copies stayed FP32
            if source is None and name in graph_input_names:
                continue  # FP32 graph input read directly
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
                # A Q's int8, a Shape's int64, MaxPool's indices: the sea reads them as
                # they are. (So does an output the graph already holds in FP16.)
                continue
            cast_name = out + "__fp16"
            make_cast(out, cast_name, TensorProto.FLOAT16, node.name)
            for user in sea_users:
                for index, name in enumerate(user.input):
                    if name == out:
                        user.input[index] = cast_name

    # --- 6. FP32 graph outputs produced by sea nodes: the producer emits FP16 under an
    # internal name, a boundary cast owns the output name, and internal consumers read
    # the FP16 tensor (PTv3's encoder re-consumes its own per-stage outputs).
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

    # --- 7. Splice the new casts in (after their producer; graph-input casts up front)
    # and drop the value_info hints: backends re-infer, and a stale FLOAT entry is a
    # hard type error in onnxruntime's loader.
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
