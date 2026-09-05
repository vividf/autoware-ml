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
  FP16, Q/DQ-island-aware when the graph is also quantized.
- Everything else takes :func:`autocast_to_fp16` (ModelOpt AutoCast, per-node).
"""

from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


#: Quantize/dequantize node spellings. INT8 exports as standard ONNX Q/DQ; FP8 exports
#: as modelopt's TRT-domain custom ops (its E4M3 symbolic bypasses standard
#: ``QuantizeLinear``, whose float8 form it never emits).
_QUANTIZE_OPS = ("QuantizeLinear", "TRT_FP8QuantizeLinear")
_DEQUANTIZE_OPS = ("DequantizeLinear", "TRT_FP8DequantizeLinear")
_QDQ_OPS = _QUANTIZE_OPS + _DEQUANTIZE_OPS


def onnx_has_qdq(onnx_path: Path) -> bool:
    """Whether the ONNX graph contains quantize/dequantize nodes (INT8 or FP8)."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return any(node.op_type in _QDQ_OPS for node in model.graph.node)


def onnx_custom_op_domains(onnx_path: Path) -> tuple[str, ...]:
    """Non-standard operator domains used by the graph's nodes.

    Nodes outside the default ONNX domain (and ``ai.onnx.*``) are runtime plugins —
    ``autoware::ImplicitGemm`` and friends. AutoCast cannot process such a graph:
    it infers types with TensorRT's ONNX parser, which rejects an op whose plugin
    is not registered in the exporting process.
    """
    import onnx

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


#: Ops TensorRT's Q/DQ propagation commutes across. The backward walk from each
#: QuantizeLinear grows the island through these so quantized chains stay castless end
#: to end (conv -> relu -> reshape -> next Q); anything else ends the region. Membership
#: means "quantization commutes with this op, so TensorRT can move the Q across it and
#: fuse an int8-out kernel": monotone/value-neutral activations, linear pooling, and
#: pure data-movement ops qualify. Deliberately ABSENT: nonlinear activations quantization
#: does not commute with (Gelu, Sigmoid, Tanh, Erf, Mul-gating ...) — TensorRT cannot
#: propagate Q across them anyway, so islanding them would only force FP32 execution of
#: ops that are cheaper left in the FP16 sea (requantized after).
#: MAINTENANCE CONTRACT: this list approximates the propagation rules of the pinned
#: TensorRT version. When a quantized chain breaks on an op missing here, the pass logs
#: a warning naming it; extend the list (with a slot entry below — enforced at import —
#: and re-run the three-model battery) then.
_QDQ_COMMUTING_OPS = frozenset(
    {
        "Relu",
        "LeakyRelu",
        "Clip",
        "Concat",
        "Add",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Reshape",
        "Transpose",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
        "Slice",
        "Gather",
        "Identity",
    }
)

#: Positions of the FLOAT-typed inputs, per island op. The boundary-cast logic must
#: never cast an integer edge (a Q/DQ zero-point, a Reshape shape, a Gather index...),
#: so every op that can be an island member has an explicit entry here: Q/DQ, the
#: commuting ops above, and ``None`` rows meaning "every input is float" (the quantized
#: compute ops and all-float pointwise). The import-time check below keeps this table
#: and the whitelist in lockstep, so extending one without the other is impossible.
_ALL_FLOAT = None
_ISLAND_FLOAT_INPUT_SLOTS: dict = {
    # Q/DQ: data + scale are float for Q; only the scale for DQ (its data is int8).
    "QuantizeLinear": (0, 1),
    "TRT_FP8QuantizeLinear": (0, 1),
    "DequantizeLinear": (1,),
    "TRT_FP8DequantizeLinear": (1,),
    # Commuting ops: data input(s) only — trailing inputs are ints (shape/axes/indices)
    # or all inputs are float.
    "Relu": (0,),
    "LeakyRelu": (0,),
    "Clip": _ALL_FLOAT,  # min/max inputs are float
    "Concat": _ALL_FLOAT,
    "Add": _ALL_FLOAT,
    "MaxPool": (0,),
    "AveragePool": (0,),
    "GlobalAveragePool": (0,),
    "Reshape": (0,),  # input[1] is the int64 shape
    "Transpose": (0,),
    "Flatten": (0,),
    "Squeeze": (0,),  # input[1] (opset 13+) is the int64 axes
    "Unsqueeze": (0,),
    "Slice": (0,),  # starts/ends/axes/steps are int64
    "Gather": (0,),  # indices are int
    "Identity": (0,),
}
_MISSING_SLOT_ENTRIES = _QDQ_COMMUTING_OPS - set(_ISLAND_FLOAT_INPUT_SLOTS)
assert not _MISSING_SLOT_ENTRIES, (
    f"_QDQ_COMMUTING_OPS entries missing a float-slot row: {sorted(_MISSING_SLOT_ENTRIES)}. "
    "Every whitelisted op needs one so island boundary casts never touch integer edges."
)

#: Ops quantization mathematically does NOT commute with: a chain deliberately ends here
#: (the op runs FP16 in the sea and the activation is requantized after), so the
#: broken-chain check reports these at DEBUG, not WARNING — only genuinely unclassified
#: ops deserve a look at the whitelist.
_KNOWN_NON_COMMUTING_OPS = frozenset(
    {
        "LayerNormalization",
        "BatchNormalization",
        "Softmax",
        "Sigmoid",
        "HardSigmoid",
        "Gelu",
        "Erf",
        "Tanh",
        "Mul",
        "Div",
        "Pow",
        "Sqrt",
        "Exp",
    }
)


def _quantized_island_names(graph) -> list[str]:
    """The nodes that must stay FP32 for the Q/DQ regions to survive an FP16 cast.

    An island is a Q/DQ pair *plus* what TensorRT's INT8 fusion pattern-matches around
    it: the scale/zero-point producers (their FP32 values are the quantization — rounding
    them through FP16 is what broke the naive cast, measured mIoU 0.545 -> 0.067), the
    consumers of DequantizeLinear outputs (the quantized GEMM itself: a Cast between
    DQ and its consumer defeats the DQ -> op -> Q fusion, measured as TensorRT's
    "Per-tensor quantization/dequantization layer should have 1 scale factor element"),
    and the pointwise chain between a quantized op and the next QuantizeLinear (a Cast
    there blocks Q propagation into the producer, forcing every quantized conv to
    materialize an FP32 output: measured 4.76 ms vs 3.87 for the same CenterPoint
    backbone when the chains stay castless).

    fp32-typed islands are deliberate and load-bearing: retyping Q/DQ to fp16 (legal
    since opset 19) hits a TensorRT 10.8/10.16 defect that emits NaN when the fp16
    combined scale s_x*s_w goes subnormal (see fp16-typed-qdq-nogo.md).
    """
    producer_of = {out: node for node in graph.node for out in node.output}
    island: dict[str, None] = {}
    dq_outputs = set()
    for node in graph.node:
        if node.op_type not in _QDQ_OPS:
            continue
        island[node.name] = None
        for name in node.input[1:]:  # scale / zero_point
            producer = producer_of.get(name)
            if producer is not None:
                island[producer.name] = None
        if node.op_type in _DEQUANTIZE_OPS:
            dq_outputs.update(node.output)
    for node in graph.node:
        if node.name not in island and any(name in dq_outputs for name in node.input):
            island[node.name] = None

    # Grow each island backward from the Q data inputs through commuting ops, so the
    # region between a quantized op and its re-quantization carries no casts. The walk
    # follows FLOAT data edges only (per the slot table): stepping through an integer
    # input (a Gather index chain, a Reshape shape) would drag int-typed glue into the
    # island and let the boundary logic cast integer edges.
    pending = [node.input[0] for node in graph.node if node.op_type in _QUANTIZE_OPS and node.input]
    while pending:
        producer = producer_of.get(pending.pop())
        if producer is None or producer.name in island:
            continue
        if producer.op_type not in _QDQ_COMMUTING_OPS:
            continue
        island[producer.name] = None
        slots = _ISLAND_FLOAT_INPUT_SLOTS[producer.op_type]
        pending.extend(
            name
            for index, name in enumerate(producer.input)
            if slots is _ALL_FLOAT or index in slots
        )
    return list(island)


def _warn_broken_quantized_chains(graph, island: set) -> None:
    """Log when a quantized chain ends on an op outside the commuting whitelist.

    A QuantizeLinear fed (via one non-island hop) from a quantized island op means the
    backward growth stopped on that hop's op type: the chain gets a cast boundary and
    the upstream quantized op materializes FP32 output instead of fusing int8-out.
    Correctness is unaffected — this is a silent-latency guard, and the fix is usually
    one entry in ``_QDQ_COMMUTING_OPS``.
    """
    producer_of = {out: node for node in graph.node for out in node.output}
    dq_consumers = {
        node.name for node in graph.node if node.name in island and node.op_type not in _QDQ_OPS
    }
    for node in graph.node:
        if node.op_type not in _QUANTIZE_OPS or not node.input:
            continue
        hop = producer_of.get(node.input[0])
        if hop is None or hop.name in island:
            continue
        if any(
            producer_of.get(name) is not None and producer_of[name].name in dq_consumers
            for name in hop.input
        ):
            if hop.op_type in _KNOWN_NON_COMMUTING_OPS:
                logger.debug(
                    "Quantized chain ends at %s %r feeding %r (known non-commuting op; "
                    "it runs FP16 and the activation is requantized after — by design).",
                    hop.op_type,
                    hop.name,
                    node.name,
                )
            else:
                logger.warning(
                    "Quantized chain breaks at %s %r feeding %r: the op is not in "
                    "_QDQ_COMMUTING_OPS, so the upstream quantized op will materialize FP32 "
                    "output instead of fusing. If quantization commutes with it, whitelist "
                    "it (with a float-slot row) and re-run the battery; if it does not, "
                    "add it to _KNOWN_NON_COMMUTING_OPS to silence this.",
                    hop.op_type,
                    hop.name,
                    node.name,
                )


def cast_graph_to_fp16(onnx_path: Path) -> None:
    """Convert a graph to FP16 in place, around FP32 Q/DQ islands, keeping the I/O FP32.

    The FP16 path for graphs AutoCast cannot process: plugin graphs (AutoCast types the
    graph with TensorRT's parser, which rejects unregistered plugin ops) and quantized
    graphs (AutoCast rejects Q/DQ models). Everything outside the quantization islands
    becomes FP16 (plugins run FP16 when their tensors are — filters and bias follow the
    feature dtype); the islands stay exactly as the checkpoint calibrated them, with
    single Casts on the float edges where an island meets the FP16 sea; ``keep_io_types``
    semantics hold the artifact ABI at FP32.

    Implemented in-house rather than via onnxconverter-common: the library inserted
    boundary casts around every blocked node (round-trip pairs inside islands), left
    stale value_info entries that hard-fail onnxruntime's loader, and needed the island
    list protected from recomputation — three patch layers this pass makes unnecessary
    by only ever creating casts at true island/IO boundaries.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

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
    island = set(_quantized_island_names(graph))
    _warn_broken_quantized_chains(graph, island)

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
    for node in graph.node:
        if in_island(node) or node.op_type != "Cast":
            continue
        if node.output[0] in island_float_inputs:
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == TensorProto.FLOAT:
                attribute.i = TensorProto.FLOAT16

    # New casts are appended with an anchor to splice after (None = graph front).
    inserted: list[tuple] = []

    def make_cast(source: str, target: str, to, anchor_name) -> None:
        inserted.append(
            (anchor_name, helper.make_node("Cast", [source], [target], to=to, name=target))
        )

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

    # --- 5. Island boundaries: a float edge entering an island from the sea gets one
    # FP32 cast; a float island output consumed by the sea gets one FP16 cast.
    for node in list(graph.node):
        if not in_island(node):
            continue
        slots = _ISLAND_FLOAT_INPUT_SLOTS.get(node.op_type)
        for index, name in enumerate(node.input):
            if slots is not None and index not in slots:
                continue
            source = producer_of.get(name)
            if source is not None and in_island(source):
                continue  # island-internal edge: castless by construction
            if source is None and name not in graph_input_names:
                continue  # initializer: island copies stayed FP32
            if source is None and name in graph_input_names:
                continue  # FP32 graph input read directly
            cast_name = name + "__fp32"
            if cast_name not in node_by_name:
                make_cast(name, cast_name, TensorProto.FLOAT, source.name)
                node_by_name[cast_name] = True
            node.input[index] = cast_name
        if node.op_type in _QUANTIZE_OPS:
            continue  # integer outputs, always island-internal (feed DQ)
        for out in node.output:
            sea_users = [n for n in consumers_of.get(out, []) if not in_island(n)]
            if not sea_users:
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
        "Cast %s to FP16 around %d island node(s); graph I/O kept FP32.",
        onnx_path.name,
        len(island),
    )
