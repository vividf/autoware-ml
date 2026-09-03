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
from typing import Any, Mapping

import torch

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


def _quantized_island_names(graph) -> list[str]:
    """The nodes that must stay FP32 for the Q/DQ regions to survive an FP16 cast.

    An island is a Q/DQ pair *plus* what TensorRT's INT8 fusion pattern-matches around
    it: the scale/zero-point producers (their FP32 values are the quantization — rounding
    them through FP16 is what broke the naive cast, measured mIoU 0.545 -> 0.067), and
    the consumers of DequantizeLinear outputs (the quantized GEMM itself: a Cast between
    DQ and its consumer defeats the DQ -> op -> Q fusion, measured as TensorRT's
    "Per-tensor quantization/dequantization layer should have 1 scale factor element").
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
    return list(island)


def _strip_fp16_round_trips(graph) -> int:
    """Remove ``Cast(fp16) -> Cast(fp32)`` chains, reconnecting to the FP32 source.

    The converter inserts boundary casts around *every* blocked node, including between
    two blocked neighbors, so island-internal edges become FP32 -> FP16 -> FP32 round
    trips: the values (Q/DQ scales above all) come back rounded, and the extra nodes sit
    inside TensorRT's fusion patterns. A genuine island boundary is a single Cast and is
    left alone; only the adjacent pairs are collapsed.
    """
    from onnx import TensorProto

    def cast_target(node):
        return next((a.i for a in node.attribute if a.name == "to"), None)

    producer_of = {out: node for node in graph.node for out in node.output}
    graph_outputs = {output.name for output in graph.output}
    removed_pairs = 0
    for down in list(graph.node):
        if down.op_type != "Cast" or cast_target(down) != TensorProto.FLOAT:
            continue
        if down.output[0] in graph_outputs:
            continue
        up = producer_of.get(down.input[0])
        if up is None or up.op_type != "Cast" or cast_target(up) != TensorProto.FLOAT16:
            continue
        source = up.input[0]
        for node in graph.node:
            for index, name in enumerate(node.input):
                if name == down.output[0]:
                    node.input[index] = source
        graph.node.remove(down)
        removed_pairs += 1
        if not any(up.output[0] in node.input for node in graph.node) and (
            up.output[0] not in graph_outputs
        ):
            graph.node.remove(up)
    return removed_pairs


def _island_input_names(graph) -> set[str]:
    """Tensors consumed by island nodes in the converted graph (its FP32 entries)."""
    island = set(_quantized_island_names(graph))
    return {name for node in graph.node if node.name in island for name in node.input}


def cast_graph_to_fp16(onnx_path: Path) -> None:
    """Convert a whole graph to FP16 in place, keeping the I/O tensors FP32.

    The FP16 path for graphs AutoCast cannot process: AutoCast types the graph with
    TensorRT's parser and calibrates per node, which needs every operator implemented in
    the exporting process, while a plugin graph's compute lives almost entirely in its
    plugin nodes anyway — per-node selection has nothing meaningful to keep in FP32. So
    such graphs get the blunt conversion: every float initializer and internal tensor
    becomes FP16 (the plugins run FP16 when their tensors are — filters and bias follow
    the feature dtype), engines still build strongly typed, and ``keep_io_types`` holds
    the artifact ABI at FP32.

    A quantized (Q/DQ) graph converts too, as FP16 *around* FP32 quantization islands:
    the Q/DQ nodes, their scale/zero-point constants, and the GEMMs consuming the
    dequantized tensors stay exactly as the checkpoint calibrated them (see
    :func:`_quantized_island_names`), everything else becomes FP16, and the island
    boundaries carry single Casts on the data path only.
    """
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16

    model = onnx.load(str(onnx_path))
    _assign_missing_node_names(model.graph)
    island_names = _quantized_island_names(model.graph)
    converted = float16.convert_float_to_float16(
        model, keep_io_types=True, node_block_list=island_names or None
    )
    if island_names:
        removed = _strip_fp16_round_trips(converted.graph)
        logger.info(
            "Kept %d node(s) in the FP32 quantization islands; collapsed %d island-internal "
            "FP16 round trip(s).",
            len(island_names),
            removed,
        )

    # The converter rewrites float tensors and initializers but leaves pre-existing
    # int-to-FLOAT Cast nodes at FLOAT, which then meet FP16 tensors downstream
    # ("DIV must have same input types"). After a whole-graph conversion the only
    # legitimate FLOAT casts are the boundary ones feeding the kept-FP32 graph outputs.
    graph_outputs = {output.name for output in converted.graph.output}
    island_inputs = _island_input_names(converted.graph) if island_names else set()
    for node in converted.graph.node:
        if node.op_type != "Cast" or node.output[0] in graph_outputs:
            continue
        if node.output[0] in island_inputs:
            # A boundary cast feeding a quantization island: FP32 by design.
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == TensorProto.FLOAT:
                attribute.i = TensorProto.FLOAT16

    # A kept-FP32 graph output can also be consumed *inside* the graph (PTv3's encoder
    # emits its per-stage point features and keeps pooling them). ``keep_io_types``
    # inserts the boundary Cast under the output's name, so those internal consumers
    # would read the FP32 copy and meet FP16 weights ("must have same input types").
    # The boundary cast belongs to the output alone: rewire internal consumers to the
    # FP16 tensor it came from.
    boundary_sources: dict[str, str] = {}
    for node in converted.graph.node:
        if node.op_type != "Cast" or node.output[0] not in graph_outputs:
            continue
        if any(
            attribute.name == "to" and attribute.i == TensorProto.FLOAT
            for attribute in node.attribute
        ):
            boundary_sources[node.output[0]] = node.input[0]
    for node in converted.graph.node:
        if node.op_type == "Cast" and node.output[0] in boundary_sources:
            continue
        for index, name in enumerate(node.input):
            if name in boundary_sources:
                node.input[index] = boundary_sources[name]

    # The conversion leaves stale FLOAT value_info entries behind for tensors that now
    # carry FP16 — e.g. around the converter's own op-block-listed nodes (Max, TopK...),
    # whose FLOAT boundary casts the retargeting above flips to FP16. value_info is an
    # optional hint, but every stale entry is a hard type error in onnxruntime's loader
    # (TensorRT's parser ignores them), so drop the hints and let backends re-infer.
    del converted.graph.value_info[:]

    onnx.save(converted, str(onnx_path))
    logger.info("Cast %s to FP16 (graph I/O kept FP32).", onnx_path.name)


def autocast_to_fp16(onnx_path: Path, sample_inputs: Mapping[str, Any]) -> None:
    """Convert an exported FP32 ONNX graph to mixed FP16 in place (ModelOpt AutoCast).

    TensorRT engines build strongly typed, so FP16 must live in the graph itself; this
    is the official replacement for the removed ``BuilderFlag.FP16`` weak-typing path.
    I/O tensor types are preserved (``keep_io_types=True``) so the artifact ABI —
    what the Autoware runtime binds against — does not change with the precision.

    ``sample_inputs`` (the stage's trace inputs) drive AutoCast's reference run: its
    magnitude-based node classification then sees real activations, and graphs with
    dynamic spatial dims get valid shapes (AutoCast's random fallback fills dynamic
    dims with 1, which breaks strided convolutions). Same inputs → same partition,
    so the conversion is reproducible.

    Quantized graphs must not pass through here: AutoCast does not support Q/DQ models
    (the caller gates on :func:`onnx_has_qdq`).

    Args:
        onnx_path: Exported FP32 ``.onnx``, overwritten with the mixed-FP16 graph.
        sample_inputs: ONNX input name -> tensor/array with concrete shapes (one batch).
    """
    from modelopt.onnx.autocast import convert_to_mixed_precision

    import numpy as np
    import onnx

    feed = {}
    for name, value in sample_inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        feed[name] = np.asarray(value)
    calibration_path = onnx_path.with_suffix(".autocast_inputs.npz")
    np.savez(calibration_path, **feed)

    logger.info("AutoCast: converting %s to mixed FP16 (I/O types preserved)...", onnx_path.name)
    try:
        model = convert_to_mixed_precision(
            onnx_path=str(onnx_path),
            low_precision_type="fp16",
            keep_io_types=True,
            calibration_data=str(calibration_path),
        )
    finally:
        calibration_path.unlink(missing_ok=True)
    onnx.save(model, str(onnx_path))
    logger.info("AutoCast: wrote mixed-FP16 graph back to %s", onnx_path)


def keep_topk_in_fp16(onnx_path: Path) -> Path:
    """Let TopK read its FP16 tensor directly instead of an FP32 copy.

    AutoCast pins TopK to FP32, so in a mixed-FP16 graph the selection input arrives
    through a Cast — for a proposal head that means casting the *entire* flattened
    heatmap before selecting a few hundred elements (BEVFusion: 3.24M elements,
    measured 0.81 ms -> 0.45 ms on the dense graph by bypassing it; the ``sorted``
    attribute measured as irrelevant to TensorRT).

    A stage declares this transform (``GraphStage.onnx_transforms``) rather than the
    framework applying it globally, because ranking scores in FP16 is a per-model
    accuracy judgement: near-ties may reorder (BEVFusion already declares proposal
    ties in its ``verification_caveat``), and the gate is the evaluated metric.

    No-op when no FP32 cast feeds a TopK (fp32 exports, Q/DQ graphs).
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(onnx_path))
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}

    def cast_target(node) -> int | None:
        return next((a.i for a in node.attribute if a.name == "to"), None)

    bypassed = 0
    for node in graph.node:
        if node.op_type != "TopK":
            continue
        upstream = producers.get(node.input[0])
        if (
            upstream is None
            or upstream.op_type != "Cast"
            or cast_target(upstream) != TensorProto.FLOAT
        ):
            continue
        node.input[0] = upstream.input[0]
        bypassed += 1
        # The values output follows the input dtype now.
        for value_info in graph.value_info:
            if (
                value_info.name == node.output[0]
                and value_info.type.tensor_type.elem_type == TensorProto.FLOAT
            ):
                value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
    if bypassed:
        onnx.save(model, str(onnx_path))
        logger.info(
            "keep_topk_in_fp16: %d TopK input cast(s) bypassed in %s.", bypassed, onnx_path.name
        )
    return onnx_path
