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

"""The AutoCast FP16 path for plain graphs (no plugins, no Q/DQ).

modelopt's AutoCast converts with a numeric gate (per-node output comparison against
a tolerance), which the in-house island pass in :mod:`.precision` deliberately does
not have — but AutoCast types the graph through TensorRT's parser, so it rejects
plugin ops, and it refuses Q/DQ models outright; those graphs take the island pass
instead (see the routing table in ``deployment/export.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnx
import torch
from modelopt.onnx.autocast import convert_to_mixed_precision
from onnx import TensorProto, helper

from autoware_ml.deployment.onnx.dtypes import tensor_types

logger = logging.getLogger(__name__)


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

    Selecting in FP16 makes the ``values`` output FP16, and every consumer of it — an
    FP32 operand downstream, or the artifact's declared FLOAT output — was written
    against FP32. So the transform casts the *selected* values back to FP32 under their
    original name (k elements, not the heatmap): consumer contracts and the graph-output
    ABI are unchanged, and only the big cast is gone. Values nobody reads stay FP16.

    A stage declares this transform (``GraphStage.onnx_transforms``) rather than the
    framework applying it globally, because ranking scores in FP16 is a per-model
    accuracy judgement: near-ties may reorder (BEVFusion already declares proposal
    ties in its ``verification_caveat``), and the gate is the evaluated metric.

    No-op when no FP32 cast *of an FP16 tensor* feeds a TopK (fp32 exports, Q/DQ
    graphs, a Cast lifting integers); running it twice changes nothing more.
    """

    model = onnx.load(str(onnx_path))
    graph = model.graph
    types = tensor_types(model)
    producers = {output: node for node in graph.node for output in node.output}
    graph_outputs = {output.name for output in graph.output}

    def cast_target(node) -> int | None:
        return next((a.i for a in node.attribute if a.name == "to"), None)

    def consumed(name: str) -> bool:
        return name in graph_outputs or any(name in node.input for node in graph.node)

    bypassed_casts: list = []
    cast_backs: dict[int, onnx.NodeProto] = {}  # TopK position -> cast to splice after it
    for position, node in enumerate(graph.node):
        if node.op_type != "TopK":
            continue
        upstream = producers.get(node.input[0])
        if (
            upstream is None
            or upstream.op_type != "Cast"
            or cast_target(upstream) != TensorProto.FLOAT
        ):
            continue
        source = upstream.input[0]
        if types.get(source) != TensorProto.FLOAT16:
            logger.info(
                "keep_topk_in_fp16: %s keeps its FP32 input — the Cast it reads lifts %r "
                "(type %s), not an FP16 tensor.",
                node.name or node.op_type,
                source,
                types.get(source),
            )
            continue
        node.input[0] = source
        bypassed_casts.append(upstream)
        values = node.output[0]
        if values and consumed(values):
            internal = values + "__fp16"
            node.output[0] = internal
            cast_backs[position] = helper.make_node(
                "Cast", [internal], [values], to=TensorProto.FLOAT, name=values
            )
    if not bypassed_casts:
        return onnx_path

    dropped = {node.output[0] for node in bypassed_casts if not consumed(node.output[0])}
    rebuilt = []
    for position, node in enumerate(graph.node):
        if node.op_type == "Cast" and node.output[0] in dropped:
            continue  # the heatmap cast nobody reads any more
        rebuilt.append(node)
        if position in cast_backs:
            rebuilt.append(cast_backs[position])
    del graph.node[:]
    graph.node.extend(rebuilt)
    # value_info: `values` is still FLOAT (the cast-back produces it); the dropped
    # cast's tensor no longer exists, and nothing declares the FP16 selection.
    kept_info = [info for info in graph.value_info if info.name not in dropped]
    del graph.value_info[:]
    graph.value_info.extend(kept_info)

    onnx.save(model, str(onnx_path))
    logger.info(
        "keep_topk_in_fp16: %d TopK input cast(s) bypassed in %s (%d values output(s) "
        "cast back to FP32 for their consumers).",
        len(bypassed_casts),
        onnx_path.name,
        len(cast_backs),
    )
    return onnx_path
