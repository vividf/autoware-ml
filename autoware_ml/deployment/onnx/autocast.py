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

import torch

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
