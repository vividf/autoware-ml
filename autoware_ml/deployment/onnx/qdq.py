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


"""Q/DQ scales and zero points as initializers instead of constant helper chains.

modelopt's symbolics emit a quantizer's scale as a ``Constant`` and its zero point as
``Constant -> Cast``, and with ``do_constant_folding: false`` (the tracer folds some of the
exported ops wrongly) the chains reach the artifact. Netron inlines a ``Constant`` only
into a single consumer input, so neither value shows on the Q/DQ nodes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from autoware_ml.deployment.onnx.inspect import QDQ_OPS

logger = logging.getLogger(__name__)

#: The only ops the pass evaluates or deletes; the tracer occasionally adds an ``Identity``.
_FOLDABLE_OPS = frozenset({"Constant", "Cast", "Identity"})

#: Parameter chains are two or three ops deep; longer ones are left alone.
_MAX_CHAIN_DEPTH = 8

#: Scale and zero point; modelopt's TRT-domain FP8 Q/DQ have the scale only.
_PARAM_SLOTS = slice(1, 3)


def _constant_value(node: onnx.NodeProto) -> np.ndarray | None:
    """Value of a ``Constant`` node, or ``None`` for a form not read here."""
    for attribute in node.attribute:
        if attribute.name == "value":
            return numpy_helper.to_array(attribute.t)
        if attribute.name == "value_float":
            return np.array(attribute.f, dtype=np.float32)
        if attribute.name == "value_floats":
            return np.array(list(attribute.floats), dtype=np.float32)
        if attribute.name == "value_int":
            return np.array(attribute.i, dtype=np.int64)
        if attribute.name == "value_ints":
            return np.array(list(attribute.ints), dtype=np.int64)
    return None


def _evaluate(
    name: str,
    producer_of: dict[str, onnx.NodeProto],
    depth: int = 0,
) -> np.ndarray | None:
    """Evaluate the helper chain producing ``name``, or return ``None`` to leave it alone."""
    node = producer_of.get(name)
    if node is None or depth > _MAX_CHAIN_DEPTH or node.op_type not in _FOLDABLE_OPS:
        return None
    if node.op_type == "Constant":
        return _constant_value(node)
    if not node.input:
        return None
    if node.op_type == "Identity":
        return _evaluate(node.input[0], producer_of, depth + 1)

    value = _evaluate(node.input[0], producer_of, depth + 1)
    target = next((a.i for a in node.attribute if a.name == "to"), 0)
    if value is None or not target:
        return None
    try:
        return value.astype(helper.tensor_dtype_to_np_dtype(target))
    except (KeyError, TypeError, ValueError):
        return None  # FP8 / bfloat16 have no numpy dtype


def fold_qdq_params(onnx_path: Path) -> None:
    """Rewrite the constant scales and zero points of Q/DQ nodes as initializers, in place.

    Idempotent. The values stay bit-identical (a ``Cast`` in a chain is applied), and a
    helper node that another node still reads is kept.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    if not any(node.op_type in QDQ_OPS for node in graph.node):
        return

    producer_of = {out: node for node in graph.node for out in node.output}
    reserved = {init.name for init in graph.initializer}
    reserved |= {value.name for value in graph.input}
    # A graph output must stay a node output.
    protected = {value.name for value in graph.output}

    folded: dict[str, np.ndarray] = {}
    for node in graph.node:
        if node.op_type not in QDQ_OPS:
            continue
        for name in node.input[_PARAM_SLOTS]:
            if not name or name in reserved or name in protected or name in folded:
                continue
            value = _evaluate(name, producer_of)
            if value is not None:
                folded[name] = value
    if not folded:
        return

    graph.initializer.extend(numpy_helper.from_array(value, name) for name, value in folded.items())
    # Drop the producers of the folded tensors (a name cannot be defined twice), then every
    # helper left unread.
    kept = [node for node in graph.node if not (node.output and set(node.output) <= folded.keys())]
    while True:
        read = {name for node in kept for name in node.input} | protected
        alive = [
            node
            for node in kept
            if node.op_type not in _FOLDABLE_OPS or any(out in read for out in node.output)
        ]
        if len(alive) == len(kept):
            break
        kept = alive

    removed = len(graph.node) - len(kept)
    del graph.node[:]
    graph.node.extend(kept)
    onnx.save(model, str(onnx_path))
    logger.info(
        "Folded %d Q/DQ scale/zero-point tensor(s) of %s into initializers, dropping %d "
        "helper node(s).",
        len(folded),
        onnx_path.name,
        removed,
    )
