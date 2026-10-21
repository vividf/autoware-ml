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


"""Readable quantization parameters on every exported Q/DQ node.

modelopt's ONNX symbolics spell a quantizer's scale and zero point as little constant
sub-graphs rather than as tensors: ``Constant`` for the scale and ``Constant -> Cast``
for the zero point (``modelopt/torch/quantization/export_onnx.py`` builds the zero point
in the quantizer's compute dtype, then casts it to int8/uint8). The framework exports
with ``do_constant_folding: false`` -- the traced graphs carry ops whose folding the
tracer gets wrong -- so those helper nodes survive into the artifact, and the values that
*are* the quantization become unreadable where it matters:

- Netron inlines a ``Constant`` into its consumer only when the constant's output feeds
  exactly one node input (``onnx.js``, ``Graph.push``). A Q/DQ pair shares one scale
  constant, so the scale renders as a separate ``Constant`` box instead of a value on the
  node, and the zero point -- one ``Cast`` hop away -- cannot be inlined at all. The Q/DQ
  node panel shows ``y_scale``/``y_zero_point`` as plain edges, with no value or dtype.
- Graph passes that need a Q/DQ's parameter types have to walk the chain themselves
  (:func:`~autoware_ml.deployment.onnx.dtypes.tensor_types` cannot seed a quantize
  output's type from a zero point it can only see through a ``Cast``).

:func:`fold_qdq_params` evaluates those chains once, after the export, and writes the
results back as graph initializers under the same tensor names -- the spelling a
hand-written Q/DQ graph uses, and the one AWML's exports happen to have for the scale.
The graph keeps identical numerics (the ``Cast`` is applied, not dropped) and loses the
helper nodes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from autoware_ml.deployment.onnx.inspect import QDQ_OPS

logger = logging.getLogger(__name__)

#: Helper ops a quantization parameter may travel through, and the only ones this pass
#: ever evaluates or deletes. ``Constant`` is the leaf, ``Cast`` retypes the zero point,
#: ``Identity`` is the tracer's occasional pass-through.
_FOLDABLE_OPS = frozenset({"Constant", "Cast", "Identity"})

#: Chain-length guard: a quantization parameter is a constant two or three hops deep.
#: Anything longer is not the pattern this pass recognizes, so it is left alone.
_MAX_CHAIN_DEPTH = 8

#: Scale and zero point, by input position -- identical for the standard ops and for
#: modelopt's TRT-domain FP8 spellings (which carry the scale only).
_PARAM_SLOTS = slice(1, 3)


def _constant_value(node: onnx.NodeProto) -> np.ndarray | None:
    """The value a ``Constant`` node holds, or ``None`` for a spelling we do not read."""
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
    return None  # sparse_value, or a Constant of a shape/dtype-only form


def _evaluate(
    name: str,
    producer_of: dict[str, onnx.NodeProto],
    depth: int = 0,
) -> np.ndarray | None:
    """Evaluate a constant helper chain, or ``None`` when the tensor is not one.

    ``None`` is the "leave it alone" answer for everything this pass must not touch: a
    genuinely computed scale, a chain through an op outside :data:`_FOLDABLE_OPS`, or a
    cast whose target dtype numpy cannot express.
    """
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
        return None  # an FP8/BFLOAT16 cast without a numpy dtype behind it


def fold_qdq_params(onnx_path: Path) -> None:
    """Rewrite every Q/DQ node's constant scale and zero point as graph initializers.

    In-place and idempotent: a parameter that is already an initializer, a graph input or
    a graph output is left as it is, and so is one this pass cannot evaluate exactly (see
    :func:`_evaluate`). Helper nodes are deleted only once nothing reads them, so a
    ``Constant`` shared with a non-Q/DQ consumer survives.

    The numbers do not change -- a ``Cast`` in the chain is applied rather than dropped --
    so the pass is safe to run on a calibrated graph: Q/DQ values stay bit-identical to
    what the checkpoint calibrated, only their spelling changes.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    graph = model.graph
    if not any(node.op_type in QDQ_OPS for node in graph.node):
        return

    producer_of = {out: node for node in graph.node for out in node.output}
    # Names the pass must not claim: an initializer or graph input already *is* a tensor,
    # and a graph output has to stay a produced tensor.
    reserved = {init.name for init in graph.initializer}
    reserved |= {value.name for value in graph.input}
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
    # The folded tensors are initializers now, so their producers must go (two
    # definitions of one tensor name is an invalid graph), and then every helper node
    # left without a consumer goes with them.
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
