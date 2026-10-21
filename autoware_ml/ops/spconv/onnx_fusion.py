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

"""Fold the bias ``Add`` and ``Relu`` after each ``autoware::ImplicitGemm`` into the node.

TensorRT cannot fuse standard ops into a plugin node, but the plugin applies a bias
(optional sixth input) and an activation (``act_type``) itself. Only the exported graph
changes; the PyTorch modules keep the explicit bias add and ReLU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
from onnx import helper

from autoware_ml.ops.spconv.contract import (
    ACT_RELU,
    ACT_TYPE_ATTR,
    AUTOWARE_DOMAIN,
    IMPLICIT_GEMM_INPUTS_WITHOUT_BIAS,
    IMPLICIT_GEMM_OP,
)

logger = logging.getLogger(__name__)

_INPUTS_WITHOUT_BIAS = IMPLICIT_GEMM_INPUTS_WITHOUT_BIAS
_STANDARD_DOMAINS = ("", "ai.onnx")


def _is_implicit_gemm(node: onnx.NodeProto) -> bool:
    return node.op_type == IMPLICIT_GEMM_OP and node.domain == AUTOWARE_DOMAIN


def _is_standard_op(node: onnx.NodeProto, op_type: str) -> bool:
    return node.op_type == op_type and node.domain in _STANDARD_DOMAINS


def replace_attribute(node: onnx.NodeProto, name: str, value: int | float) -> None:
    """Set ``name`` on ``node``, replacing an existing attribute of that name."""
    kept = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def fuse_implicit_gemm_bias_activation(model: onnx.ModelProto) -> tuple[int, int]:
    """Fold trailing bias adds and ReLUs into ``autoware::ImplicitGemm`` nodes, in place.

    Each ``ImplicitGemm -> Add(bias initializer) [-> Relu]`` chain whose intermediate values
    feed only the next node becomes one ``ImplicitGemm`` with the bias as its sixth input and,
    with the ReLU, ``act_type = ACT_RELU``. Only this order folds: the plugin computes
    ``relu(gemm(x) + b)``.

    Returns:
        Number of folded biases and number of folded activations.
    """
    graph = model.graph
    initializers = {initializer.name for initializer in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)
    graph_outputs = {output.name for output in graph.output}

    removed: list[onnx.NodeProto] = []
    fused_biases = 0
    fused_activations = 0

    for gemm in graph.node:
        if not _is_implicit_gemm(gemm) or len(gemm.input) != _INPUTS_WITHOUT_BIAS:
            continue

        gemm_output = gemm.output[0]
        following = consumers.get(gemm_output, [])
        if len(following) != 1 or gemm_output in graph_outputs:
            continue
        add = following[0]
        if not _is_standard_op(add, "Add"):
            continue
        bias = next((tensor for tensor in add.input if tensor in initializers), None)
        if bias is None or len(add.input) != 2:
            continue

        gemm.input.append(bias)
        fused_biases += 1
        last, last_output = add, add.output[0]

        activation = consumers.get(add.output[0], [])
        if (
            len(activation) == 1
            and _is_standard_op(activation[0], "Relu")
            and add.output[0] not in graph_outputs
        ):
            relu = activation[0]
            replace_attribute(gemm, ACT_TYPE_ATTR, ACT_RELU)
            fused_activations += 1
            removed.append(relu)
            last, last_output = relu, relu.output[0]

        # Take over the chain's output name so downstream readers stay connected.
        removed.append(add)
        gemm.output[0] = last_output
        del last

    if removed:
        # By identity: `in` compares protobuf messages by value.
        removed_ids = {id(node) for node in removed}
        kept = [node for node in graph.node if id(node) not in removed_ids]
        del graph.node[:]
        graph.node.extend(kept)

    return fused_biases, fused_activations


def fuse_sparse_graph(onnx_path: str | Path) -> Path:
    """Apply :func:`fuse_implicit_gemm_bias_activation` to an ONNX file in place.

    A no-op for graphs without ``autoware::ImplicitGemm`` nodes. Returns the path.
    """
    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    fused_biases, fused_activations = fuse_implicit_gemm_bias_activation(model)
    if fused_biases or fused_activations:
        onnx.save(model, str(onnx_path))
        logger.info(
            "Sparse fusion in %s: folded %d bias add(s) and %d ReLU(s) into ImplicitGemm.",
            onnx_path.name,
            fused_biases,
            fused_activations,
        )
    # An unfolded node is still correct but costs extra kernels per block, and nothing else
    # would show that the pattern stopped matching the export.
    unfused = sum(
        1
        for node in model.graph.node
        if _is_implicit_gemm(node) and len(node.input) == _INPUTS_WITHOUT_BIAS
    )
    if unfused:
        logger.warning(
            "Sparse fusion in %s: %d ImplicitGemm node(s) kept their bias Add outside the "
            "plugin (%d folded). The graph is correct but slower than the fused form — the "
            "exported bias/activation shape no longer matches what this pass expects.",
            onnx_path.name,
            unfused,
            fused_biases,
        )
    return onnx_path
