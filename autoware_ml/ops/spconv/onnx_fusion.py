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

"""Post-export fusion for exported sparse-convolution graphs.

TensorRT cannot fuse a standard ONNX operator into a plugin node, so a traced sparse
encoder leaves the per-channel bias and the block's ReLU as separate ``Add`` and ``Relu``
nodes after every ``autoware::ImplicitGemm``. The runtime plugin computes both itself —
an optional sixth input carries the bias and ``act_type`` selects the activation — so
this module rewrites the exported graph into that form.

The PyTorch side is deliberately untouched: it keeps the explicit bias add and the
block's ReLU, and the exported artifact is the only thing that changes.

Order matters. The activation can only be folded into the plugin once the bias is folded
too, because the plugin applies the activation *after* its bias add, while the traced
graph adds the bias after the GEMM: ``relu(gemm(x) + b) != relu(gemm(x)) + b``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
from onnx import helper

logger = logging.getLogger(__name__)

IMPLICIT_GEMM_OP = "ImplicitGemm"
AUTOWARE_DOMAIN = "autoware"
#: Mirrors the plugin's activation enum (cumm ``tv::gemm::Activation``).
ACT_NONE = 0
ACT_RELU = 1
#: The plugin's optional bias is its sixth input.
_INPUTS_WITHOUT_BIAS = 5
_STANDARD_DOMAINS = ("", "ai.onnx")


def _is_implicit_gemm(node: onnx.NodeProto) -> bool:
    return node.op_type == IMPLICIT_GEMM_OP and node.domain == AUTOWARE_DOMAIN


def _is_standard_op(node: onnx.NodeProto, op_type: str) -> bool:
    return node.op_type == op_type and node.domain in _STANDARD_DOMAINS


def _set_attribute(node: onnx.NodeProto, name: str, value: int) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def fuse_implicit_gemm_bias_activation(model: onnx.ModelProto) -> tuple[int, int]:
    """Fold trailing bias adds and ReLUs into ``autoware::ImplicitGemm`` nodes.

    Rewrites each ``ImplicitGemm -> Add(bias initializer) [-> Relu]`` chain, where every
    intermediate value feeds only the next node in the chain, into a single
    ``ImplicitGemm`` carrying the bias as its sixth input and, when a ReLU was folded,
    ``act_type = ACT_RELU``.

    Args:
        model: Exported graph, modified in place.

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

        # The bias becomes the plugin's optional sixth input.
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
            _set_attribute(gemm, "act_type", ACT_RELU)
            fused_activations += 1
            removed.append(relu)
            last, last_output = relu, relu.output[0]

        # The plugin now produces what the folded chain produced, so it takes over the
        # chain's output name and every downstream reader follows unchanged.
        removed.append(add)
        gemm.output[0] = last_output
        del last

    if removed:
        kept = [node for node in graph.node if node not in removed]
        del graph.node[:]
        graph.node.extend(kept)

    return fused_biases, fused_activations


def fuse_sparse_graph(onnx_path: str | Path) -> Path:
    """Apply :func:`fuse_implicit_gemm_bias_activation` to an exported ONNX file.

    A no-op for graphs without ``autoware::ImplicitGemm`` nodes, so it is safe to run
    over any stage.

    Args:
        onnx_path: Exported graph, rewritten in place.

    Returns:
        The same path.
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
    return onnx_path
