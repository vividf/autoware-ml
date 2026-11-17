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


"""Inspect exported ONNX graphs."""

from __future__ import annotations

import onnx

#: Standard-domain and TensorRT-domain quantize / dequantize operators.
QUANTIZE_OPS = frozenset({"QuantizeLinear", "TRT_FP8QuantizeLinear"})
DEQUANTIZE_OPS = frozenset({"DequantizeLinear", "TRT_FP8DequantizeLinear"})
QDQ_OPS = QUANTIZE_OPS | DEQUANTIZE_OPS


def has_qdq(model: onnx.ModelProto) -> bool:
    """Whether the graph contains quantize / dequantize nodes (INT8 or FP8)."""
    return any(node.op_type in QDQ_OPS for node in model.graph.node)


def custom_op_domains(model: onnx.ModelProto) -> tuple[str, ...]:
    """Sorted domains of the graph's non-standard nodes, such as runtime plugins."""
    domains = {
        node.domain
        for node in model.graph.node
        if node.domain and not node.domain.startswith("ai.onnx")
    }
    return tuple(sorted(domains))
