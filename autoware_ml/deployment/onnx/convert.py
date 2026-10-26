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


"""Precision conversion of an exported graph, with the fp16 path chosen from the graph.

Graphs with Q/DQ nodes or plugin ops take
:func:`~autoware_ml.deployment.onnx.precision.cast_graph_to_fp16`: onnxconverter-common
cannot keep a calibrated Q/DQ pair exact, type a plugin op or keep an INT8 plugin's scale
inputs fp32. Other graphs take :func:`autoware_ml.utils.onnx_precision.convert_onnx_precision`.
The dispatch lives here to keep ``autoware_ml.utils`` free of deployment imports.
"""

from __future__ import annotations

import logging
from pathlib import Path

from autoware_ml.deployment.onnx.inspect import onnx_custom_op_domains, onnx_has_qdq
from autoware_ml.deployment.onnx.precision import cast_graph_to_fp16
from autoware_ml.utils.onnx_precision import OnnxPrecision
from autoware_ml.utils.onnx_precision import convert_onnx_precision as convert_plain_onnx_precision

logger = logging.getLogger(__name__)


def convert_onnx_precision(onnx_path: str | Path, precision: OnnxPrecision) -> Path:
    """Convert an exported ONNX to ``precision`` in place and return its path."""
    if precision != OnnxPrecision.FP16:
        raise ValueError(f"No conversion implemented for precision '{precision}'.")
    onnx_path = Path(onnx_path)
    domains = onnx_custom_op_domains(onnx_path)
    has_qdq = onnx_has_qdq(onnx_path)
    if domains or has_qdq:
        logger.info(
            "Converting %s to fp16 with the island-aware whole-graph cast (%s).",
            onnx_path.name,
            ", ".join(
                filter(
                    None,
                    [
                        f"plugin domains: {', '.join(domains)}" if domains else "",
                        "Q/DQ nodes" if has_qdq else "",
                    ],
                )
            ),
        )
        cast_graph_to_fp16(onnx_path)
        return onnx_path
    return convert_plain_onnx_precision(onnx_path, precision)
