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


"""Precision conversion of an exported graph, chosen from the graph itself.

Two fp16 paths:

- A graph carrying quantize/dequantize nodes or custom-domain plugin ops takes the
  island-aware whole-graph cast (:func:`autoware_ml.deployment.onnx.precision.cast_graph_to_fp16`):
  the generic converter has no rule keeping a calibrated Q/DQ pair exact and cannot type a
  plugin op. Conv-family Q/DQ go fp16 with the graph; Q/DQ feeding a Gemm/MatMul stay
  fp32-typed islands; the graph I/O stays fp32.
- Every other graph takes the onnxconverter-common path of
  :func:`autoware_ml.utils.onnx_precision.convert_onnx_precision`, unchanged.

Deployment-only: the deployment package reads the graph inspection and the cast, and
``autoware_ml.utils`` stays free of deployment imports.
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
    """Convert an exported ONNX to ``precision`` in place and return its path.

    Args:
        onnx_path: The exported graph, rewritten in place.
        precision: The explicit precision the shipped graph carries.

    Returns:
        The path of the converted graph.

    Raises:
        ValueError: If no conversion is implemented for ``precision``.
    """
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
