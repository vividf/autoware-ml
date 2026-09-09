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

"""Quantized sparse convolution — the ``spconv`` module kind, on modelopt's registry.

A sparse convolution is a GEMM like any other, but it carries its activations inside a
``SparseConvTensor`` rather than a plain tensor, so modelopt's stock ``QuantConv2d`` cannot
be reused: the input quantizer has to reach ``input.features``. Everything else is
modelopt's — ``QuantLinearConvBase`` supplies the ``weight_quantizer`` (through the dynamic
``weight`` attribute) and the disabled ``output_quantizer``, so a converted sparse
convolution carries exactly the ``input_quantizer`` / ``weight_quantizer`` state-dict keys
every other quantized module does, and the framework's calibration walks it unchanged.

The deployed form is *not* Q/DQ: the sparse tower runs as ``autoware::ImplicitGemm`` plugin
nodes, and the plugin does its own quantization from per-layer scales
(:mod:`autoware_ml.ops.spconv.onnx_int8` writes them into the exported graph, derived from
the ``amax`` values calibrated here). So the fake quantization in this module exists to
(a) calibrate those scales and (b) let the PyTorch backend show the INT8 accuracy the
engine will have — not to be exported.

Registration happens at import time and only when ``spconv`` is installed; the module is a
no-op otherwise, which is what lets a CPU/import-only environment load the framework.
"""

from __future__ import annotations

import logging
from typing import Any

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_module import QuantLinearConvBase

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

logger = logging.getLogger(__name__)

#: Whether the sparse-convolution quantized module is registered in this process.
SPCONV_QUANT_REGISTERED = False

if IS_SPCONV_AVAILABLE:  # pragma: no branch - environment-dependent
    from spconv.pytorch.conv import SparseConvolution

    @QuantModuleRegistry.register({SparseConvolution: "spconv.SparseConvolution"})
    class _QuantSparseConvolution(QuantLinearConvBase):
        """Sparse convolution with a quantized input and weight.

        Registered for ``spconv.pytorch.conv.SparseConvolution``, so every subclass
        (``SubMConv3d``, ``SparseConv3d``, ...) converts through the same rule.
        """

        def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
            """Quantize ``input.features`` and the weight, then run the sparse convolution.

            Args:
                input: The layer's ``SparseConvTensor``.
                *args: Forwarded to the original convolution (``add_input``).
                **kwargs: Forwarded to the original convolution.

            Returns:
                The convolution's output ``SparseConvTensor``.
            """
            quantized = input.replace_feature(self.input_quantizer(input.features))
            # `quantize_weight` is what makes the dynamic `weight` attribute return the
            # fake-quantized tensor; the base class's forward cannot be reused because it
            # would quantize the SparseConvTensor itself.
            with self.quantize_weight():
                return SparseConvolution.forward(self, quantized, *args, **kwargs)

    SPCONV_QUANT_REGISTERED = True
