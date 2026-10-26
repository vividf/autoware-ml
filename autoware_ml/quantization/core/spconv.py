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

"""Quantized sparse convolution: the ``spconv`` module kind, on modelopt's registry.

modelopt's ``QuantConv2d`` cannot be reused: the activations sit in a ``SparseConvTensor``, and
the input quantizer has to reach ``input.features``. ``QuantLinearConvBase`` provides the rest,
so a converted sparse convolution has the same quantizer ``state_dict`` keys as any other
quantized module.

The deployed sparse tower is not Q/DQ: the ``autoware::ImplicitGemm`` plugin quantizes from
per-layer scales derived from the ``amax`` calibrated here. The fake quantization calibrates
those scales and shows the engine's INT8 accuracy on the PyTorch backend; it is not exported.
"""

from __future__ import annotations

import logging
from typing import Any

from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_module import QuantLinearConvBase

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

logger = logging.getLogger(__name__)

SPCONV_QUANT_REGISTERED = False

if IS_SPCONV_AVAILABLE:  # pragma: no branch - environment-dependent
    from spconv.pytorch.conv import SparseConvolution

    @QuantModuleRegistry.register({SparseConvolution: "spconv.SparseConvolution"})
    class _QuantSparseConvolution(QuantLinearConvBase):
        """Sparse convolution with a quantized input and weight.

        Registered for ``SparseConvolution``; modelopt's registry also converts the subclasses
        that inherit its forward (``SubMConv3d``, ``SparseConv3d``, ...).
        """

        def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
            """Quantize ``input.features`` and the weight, then run the sparse convolution."""
            quantized = input.replace_feature(self.input_quantizer(input.features))
            # Inside quantize_weight(), the dynamic `weight` attribute returns the
            # fake-quantized tensor.
            with self.quantize_weight():
                return SparseConvolution.forward(self, quantized, *args, **kwargs)

    SPCONV_QUANT_REGISTERED = True
