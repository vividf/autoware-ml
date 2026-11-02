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

The export does not trace these quantizers: the sparse INT8 pass writes the ``amax``
calibrated here into the ``autoware::ImplicitGemm`` node (a QuantizeLinear in front, INT8
filters, per-channel scales). The fake quantization calibrates those scales and shows the
engine's INT8 accuracy on the PyTorch backend.
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

    from autoware_ml.ops.submanifold_conv_spconv import SpconvSubMConv3d

    @QuantModuleRegistry.register(
        {
            SparseConvolution: "spconv.SparseConvolution",
            SpconvSubMConv3d: "autoware_ml.SpconvSubMConv3d",
        }
    )
    class _QuantSparseConvolution(QuantLinearConvBase):
        """Sparse convolution with a quantized input and weight.

        modelopt's registry converts a subclass of a registered class only when it inherits
        that class's forward: ``SubMConv3d`` and ``SparseConv3d`` come with
        ``SparseConvolution``, but ``SpconvSubMConv3d`` overrides forward (autocast off,
        features in the weight dtype) and is registered by name.
        """

        def forward(self, input: Any, *args: Any, **kwargs: Any) -> Any:
            """Quantize ``input.features`` and the weight, then run the module's own forward."""
            quantized = input.replace_feature(self.input_quantizer(input.features))
            # Inside quantize_weight(), the dynamic `weight` attribute returns the
            # fake-quantized tensor.
            with self.quantize_weight():
                return self.original_cls.forward(self, quantized, *args, **kwargs)

    SPCONV_QUANT_REGISTERED = True
