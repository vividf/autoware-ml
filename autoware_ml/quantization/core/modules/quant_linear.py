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

"""Quantized Linear module (PillarFeatureNet PFN layers, ConvNeXt pointwise).

Descriptor choice lives in :mod:`..descriptors`; the builder passes descriptors to
:meth:`init_quantizer` explicitly — see :mod:`.quant_conv` for the rationale.
"""

import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.quantization.core import backend as quant_backend


class QuantLinear(nn.Linear):
    """Quantized Linear (fake-quant on input activations and weights).

    Args:
        Same as nn.Linear.

    Attributes:
        _input_quantizer: TensorQuantizer for input activations (after ``init_quantizer``).
        _weight_quantizer: TensorQuantizer for weights (after ``init_quantizer``).
    """

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        self._input_quantizer = None
        self._weight_quantizer = None

    def init_quantizer(self, quant_desc_input, quant_desc_weight):
        """Attach input and weight quantizers built from the given descriptors."""
        TensorQuantizer = quant_backend.get_tensor_quantizer_cls()
        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._weight_quantizer = TensorQuantizer(quant_desc_weight)

    def forward(self, x):
        """Forward with quantized input and weights."""
        if self._input_quantizer is not None and self._weight_quantizer is not None:
            quant_input = self._input_quantizer(x)
            quant_weight = self._weight_quantizer(self.weight)
        else:
            quant_input = x
            quant_weight = self.weight

        return F.linear(quant_input, quant_weight, self.bias)
