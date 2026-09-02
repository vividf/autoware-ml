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

"""Quantized Conv2d and ConvTranspose2d modules.

The modules carry no descriptor defaults of their own: the descriptor choice per
precision lives in :mod:`..descriptors`, and whoever builds a quantized module
(the replace engine, a recipe) passes the descriptors to :meth:`init_quantizer`
explicitly. modelopt's ``TensorQuantizer`` traces to Q/DQ ONNX ops natively, so
the forward applies fake-quant unconditionally whenever quantizers are attached.
"""

import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.quantization.core import modelopt as quant_backend


class QuantConv2d(nn.Conv2d):
    """Quantized Conv2d (fake-quant on input activations and weights).

    Args:
        Same as nn.Conv2d.

    Attributes:
        _input_quantizer: TensorQuantizer for input activations (after ``init_quantizer``).
        _weight_quantizer: TensorQuantizer for weights (after ``init_quantizer``).
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
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

        return self._conv_forward(quant_input, quant_weight, self.bias)


class QuantConvTranspose2d(nn.ConvTranspose2d):
    """Quantized ConvTranspose2d (FPN upsample layers).

    Args:
        Same as nn.ConvTranspose2d.

    Attributes:
        _input_quantizer: TensorQuantizer for input activations (after ``init_quantizer``).
        _weight_quantizer: TensorQuantizer for weights (after ``init_quantizer``).
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self._input_quantizer = None
        self._weight_quantizer = None

    def init_quantizer(self, quant_desc_input, quant_desc_weight):
        """Attach input and weight quantizers built from the given descriptors."""
        TensorQuantizer = quant_backend.get_tensor_quantizer_cls()
        self._input_quantizer = TensorQuantizer(quant_desc_input)
        self._weight_quantizer = TensorQuantizer(quant_desc_weight)

    def forward(self, x, output_size=None):
        """Forward with quantized input and weights."""
        if self._input_quantizer is not None and self._weight_quantizer is not None:
            quant_input = self._input_quantizer(x)
            quant_weight = self._weight_quantizer(self.weight)
        else:
            quant_input = x
            quant_weight = self.weight

        # Compute output padding
        if output_size is None:
            output_padding = self.output_padding
        else:
            output_padding = self._output_padding(
                quant_input,
                output_size,
                self.stride,
                self.padding,
                self.kernel_size,
                num_spatial_dims=2,
                dilation=self.dilation,
            )

        return F.conv_transpose2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
