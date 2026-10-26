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

"""Quantizer descriptors (modelopt ``QuantizerAttributeConfig``), keyed by :class:`Precision`.

FP8 is E4M3 (``num_bits=(4, 3)``) with max-calibrated activations, as in modelopt's
``FP8_DEFAULT_CFG``. FP8 weights are per-tensor because modelopt's E4M3 ONNX export asserts a
scalar amax (``TensorQuantizer._check_onnx_readiness``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.config import QuantizerAttributeConfig

from autoware_ml.quantization.config import Precision

#: Activation bit widths; activations are quantized per tensor for every layer type.
_INPUT_BITS: Mapping[Precision, Any] = {
    Precision.INT8: 8,
    Precision.FP8: (4, 3),
}

#: INT8 is per output channel, the modelopt preset the production checkpoints were
#: calibrated with.
_CONV2D_WEIGHT: Mapping[Precision, QuantizerAttributeConfig] = {
    Precision.INT8: tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL,
    Precision.FP8: QuantizerAttributeConfig(num_bits=(4, 3)),
}

#: Per-tensor: TensorRT INT8 transposed convolution can fail the engine build with
#: per-channel weight scales (``vol == 1`` / ``Could not find any implementation``).
_CONV_TRANSPOSE2D_WEIGHT: Mapping[Precision, QuantizerAttributeConfig] = {
    Precision.INT8: tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
    Precision.FP8: QuantizerAttributeConfig(num_bits=(4, 3)),
}

#: The weight is ``[C_out, k1, k2, k3, C_in]``, so ``axis=0`` gives the per-output-channel
#: scales the ImplicitGemm plugin's ``channel_scale`` input takes. No FP8: the plugin has an
#: INT8 path only.
_SPCONV_WEIGHT: Mapping[Precision, QuantizerAttributeConfig] = {
    Precision.INT8: QuantizerAttributeConfig(num_bits=8, axis=0),
}

_LINEAR_WEIGHT: Mapping[Precision, QuantizerAttributeConfig] = {
    Precision.INT8: QuantizerAttributeConfig(num_bits=8, axis=0),
    Precision.FP8: QuantizerAttributeConfig(num_bits=(4, 3)),
}


def _lookup(table: Mapping[Precision, Any], precision: Precision, what: str) -> Any:
    try:
        return table[precision]
    except KeyError:
        raise NotImplementedError(
            f"No {what} descriptor is defined for precision {precision.value!r}; "
            f"supported: {[p.value for p in table]}."
        ) from None


def input_desc(precision: Precision, calibrator: str = "histogram") -> QuantizerAttributeConfig:
    """Activation descriptor for ``precision``; FP8 ignores ``calibrator`` and always uses max."""
    bits = _lookup(_INPUT_BITS, precision, "activation")
    if precision is Precision.FP8:
        calibrator = "max"
    return QuantizerAttributeConfig(num_bits=bits, calibrator=calibrator)


def conv2d_weight_desc(precision: Precision) -> QuantizerAttributeConfig:
    """Conv2d weight descriptor for ``precision``."""
    return _lookup(_CONV2D_WEIGHT, precision, "Conv2d weight")


def conv_transpose2d_weight_desc(precision: Precision) -> QuantizerAttributeConfig:
    """ConvTranspose2d weight descriptor for ``precision``."""
    return _lookup(_CONV_TRANSPOSE2D_WEIGHT, precision, "ConvTranspose2d weight")


def linear_weight_desc(precision: Precision) -> QuantizerAttributeConfig:
    """Linear weight descriptor for ``precision``."""
    return _lookup(_LINEAR_WEIGHT, precision, "Linear weight")


def spconv_weight_desc(precision: Precision) -> QuantizerAttributeConfig:
    """Sparse-convolution weight descriptor for ``precision`` (INT8 only)."""
    return _lookup(_SPCONV_WEIGHT, precision, "sparse convolution weight")
