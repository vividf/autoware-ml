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

The single source of *which* descriptor each layer type uses — nothing else in the
framework spells bit widths. The module-replacement engine (:mod:`.replace`) and the
architecture recipes (:mod:`..recipes.attach`) request descriptors here with the
precision the plan hands them.

Adding a precision = adding its row to each table below (plus the :class:`Precision`
enum member). FP8 rows are E4M3 (``num_bits=(4, 3)``): weights per-tensor because
modelopt's E4M3 ONNX export asserts a scalar amax (``TensorQuantizer._check_onnx_readiness``),
activations with max calibration (scales from tensor maxima, the modelopt ``FP8_DEFAULT_CFG``
convention).

Leaf module: imports only modelopt's config types and the config enum.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modelopt.torch.quantization import tensor_quant
from modelopt.torch.quantization.config import QuantizerAttributeConfig

from autoware_ml.quantization.config import Precision

#: Activation (input) descriptor arguments, shared by Conv2d / ConvTranspose2d / Linear
#: inputs AND the recipe quantizers (residual / pool) — sharing the same parameters keeps
#: their calibration consistent with the conv inputs. Per-tensor. The calibrator kind
#: (``histogram`` vs ``max``) is the config's choice for INT8 (:func:`input_desc`);
#: FP8 is always max.
_INPUT_BITS: Mapping[Precision, Any] = {
    Precision.INT8: 8,
    Precision.FP8: (4, 3),
}

#: Per-output-channel weight descriptor for Conv2d. INT8 keeps the modelopt preset the
#: calibrated production checkpoints were built with; FP8 is per-tensor (see module doc).
_CONV2D_WEIGHT: Mapping[Precision, QuantizerAttributeConfig] = {
    Precision.INT8: tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL,
    Precision.FP8: QuantizerAttributeConfig(num_bits=(4, 3)),
}

#: Per-tensor weight descriptor for ConvTranspose2d. TensorRT INT8 transposed conv is
#: fragile with per-channel weight scales (it can fail the engine build with
#: ``vol == 1`` / ``Could not find any implementation``), so weights are per-tensor.
_CONV_TRANSPOSE2D_WEIGHT: Mapping[Precision, QuantizerAttributeConfig] = {
    Precision.INT8: tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
    Precision.FP8: QuantizerAttributeConfig(num_bits=(4, 3)),
}

#: Weight descriptor for Linear: INT8 per-output-channel (per-row); FP8 per-tensor.
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
    """Activation (input) descriptor for ``precision``.

    Args:
        precision: Target precision.
        calibrator: ``"histogram"`` or ``"max"`` — the INT8 activation calibrator the config
            asked for (:attr:`CalibrationConfig.activation_calibrator`). FP8 ignores it and
            always calibrates with max.
    """
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
