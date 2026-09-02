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

"""Quantization descriptor choices, keyed by :class:`~autoware_ml.quantization.config.Precision`.

The single source of *which* quantizer descriptor each layer type uses. The
module-replacement engine (:mod:`.replace`) and the architecture recipes
(:mod:`..recipes.attach`) request descriptors here with the precision the plan
hands them — nothing else in the framework spells bit widths.

Adding a precision = adding its row to each table below (plus the :class:`Precision`
enum member and, if modelopt needs new descriptor fields, a translation in
:func:`autoware_ml.quantization.core.modelopt.make_quant_desc`). FP8 rows are E4M3
(``num_bits=(4, 3)``) with max calibration — the modelopt convention for FP8, whose
scales come from tensor maxima rather than histograms.

Descriptors are built through :mod:`.backend`, which translates the framework's
historical vocabulary to modelopt's ``QuantizerAttributeConfig``.

Leaf module: it imports only :mod:`.backend` and the config enum, so every core
submodule can import it without a cycle.
"""

from __future__ import annotations

from typing import Any, Mapping

from autoware_ml.quantization.config import Precision

from . import modelopt as _backend

#: Per-tensor histogram activation descriptor, shared by Conv2d / ConvTranspose2d /
#: Linear inputs AND the recipe quantizers (residual / eSE / pool) — sharing the same
#: parameters keeps their calibration consistent with the conv inputs.
_INPUT_DESC_ARGS: Mapping[Precision, dict[str, Any]] = {
    Precision.INT8: dict(num_bits=8, calib_method="histogram"),
    Precision.FP8: dict(num_bits=(4, 3), calib_method="max"),
}

#: Per-output-channel weight descriptor for Conv2d. INT8 keeps the modelopt preset the
#: calibrated production checkpoints were built with. FP8 weights are per-tensor:
#: modelopt's E4M3 ONNX export requires a scalar amax (``TensorQuantizer.
#: _check_onnx_readiness`` asserts on per-channel), matching its ``FP8_DEFAULT_CFG``.
_CONV2D_WEIGHT_PRESET: Mapping[Precision, str] = {
    Precision.INT8: "QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL",
}
_CONV2D_WEIGHT_ARGS: Mapping[Precision, dict[str, Any]] = {
    Precision.FP8: dict(num_bits=(4, 3)),
}

#: Per-tensor weight descriptor for ConvTranspose2d. TensorRT INT8 transposed conv is
#: fragile with per-channel weight scales (it can fail the engine build with
#: ``vol == 1`` / ``Could not find any implementation``), so weights are per-tensor.
_CONV_TRANSPOSE2D_WEIGHT_PRESET: Mapping[Precision, str] = {
    Precision.INT8: "QUANT_DESC_8BIT_PER_TENSOR",
}
_CONV_TRANSPOSE2D_WEIGHT_ARGS: Mapping[Precision, dict[str, Any]] = {
    Precision.FP8: dict(num_bits=(4, 3)),
}

#: Weight descriptor for Linear: INT8 per-output-channel (per-row); FP8 per-tensor
#: (modelopt's E4M3 ONNX export requires scalar amax — see the Conv2d note).
_LINEAR_WEIGHT_ARGS: Mapping[Precision, dict[str, Any]] = {
    Precision.INT8: dict(num_bits=8, axis=(0,)),
    Precision.FP8: dict(num_bits=(4, 3)),
}


def _lookup(table: Mapping[Precision, Any], precision: Precision, what: str) -> Any:
    try:
        return table[precision]
    except KeyError:
        raise NotImplementedError(
            f"No {what} descriptor is defined for precision {precision.value!r}; "
            f"supported: {[p.value for p in table]}."
        ) from None


def input_desc(precision: Precision) -> Any:
    """Activation (input) descriptor for ``precision`` — see :data:`_INPUT_DESC_ARGS`."""
    return _backend.make_quant_desc(**_lookup(_INPUT_DESC_ARGS, precision, "activation"))


def conv2d_weight_desc(precision: Precision) -> Any:
    """Conv2d weight descriptor for ``precision``."""
    if precision in _CONV2D_WEIGHT_PRESET:
        return _backend.get_preset_desc(_CONV2D_WEIGHT_PRESET[precision])
    return _backend.make_quant_desc(**_lookup(_CONV2D_WEIGHT_ARGS, precision, "Conv2d weight"))


def conv_transpose2d_weight_desc(precision: Precision) -> Any:
    """ConvTranspose2d weight descriptor for ``precision``."""
    if precision in _CONV_TRANSPOSE2D_WEIGHT_PRESET:
        return _backend.get_preset_desc(_CONV_TRANSPOSE2D_WEIGHT_PRESET[precision])
    return _backend.make_quant_desc(
        **_lookup(_CONV_TRANSPOSE2D_WEIGHT_ARGS, precision, "ConvTranspose2d weight")
    )


def linear_weight_desc(precision: Precision) -> Any:
    """Linear weight descriptor for ``precision``."""
    return _backend.make_quant_desc(**_lookup(_LINEAR_WEIGHT_ARGS, precision, "Linear weight"))
