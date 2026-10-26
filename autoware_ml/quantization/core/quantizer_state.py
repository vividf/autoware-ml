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

"""Quantizer state operations: enable/disable, amax validation, status reporting.

None of them changes ``state_dict`` keys.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from modelopt.torch.quantization.nn import TensorQuantizer
from torch import nn

logger = logging.getLogger(__name__)


def set_quantizers_enabled(module: nn.Module, enabled: bool) -> int:
    """Enable or disable every ``TensorQuantizer`` under ``module`` and return how many."""
    count = 0
    for _name, submodule in module.named_modules():
        if isinstance(submodule, TensorQuantizer):
            submodule._disabled = not enabled
            count += 1
    return count


@contextmanager
def quantizers_disabled(model: nn.Module):
    """Disable every quantizer under ``model`` inside the block, then enable all of them."""
    set_quantizers_enabled(model, False)
    try:
        yield model
    finally:
        set_quantizers_enabled(model, True)


def disable_quantizers_in(model: nn.Module, module_names: Iterable[str]) -> int:
    """Disable the quantizers under each named module and return how many modules were found.

    ``module_names`` are exact names, such as the output of
    :func:`~autoware_ml.quantization.core.replace.expand_skip_quantize`; a missing one is
    logged and skipped.
    """
    modules = dict(model.named_modules())
    count = 0
    for name in sorted(module_names):
        module = modules.get(name)
        if module is None:
            logger.warning("disable_quantizers_in: module not found, skipping: %s", name)
            continue
        set_quantizers_enabled(module, False)
        count += 1
    if count:
        logger.info("Disabled quantizers in %d skip_quantize module(s)", count)
    return count


def validate_quantizer_amax(model: nn.Module) -> None:
    """Check the ``amax`` of every enabled quantizer; TensorRT needs positive finite scales.

    A missing or non-finite ``amax`` raises a ``RuntimeError``. A non-positive one (a dead
    channel) is clamped to 1e-8 with a warning: the channel quantizes to zero either way, and
    TensorRT rejects a zero scale. Disabled quantizers, which may have no ``amax``, are skipped.
    """
    fatal: list[tuple[str, str]] = []
    clamped: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer) or module._disabled:
            continue
        amax = getattr(module, "_amax", None)
        if amax is None:
            fatal.append((name, "amax=None (never calibrated)"))
            continue
        if not torch.is_tensor(amax):
            amax = torch.as_tensor(amax)
        if not torch.isfinite(amax).all():
            fatal.append((name, "amax has NaN/Inf (poisoned calibration input)"))
        elif float(amax.min()) <= 0.0:
            module._amax = amax.clamp(min=1e-8)
            clamped.append(name)

    if clamped:
        logger.warning(
            "Clamped non-positive amax to 1e-8 in %d quantizer(s) (dead/all-zero channels): %s%s",
            len(clamped),
            clamped[:10],
            " ..." if len(clamped) > 10 else "",
        )
    if fatal:
        preview = "\n  ".join(f"{n}: {r}" for n, r in fatal[:20])
        raise RuntimeError(
            f"Found {len(fatal)} TensorQuantizer(s) with unusable amax; fake-quant would emit "
            f"NaN/Inf and an exported graph would bake invalid Q/DQ scales. First offenders:\n  {preview}\n"
            "Fixes: calibrate on clean val data (the default), or add the layer to skip_quantize."
        )


def print_quantizer_status(model: nn.Module) -> None:
    """Log the enabled, disabled and calibrated quantizer counts, and each quantizer at DEBUG."""
    enabled = disabled = calibrated = 0
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        if module._disabled:
            disabled += 1
        else:
            enabled += 1
        amax = getattr(module, "_amax", None)
        if amax is None:
            detail = "amax=None"
        else:
            calibrated += 1
            if amax.numel() == 1:
                detail = f"amax={amax.item():.6f}"
            else:
                detail = (
                    f"amax=[{amax.numel()} elements] "
                    f"min={amax.min().item():.6f}, max={amax.max().item():.6f}"
                )
        status = "DISABLED" if module._disabled else "ENABLED"
        logger.debug("Quantizer %s: %s, %s", name, status, detail)

    logger.info(
        "Quantizer status: %d enabled, %d disabled, %d calibrated (%d total)",
        enabled,
        disabled,
        calibrated,
        enabled + disabled,
    )


def count_quantizers(model: nn.Module) -> dict:
    """Count the enabled and disabled quantizers in the model."""
    enabled = 0
    disabled = 0
    for _name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if module._disabled:
                disabled += 1
            else:
                enabled += 1
    return {"enabled": enabled, "disabled": disabled, "total": enabled + disabled}
