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

Everything here inspects or toggles the ``TensorQuantizer`` modules of an already
prepared tree — nothing changes ``state_dict`` keys.
"""

from contextlib import contextmanager
import logging
from typing import Iterable, Type, Union

import torch
import torch.nn as nn

from autoware_ml.quantization.core import backend as quant_backend

TensorQuantizer = quant_backend.get_tensor_quantizer_cls()

logger = logging.getLogger(__name__)


def restore_root_logging() -> None:
    """Undo the ``absl.logging`` root-logger hijack pulled in by the quantization backend.

    Importing modelopt can import ``absl.logging``, which installs its own handler on the
    root logger (only WARNING+ reaches stderr) — silently swallowing every later log record
    of ONNX/TensorRT export and evaluation. This removes absl's handlers and restores the
    CLI's ``logging.basicConfig`` shape when absl left the root logger bare. It is a no-op
    when absl never hijacked (unit tests, plain training).
    """
    root = logging.getLogger()
    absl_handlers = [h for h in root.handlers if type(h).__module__.startswith("absl")]
    for handler in absl_handlers:
        root.removeHandler(handler)
    if absl_handlers and not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )
    if absl_handlers and root.level > logging.INFO:
        root.setLevel(logging.INFO)


def tensor_quantizer_cls() -> Type:
    """Return the backend ``TensorQuantizer`` class.

    Also re-asserts the root logging configuration (see :func:`restore_root_logging`) —
    kept here so every call site that touches the backend re-checks for the absl hijack.
    """
    restore_root_logging()
    return TensorQuantizer


def move_quantizer_amax_to_device(model: nn.Module, device: Union[str, torch.device]) -> int:
    """Move every ``TensorQuantizer._amax`` tensor to ``device`` (post checkpoint-load fixup).

    Shared by the CenterPoint and BEVFusion deploy loaders after ``load_state_dict``.

    Returns:
        Number of amax tensors moved.
    """
    quantizer_cls = tensor_quantizer_cls()
    device = torch.device(device)
    moved = 0
    for _name, module in model.named_modules():
        if isinstance(module, quantizer_cls):
            if getattr(module, "_amax", None) is not None and module._amax.device != device:
                module._amax = module._amax.to(device)
                moved += 1
    if moved:
        logger.info("Moved %d quantizer amax tensors to %s", moved, device)
    return moved


def setup_quantization_for_onnx_export() -> None:
    """Configure the quantization backend for proper ONNX export.

    modelopt's ``TensorQuantizer`` traces to QuantizeLinear/DequantizeLinear ONNX ops natively,
    so there is nothing to switch — kept as the single pre-export call site so the logging
    re-assert below still runs.
    """
    # Re-assert deployment logging first (same absl-hijack concern as tensor_quantizer_cls).
    tensor_quantizer_cls()
    quant_backend.setup_onnx_export()


def set_quantizers_enabled(module: nn.Module, enabled: bool) -> int:
    """Enable or disable every ``TensorQuantizer`` under ``module`` (itself included).

    Args:
        module: Model or submodule whose quantizers to toggle.
        enabled: ``False`` disables fake-quant (the quantizers pass through),
            ``True`` re-enables it.

    Returns:
        Number of quantizers toggled.
    """
    count = 0
    for _name, submodule in module.named_modules():
        if isinstance(submodule, TensorQuantizer):
            submodule._disabled = not enabled
            count += 1
    return count


@contextmanager
def quantizers_disabled(model: nn.Module):
    """Context manager: run with every quantizer under ``model`` disabled, then re-enable.

    Example:
        >>> with quantizers_disabled(model):
        ...     fp_output = model(batch)  # FP forward through the quantized tree
    """
    set_quantizers_enabled(model, False)
    try:
        yield model
    finally:
        set_quantizers_enabled(model, True)


def disable_quantizers_in(model: nn.Module, module_names: Iterable[str]) -> int:
    """Disable every ``TensorQuantizer`` inside the named modules — the ``skip_quantize`` disable loop.

    The single spelling of "turn the ``skip_quantize`` subtrees off after calibration / checkpoint load,"
    shared by the quantize stage and the deploy loaders. ``module_names`` is the concrete
    set produced by :func:`~autoware_ml.quantization.core.replace.expand_skip_quantize` (matched modules
    plus all descendants), so an exact ``named_modules()`` lookup per name is sufficient;
    :func:`set_quantizers_enabled` then recursively disables the quantizers under each hit.

    Args:
        model: Model whose quantizers to disable.
        module_names: Concrete dotted module names (typically from ``expand_skip_quantize``).

    Returns:
        Number of named modules found and disabled. Names not present in the model are logged as
        warnings and skipped (an expanded set can never miss, so a miss means stale input).
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
    """Validate every enabled ``TensorQuantizer``'s ``amax`` (TensorRT needs positive finite scales).

    The one amax health policy, shared by the PTQ producer, the QAT callback, and the
    checkpoint loader:

    - ``None`` (never calibrated) or non-finite (NaN/Inf — poisoned calibration input) is
      fatal: fake-quant would emit NaN and an exported graph would bake invalid scales.
    - A finite but non-positive amax (a genuinely dead / all-zero channel) is clamped to a
      small epsilon and warned: that channel quantizes to ~0 either way, and TensorRT
      rejects a zero scale.

    Disabled quantizers are skipped: they are not used in forward and may legitimately
    carry ``amax=None`` (e.g. inside ``skip_quantize`` subtrees).

    Raises:
        RuntimeError: If any enabled quantizer has ``amax`` that is ``None`` or non-finite.
    """
    quantizer_cls = tensor_quantizer_cls()

    fatal: list[tuple[str, str]] = []
    clamped: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, quantizer_cls) or getattr(module, "_disabled", False):
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
    """
    Log the status of all TensorQuantizers in the model.

    One INFO summary line (enabled / disabled / calibrated counts); the per-quantizer
    name, status, and amax details are emitted at DEBUG for debugging placement.

    Args:
        model: PyTorch model
    """
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
                # Scalar amax (per-tensor quantization)
                detail = f"amax={amax.item():.6f}"
            else:
                # Multi-element amax (per-channel quantization)
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
    """
    Count enabled and disabled quantizers in the model.

    Args:
        model: PyTorch model

    Returns:
        Dict with 'enabled', 'disabled', and 'total' counts
    """
    enabled = 0
    disabled = 0

    for _name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            if module._disabled:
                disabled += 1
            else:
                enabled += 1

    return {"enabled": enabled, "disabled": disabled, "total": enabled + disabled}
