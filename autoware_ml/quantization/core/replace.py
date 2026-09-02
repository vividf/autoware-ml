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

"""Generic module-replacement engine for quantization — the ReplaceModule transform.

Model-agnostic Q/DQ insertion: swap ``nn.Conv2d`` / ``nn.ConvTranspose2d`` / ``nn.Linear`` for their
quantized subclasses via one kind-table-driven walker (:func:`replace_quantizable_modules`).
Architecture-specific placement (residual-add / eSE / OSA forward hooks) lives in
:mod:`autoware_ml.quantization.recipes`; which submodules get which kinds is a model's
:class:`~autoware_ml.quantization.plan.QuantRules` declaration (e.g. CenterPoint's in
``models/detection3d/main_modules/centerpoint_quantization.py``).
"""

import logging
from fnmatch import fnmatch
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple, Type

import torch
import torch.nn as nn

from autoware_ml.quantization.config import Precision

from .descriptors import (
    conv2d_weight_desc,
    conv_transpose2d_weight_desc,
    input_desc,
    linear_weight_desc,
)
from .modules import QuantConv2d, QuantConvTranspose2d, QuantLinear

logger = logging.getLogger(__name__)


def match_skip_quantize_roots(
    model: nn.Module, patterns: Iterable[str], *, log: bool = True
) -> List[Tuple[str, str]]:
    """Match ``skip_quantize`` glob patterns against the model — the single matching step.

    Each pattern is matched with :func:`fnmatch.fnmatch` against ``model.named_modules()``;
    a bare name (no glob metacharacters) matches that module exactly. Per-pattern match
    counts are logged and a pattern that matches **nothing** raises a warning — this fixes
    modelopt's silent-no-match footgun and catches typos in ``skip_quantize`` immediately.

    Args:
        model: The model whose ``named_modules()`` the patterns are resolved against.
        patterns: ``skip_quantize`` glob patterns (dotted module names, ``fnmatch`` syntax).
        log: Emit per-pattern match-count info and the zero-match warning (default True).

    Returns:
        ``(pattern, matched_module_name)`` pairs, in pattern order then model order —
        the *roots* only; subtree expansion is :func:`expand_skip_quantize`'s job.
    """
    all_names = [name for name, _ in model.named_modules() if name]
    matches: List[Tuple[str, str]] = []
    for pattern in patterns:
        matched = [name for name in all_names if name == pattern or fnmatch(name, pattern)]
        if log:
            if matched:
                logger.info(
                    "[skip_quantize] pattern %r matched %d module(s)", pattern, len(matched)
                )
            else:
                logger.warning(
                    "[skip_quantize] pattern %r matched NOTHING — check for a typo (it will exclude no layer from quantization)",
                    pattern,
                )
        matches.extend((pattern, name) for name in matched)
    return matches


def expand_skip_quantize(
    model: nn.Module, patterns: Iterable[str], *, log: bool = True
) -> Set[str]:
    """Resolve ``skip_quantize`` glob patterns into the concrete set of module names left un-quantized.

    This is the single place ``skip_quantize`` subtree semantics live (matching itself lives in
    :func:`match_skip_quantize_roots`). The result is the **subtree**: every matched module **plus
    all its descendants**. Materializing descendants is what lets a subtree-root entry (e.g.
    ``"pts_voxel_encoder"``) actually skip the subtree even though the engine walks each submodule
    *from its root* and only tests descendant names by exact match.

    Args:
        model: The model whose ``named_modules()`` the patterns are resolved against.
        patterns: ``skip_quantize`` glob patterns (dotted module names, ``fnmatch`` syntax).
        log: Emit per-pattern match-count info and the zero-match warning (default True).

    Returns:
        The set of dotted module names (matched modules and all their descendants) to exclude
        from quantization (their runtime precision follows the deploy ONNX precision).
        Suitable as ``skip_names`` for the replace/attach helpers and for the disable loops.
    """
    all_names = [name for name, _ in model.named_modules() if name]
    skip: Set[str] = set()
    for _pattern, name in match_skip_quantize_roots(model, patterns, log=log):
        skip.add(name)
        prefix = name + "."
        skip.update(child for child in all_names if child.startswith(prefix))
    return skip


def _rebuild_conv2d_as_quant(conv: nn.Conv2d, precision: Precision) -> QuantConv2d:
    """Build QuantConv2d via ``__init__`` + weight copy (no ``__dict__`` transplant).

    Copying ``vars(conv)`` onto a ``QuantConv2d`` shell can carry MMEngine/spconv hooks or
    half-initialized state that interacts badly with fake tensors during
    ``TensorQuantizer`` setup. PTQ deploy load uses this path for robustness.
    """
    q = QuantConv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    q = q.to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        q.weight.copy_(conv.weight)
        if conv.bias is not None:
            q.bias.copy_(conv.bias)
    q.init_quantizer(input_desc(precision), conv2d_weight_desc(precision))
    return q


def _rebuild_conv_transpose2d_as_quant(
    conv: nn.ConvTranspose2d, precision: Precision
) -> QuantConvTranspose2d:
    """Same as :func:`_rebuild_conv2d_as_quant` for transposed conv (FPN upsample)."""
    q = QuantConvTranspose2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=conv.output_padding,
        groups=conv.groups,
        bias=conv.bias is not None,
        dilation=conv.dilation,
        padding_mode=conv.padding_mode,
    )
    q = q.to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        q.weight.copy_(conv.weight)
        if conv.bias is not None:
            q.bias.copy_(conv.bias)
    q.init_quantizer(input_desc(precision), conv_transpose2d_weight_desc(precision))
    return q


def _clear_module_hooks(module: nn.Module) -> None:
    """Remove forward/backward/state_dict hooks copied from the original nn.Conv2d.

    Rare third-party / registry hooks can interact badly with quantization init; hooks are not needed
    on the quantized clone for deployment load.
    """
    for name in (
        "_forward_hooks",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_state_dict_hooks",
        "_load_state_dict_pre_hooks",
        "_load_state_dict_post_hooks",
    ):
        d = getattr(module, name, None)
        if d is not None and hasattr(d, "clear"):
            d.clear()


def clone_as_quant_by_transplant(
    nn_instance: nn.Module,
    quant_module: Type,
    quant_desc_input,
    quant_desc_weight,
) -> nn.Module:
    """
    Transfer weights and attributes from original module to quantized version.

    This function creates a new quantized module instance (via ``__new__``) and copies all
    attributes from the original module (``vars()`` transplant), then initializes the
    quantizers from the given descriptors.

    Two clone mechanisms exist on purpose: Conv/ConvTranspose go through the clean
    ``_rebuild_*_as_quant`` path (``__init__`` + weight copy) because their PTQ deploy load hit
    MMEngine/spconv hook + fake-tensor issues with the transplant; Linear stays on this transplant
    path, which has been stable for the voxel-encoder/ConvNeXt Linears. TODO: a
    ``_rebuild_linear_as_quant`` for symmetry is a candidate follow-up — verify mAP unchanged
    before switching.

    Args:
        nn_instance: Original PyTorch module (Conv2d, Linear, etc.)
        quant_module: Quantized module class (QuantConv2d, QuantLinear, etc.)
        quant_desc_input: Activation quantizer descriptor.
        quant_desc_weight: Weight quantizer descriptor.

    Returns:
        Quantized module with copied weights and initialized quantizers
    """
    # Create new instance without calling __init__
    quant_instance = quant_module.__new__(quant_module)

    # Copy all attributes from original module
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)

    _clear_module_hooks(quant_instance)

    quant_instance.init_quantizer(quant_desc_input, quant_desc_weight)

    return quant_instance


def _rebuild_linear_as_quant(linear: nn.Linear, precision: Precision) -> QuantLinear:
    """Linear -> QuantLinear via the transplant clone (see its docstring for why)."""
    return clone_as_quant_by_transplant(
        linear, QuantLinear, input_desc(precision), linear_weight_desc(precision)
    )


# The kind table: everything the ReplaceModule transform can swap, as
# (source class, quantized subclass, rebuild function) rules. A module is
# replaced by the FIRST rule whose source class it is an instance of (and whose
# quantized subclass it is not already an instance of). Rebuild functions take
# ``(module, precision)`` and fetch their descriptors from :mod:`.descriptors`.
_REPLACEMENT_KINDS: dict = {
    "conv": (
        (nn.Conv2d, QuantConv2d, _rebuild_conv2d_as_quant),
        (nn.ConvTranspose2d, QuantConvTranspose2d, _rebuild_conv_transpose2d_as_quant),
    ),
    "linear": ((nn.Linear, QuantLinear, _rebuild_linear_as_quant),),
}

#: Module types the walker must never swap, whatever the kind rules say.
#: ``nn.MultiheadAttention.out_proj`` is a ``NonDynamicallyQuantizableLinear`` whose
#: forward the attention fast path bypasses (``F.multi_head_attention_forward`` reads
#: ``.weight`` directly), so a quantizer planted there never sees a calibration batch
#: and silently vanishes from any export that rebuilds the attention module — a
#: calibrated-looking checkpoint that quantizes nothing.
_NEVER_REPLACE: tuple = (nn.modules.linear.NonDynamicallyQuantizableLinear,)


def replace_quantizable_modules(
    model: nn.Module,
    kinds: Sequence[str],
    skip_names: Optional[Set[str]] = None,
    prefix: str = "",
    on_replace: Optional[Callable[[str, nn.Module, nn.Module], None]] = None,
    *,
    precision: Precision,
) -> None:
    """Recursively swap every module of the requested kinds for its quantized subclass.

    The one ReplaceModule walker: it traverses the tree bottom-up and replaces each
    matching leaf via the kind table (:data:`_REPLACEMENT_KINDS`), except modules whose
    full dotted names are in ``skip_names`` — a skipped name skips its **whole subtree**
    (which is what lets ``expand_skip_quantize`` container entries like
    ``'pts_backbone.blocks.0'`` exclude a whole block).

    Args:
        model: (Sub)model to modify in place. ``None`` / non-modules are ignored.
        kinds: Module kinds to replace — keys of the kind table (``"conv"``, ``"linear"``).
        skip_names: Full dotted module names (from the model root) to leave untouched.
        prefix: Dotted name of ``model`` itself, so reported names are root-relative.
        on_replace: Optional callback ``(full_name, original, replacement)`` invoked
            after each swap — the placement recording hook.
        precision: Target precision; selects the descriptors of every inserted quantizer.

    Raises:
        KeyError: On an unknown kind name (kind vocabulary lives in the table).
    """
    rules = tuple(rule for kind in kinds for rule in _REPLACEMENT_KINDS[kind])
    _replace_walk(model, rules, skip_names or set(), prefix, on_replace, precision)


def _replace_walk(
    model: nn.Module,
    rules: tuple,
    skip_names: Set[str],
    prefix: str,
    on_replace: Optional[Callable[[str, nn.Module, nn.Module], None]],
    precision: Precision,
) -> None:
    """Recursive body of :func:`replace_quantizable_modules`."""
    if model is None or not isinstance(model, nn.Module):
        return

    for name in list(model._modules.keys()):
        submodule = model._modules[name]
        full_name = f"{prefix}.{name}" if prefix else name

        # Skip entire subtree if this module name is in the skip list.
        if full_name in skip_names:
            continue

        if submodule is not None:
            _replace_walk(submodule, rules, skip_names, full_name, on_replace, precision)

        if isinstance(submodule, _NEVER_REPLACE):
            continue

        for source_cls, quant_cls, rebuild in rules:
            if isinstance(submodule, source_cls) and not isinstance(submodule, quant_cls):
                replacement = rebuild(submodule, precision)
                model._modules[name] = replacement
                if on_replace is not None:
                    on_replace(full_name, submodule, replacement)
                break
