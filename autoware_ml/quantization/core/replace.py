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

"""Module replacement — the ReplaceModule transform, on modelopt's ``QuantModuleRegistry``.

Model-agnostic Q/DQ insertion: every ``nn.Conv2d`` / ``nn.ConvTranspose2d`` / ``nn.Linear``
of the requested kinds is converted **in place** into its modelopt quantized counterpart
(``QuantModuleRegistry.convert`` patches the instance's class — no rebuild, no weight copy,
no ``__dict__`` transplant), then its ``input_quantizer`` / ``weight_quantizer`` take the
framework's descriptors from :mod:`.descriptors`. The converted module keeps its type
identity (``isinstance(m, nn.Conv2d)`` stays true) and gains ``input_quantizer``,
``weight_quantizer`` and a disabled ``output_quantizer``.

Architecture-specific placement (residual-add / pool) lives in
:mod:`autoware_ml.quantization.recipes`; which submodules get which kinds is a model's
:class:`~autoware_ml.quantization.plan.QuantRules` declaration (e.g. CenterPoint's in
``models/detection3d/main_modules/centerpoint/quantization.py``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from fnmatch import fnmatch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
from torch import nn

from autoware_ml.quantization.config import Precision

from .descriptors import (
    conv2d_weight_desc,
    conv_transpose2d_weight_desc,
    input_desc,
    linear_weight_desc,
)

logger = logging.getLogger(__name__)


def match_skip_quantize_roots(
    model: nn.Module, patterns: Iterable[str], *, log: bool = True
) -> list[tuple[str, str]]:
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
    matches: list[tuple[str, str]] = []
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
) -> set[str]:
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
    skip: set[str] = set()
    for _pattern, name in match_skip_quantize_roots(model, patterns, log=log):
        skip.add(name)
        prefix = name + "."
        skip.update(child for child in all_names if child.startswith(prefix))
    return skip


#: A weight-descriptor chooser: ``fn(precision) -> QuantizerAttributeConfig``.
WeightDesc = Callable[[Precision], QuantizerAttributeConfig]

# The kind table: everything the ReplaceModule transform can convert, as
# (source class, weight descriptor) rules. A module is converted by the FIRST rule whose
# source class it is an instance of. The quantized class itself comes from modelopt's
# registry (``nn.Conv2d`` -> ``QuantConv2d`` etc.); only the descriptors are ours.
_REPLACEMENT_KINDS: dict[str, tuple[tuple[type[nn.Module], WeightDesc], ...]] = {
    "conv": (
        (nn.Conv2d, conv2d_weight_desc),
        (nn.ConvTranspose2d, conv_transpose2d_weight_desc),
    ),
    "linear": ((nn.Linear, linear_weight_desc),),
}

#: Module types the walker must never convert, whatever the kind rules say.
#: ``nn.MultiheadAttention.out_proj`` is a ``NonDynamicallyQuantizableLinear`` whose
#: forward the attention fast path bypasses (``F.multi_head_attention_forward`` reads
#: ``.weight`` directly), so a quantizer planted there never sees a calibration batch
#: and silently vanishes from any export that rebuilds the attention module — a
#: calibrated-looking checkpoint that quantizes nothing. (modelopt's registry would
#: happily convert it: it shares ``nn.Linear.forward``.)
_NEVER_REPLACE: tuple = (nn.modules.linear.NonDynamicallyQuantizableLinear,)


def convert_to_quant_module(
    module: nn.Module,
    weight_desc: WeightDesc,
    precision: Precision,
    calibrator: str,
) -> QuantModule:
    """Convert one leaf module in place and apply the framework's descriptors.

    Args:
        module: An ``nn.Conv2d`` / ``nn.ConvTranspose2d`` / ``nn.Linear`` (registered in
            modelopt's ``QuantModuleRegistry``).
        weight_desc: Weight-descriptor chooser for this module kind.
        precision: Target precision; selects both descriptors.
        calibrator: Activation calibrator kind (``"histogram"`` / ``"max"``).

    Returns:
        ``module`` itself, now a ``QuantModule`` subclass instance.
    """
    quant = QuantModuleRegistry.convert(module)
    quant.input_quantizer.set_from_attribute_config(input_desc(precision, calibrator))
    quant.weight_quantizer.set_from_attribute_config(weight_desc(precision))
    return quant


def replace_quantizable_modules(
    model: nn.Module,
    kinds: Sequence[str],
    skip_names: set[str] | None = None,
    prefix: str = "",
    on_replace: Callable[[str, str, nn.Module], None] | None = None,
    *,
    precision: Precision,
    calibrator: str = "histogram",
) -> None:
    """Recursively convert every module of the requested kinds into its quantized counterpart.

    The one ReplaceModule walker: it traverses the tree bottom-up and converts each
    matching leaf via the kind table (:data:`_REPLACEMENT_KINDS`), except modules whose
    full dotted names are in ``skip_names`` — a skipped name skips its **whole subtree**
    (which is what lets ``expand_skip_quantize`` container entries like
    ``'pts_backbone.blocks.0'`` exclude a whole block).

    Args:
        model: (Sub)model to modify in place. ``None`` / non-modules are ignored.
        kinds: Module kinds to convert — keys of the kind table (``"conv"``, ``"linear"``).
        skip_names: Full dotted module names (from the model root) to leave untouched.
        prefix: Dotted name of ``model`` itself, so reported names are root-relative.
        on_replace: Optional callback ``(full_name, original_class_name, module)`` invoked
            after each conversion — the placement recording hook.
        precision: Target precision; selects the descriptors of every inserted quantizer.
        calibrator: Activation calibrator kind (``"histogram"`` / ``"max"``), from
            :attr:`CalibrationConfig.activation_calibrator`.

    Raises:
        KeyError: On an unknown kind name (kind vocabulary lives in the table).
    """
    rules = tuple(rule for kind in kinds for rule in _REPLACEMENT_KINDS[kind])
    _replace_walk(model, rules, skip_names or set(), prefix, on_replace, precision, calibrator)


def _replace_walk(
    model: nn.Module,
    rules: tuple,
    skip_names: set[str],
    prefix: str,
    on_replace: Callable[[str, str, nn.Module], None] | None,
    precision: Precision,
    calibrator: str,
) -> None:
    """Recursive body of :func:`replace_quantizable_modules`."""
    if model is None or not isinstance(model, nn.Module):
        return

    for name, submodule in list(model._modules.items()):
        full_name = f"{prefix}.{name}" if prefix else name

        # Skip entire subtree if this module name is in the skip list.
        if full_name in skip_names or submodule is None:
            continue

        _replace_walk(submodule, rules, skip_names, full_name, on_replace, precision, calibrator)

        if isinstance(submodule, (_NEVER_REPLACE, QuantModule)):
            continue

        for source_cls, weight_desc in rules:
            if isinstance(submodule, source_cls):
                original = type(submodule).__name__
                convert_to_quant_module(submodule, weight_desc, precision, calibrator)
                if on_replace is not None:
                    on_replace(full_name, original, submodule)
                break
