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

"""Q/DQ insertion through modelopt's ``QuantModuleRegistry``.

``QuantModuleRegistry.convert`` swaps a module's class in place: the module keeps its weights,
still passes ``isinstance(m, nn.Conv2d)``, and gains ``input_quantizer``, ``weight_quantizer``
and a disabled ``output_quantizer``, set up from :mod:`.descriptors`. The ``conv`` kind covers
``Conv2d`` and ``ConvTranspose2d`` only; ``Conv1d`` heads (TransFusion's prediction branches)
stay un-quantized.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from fnmatch import fnmatch

from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
from torch import nn

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.quantization.config import Precision

if IS_SPCONV_AVAILABLE:
    # Guarded at the top rather than imported inside a function: spconv is the one optional
    # dependency. Importing core.spconv registers the quantized sparse convolution with modelopt.
    from spconv.pytorch.conv import SparseConvolution

    from autoware_ml.quantization.core import spconv as _spconv_quant_registration  # noqa: F401

from .descriptors import (
    conv2d_weight_desc,
    conv_transpose2d_weight_desc,
    input_desc,
    linear_weight_desc,
    spconv_weight_desc,
)

logger = logging.getLogger(__name__)


def match_skip_quantize_roots(
    model: nn.Module, patterns: Iterable[str], *, log: bool = True
) -> list[tuple[str, str]]:
    """Return ``(pattern, module name)`` for each module a ``skip_quantize`` glob matches.

    A pattern that matches nothing raises: unlike modelopt, which ignores such a pattern, a
    typo must not silently quantize the layers it meant to exclude. ``log=False`` skips the
    logging and this check, for patterns already checked. Only the matched modules are
    returned; :func:`expand_skip_quantize` adds their subtrees.
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
                raise ValueError(
                    f"[skip_quantize] pattern {pattern!r} matched no module of "
                    f"{type(model).__name__}: it would exclude no layer from quantization. "
                    "Check the module path (a typo, or a module tree that was renamed)."
                )
        matches.extend((pattern, name) for name in matched)
    return matches


def expand_skip_quantize(
    model: nn.Module, patterns: Iterable[str], *, log: bool = True
) -> set[str]:
    """Return the names of the modules ``patterns`` match, plus all their descendants.

    The descendants are listed because the replace walker starts below each declared
    submodule and tests names by exact match: a pattern naming a declared submodule, such as
    ``"pts_voxel_encoder"``, takes effect only through its descendants' names.
    """
    all_names = [name for name, _ in model.named_modules() if name]
    skip: set[str] = set()
    for _pattern, name in match_skip_quantize_roots(model, patterns, log=log):
        skip.add(name)
        prefix = name + "."
        skip.update(child for child in all_names if child.startswith(prefix))
    return skip


WeightDesc = Callable[[Precision], QuantizerAttributeConfig]

# Module kind -> (source class, weight descriptor) rules; the first rule a module is an
# instance of converts it. The quantized class comes from modelopt's registry.
_REPLACEMENT_KINDS: dict[str, tuple[tuple[type[nn.Module], WeightDesc], ...]] = {
    "conv": (
        (nn.Conv2d, conv2d_weight_desc),
        (nn.ConvTranspose2d, conv_transpose2d_weight_desc),
    ),
    "linear": ((nn.Linear, linear_weight_desc),),
}

if IS_SPCONV_AVAILABLE:
    _REPLACEMENT_KINDS["spconv"] = ((SparseConvolution, spconv_weight_desc),)

#: Never converted, whatever the kinds say. ``nn.MultiheadAttention`` reads the weight of its
#: ``out_proj`` (a ``NonDynamicallyQuantizableLinear``) in ``F.multi_head_attention_forward``
#: without calling its forward, so a quantizer there never sees a calibration batch and is
#: lost on export. modelopt's registry would convert it like any ``nn.Linear``.
_NEVER_REPLACE: tuple = (nn.modules.linear.NonDynamicallyQuantizableLinear,)


def convert_to_quant_module(
    module: nn.Module,
    weight_desc: WeightDesc,
    precision: Precision,
    calibrator: str,
) -> QuantModule:
    """Convert ``module`` in place to its modelopt quantized class and set up its quantizers."""
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
    """Convert every module of ``kinds`` under ``model`` in place, bottom-up.

    Names are dotted from the model root, ``prefix`` being the name of ``model``. A name in
    ``skip_names`` skips its whole subtree. ``on_replace(name, original_class_name, module)``
    is called after each conversion.
    """
    missing = [kind for kind in kinds if kind not in _REPLACEMENT_KINDS]
    if missing == ["spconv"]:
        raise RuntimeError(
            "The model declares the 'spconv' module kind but spconv is not installed, so its "
            "sparse layers cannot be quantized. Run in the deployment image, or drop the "
            "sparse submodule from the model's QuantRules."
        )
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
