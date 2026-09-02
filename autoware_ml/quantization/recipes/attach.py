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

"""Architecture recipes — the PatchBlockForward / WrapModule transforms.

Each recipe is a **matcher + action** pair: the matcher identifies a block by class
(the :data:`_RESIDUAL_SPECS` registry for residual blocks; exact classes for eSE and
MaxPool), and the action attaches :class:`~modelopt.torch.quantization.nn.TensorQuantizer`
modules at TensorRT-friendly locations and swaps the block's ``forward`` for the rewrite in
:mod:`.quant_forwards` (or wraps the module, for pools). Supporting a new residual block
type means adding one :class:`ResidualBlockSpec` row, not another ``elif``.

Recipes are class-gated and span architectures on purpose: each fires only where the
model has that block, so zero matches are normal (a plain SECOND backbone matches none
of them). Every attach function takes an optional ``on_apply(module, transform, reason,
detail)`` callback — the placement recording hook of
:class:`~autoware_ml.quantization.plan.QuantizationPlan`.
"""

from dataclasses import dataclass
import logging
from typing import Callable, Dict, Optional, Set, Tuple, Type

import torch.nn as nn

from autoware_ml.quantization.config import Precision
from autoware_ml.quantization.core import modelopt as quant_backend
from autoware_ml.quantization.core.descriptors import input_desc

from .quant_forwards import (
    QuantBasicBlockForward,
    QuantConvNeXtBlockForward,
    QuantOSAModuleForward,
    QuantBeforePool,
    QuantSparseBasicBlockForward,
    QuantESEModuleForward,
)

logger = logging.getLogger(__name__)


def _new_input_quantizer(precision: Precision):
    """Create a fresh ``TensorQuantizer`` on the conv-input activation descriptor.

    The recipes use the same descriptor parameters as the conv/linear input quantizers
    (:func:`~autoware_ml.quantization.core.descriptors.input_desc`), so the residual /
    eSE / pool quantizers calibrate consistently with the layers around them.
    """
    TensorQuantizer = quant_backend.get_tensor_quantizer_cls()

    return TensorQuantizer(input_desc(precision))


def _replace_block_forward(module: nn.Module, forward_cls) -> None:
    """Replace ``module.forward`` with ``forward_cls(module)``, saving the original once (idempotent)."""
    if isinstance(module.forward, forward_cls):
        return
    if not hasattr(module, "_original_forward"):
        module._original_forward = module.forward
    module.forward = forward_cls(module)


#: Placement recording hook: ``on_apply(module_name, transform, reason, detail)``.
OnApply = Callable[[str, str, str, str], None]


@dataclass(frozen=True)
class ResidualBlockSpec:
    """How one residual block class gets its INT8 residual placement.

    Attributes:
        class_name: Block class name to match against ``type(module).__name__``.
        quant_forward: ``Quant*Forward`` class that replaces matched blocks' ``forward``.
        share_from: Submodule paths (dots index into containers, e.g. ``"concat.0"``)
            whose ``_input_quantizer`` the ``residual_quantizer`` reuses, in priority
            order. Sharing keeps the residual scale identical to the branch input scale
            (same calibration data) — the lidar-ai-solution / CUDA-BEVFusion recipe.
        exact: Match the class name exactly (no subclass-by-name matching).
        fresh_if_downsample: A block with a ``downsample`` branch gets a fresh
            quantizer instead of sharing (the identity passes through the downsample,
            so the branch-input scale no longer applies).
        osa_concat: Also attach per-branch ``concat_input_quantizers`` and, for
            ``identity=False`` blocks, replace ``forward`` without a residual quantizer.
    """

    class_name: str
    quant_forward: Type
    share_from: Tuple[str, ...]
    exact: bool = False
    fresh_if_downsample: bool = False
    osa_concat: bool = False


# The residual-recipe registry: first matching spec wins. Non-exact specs match
# subclassed block names by substring ("MyConvNeXtBlock" still matches), which is
# why "SparseBasicBlock" must precede "BasicBlock". The full supported set; this
# deliberately spans architectures.
_RESIDUAL_SPECS: Tuple[ResidualBlockSpec, ...] = (
    ResidualBlockSpec(
        "_OSA_module",
        quant_forward=QuantOSAModuleForward,
        share_from=("concat.0",),
        exact=True,
        fresh_if_downsample=True,
        osa_concat=True,
    ),
    ResidualBlockSpec(
        "ConvNeXtBlock",
        quant_forward=QuantConvNeXtBlockForward,
        share_from=("depthwise_conv",),
        fresh_if_downsample=True,
    ),
    ResidualBlockSpec(
        "SparseBasicBlock",
        quant_forward=QuantSparseBasicBlockForward,
        share_from=("conv1",),
        fresh_if_downsample=True,
    ),
    ResidualBlockSpec(
        "BasicBlock",
        quant_forward=QuantBasicBlockForward,
        share_from=("conv1",),
        fresh_if_downsample=True,
    ),
)


def _match_residual_spec(cls_name: str) -> Optional[ResidualBlockSpec]:
    """Return the first registry spec matching a block class name (``None`` if no recipe)."""
    for spec in _RESIDUAL_SPECS:
        if cls_name == spec.class_name or (not spec.exact and spec.class_name in cls_name):
            return spec
    return None


def _submodule_by_path(module: nn.Module, path: str) -> Optional[nn.Module]:
    """Resolve a dotted path relative to ``module`` (digits index into containers)."""
    current: Optional[nn.Module] = module
    for part in path.split("."):
        if current is None:
            return None
        try:
            current = current[int(part)] if part.isdigit() else getattr(current, part, None)
        except (IndexError, KeyError, TypeError):
            return None
    return current


def _resolve_residual_quantizer(
    module: nn.Module, spec: ResidualBlockSpec
) -> Tuple[Optional[nn.Module], str]:
    """Pick the residual quantizer for a matched block per its spec.

    Returns:
        ``(shared_quantizer, how)`` — the quantizer to reuse (``None`` means create a
        fresh one) and a human-readable description for the placement record.
    """
    if spec.fresh_if_downsample and getattr(module, "downsample", None) is not None:
        return None, "fresh (block has a downsample branch)"
    for path in spec.share_from:
        submodule = _submodule_by_path(module, path)
        quantizer = getattr(submodule, "_input_quantizer", None) if submodule is not None else None
        if quantizer is not None:
            return quantizer, f"shared from {path}._input_quantizer"
    return None, "fresh (no shareable input quantizer)"


def attach_residual_add_recipe(
    model: nn.Module, precision: Precision, on_apply: Optional[OnApply] = None
):
    """
    Attach residual_quantizer to modules that perform residual add and replace their forward methods.

    This follows the same approach as lidar-ai-solution (CUDA-BEVFusion):
    - Only quantize the identity branch (residual connection), not the conv path output
    - This enables TensorRT to fuse Conv+Add operations, reducing reformat operations
    - The residual_quantizer uses the same quant descriptor as conv layers for consistency

    Which blocks match and where their residual quantizer comes from is the
    :data:`_RESIDUAL_SPECS` registry's job — this function is just the walk.

    Args:
        model: Model whose residual blocks get the recipe.
        precision: Target precision of the attached quantizers.
        on_apply: Optional placement recording hook.
    """
    attached_count = 0
    for name, module in model.named_modules():
        spec = _match_residual_spec(module.__class__.__name__)
        if spec is None:
            continue
        detail_parts = [spec.quant_forward.__name__]

        if spec.osa_concat:
            # Branch inputs get Q/DQ before Concat: skip connections are x + layer0..layer(n-2);
            # the main path (layer(n-1) output) stays un-quantized, like the ResNet Add.
            n_branch_inputs = len(module.layers)
            if (
                not hasattr(module, "concat_input_quantizers")
                or len(module.concat_input_quantizers) != n_branch_inputs
            ):
                concat_quantizers = nn.ModuleList(
                    [_new_input_quantizer(precision) for _ in range(n_branch_inputs)]
                )
                module.add_module("concat_input_quantizers", concat_quantizers)
            detail_parts.append(f"concat_input_quantizers[{n_branch_inputs}]")
            # When identity=True the quant forward reuses concat_input_quantizers[0] as the single Q
            # for the block input (no extra module); identity=False needs no residual Q at all.
            if not getattr(module, "identity", False):
                _replace_block_forward(module, spec.quant_forward)
                if on_apply is not None:
                    detail_parts.append("identity=False: no residual quantizer")
                    on_apply(
                        name,
                        "patch_forward",
                        f"recipe 'residual_add': matched {spec.class_name}",
                        "; ".join(detail_parts),
                    )
                continue

        # Attach residual_quantizer if not already present. Reused quantizers are assigned
        # as plain attributes (not add_module) because a TensorQuantizer cannot be a
        # submodule of two parents; the replaced forward still calls it so ONNX export traces
        # the Q/DQ.
        if not hasattr(module, "residual_quantizer"):
            shared, how = _resolve_residual_quantizer(module, spec)
            if shared is None:
                module.add_module("residual_quantizer", _new_input_quantizer(precision))
            else:
                module.residual_quantizer = shared
            attached_count += 1
            detail_parts.append(f"residual_quantizer: {how}")

        # Replace forward with the block-specific rewrite (quantizes only the residual branch).
        _replace_block_forward(module, spec.quant_forward)
        if on_apply is not None:
            on_apply(
                name,
                "patch_forward",
                f"recipe 'residual_add': matched {spec.class_name}",
                "; ".join(detail_parts),
            )

    if attached_count > 0:
        logger.info("Attached residual_quantizer to %d residual blocks", attached_count)


def attach_ese_recipe(
    model: nn.Module, precision: Precision, on_apply: Optional[OnApply] = None
) -> int:
    """
    Set up the single-Q-at-input eSE recipe on every ``eSEModule`` (one call, no ordering contract).

    Per module: attach ``pool_input_quantizer`` — the ONE Q/DQ at the eSE input, whose output ``qx``
    is shared by the pooling branch (``avg_pool → fc → hsigmoid``) *and* the ``Mul`` bypass — plus
    ``mul_gate_quantizer`` for the gate operand, then install :class:`QuantESEModuleForward` once.
    Result: both ``Mul`` operands are INT8 with a single FP32→INT8 reformat at the eSE input.

    (The legacy order-dependent two-Q path — a separate ``mul_identity_quantizer``, i.e. a second
    reformat with the pool branch left unquantized — has been removed; no shipping config used it.)

    Args:
        model: Model whose ``eSEModule`` blocks get the recipe.
        precision: Target precision of the attached quantizers.
        on_apply: Optional placement recording hook.

    Returns:
        Number of eSEModules set up.
    """
    count = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ != "eSEModule":
            continue
        if getattr(module, "pool_input_quantizer", None) is None:
            module.add_module("pool_input_quantizer", _new_input_quantizer(precision))
        if getattr(module, "mul_gate_quantizer", None) is None:
            module.add_module("mul_gate_quantizer", _new_input_quantizer(precision))
        _replace_block_forward(module, QuantESEModuleForward)
        count += 1
        if on_apply is not None:
            on_apply(
                name,
                "patch_forward",
                "recipe 'ese': matched eSEModule",
                "QuantESEModuleForward; pool_input_quantizer + mul_gate_quantizer "
                "(single Q at the eSE input, both Mul operands INT8)",
            )
    if count > 0:
        logger.info(
            "Attached single-Q eSE quantizers (pool_input + mul_gate) to %d eSEModules", count
        )
    return count


def attach_maxpool_recipe(
    model: nn.Module,
    precision: Precision,
    skip_names: Optional[Set[str]] = None,
    on_apply: Optional[OnApply] = None,
) -> int:
    """
    Replace nn.MaxPool2d modules with QuantBeforePool(quantizer, pool) so QDQ is applied before MaxPool.

    VoVNet _OSA_stage uses "Pooling" (MaxPool2d) before the first OSA block in stage3/stage4.
    This adds QDQ on the pool input so the MaxPool layer has quantized input in the ONNX graph.

    Args:
        model: Model whose MaxPool2d modules get wrapped.
        precision: Target precision of the attached quantizers.
        skip_names: skip_quantize subtree names to leave untouched (boundary-safe match).
        on_apply: Optional placement recording hook.

    Returns:
        Number of MaxPool2d modules replaced with QuantBeforePool.
    """
    skip_names = skip_names or set()
    name_to_module = dict(model.named_modules())
    to_replace = []  # (full_name, parent_module, child_name, pool_module)

    for name, module in model.named_modules():
        if not isinstance(module, nn.MaxPool2d):
            continue
        if isinstance(module, QuantBeforePool):
            continue
        # Boundary-safe subtree match: "backbone.block1" must not match "backbone.block10".
        if any(name == s or name.startswith(s + ".") for s in skip_names):
            continue
        parts = name.split(".")
        if not parts:
            continue
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = name_to_module.get(parent_name) if parent_name else model
        if parent is None:
            continue
        to_replace.append((name, parent, child_name, module))

    count = 0
    for full_name, parent, child_name, pool_module in to_replace:
        setattr(parent, child_name, QuantBeforePool(_new_input_quantizer(precision), pool_module))
        count += 1
        if on_apply is not None:
            on_apply(
                full_name,
                "wrap_module",
                "recipe 'maxpool': MaxPool2d input Q/DQ",
                "MaxPool2d -> QuantBeforePool",
            )

    if count > 0:
        logger.info("Attached QDQ before %d MaxPool2d modules", count)
    return count


#: Uniform attacher signature the plan calls: ``fn(model, skip_names, on_apply, precision)``.
RecipeAttacher = Callable[[nn.Module, Set[str], Optional[OnApply], Precision], object]

#: The recipe registry: recipe name -> attacher. ``QuantizationPlan.prepare`` applies
#: these in ``config.VALID_RECIPES`` order; adding a recipe means one entry here plus
#: its name in ``VALID_RECIPES`` (``plan.py`` verifies the two sets match at import).
#:
#: ``residual_add`` and ``ese`` ignore ``skip_names`` on purpose: their quantizers are
#: still attached inside skip_quantize subtrees (the state_dict layout must not depend
#: on ``skip_quantize``) and are disabled after load instead. ``maxpool`` must honor
#: it because wrapping a pool changes the module tree itself.
RECIPE_ATTACHERS: Dict[str, RecipeAttacher] = {
    "residual_add": lambda model, skip_names, on_apply, precision: attach_residual_add_recipe(
        model, precision, on_apply=on_apply
    ),
    "ese": lambda model, skip_names, on_apply, precision: attach_ese_recipe(
        model, precision, on_apply=on_apply
    ),
    "maxpool": lambda model, skip_names, on_apply, precision: attach_maxpool_recipe(
        model, precision, skip_names, on_apply=on_apply
    ),
}
