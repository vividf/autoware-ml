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

"""Architecture recipes — the ConvertBlock / WrapModule transforms.

Each recipe is a **matcher + action** pair. ``residual_add`` matches residual blocks by
class (:class:`ResidualBlockSpec` rows: the block class, its quantized counterpart in
:mod:`.quant_blocks`, and where its residual quantizer comes from) and converts them in
place through :data:`~.quant_blocks.QuantBlockRegistry`; ``ese`` does the same for VoVNet
eSE blocks (:class:`ESEBlockSpec`: single Q at the eSE input + gate quantizer); ``maxpool``
wraps every ``nn.MaxPool2d`` so Q/DQ lands on its input.

Recipes are class-gated and span architectures on purpose: each fires only where the
model has that block, so zero matches are normal (a plain SECOND backbone matches none of
them). They are also **scoped to the submodules the model's rules quantize**
(``roots`` = the present keys of ``QuantRules.quantize_submodules``): a residual block whose
convolutions are not quantized gets no residual Q/DQ either — BEVFusion's spconv encoder
holds ``SparseBasicBlock`` modules but deploys through the libspconv exporter, so a quantizer
there would fake-quantize in PyTorch and never reach the engine. Every attacher has the one
:data:`RecipeAttacher` signature ``(model, ctx: RecipeContext) -> count``; the context carries
the ``on_apply(module, transform, reason, detail)`` placement recording hook of
:class:`~autoware_ml.quantization.plan.QuantizationPlan`.

Block specs come from two places: the model's :attr:`QuantRules.residual_blocks` /
:attr:`QuantRules.ese_blocks` (a model declares the block classes it owns — VoVNet's
``_OSA_module`` / ``eSEModule`` live in the model package, not here) and
:func:`default_block_specs` (blocks the repo itself defines — today the spconv
``SparseBasicBlock``). Matching is ``isinstance`` on the declared class, first spec wins;
there is no name or substring matching.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
from torch import nn

from autoware_ml.quantization.config import Precision
from autoware_ml.quantization.core.descriptors import input_desc

from .quant_blocks import QuantBeforePool, QuantBlockRegistry, QuantSparseBasicBlock

logger = logging.getLogger(__name__)

#: Placement recording hook: ``on_apply(module_name, transform, reason, detail)``.
OnApply = Callable[[str, str, str, str], None]


# --------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class _BlockSpec:
    """A block class and the ``QuantModule`` it is converted into (in place)."""

    block_cls: type[nn.Module]
    quant_block_cls: type[QuantModule]

    def ensure_registered(self) -> None:
        """Register ``block_cls -> quant_block_cls`` in :data:`QuantBlockRegistry` (idempotent)."""
        if self.block_cls not in QuantBlockRegistry:
            QuantBlockRegistry.register({self.block_cls: self.block_cls.__name__})(
                self.quant_block_cls
            )


@dataclass(frozen=True)
class ResidualBlockSpec(_BlockSpec):
    """How one residual block class gets its residual-branch Q/DQ (``residual_add`` recipe).

    Attributes:
        block_cls: The block class to match (``isinstance``; subclasses that keep the
            block's ``forward`` match too, subclasses with their own ``forward`` do not —
            modelopt's registry rule, since the quantized forward mirrors the original).
        quant_block_cls: ``QuantModule`` subclass whose ``forward`` is the quantized rewrite
            (see :mod:`.quant_blocks`).
        share_from: Submodule paths (dots index into containers, e.g. ``"concat.0"``)
            whose ``input_quantizer`` the ``residual_quantizer`` reuses, in priority
            order. Sharing keeps the residual scale identical to the branch input scale
            (same calibration data) — the lidar-ai-solution / CUDA-BEVFusion recipe.
        fresh_if_downsample: A block with a ``downsample`` branch gets a fresh
            quantizer instead of sharing (the identity passes through the downsample,
            so the branch-input scale no longer applies).
        osa_concat: VoVNet ``_OSA_module`` placement: attach ``concat_input_quantizers``
            (one per Concat skip input, ``len(block.layers)``) instead of a residual
            quantizer — with ``identity=True`` the first one is the single Q at the block
            input (see :class:`~.quant_blocks.QuantOSAModule`).
    """

    share_from: tuple[str, ...] = ("conv1",)
    fresh_if_downsample: bool = True
    osa_concat: bool = False


@dataclass(frozen=True)
class ESEBlockSpec(_BlockSpec):
    """A VoVNet ``eSEModule`` class and its quantized rewrite (``ese`` recipe).

    The recipe attaches ``pool_input_quantizer`` (the ONE Q at the eSE input, fanned out to
    the gate path and the ``Mul`` bypass) and ``mul_gate_quantizer`` (the gate operand), so
    both ``Mul`` operands are INT8 with a single FP32 -> INT8 reformat.
    """


@dataclass(frozen=True)
class BlockSpecs:
    """All block specs a plan applies: the model's declarations plus the repo defaults."""

    residual: tuple[ResidualBlockSpec, ...] = ()
    ese: tuple[ESEBlockSpec, ...] = ()

    def __add__(self, other: BlockSpecs) -> BlockSpecs:
        return BlockSpecs(residual=self.residual + other.residual, ese=self.ese + other.ese)


def default_block_specs() -> BlockSpecs:
    """Block specs for the blocks this repo defines itself.

    ``SparseBasicBlock`` lives next to spconv; when spconv is not installed no such block
    can exist in any model, so the spec set is simply empty. VoVNet blocks are not defined
    in this repo: the model that brings them declares their specs.
    """
    try:
        from autoware_ml.models.detection3d.encoders.sparse import SparseBasicBlock
    except ImportError:  # pragma: no cover — spconv-less environments
        return BlockSpecs()
    return BlockSpecs(
        residual=(
            ResidualBlockSpec(
                SparseBasicBlock,
                QuantSparseBasicBlock,
                share_from=("conv1",),
                fresh_if_downsample=True,
            ),
        )
    )


@dataclass(frozen=True)
class RecipeContext:
    """Everything an attacher needs besides the model — built once per ``prepare``.

    Attributes:
        precision: Target precision of every quantizer a recipe creates.
        calibrator: Activation calibrator kind (``"histogram"`` / ``"max"``).
        roots: Dotted names of the quantized submodules (the recipe scope).
        skip_names: Expanded ``skip_quantize`` set. Honored by ``maxpool`` only: the block
            recipes convert inside skipped subtrees too (so the state_dict layout never
            depends on ``skip_quantize``) and their quantizers are disabled after load.
        specs: Block specs to match (model declarations + repo defaults).
        on_apply: Optional placement recording hook.
    """

    precision: Precision
    calibrator: str
    roots: tuple[str, ...]
    skip_names: frozenset[str] = frozenset()
    specs: BlockSpecs = BlockSpecs()
    on_apply: OnApply | None = None

    def new_input_quantizer(self) -> TensorQuantizer:
        """A fresh ``TensorQuantizer`` on the conv-input activation descriptor.

        Recipes use the same descriptor as the conv/linear input quantizers
        (:func:`~autoware_ml.quantization.core.descriptors.input_desc`), so residual / pool /
        eSE quantizers calibrate consistently with the layers around them.
        """
        return TensorQuantizer(input_desc(self.precision, self.calibrator))

    def record(self, module: str, transform: str, reason: str, detail: str) -> None:
        """Forward one placement decision to ``on_apply`` (no-op without a hook)."""
        if self.on_apply is not None:
            self.on_apply(module, transform, reason, detail)


# --------------------------------------------------------------------------- matching


def _submodule_by_path(module: nn.Module, path: str) -> nn.Module | None:
    """Resolve a dotted path relative to ``module`` (digits index into containers)."""
    current: nn.Module | None = module
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
) -> tuple[TensorQuantizer | None, str]:
    """Pick the residual quantizer for a matched block per its spec.

    Returns:
        ``(shared_quantizer, how)`` — the quantizer to reuse (``None`` means create a
        fresh one) and a human-readable description for the placement record.
    """
    if spec.fresh_if_downsample and getattr(module, "downsample", None) is not None:
        return None, "fresh (block has a downsample branch)"
    for path in spec.share_from:
        submodule = _submodule_by_path(module, path)
        quantizer = getattr(submodule, "input_quantizer", None) if submodule is not None else None
        if isinstance(quantizer, TensorQuantizer):
            return quantizer, f"shared from {path}.input_quantizer"
    return None, "fresh (no shareable input quantizer)"


def _in_scope(name: str, roots: tuple[str, ...]) -> bool:
    """Whether dotted module ``name`` lies inside one of the quantized submodule ``roots``."""
    return any(name == root or name.startswith(root + ".") for root in roots)


def _convertible_blocks(
    model: nn.Module, roots: tuple[str, ...], specs: tuple[_BlockSpec, ...]
) -> Iterator[tuple[str, nn.Module, _BlockSpec]]:
    """Yield ``(name, module, spec)`` for every not-yet-converted block under ``roots`` a spec matches."""
    for name, module in list(model.named_modules()):
        if isinstance(module, QuantModule) or not _in_scope(name, roots):
            continue  # already converted (a leaf Conv/Linear, or a block from a previous prepare)
        for spec in specs:
            if isinstance(module, spec.block_cls):
                yield name, module, spec
                break


# --------------------------------------------------------------------------- recipes


def attach_residual_add_recipe(model: nn.Module, ctx: RecipeContext) -> int:
    """Convert every matched residual block under ``ctx.roots`` and give it its residual-branch Q/DQ.

    Follows lidar-ai-solution (CUDA-BEVFusion): quantize only the identity branch, not the
    conv-path output, so TensorRT fuses Conv+Add. Blocks inside ``skip_quantize`` subtrees
    are converted too (see :class:`RecipeContext`).

    Args:
        model: Model whose residual blocks get the recipe.
        ctx: Recipe context; only :attr:`BlockSpecs.residual` of its specs is read.

    Returns:
        Number of blocks converted.
    """
    count = 0
    for name, module, spec in _convertible_blocks(model, ctx.roots, ctx.specs.residual):
        assert isinstance(spec, ResidualBlockSpec)
        spec.ensure_registered()
        original = type(module).__name__
        QuantBlockRegistry.convert(module)

        if spec.osa_concat:
            # One Q per Concat skip input (block input + every layer output but the last).
            n_inputs = len(module.layers)
            module.add_module(
                "concat_input_quantizers",
                nn.ModuleList([ctx.new_input_quantizer() for _ in range(n_inputs)]),
            )
            how = f"concat_input_quantizers[{n_inputs}]; " + (
                "identity=True: [0] is the single Q at the block input (no residual quantizer)"
                if getattr(module, "identity", False)
                else "identity=False: no residual Add"
            )
        else:
            shared, how = _resolve_residual_quantizer(module, spec)
            if shared is None:
                module.add_module("residual_quantizer", ctx.new_input_quantizer())
            else:
                # A TensorQuantizer cannot be the child of two parents: bind the shared one as
                # a plain attribute. The quantized forward still calls it, so ONNX tracing sees
                # the Q/DQ.
                object.__setattr__(module, "residual_quantizer", shared)
            how = f"residual_quantizer: {how}"
        count += 1
        ctx.record(
            name,
            "convert_block",
            f"recipe 'residual_add': matched {spec.block_cls.__name__}",
            f"{original} -> {type(module).__name__}; {how}",
        )
    if count:
        logger.info("Converted %d residual blocks (residual_add recipe)", count)
    return count


def attach_ese_recipe(model: nn.Module, ctx: RecipeContext) -> int:
    """Convert every matched eSE block under ``ctx.roots``: single Q at the input + gate quantizer.

    Both quantizers are fresh submodules (their ``amax`` lands in the state_dict). Blocks
    inside ``skip_quantize`` subtrees are converted too (see :class:`RecipeContext`).

    Args:
        model: Model whose eSE blocks get the recipe.
        ctx: Recipe context; only :attr:`BlockSpecs.ese` of its specs is read.

    Returns:
        Number of blocks converted.
    """
    count = 0
    for name, module, spec in _convertible_blocks(model, ctx.roots, ctx.specs.ese):
        spec.ensure_registered()
        original = type(module).__name__
        QuantBlockRegistry.convert(module)
        module.add_module("pool_input_quantizer", ctx.new_input_quantizer())
        module.add_module("mul_gate_quantizer", ctx.new_input_quantizer())
        count += 1
        ctx.record(
            name,
            "convert_block",
            f"recipe 'ese': matched {spec.block_cls.__name__}",
            f"{original} -> {type(module).__name__}; pool_input_quantizer + mul_gate_quantizer "
            "(single Q at the eSE input, both Mul operands INT8)",
        )
    if count:
        logger.info("Converted %d eSE blocks (ese recipe)", count)
    return count


def attach_maxpool_recipe(model: nn.Module, ctx: RecipeContext) -> int:
    """Replace every ``nn.MaxPool2d`` under ``ctx.roots`` with ``QuantBeforePool(quantizer, pool)``.

    Adds Q/DQ on the pool input so MaxPool runs on INT8 in the TensorRT graph. Honors
    ``ctx.skip_names`` (boundary-safe subtree match) because wrapping changes the module tree.

    Args:
        model: Model whose MaxPool2d modules get wrapped.
        ctx: Recipe context (scope and ``skip_names``; specs unused).

    Returns:
        Number of MaxPool2d modules wrapped.
    """
    name_to_module = dict(model.named_modules())
    to_replace = []  # (full_name, parent_module, child_name, pool_module)
    for name, module in model.named_modules():
        if not isinstance(module, nn.MaxPool2d) or isinstance(module, QuantBeforePool):
            continue
        # Boundary-safe subtree matches: "backbone.block1" must not match "backbone.block10".
        if not _in_scope(name, ctx.roots) or _in_scope(name, tuple(ctx.skip_names)):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = name_to_module.get(parent_name) if parent_name else model
        if parent is None or isinstance(parent, QuantBeforePool):
            continue  # already wrapped (idempotent on a second prepare)
        to_replace.append((name, parent, child_name, module))

    for full_name, parent, child_name, pool_module in to_replace:
        parent.add_module(child_name, QuantBeforePool(ctx.new_input_quantizer(), pool_module))
        ctx.record(
            full_name,
            "wrap_module",
            "recipe 'maxpool': MaxPool2d input Q/DQ",
            "MaxPool2d -> QuantBeforePool",
        )
    if to_replace:
        logger.info("Attached Q/DQ before %d MaxPool2d modules", len(to_replace))
    return len(to_replace)


#: Uniform attacher signature the plan calls.
RecipeAttacher = Callable[[nn.Module, RecipeContext], int]

#: The recipe registry: recipe name -> attacher. ``QuantizationPlan.prepare`` applies
#: these in ``config.VALID_RECIPES`` order (``residual_add`` before ``ese`` so a VoVNet OSA
#: block is converted before the eSE nested inside it); adding a recipe means one entry
#: here plus its name in ``VALID_RECIPES`` (``plan.py`` verifies the two sets match at import).
RECIPE_ATTACHERS: dict[str, RecipeAttacher] = {
    "residual_add": attach_residual_add_recipe,
    "ese": attach_ese_recipe,
    "maxpool": attach_maxpool_recipe,
}
