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

"""Quantization plans: declare rules, apply transforms, record every decision.

This module is the single interface between deployment stages and quantization
(see ``autoware_ml/quantization/README.md`` for the design rationale):

- :class:`QuantRules` — a model's *declaration*: which top-level submodules get
  which module kinds replaced, and which architecture recipes apply. One model
  = one rules object next to the model (e.g. CenterPoint's in
  ``models/detection3d/main_modules/centerpoint/quantization.py``).
- :class:`QuantizationPlan` — rules + parsed config, bound together. Its
  :meth:`~QuantizationPlan.prepare` fuses BN and inserts Q/DQ **and records
  every placement decision it makes**.
- :class:`PlacementDecision` / :class:`PlacementRecord` — that record: which
  module, which transform, why, with what outcome. Serializable, so the
  quantize stage embeds it in the checkpoint and the loader verifies its own
  rebuilt module tree against it (:meth:`PlacementRecord.verify_matches`) — the
  same-plan-same-tree invariant becomes a machine check instead of discipline.

Transforms a placement record can contain (one entry kind each; see each
transform's home module for its mechanics):

- ``fuse_bn``        — Conv+BN weight fold, BN becomes ``nn.Identity``
  (:mod:`.core.fusion`).
- ``skip_quantize``  — a matched module and its whole subtree stay un-quantized
  (:func:`.core.replace.expand_skip_quantize`).
- ``replace_module`` — ``nn.Conv2d``/``nn.ConvTranspose2d``/``nn.Linear``
  converted in place into its modelopt quantized class (:mod:`.core.replace`).
- ``wrap_module``    — a pool wrapped so Q/DQ lands on its input
  (:class:`~.recipes.quant_blocks.QuantBeforePool`).
- ``convert_block``  — a residual block converted in place into its ``Quant*`` block
  class, with a ``residual_quantizer`` attached (:mod:`.recipes.attach`).

Stage code (the quantize entrypoints and the deploy loader) holds a plan and calls
``prepare`` — it never sees quantization internals. The record covers module-tree
*construction* only; the post-load ``disable_quantizers_in`` pass changes no
``state_dict`` keys and is deliberately outside it.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from autoware_ml.quantization.config import (
    VALID_MODULE_KINDS,
    VALID_RECIPES,
    Precision,
    QuantizationConfig,
)
from autoware_ml.quantization.core.fusion import find_conv_bn_pairs, fuse_model_bn
from autoware_ml.quantization.core.replace import (
    expand_skip_quantize,
    match_skip_quantize_roots,
    replace_quantizable_modules,
)
from autoware_ml.quantization.recipes.attach import (
    RECIPE_ATTACHERS,
    BlockSpecs,
    ESEBlockSpec,
    RecipeContext,
    ResidualBlockSpec,
    default_block_specs,
)

logger = logging.getLogger(__name__)

# Adding a recipe means registering its attacher AND listing it in VALID_RECIPES
# (the canonical apply order); catch a mismatch at import time, not mid-prepare.
if set(RECIPE_ATTACHERS) != set(VALID_RECIPES):
    raise RuntimeError(
        f"Recipe registry drift: RECIPE_ATTACHERS={sorted(RECIPE_ATTACHERS)} vs "
        f"VALID_RECIPES={sorted(VALID_RECIPES)}. Register every recipe in both places."
    )


@dataclass(frozen=True)
class PlacementDecision:
    """One recorded quantization decision: which module, which transform, why.

    Attributes:
        module: Dotted module name from the model root.
        transform: One of the transform names (module docstring).
        reason: Why the transform applies (submodule rule / pattern / recipe match).
        detail: Outcome detail (classes swapped, quantizer shared from where, ...).
    """

    module: str
    transform: str
    reason: str
    detail: str = ""


class PlacementRecord:
    """The recorded outcome of one plan ``prepare``: an ordered list of decisions.

    Two records are considered equal when they contain the same decision
    *multiset* — apply order does not affect the resulting module tree, so
    :meth:`diff` is order-insensitive on purpose.
    """

    def __init__(self, decisions: Sequence[PlacementDecision] = ()) -> None:
        self.decisions: list[PlacementDecision] = list(decisions)

    def add(self, module: str, transform: str, reason: str, detail: str = "") -> None:
        """Append one decision."""
        self.decisions.append(
            PlacementDecision(module=module, transform=transform, reason=reason, detail=detail)
        )

    def __len__(self) -> int:
        return len(self.decisions)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict.

        Versioning lives one level up: the embedding checkpoint payload carries the
        single format version for the whole ``quantization`` entry.
        """
        return {"decisions": [asdict(decision) for decision in self.decisions]}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> PlacementRecord:
        """Deserialize from :meth:`to_json_dict` output."""
        return cls([PlacementDecision(**entry) for entry in data.get("decisions", [])])

    def diff(
        self, other: PlacementRecord
    ) -> tuple[list[PlacementDecision], list[PlacementDecision]]:
        """Compare decision multisets (order-insensitive).

        Returns:
            ``(only_in_self, only_in_other)`` — both empty when the records
            describe the same tree construction.
        """
        mine = Counter(self.decisions)
        theirs = Counter(other.decisions)
        only_in_self = sorted((mine - theirs).elements(), key=lambda d: (d.module, d.transform))
        only_in_other = sorted((theirs - mine).elements(), key=lambda d: (d.module, d.transform))
        return only_in_self, only_in_other

    def verify_matches(self, produced: PlacementRecord, source: str) -> None:
        """Raise unless ``self`` (a rebuilt tree) describes the same construction as ``produced``.

        Any drift means the ``load_state_dict`` that follows would silently mis-map
        calibrated weights, so this raises instead.

        Args:
            produced: The record the quantize stage produced (embedded in the checkpoint).
            source: Where ``produced`` came from, for the error message.

        Raises:
            RuntimeError: When the decision multisets differ.
        """
        only_rebuilt, only_produced = self.diff(produced)
        if only_rebuilt or only_produced:
            preview = "\n  ".join(
                [f"rebuilt only: {d}" for d in only_rebuilt[:5]]
                + [f"saved only: {d}" for d in only_produced[:5]]
            )
            raise RuntimeError(
                f"Quantized tree drift: the rebuilt tree does not match the placement record from "
                f"{source} ({len(only_rebuilt)} decision(s) only here, {len(only_produced)} only in "
                f"the saved record). Loading would silently mis-map calibrated weights. "
                f"First differences:\n  {preview}"
            )
        logger.info(
            "Placement record verified: rebuilt tree matches %s (%d decisions).",
            source,
            len(produced),
        )

    def log_summary(self) -> None:
        """Log one line per transform kind with its decision count."""
        counts = Counter(decision.transform for decision in self.decisions)
        summary = ", ".join(f"{transform}={count}" for transform, count in sorted(counts.items()))
        logger.info(
            "[quant-plan] placement record: %d decisions (%s)", len(self), summary or "empty"
        )

    def log_table(self) -> None:
        """Log the full per-module placement table (the dry-run report)."""
        logger.info("[quant-plan] placement record (%d decisions):", len(self))
        logger.info("    %-52s %-16s %s", "module", "transform", "reason / detail")
        for decision in self.decisions:
            note = f"{decision.reason}" + (f" — {decision.detail}" if decision.detail else "")
            logger.info("    %-52s %-16s %s", decision.module, decision.transform, note)


@dataclass(frozen=True)
class QuantRules:
    """A model's quantization declaration: what gets quantized, which recipes apply.

    Per-submodule module kinds are architecture facts and belong in the model's
    rules, not in config; config (``skip_quantize`` / ``disable_recipes``) only
    subtracts from what the rules declare.

    Attributes:
        quantize_submodules: Top-level model attribute name -> module kinds to
            replace inside it. Two spellings:

            - ``("conv", "linear")`` — every kind at the config's
              ``default_precision`` (the common case);
            - ``{"conv": "int8", "linear": "fp8"}`` — per-kind precision, for a model
              whose layer families tolerate different precisions. A kind mapped to
              ``None`` follows ``default_precision``.

            A submodule absent on the model is skipped silently, so one rules object
            can serve model variants.
        recipes: Architecture recipes to attach (subset of :data:`VALID_RECIPES`;
            default: all). Recipes are class-gated: each fires only where the
            architecture has that block, so zero matches are normal. Applied in
            canonical :data:`VALID_RECIPES` order regardless of declaration order.
            Recipe quantizers always follow ``default_precision`` (they are
            activation-side glue shared with the conv inputs, not per-kind weights).
        residual_blocks: Extra :class:`~.recipes.attach.ResidualBlockSpec` rows for the
            ``residual_add`` recipe — the residual blocks this model owns (mmpretrain
            ``ConvNeXtBlock``, VoVNet ``_OSA_module`` ...), matched before the repo-wide
            defaults (:func:`~.recipes.attach.default_block_specs`).
        ese_blocks: :class:`~.recipes.attach.ESEBlockSpec` rows for the ``ese`` recipe
            (VoVNet ``eSEModule``).
    """

    quantize_submodules: Mapping[str, tuple[str, ...] | Mapping[str, str | None]]
    recipes: tuple[str, ...] = VALID_RECIPES
    residual_blocks: tuple[ResidualBlockSpec, ...] = ()
    ese_blocks: tuple[ESEBlockSpec, ...] = ()

    def __post_init__(self) -> None:
        for submodule_name, kinds in self.quantize_submodules.items():
            unknown = set(kinds) - set(VALID_MODULE_KINDS)
            if unknown:
                raise ValueError(
                    f"QuantRules submodule {submodule_name!r} declares unknown module kind(s) "
                    f"{sorted(unknown)}; valid kinds: {list(VALID_MODULE_KINDS)}."
                )
            if isinstance(kinds, Mapping):
                for precision_name in kinds.values():
                    if precision_name is not None:
                        Precision(precision_name)  # raises ValueError on an unknown precision
        unknown_recipes = set(self.recipes) - set(VALID_RECIPES)
        if unknown_recipes:
            raise ValueError(
                f"QuantRules declares unknown recipe(s) {sorted(unknown_recipes)}; "
                f"valid recipes: {list(VALID_RECIPES)}."
            )

    def resolved_kinds(
        self, submodule_name: str, default_precision: Precision
    ) -> Mapping[str, Precision]:
        """The submodule's kinds with every precision resolved.

        Args:
            submodule_name: Key of :attr:`quantize_submodules`.
            default_precision: Config precision used for kinds without their own.
        """
        kinds = self.quantize_submodules[submodule_name]
        if isinstance(kinds, Mapping):
            return {
                kind: (Precision(name) if name is not None else default_precision)
                for kind, name in kinds.items()
            }
        return {kind: default_precision for kind in kinds}


class QuantizationPlan:
    """Rules + config bound together; ``prepare`` builds the tree and the record.

    The same-plan-everywhere invariant: the quantize stage (PTQ / QAT) and the
    deploy loader all build the quantized module tree by calling the *same*
    model-provided plan's :meth:`prepare`, so the calibrated ``state_dict`` and
    the deploy ``load_state_dict`` line up by construction — and the placement
    record lets the loader *verify* that instead of trusting it.

    Args:
        rules: The model's :class:`QuantRules` declaration.
        config: Parsed ``quantization`` config block.
    """

    def __init__(self, rules: QuantRules, config: QuantizationConfig) -> None:
        self.rules = rules
        self.config = config
        #: Placement record of the last :meth:`prepare` call (``None`` until then).
        self.placement_record: PlacementRecord | None = None

    def prepare(self, model: Any) -> Any:
        """Fuse BN and insert Q/DQ in place, recording every decision.

        Steps, in order (each earlier step can change module names/types the
        later steps see, so the order is part of the contract):

        1. BN fusion across the whole model (when ``config.fuse_bn``) — fusing
           an un-quantized module's BN is an inference identity but *changes
           state_dict keys*, so PTQ and deploy must fuse the exact same set;
           ``skip_quantize`` only subtracts from the *quantized* set.
        2. ``skip_quantize`` resolution into a concrete skip set (subtree match).
        3. Module replacement per :attr:`rules.quantize_submodules` (minus the
           skip set).
        4. Architecture recipes in canonical order, minus
           ``config.disable_recipes``, scoped to the submodules of step 3.

        The activation calibrator kind (histogram vs max) follows
        ``config.calibration``; it changes no state_dict key, so a checkpoint
        calibrated with one method loads into a tree prepared for another.

        Returns:
            ``model`` (mutated in place) for chaining convenience.
        """
        record = PlacementRecord()
        if self.config.fuse_bn:
            self._fuse_bn(model, record)
        skip_names = self._resolve_skip_quantize(model, record)
        roots = self._replace_modules(model, skip_names, record)
        self._apply_recipes(model, roots, skip_names, record)
        self.placement_record = record
        record.log_summary()
        return model

    # ------------------------------------------------------------------ prepare steps

    @staticmethod
    def _fuse_bn(model: Any, record: PlacementRecord) -> None:
        """Step 1: fold every adjacent Conv+BN pair (whole model, independent of skip_quantize)."""
        model.eval()
        for conv_name, bn_name in find_conv_bn_pairs(model):
            record.add(
                conv_name,
                "fuse_bn",
                reason="adjacent Conv+BN pair",
                detail=f"folds {bn_name}; BN becomes Identity",
            )
        fuse_model_bn(model)

    def _resolve_skip_quantize(self, model: Any, record: PlacementRecord) -> set[str]:
        """Step 2: record the matched skip roots and return the expanded skip set."""
        for pattern, root_name in match_skip_quantize_roots(model, self.config.skip_quantize):
            record.add(
                root_name,
                "skip_quantize",
                reason=f"skip_quantize pattern {pattern!r}",
                detail="module and all descendants stay un-quantized",
            )
        return expand_skip_quantize(model, self.config.skip_quantize, log=False)

    def _replace_modules(
        self, model: Any, skip_names: set[str], record: PlacementRecord
    ) -> tuple[str, ...]:
        """Step 3: convert the declared module kinds under each declared submodule.

        Returns:
            The declared submodule names present on the model — the recipe scope.
        """
        default_precision = self.config.default_precision
        calibrator = self.config.calibration.activation_calibrator
        roots: list[str] = []
        for submodule_name in self.rules.quantize_submodules:
            submodule = getattr(model, submodule_name, None)
            if submodule is None:
                continue  # one rules object serves model variants
            roots.append(submodule_name)
            by_precision: dict[Precision, list[str]] = {}
            for kind, precision in self.rules.resolved_kinds(
                submodule_name, default_precision
            ).items():
                by_precision.setdefault(precision, []).append(kind)
            for precision, kinds in by_precision.items():
                reason = f"submodule rule: {submodule_name} ({', '.join(kinds)})"
                # The precision appears in the recorded detail only when it deviates from
                # the default, so records of single-precision checkpoints stay identical.
                suffix = "" if precision is default_precision else f" @{precision.value}"

                def on_replace(name: str, original: str, new: Any, reason=reason, suffix=suffix):
                    record.add(
                        name,
                        "replace_module",
                        reason=reason,
                        detail=f"{original} -> {type(new).__name__}{suffix}",
                    )

                replace_quantizable_modules(
                    submodule,
                    kinds=tuple(kinds),
                    skip_names=skip_names,
                    prefix=submodule_name,
                    on_replace=on_replace,
                    precision=precision,
                    calibrator=calibrator,
                )
        return tuple(roots)

    def _apply_recipes(
        self, model: Any, roots: tuple[str, ...], skip_names: set[str], record: PlacementRecord
    ) -> None:
        """Step 4: architecture recipes in canonical order, scoped to ``roots``."""
        context = RecipeContext(
            precision=self.config.default_precision,
            calibrator=self.config.calibration.activation_calibrator,
            roots=roots,
            skip_names=frozenset(skip_names),
            specs=BlockSpecs(
                residual=tuple(self.rules.residual_blocks), ese=tuple(self.rules.ese_blocks)
            )
            + default_block_specs(),
            on_apply=record.add,
        )
        disabled = set(self.config.disable_recipes)
        for recipe_name in VALID_RECIPES:
            if recipe_name in self.rules.recipes and recipe_name not in disabled:
                RECIPE_ATTACHERS[recipe_name](model, context)
