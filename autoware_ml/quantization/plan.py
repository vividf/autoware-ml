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
(see ``docs/user-guide/quantization.md`` for the design rationale):

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
  (:mod:`autoware_ml.utils.bn_fusion`).
- ``skip_quantize``  — a matched module and its whole subtree stay un-quantized
  (:func:`.core.replace.expand_skip_quantize`).
- ``replace_module`` — ``nn.Conv2d``/``nn.ConvTranspose2d``/``nn.Linear``/sparse conv
  converted in place into its modelopt quantized class (:mod:`.core.replace`).

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

from autoware_ml.quantization.config import Precision, QuantizationConfig
from autoware_ml.quantization.core.replace import (
    expand_skip_quantize,
    match_skip_quantize_roots,
    replace_quantizable_modules,
)
from autoware_ml.quantization.rules import QuantRules
from autoware_ml.utils.bn_fusion import find_conv_bn_pairs, fuse_model_bn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlacementDecision:
    """One recorded quantization decision: which module, which transform, why.

    Attributes:
        module: Dotted module name from the model root.
        transform: One of the transform names (module docstring).
        reason: Why the transform applies (submodule rule / skip pattern).
        detail: Outcome detail (classes swapped, quantizer shared from where, ...).
    """

    module: str
    transform: str
    reason: str
    detail: str = ""

    @property
    def structure(self) -> tuple[str, str]:
        """The part of the decision that shapes the module tree: ``(module, transform)``.

        ``reason`` and ``detail`` are prose written for humans reading the record. They
        explain a decision; they cannot change it. Equality checks that gate *loading* use
        this instead, so rewording a reason — or modelopt renaming a quantized class, which
        lands verbatim in ``detail`` — does not condemn every checkpoint calibrated before
        the rewording.
        """
        return (self.module, self.transform)


class PlacementRecord:
    """The recorded outcome of one plan ``prepare``: an ordered list of decisions.

    Two records are considered equal when they contain the same decision
    *multiset* — apply order does not affect the resulting module tree, so
    :meth:`diff` is order-insensitive on purpose. Equality that gates loading is
    narrower still: only ``(module, transform)`` (see :attr:`PlacementDecision.structure`).
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

        Versioning lives one level up: the embedding checkpoint payload
        (:class:`~autoware_ml.quantization.checkpoint.QuantizationDescription`) carries
        the single ``format`` version for the whole ``quantization`` entry.
        """
        return {"decisions": [asdict(decision) for decision in self.decisions]}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> PlacementRecord:
        """Deserialize from :meth:`to_json_dict` output."""
        return cls([PlacementDecision(**entry) for entry in data.get("decisions", [])])

    def diff(
        self, other: PlacementRecord, *, structural_only: bool = False
    ) -> tuple[list[PlacementDecision], list[PlacementDecision]]:
        """Compare decision multisets (order-insensitive).

        Args:
            other: The record to compare against.
            structural_only: Match on ``(module, transform)`` alone, ignoring the prose
                ``reason``/``detail``. This is the comparison that gates loading
                (:meth:`verify_matches`); the full comparison is for reporting, where the
                prose is the useful part.

        Returns:
            ``(only_in_self, only_in_other)`` — both empty when the records
            describe the same tree construction.
        """
        key = (lambda d: d.structure) if structural_only else (lambda d: d)
        mine = Counter(key(decision) for decision in self.decisions)
        theirs = Counter(key(decision) for decision in other.decisions)
        by_key: dict[Any, list[PlacementDecision]] = {}
        for decision in [*self.decisions, *other.decisions]:
            by_key.setdefault(key(decision), []).append(decision)

        def expand(counter: Counter) -> list[PlacementDecision]:
            return sorted(
                (by_key[k][0] for k in counter.elements()),
                key=lambda d: (d.module, d.transform),
            )

        return expand(mine - theirs), expand(theirs - mine)

    def verify_matches(self, produced: PlacementRecord, source: str) -> None:
        """Raise unless ``self`` (a rebuilt tree) describes the same construction as ``produced``.

        Any drift means the ``load_state_dict`` that follows would silently mis-map
        calibrated weights, so this raises instead. The comparison is structural —
        ``(module, transform)`` — because that is what decides the tree; the recorded
        prose is compared by nobody, so improving an explanation (or upgrading modelopt,
        whose class names land in ``detail``) never invalidates an existing checkpoint.

        Args:
            produced: The record the quantize stage produced (embedded in the checkpoint).
            source: Where ``produced`` came from, for the error message.

        Raises:
            RuntimeError: When the decision multisets differ.
        """
        only_rebuilt, only_produced = self.diff(produced, structural_only=True)
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

    @classmethod
    def for_model(cls, model: Any, config: QuantizationConfig) -> QuantizationPlan:
        """Bind a model's declared rules to a parsed ``quantization`` config.

        The one constructor every stage uses — PTQ, QAT and the deploy / test loader —
        so the same model and config always build the same quantized module tree.

        Raises:
            NotImplementedError: When the model declares no quantization rules
                (``build_quantization_rules()`` returns ``None``).
        """
        rules = model.build_quantization_rules()
        if rules is None:
            raise NotImplementedError(
                f"{type(model).__name__} declares no quantization rules "
                "(build_quantization_rules returned None), so it cannot be quantized."
            )
        return cls(rules=rules, config=config)

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

        The activation calibrator kind (histogram vs max) follows
        ``config.calibration``; it changes no state_dict key, so a checkpoint
        calibrated with one method loads into a tree prepared for another.

        Returns:
            ``model`` (mutated in place) for chaining convenience.
        """
        record = PlacementRecord()
        if self.config.fuse_bn:
            self._fuse_bn(model, record)
        elif find_conv_bn_pairs(model):
            # Export always folds BN into the deployed graph (deployment/export_specs.py);
            # weight quantizers calibrated on unfolded weights would then ship scales that
            # do not match the folded weights. Nothing needs unfolded quantization.
            raise ValueError(
                "quantization.fuse_bn=false on a model with Conv+BN pairs: export folds "
                "BatchNorm into the deployed graph, so the calibrated weight scales would not "
                "match the exported weights. Keep fuse_bn=true."
            )
        skip_names = self._resolve_skip_quantize(model, record)
        self._replace_modules(model, skip_names, record)
        self.placement_record = record
        record.log_summary()
        return model

    # ------------------------------------------------------------------ prepare steps

    @staticmethod
    def _fuse_bn(model: Any, record: PlacementRecord) -> None:
        """Step 1: fold every structurally proven Conv+BN pair (whole model, regardless of skip)."""
        model.eval()
        for conv_name, bn_name in find_conv_bn_pairs(model):
            record.add(
                conv_name,
                "fuse_bn",
                reason="Conv+BN pair (sequential container or declared bn_fusion_pairs)",
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

    def _replace_modules(self, model: Any, skip_names: set[str], record: PlacementRecord) -> None:
        """Step 3: convert the declared module kinds under each declared submodule."""
        default_precision = self.config.default_precision
        calibrator = self.config.calibration.activation_calibrator
        for submodule_name in self.rules.quantize_submodules:
            submodule = getattr(model, submodule_name, None)
            if submodule is None:
                continue  # one rules object serves model variants
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
