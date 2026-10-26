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

"""Quantization plans: build a model's quantized tree and record every placement decision.

The record covers only what shapes the tree: disabling the quantizers of the ``skip_quantize``
subtrees after loading changes no ``state_dict`` key and is not recorded.
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
from autoware_ml.utils.bn_fusion import fuse_model_bn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlacementDecision:
    """One recorded quantization decision: which module, which transform, why.

    Attributes:
        module: Dotted module name from the model root.
        transform: ``fuse_bn`` (Conv+BN fold, the BN becomes ``nn.Identity``),
            ``skip_quantize`` (the module's subtree stays un-quantized) or ``replace_module``
            (converted in place to its modelopt quantized class).
        reason: Why the transform applies.
        detail: What it did, e.g. the classes swapped.
    """

    module: str
    transform: str
    reason: str
    detail: str = ""

    @property
    def structure(self) -> tuple[str, str]:
        """``(module, transform)``: the part of the decision that shapes the module tree.

        Loading compares only this, so rewording a ``reason``, or modelopt renaming a
        quantized class that lands in ``detail``, does not invalidate existing checkpoints.
        """
        return (self.module, self.transform)


class PlacementRecord:
    """The decisions of one :meth:`QuantizationPlan.prepare`, in the order they were made.

    Comparisons ignore that order: it does not change the resulting tree.
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
        """Serialize to a JSON-ready dict; the checkpoint payload carries the format version."""
        return {"decisions": [asdict(decision) for decision in self.decisions]}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> PlacementRecord:
        """Deserialize from :meth:`to_json_dict` output."""
        return cls([PlacementDecision(**entry) for entry in data.get("decisions", [])])

    def diff(
        self, other: PlacementRecord, *, structural_only: bool = False
    ) -> tuple[list[PlacementDecision], list[PlacementDecision]]:
        """Return ``(only_in_self, only_in_other)``, comparing the decisions as multisets.

        ``structural_only`` compares :attr:`PlacementDecision.structure` alone, as loading does.
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
        """Raise unless this record, of a rebuilt tree, matches ``produced`` structurally."""
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
        """Log the decision count of each transform on one line."""
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
    """A model's :class:`QuantRules` bound to a parsed ``quantization`` config.

    The quantize stage and the loader build the quantized tree through the same
    :meth:`prepare`, so the calibrated ``state_dict`` keys match on load; the placement record
    lets the loader check that.
    """

    def __init__(self, rules: QuantRules, config: QuantizationConfig) -> None:
        self.rules = rules
        self.config = config
        #: Placement record of the last :meth:`prepare` call (``None`` until then).
        self.placement_record: PlacementRecord | None = None

    @classmethod
    def for_model(cls, model: Any, config: QuantizationConfig) -> QuantizationPlan:
        """Bind ``model.build_quantization_rules()`` to ``config``."""
        rules = model.build_quantization_rules()
        if rules is None:
            raise NotImplementedError(
                f"{type(model).__name__} declares no quantization rules "
                "(build_quantization_rules returned None), so it cannot be quantized."
            )
        return cls(rules=rules, config=config)

    def prepare(self, model: Any) -> Any:
        """Fuse BN and insert Q/DQ in place, recording every decision; return ``model``.

        TensorRT folds BN into the convolution, so a weight quantizer must calibrate on the
        folded weight. The fold is an inference identity and covers the whole model,
        ``skip_quantize`` subtrees included; it changes ``state_dict`` keys, so the quantize
        stage and the loader must fold the same set. The calibration method changes no key: a
        checkpoint loads into a tree prepared for another method.
        """
        record = PlacementRecord()
        self._fuse_bn(model, record)
        skip_names = self._resolve_skip_quantize(model, record)
        self._replace_modules(model, skip_names, record)
        self.placement_record = record
        record.log_summary()
        return model

    @staticmethod
    def _fuse_bn(model: Any, record: PlacementRecord) -> None:
        model.eval()
        for conv_name, bn_name in fuse_model_bn(model):
            record.add(
                conv_name,
                "fuse_bn",
                reason="Conv+BN pair (sequential container or declared bn_fusion_pairs)",
                detail=f"folds {bn_name}; BN becomes Identity",
            )

    def _resolve_skip_quantize(self, model: Any, record: PlacementRecord) -> set[str]:
        """Record the matched skip roots and return the skip set with their subtrees."""
        for pattern, root_name in match_skip_quantize_roots(model, self.config.skip_quantize):
            record.add(
                root_name,
                "skip_quantize",
                reason=f"skip_quantize pattern {pattern!r}",
                detail="module and all descendants stay un-quantized",
            )
        return expand_skip_quantize(model, self.config.skip_quantize, log=False)

    def _replace_modules(self, model: Any, skip_names: set[str], record: PlacementRecord) -> None:
        default_precision = self.config.default_precision
        calibrator = self.config.calibration.activation_calibrator
        for submodule_name in self.rules.quantize_submodules:
            try:
                submodule = model.get_submodule(submodule_name)
            except AttributeError as error:
                raise ValueError(
                    f"{type(model).__name__} declares the quantizable submodule "
                    f"'{submodule_name}' in its QuantRules, but the model has no such "
                    "submodule. The rules name dotted paths of the module tree."
                ) from error
            by_precision: dict[Precision, list[str]] = {}
            for kind, precision in self.rules.resolved_kinds(
                submodule_name, default_precision
            ).items():
                by_precision.setdefault(precision, []).append(kind)
            for precision, kinds in by_precision.items():
                reason = f"submodule rule: {submodule_name} ({', '.join(kinds)})"
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
