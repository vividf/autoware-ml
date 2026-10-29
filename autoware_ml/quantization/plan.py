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


"""Build a model's quantized tree and record every placement decision.

The record covers only what shapes the tree: disabling the quantizers of the ``skip_quantize``
subtrees after loading changes no ``state_dict`` key and is not recorded.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from torch import nn

from autoware_ml.quantization.config import Precision, QuantizationConfig
from autoware_ml.quantization.core.replace import (
    expand_skip_quantize,
    match_skip_quantize_roots,
    replace_quantizable_modules,
)
from autoware_ml.quantization.rules import QuantRules
from autoware_ml.utils.bn_fusion import fuse_model_bn

if TYPE_CHECKING:
    from autoware_ml.models.module_base_model import ModuleBaseModel

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


class PlacementRecord:
    """The decisions of one :func:`prepare_quantized_tree`, in the order they were made."""

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

    def verify_matches(self, produced: PlacementRecord, source: str) -> None:
        """Raise unless this record, of a rebuilt tree, matches ``produced`` structurally.

        Only ``(module, transform)`` is compared, in any order: a reworded ``reason``, or a
        modelopt release that renamed a quantized class in ``detail``, does not invalidate a
        checkpoint. A decision made twice is a difference.
        """
        mine = Counter((d.module, d.transform) for d in self.decisions)
        theirs = Counter((d.module, d.transform) for d in produced.decisions)
        only_rebuilt = sorted((mine - theirs).elements())
        only_produced = sorted((theirs - mine).elements())
        if only_rebuilt or only_produced:
            preview = "\n  ".join(
                [f"rebuilt only: {module} ({transform})" for module, transform in only_rebuilt[:5]]
                + [f"saved only: {module} ({transform})" for module, transform in only_produced[:5]]
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


@dataclass(frozen=True)
class PreparedTree:
    """What :func:`prepare_quantized_tree` did to the model.

    Attributes:
        placement_record: Every decision, for the checkpoint and for the loader's check.
        skip_names: The ``skip_quantize`` modules with their descendants, named on the
            prepared tree: the quantizers under them are disabled after calibration.
    """

    placement_record: PlacementRecord
    skip_names: frozenset[str]


def prepare_quantized_tree(
    model: ModuleBaseModel | nn.Module,
    config: QuantizationConfig,
    rules: QuantRules | None = None,
) -> PreparedTree:
    """Fuse BN and insert the quantizers in place, recording every decision.

    The quantize stage, QAT and the loader build the tree through this one function, so the
    calibrated ``state_dict`` keys match on load; the placement record lets the loader check
    that. TensorRT folds BN into the convolution, so a weight quantizer must calibrate on the
    folded weight: the fold is an inference identity and covers the whole model,
    ``skip_quantize`` subtrees included. The calibration method changes no key: a checkpoint
    loads into a tree prepared for another method.

    Args:
        model: The model to prepare, in place.
        config: The ``quantization`` section.
        rules: The model's declaration; ``model.build_quantization_rules()`` by default.

    Raises:
        NotImplementedError: When the model declares no quantization rules.
        ValueError: When the rules name a submodule the model does not have, or a
            ``skip_quantize`` pattern matches nothing.
    """
    if rules is None:
        rules = model.build_quantization_rules()
        if rules is None:
            raise NotImplementedError(
                f"{type(model).__name__} declares no quantization rules "
                "(build_quantization_rules returned None), so it cannot be quantized."
            )
    record = PlacementRecord()
    model.eval()
    for conv_name, bn_name in fuse_model_bn(model):
        record.add(
            conv_name,
            "fuse_bn",
            reason="Conv+BN pair (sequential container or declared bn_fusion_pairs)",
            detail=f"folds {bn_name}; BN becomes Identity",
        )
    for pattern, root_name in match_skip_quantize_roots(model, config.skip_quantize):
        record.add(
            root_name,
            "skip_quantize",
            reason=f"skip_quantize pattern {pattern!r}",
            detail="module and all descendants stay un-quantized",
        )
    skip_names = expand_skip_quantize(model, config.skip_quantize, log=False)
    _replace_modules(model, rules, config, skip_names, record)
    # Named on the prepared tree: the replacement adds the quantizer submodules, so a glob
    # can match differently before and after.
    skip_names = expand_skip_quantize(model, config.skip_quantize, log=False)
    record.log_summary()
    return PreparedTree(placement_record=record, skip_names=frozenset(skip_names))


def _replace_modules(
    model: nn.Module,
    rules: QuantRules,
    config: QuantizationConfig,
    skip_names: set[str],
    record: PlacementRecord,
) -> None:
    calibrator = config.calibration.activation_calibrator
    for submodule_name in rules.quantize_submodules:
        try:
            submodule = model.get_submodule(submodule_name)
        except AttributeError as error:
            raise ValueError(
                f"{type(model).__name__} declares the quantizable submodule "
                f"'{submodule_name}' in its QuantRules, but the model has no such "
                "submodule. The rules name dotted paths of the module tree."
            ) from error
        by_precision: dict[Precision, list[str]] = {}
        for kind, precision in rules.resolved_kinds(
            submodule_name, config.default_precision
        ).items():
            by_precision.setdefault(precision, []).append(kind)
        for precision, kinds in by_precision.items():
            reason = f"submodule rule: {submodule_name} ({', '.join(kinds)})"
            suffix = "" if precision is config.default_precision else f" @{precision.value}"

            def on_replace(name: str, original: str, new: nn.Module, reason=reason, suffix=suffix):
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
