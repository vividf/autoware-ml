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

"""The channel table: how a pruned architecture is described and rebuilt.

FastNAS prunes widths per layer (consecutive convolutions end up with different
``out_channels``), which no model constructor argument can express. Instead of teaching
every backbone/neck/head a per-layer width list, the pruned architecture is a plain
table ``{module_name: {shape_attr: value}}`` over the layer kinds whose parameter shapes
depend on channel counts (convolutions, linears, normalizations). The table is:

- **recorded** from a pruned model (:meth:`ChannelTable.record`),
- **embedded** in the checkpoint (:mod:`autoware_ml.pruning.checkpoint`),
- **applied** to a freshly instantiated config model before its weights load
  (:func:`apply_channel_table`): every listed module is rebuilt with the same class and
  hyperparameters and the new channel counts, so ``load_state_dict`` matches by
  construction — and the pruned tree is again made of ordinary ``nn.Conv2d`` &co, which
  the quantization plan, BN fusion, and ONNX export treat like any other.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd

logger = logging.getLogger(__name__)

#: Shape-defining constructor attributes per layer kind (checked in order; first match wins).
_SHAPE_ATTRS: tuple[tuple[type[nn.Module], tuple[str, ...]], ...] = (
    (_ConvNd, ("in_channels", "out_channels")),
    (nn.Linear, ("in_features", "out_features")),
    (_BatchNorm, ("num_features",)),
    (nn.GroupNorm, ("num_channels",)),
    (nn.LayerNorm, ("normalized_shape",)),
)


def shape_attrs_of(module: nn.Module) -> tuple[str, ...] | None:
    """The channel-shape attributes of ``module``, or ``None`` for a kind the table ignores."""
    for cls, attrs in _SHAPE_ATTRS:
        if isinstance(module, cls):
            return attrs
    return None


def _shape_of(module: nn.Module, attrs: Iterable[str]) -> dict[str, Any]:
    shape = {}
    for attr in attrs:
        value = getattr(module, attr)
        shape[attr] = list(value) if isinstance(value, (tuple, list, torch.Size)) else int(value)
    return shape


class ChannelTable:
    """``{module_name: {shape_attr: value}}`` for every channel-shaped layer of a subtree."""

    def __init__(self, entries: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        self.entries: dict[str, dict[str, Any]] = {
            name: dict(shape) for name, shape in (entries or {}).items()
        }

    # ---------------------------------------------------------------- construction

    @classmethod
    def record(cls, model: nn.Module, submodules: Sequence[str] | None = None) -> ChannelTable:
        """Record the current shapes under ``submodules`` (``None`` = the whole model).

        Args:
            model: The (pruned or not) model.
            submodules: Top-level attribute names whose subtrees are recorded, in the
                model's ``named_modules`` order.
        """
        roots = tuple(submodules) if submodules is not None else ("",)
        entries: dict[str, dict[str, Any]] = {}
        for root in roots:
            root_module = model.get_submodule(root) if root else model
            for name, module in root_module.named_modules():
                attrs = shape_attrs_of(module)
                if attrs is None:
                    continue
                full = f"{root}.{name}" if root and name else (root or name)
                entries[full] = _shape_of(module, attrs)
        return cls(entries)

    # ---------------------------------------------------------------- serialization

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-friendly payload (module order preserved)."""
        return {"entries": {name: dict(shape) for name, shape in self.entries.items()}}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> ChannelTable:
        """Inverse of :meth:`to_json_dict`."""
        return cls(data.get("entries", {}))

    # ---------------------------------------------------------------- queries

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name in self.entries

    def diff(self, other: ChannelTable) -> list[str]:
        """Human-readable differences (``[]`` when equal)."""
        lines = []
        for name in sorted(set(self.entries) | set(other.entries)):
            mine, theirs = self.entries.get(name), other.entries.get(name)
            if mine != theirs:
                lines.append(f"{name}: {mine} != {theirs}")
        return lines

    def restricted_to(self, other: ChannelTable) -> ChannelTable:
        """This table's entries for the modules ``other`` lists (order of ``other``)."""
        return ChannelTable(
            {name: self.entries[name] for name in other.entries if name in self.entries}
        )

    def changed_from(self, baseline: ChannelTable) -> dict[str, tuple[dict, dict]]:
        """``{name: (baseline_shape, this_shape)}`` for the layers whose shape differs."""
        return {
            name: (baseline.entries[name], shape)
            for name, shape in self.entries.items()
            if name in baseline.entries and baseline.entries[name] != shape
        }

    def verify_matches(self, produced: ChannelTable, source: str) -> None:
        """Raise unless ``produced`` equals this table.

        Used by the loader after :func:`apply_channel_table`: a mismatch means the config
        model and the checkpoint disagree on the architecture — the weights would then
        mis-map silently, so this is a hard failure.
        """
        lines = self.diff(produced)
        if lines:
            shown = "\n  ".join(lines[:20]) + (" ..." if len(lines) > 20 else "")
            raise RuntimeError(
                f"Channel table mismatch against {source} ({len(lines)} layer(s)):\n  {shown}"
            )


# ------------------------------------------------------------------------ rebuild


def _rebuild(module: nn.Module, shape: Mapping[str, Any]) -> nn.Module:
    """A fresh module of ``type(module)`` with ``module``'s hyperparameters and ``shape``."""
    device = next(module.parameters(), torch.empty(0)).device
    if isinstance(module, _ConvNd):
        in_channels, out_channels = int(shape["in_channels"]), int(shape["out_channels"])
        groups = module.groups
        if groups > 1:
            # Depthwise-style convs scale their groups with the channels; other grouped
            # convs must stay divisible or the weight layout changes meaning.
            if groups == module.in_channels == module.out_channels:
                if in_channels != out_channels:
                    raise ValueError(
                        f"Depthwise conv cannot change in/out independently: { {**shape} }"
                    )
                groups = in_channels
            elif in_channels % groups or out_channels % groups:
                raise ValueError(
                    f"Grouped conv (groups={groups}) cannot take channels {dict(**shape)}."
                )
        kwargs: dict[str, Any] = {
            "kernel_size": module.kernel_size,
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": groups,
            "bias": module.bias is not None,
            "padding_mode": module.padding_mode,
        }
        if isinstance(module, _ConvTransposeNd):
            kwargs["output_padding"] = module.output_padding
        return type(module)(in_channels, out_channels, **kwargs).to(device)
    if isinstance(module, nn.Linear):
        return type(module)(
            int(shape["in_features"]), int(shape["out_features"]), bias=module.bias is not None
        ).to(device)
    if isinstance(module, _BatchNorm):
        return type(module)(
            int(shape["num_features"]),
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
        ).to(device)
    if isinstance(module, nn.GroupNorm):
        return type(module)(
            module.num_groups, int(shape["num_channels"]), eps=module.eps, affine=module.affine
        ).to(device)
    if isinstance(module, nn.LayerNorm):
        return type(module)(
            list(shape["normalized_shape"]),
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
        ).to(device)
    raise TypeError(f"No rebuild rule for {type(module).__name__}")


def apply_channel_table(model: nn.Module, table: ChannelTable) -> int:
    """Rebuild every module of ``table`` inside ``model`` with the table's channel counts.

    Modules already at the recorded shape are left untouched (their weights included),
    so applying a table to an already-pruned model is a no-op. Weights of rebuilt modules
    are freshly initialized — the caller loads the checkpoint afterwards.

    Args:
        model: Freshly instantiated config model.
        table: The checkpoint's channel table.

    Returns:
        Number of modules rebuilt.

    Raises:
        AttributeError: When a table entry names a module the model does not have — the
            config and the checkpoint disagree on the architecture family.
    """
    rebuilt = 0
    for name, shape in table.entries.items():
        module = model.get_submodule(name)
        attrs = shape_attrs_of(module)
        if attrs is None:
            raise TypeError(
                f"Channel table names {name!r} ({type(module).__name__}), which has no "
                "channel-shaped attributes in this build."
            )
        if _shape_of(module, attrs) == {k: shape[k] for k in attrs}:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, _rebuild(module, shape))
        rebuilt += 1
    logger.info(
        "Applied channel table: %d of %d listed modules rebuilt with pruned widths",
        rebuilt,
        len(table),
    )
    return rebuilt
