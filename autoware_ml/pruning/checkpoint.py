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

"""Self-describing pruned checkpoints.

A pruned checkpoint carries, under :data:`PRUNING_KEY`, the ``pruning`` config that
produced it and the :class:`~autoware_ml.pruning.channels.ChannelTable` of the pruned
subtree. ``build_model`` detects the payload, applies the table to the config model, then
loads the weights — deploy / test / quantize need no ``pruning`` config section.

The payload survives the later stages: the model remembers its description
(:data:`PRUNING_DESCRIPTION_ATTR`), and every checkpoint writer downstream
(``save_quantized_checkpoint``, the QAT and pruning callbacks) calls
:func:`attach_pruning_from_model`, so a pruned-then-quantized checkpoint describes both.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from autoware_ml.pruning.channels import ChannelTable
from autoware_ml.pruning.config import PruningConfig

logger = logging.getLogger(__name__)

#: Top-level checkpoint key holding the pruning description.
PRUNING_KEY = "pruning"
#: Attribute the loader sets on a model built from a pruned checkpoint.
PRUNING_DESCRIPTION_ATTR = "pruning_description"


@dataclass(frozen=True)
class PruningDescription:
    """What a pruned checkpoint says about itself."""

    config: PruningConfig
    channel_table: ChannelTable

    def to_payload(self) -> dict[str, Any]:
        """Serialize for embedding under :data:`PRUNING_KEY`."""
        return {"config": self.config.to_dict(), "channel_table": self.channel_table.to_json_dict()}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PruningDescription:
        """Deserialize an embedded payload (``KeyError`` on a layout this build predates)."""
        return cls(
            config=PruningConfig.from_dict(payload["config"]),
            channel_table=ChannelTable.from_json_dict(payload["channel_table"]),
        )


def attach_pruning(checkpoint: dict[str, Any], description: PruningDescription) -> None:
    """Embed ``description`` into a checkpoint dict in place."""
    checkpoint[PRUNING_KEY] = description.to_payload()


def attach_pruning_from_model(model: torch.nn.Module, checkpoint: dict[str, Any]) -> bool:
    """Embed the description a pruned model carries, if any. Returns whether one was."""
    description = getattr(model, PRUNING_DESCRIPTION_ATTR, None)
    if description is None:
        return False
    attach_pruning(checkpoint, description)
    return True


def save_pruned_checkpoint(
    model: torch.nn.Module, path: str | Path, description: PruningDescription
) -> Path:
    """Write ``{"state_dict", "pruning"}`` — the search stage's output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {"state_dict": model.state_dict()}
    attach_pruning(checkpoint, description)
    torch.save(checkpoint, path)
    logger.info(
        "Saved pruned checkpoint: %s (%d layers in the embedded channel table)",
        path,
        len(description.channel_table),
    )
    return path


def read_pruning(checkpoint: Mapping[str, Any]) -> PruningDescription | None:
    """Return the embedded description of a loaded checkpoint dict, or ``None``."""
    payload = checkpoint.get(PRUNING_KEY)
    if payload is None:
        return None
    return PruningDescription.from_payload(payload)


def read_pruning_from_file(path: str | Path) -> PruningDescription | None:
    """Return the embedded description of a checkpoint file (tensors memory-mapped)."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    return read_pruning(checkpoint)


def find_pruning(weight_paths: Sequence[str | Path]) -> tuple[Path, PruningDescription] | None:
    """Find the one pruned checkpoint among ``weight_paths``.

    Raises:
        ValueError: When more than one checkpoint is pruned — an architecture is a
            whole-model fact; two tables cannot merge.
    """
    found = [
        (Path(path), description)
        for path in weight_paths
        if (description := read_pruning_from_file(path)) is not None
    ]
    if len(found) > 1:
        raise ValueError(
            "More than one checkpoint carries a pruning description: "
            f"{[str(p) for p, _ in found]}. Supply exactly one pruned checkpoint."
        )
    return found[0] if found else None
