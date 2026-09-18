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

"""Self-describing quantized checkpoints.

A quantized checkpoint carries, next to its ``state_dict``, a ``quantization`` entry
holding the :class:`~autoware_ml.quantization.config.QuantizationConfig` that built
its tree and the :class:`~autoware_ml.quantization.plan.PlacementRecord` the
build recorded. That is everything a later ``build_model`` needs to rebuild the
identical quantized tree and verify it — so ``deploy`` and ``test`` never read a
``quantization`` config section, and PTQ and QAT checkpoints (a Lightning
checkpoint with the same entry) load through one path.

There are no sidecar files: the calibrated ``_amax`` buffers live in the
``state_dict`` like any other buffer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import PlacementRecord

logger = logging.getLogger(__name__)

#: Top-level checkpoint key holding the quantization description.
QUANTIZATION_KEY = "quantization"


@dataclass(frozen=True)
class QuantizationDescription:
    """What a quantized checkpoint says about itself.

    The embedded format carries no version field on purpose: checkpoints are
    reproducible artifacts (re-run ``autoware-ml quantize``), and a format drift
    fails loudly anyway — ``QuantizationConfig.from_dict`` rejects unknown keys and
    a missing key raises here.
    """

    config: QuantizationConfig
    placement_record: PlacementRecord

    def to_payload(self) -> dict[str, Any]:
        """Serialize for embedding under :data:`QUANTIZATION_KEY`."""
        return {
            "config": self.config.to_dict(),
            "placement_record": self.placement_record.to_json_dict(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> QuantizationDescription:
        """Deserialize an embedded payload.

        Raises:
            KeyError: When the payload does not have this build's layout — the
                checkpoint predates a format change; re-produce it with
                ``autoware-ml quantize``.
        """
        return cls(
            config=QuantizationConfig.from_dict(payload["config"]),
            placement_record=PlacementRecord.from_json_dict(payload["placement_record"]),
        )


def attach_quantization(checkpoint: dict[str, Any], description: QuantizationDescription) -> None:
    """Embed ``description`` into a checkpoint dict in place (used by ``on_save_checkpoint``)."""
    checkpoint[QUANTIZATION_KEY] = description.to_payload()


def save_quantized_checkpoint(
    model: torch.nn.Module, path: str | Path, description: QuantizationDescription
) -> Path:
    """Write ``{"state_dict", "quantization"}`` — the PTQ producer's output.

    The layout is a subset of a Lightning checkpoint, so PTQ and QAT checkpoints read
    identically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {"state_dict": model.state_dict()}
    attach_quantization(checkpoint, description)
    torch.save(checkpoint, path)
    logger.info(
        "Saved quantized checkpoint: %s (%d decisions in the embedded placement record)",
        path,
        len(description.placement_record),
    )
    return path


def read_quantization(checkpoint: Mapping[str, Any]) -> QuantizationDescription | None:
    """Return the embedded description of a loaded checkpoint dict, or ``None`` for an FP one."""
    payload = checkpoint.get(QUANTIZATION_KEY)
    if payload is None:
        return None
    return QuantizationDescription.from_payload(payload)


def read_quantization_from_file(path: str | Path) -> QuantizationDescription | None:
    """Return the embedded description of a checkpoint file, or ``None`` for an FP one.

    Only the payload is inspected; tensors are memory-mapped, not materialized.
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    return read_quantization(checkpoint)


def find_quantization(
    weight_paths: Sequence[str | Path],
) -> tuple[Path, QuantizationDescription] | None:
    """Find the one quantized checkpoint among ``weight_paths``.

    Returns:
        ``(path, description)`` of the quantized checkpoint, or ``None`` when every
        checkpoint is a plain FP one.

    Raises:
        ValueError: When more than one checkpoint is quantized — a quantized tree is a
            whole-model construction; merging two of them is not defined.
    """
    found = [
        (Path(path), description)
        for path in weight_paths
        if (description := read_quantization_from_file(path)) is not None
    ]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            "More than one --weights checkpoint is quantized "
            f"({[str(p) for p, _ in found]}); a quantized model loads from exactly one."
        )
    return found[0]
