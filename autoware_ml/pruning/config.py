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

"""Typed view of the Hydra ``pruning`` config section.

Mirrors :mod:`autoware_ml.quantization.config`: one parse at the entrypoint, everything
downstream receives the typed object, unknown keys are rejected (a misspelled
``channel_divisor`` would otherwise silently search a search space TensorRT INT8 kernels
dislike). An absent section yields ``enabled=False``.

The section drives the ``prune`` stage only. Deploy / test / quantize never read it: a
pruned checkpoint carries its own :class:`~autoware_ml.pruning.channels.ChannelTable`
(see :mod:`autoware_ml.pruning.checkpoint`), and ``build_model`` rebuilds the narrowed
architecture from that before loading the weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from autoware_ml.quantization.config import QATScheduleConfig
from autoware_ml.utils.config_parsing import reject_unknown_keys

VALID_MODES = ("search", "finetune")
VALID_SCORES = ("map", "proxy")


@dataclass(frozen=True)
class FinetuneConfig:
    """Knowledge-distillation fine-tune of the pruned subnet (``pruning.finetune``).

    Present only when ``pruning.mode == "finetune"``. ``epochs`` and ``lr`` are required —
    no silent recipe default. The schedule shape is the QAT one
    (:class:`~autoware_ml.quantization.config.QATScheduleConfig`; ``lr`` is its PEAK);
    unlike QAT this is real capacity recovery, so expect more epochs and a higher peak
    (M1 reference: 10 epochs, peak 1e-4 cosine on a 30-epoch / 5e-4 training).

    ``kd_weight`` scales the model's own ``distillation_loss`` before it is added to the
    task loss (``0`` = plain fine-tune, teacher still built but unused).
    ``teacher_weights`` is needed only when ``--weights`` is already a pruned checkpoint
    (no FP teacher to copy before the search).
    """

    epochs: int
    lr: float
    schedule: QATScheduleConfig = field(default_factory=QATScheduleConfig)
    kd_weight: float = 1.0
    val_check_interval: float = 0.5
    teacher_weights: str | None = None

    KNOWN_KEYS = frozenset(
        {"epochs", "lr", "schedule", "kd_weight", "val_check_interval", "teacher_weights"}
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FinetuneConfig:
        """Build from the raw ``pruning.finetune`` mapping.

        Raises:
            TypeError: If ``raw`` is not a mapping.
            ValueError: On unknown keys, missing ``epochs``/``lr``, or a bad interval.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"pruning.finetune must be a dict, got {type(raw).__name__}")
        reject_unknown_keys(raw, cls.KNOWN_KEYS, "pruning.finetune")
        missing = {k for k in ("epochs", "lr") if raw.get(k) is None}
        if missing:
            raise ValueError(
                f"pruning.finetune requires {sorted(missing)} — no silent recipe default."
            )
        val_check_interval = float(raw.get("val_check_interval", 0.5))
        if not (0.0 < val_check_interval <= 1.0):
            raise ValueError(
                f"pruning.finetune.val_check_interval must be in (0, 1], got {val_check_interval}."
            )
        kd_weight = float(raw.get("kd_weight", 1.0))
        if kd_weight < 0.0:
            raise ValueError(f"pruning.finetune.kd_weight must be >= 0, got {kd_weight}.")
        teacher = raw.get("teacher_weights")
        return cls(
            epochs=int(raw["epochs"]),
            lr=float(raw["lr"]),
            schedule=QATScheduleConfig.from_raw(raw.get("schedule")),
            kd_weight=kd_weight,
            val_check_interval=val_check_interval,
            teacher_weights=str(teacher) if teacher is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize (round-trips through :meth:`from_dict`)."""
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "schedule": self.schedule.to_dict(),
            "kd_weight": self.kd_weight,
            "val_check_interval": self.val_check_interval,
            "teacher_weights": self.teacher_weights,
        }


@dataclass(frozen=True)
class PruningConfig:
    """Typed view of the ``pruning`` config section.

    - ``flops``: FastNAS upper bound on the pruned subtree's FLOPs — a percentage of the
      original (``"60%"``) or an absolute count.
    - ``score``: what ranks candidate subnets during the search. ``map`` runs the deploy
      evaluate loop on ``score_samples`` frames (the real metric; ~2 s/eval on CenterPoint,
      gives smooth per-block widths). ``proxy`` is the relative-L2 distance to the
      un-pruned head outputs on the cached frames (seconds per search; zig-zag widths —
      smoke tests only).
    - ``calib_frames``: frames of the subtree's stage input cached for FastNAS' BN
      re-calibration (and the proxy score).
    - ``channels_ratio`` / ``channel_divisor``: the per-layer width candidates
      (fractions of the original width, rounded to the divisor — 32 keeps INT8 IMMA
      alignment).
    """

    enabled: bool = False
    mode: str = "search"
    flops: str = "60%"
    score: str = "map"
    score_samples: int = 30
    calib_frames: int = 16
    channels_ratio: tuple[float, ...] = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    channel_divisor: int = 32
    finetune: FinetuneConfig | None = None

    KNOWN_KEYS = frozenset(
        {
            "enabled",
            "mode",
            "flops",
            "score",
            "score_samples",
            "calib_frames",
            "channels_ratio",
            "channel_divisor",
            "finetune",
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> PruningConfig:
        """Build from a raw ``pruning`` mapping; ``None``/empty → disabled.

        Raises:
            ValueError: On unknown keys, an unknown ``mode``/``score``, a ``finetune``
                block under ``mode: search``, or a missing one under ``mode: finetune``.
        """
        if not raw:
            return cls()
        reject_unknown_keys(
            raw,
            cls.KNOWN_KEYS,
            "pruning",
            hint="A misspelled key would silently fall back to the default search space.",
        )
        mode = str(raw.get("mode", "search"))
        if mode not in VALID_MODES:
            raise ValueError(f"pruning.mode must be one of {VALID_MODES}, got {mode!r}.")
        score = str(raw.get("score", "map"))
        if score not in VALID_SCORES:
            raise ValueError(f"pruning.score must be one of {VALID_SCORES}, got {score!r}.")
        ratios = raw.get("channels_ratio")
        channels_ratio = (
            tuple(float(r) for r in ratios) if ratios else cls.channels_ratio  # type: ignore[misc]
        )
        if any(not (0.0 < r <= 1.0) for r in channels_ratio):
            raise ValueError(
                f"pruning.channels_ratio values must be in (0, 1], got {channels_ratio}."
            )
        divisor = int(raw.get("channel_divisor", 32))
        if divisor <= 0:
            raise ValueError(f"pruning.channel_divisor must be positive, got {divisor}.")
        finetune_raw = raw.get("finetune")
        if mode == "finetune" and finetune_raw is None:
            raise ValueError("pruning.mode='finetune' requires a pruning.finetune block.")
        if mode == "search" and finetune_raw is not None:
            raise ValueError(
                "pruning.finetune is present under mode='search' — a config lie; drop it "
                "(``finetune: null``) or switch the mode."
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=mode,
            flops=str(raw.get("flops", "60%")),
            score=score,
            score_samples=int(raw.get("score_samples", 30)),
            calib_frames=int(raw.get("calib_frames", 16)),
            channels_ratio=channels_ratio,
            channel_divisor=divisor,
            finetune=FinetuneConfig.from_dict(finetune_raw) if finetune_raw is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize (round-trips through :meth:`from_dict`)."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "flops": self.flops,
            "score": self.score,
            "score_samples": self.score_samples,
            "calib_frames": self.calib_frames,
            "channels_ratio": list(self.channels_ratio),
            "channel_divisor": self.channel_divisor,
            "finetune": self.finetune.to_dict() if self.finetune is not None else None,
        }

    def describe(self) -> str:
        """One-line summary for logs."""
        return (
            f"fastnas flops<={self.flops}, score={self.score}({self.score_samples}f), "
            f"divisor={self.channel_divisor}, ratios={list(self.channels_ratio)}"
        )
