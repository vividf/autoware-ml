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

"""Typed view of the Hydra ``quantization`` config section.

The single parse of the ``quantization`` dict: entrypoints build this once and pass
it to the model's ``build_quantization_plan`` — nothing downstream re-parses the raw
dict. Defaults are chosen so an absent section yields a fully-disabled config
(``enabled=False``), leaving non-quantized configs unaffected.

Precision placement is declarative (modelopt-style): everything the plan reaches is
``default_precision`` (INT8), and ``skip_quantize`` lists glob patterns (subtree match)
excluded from quantization — an excluded module's runtime precision follows the deploy
``onnx.precision`` (FP16 via AutoCast). Architecture recipes are always-on and class-gated;
``disable_recipes`` opts a config out of one. ``calibration`` picks the amax algorithm.

The FP input checkpoint, the training config, and
the work directory are NOT config keys here: the checkpoint arrives via ``--weights``,
the training setup via the Hydra experiment config, and artifact placement via the
MLflow run context.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from autoware_ml.utils.config_parsing import reject_unknown_keys

logger = logging.getLogger(__name__)

#: Module kinds a submodule rule may request.
VALID_MODULE_KINDS = ("conv", "linear")
#: Architecture recipes, in the canonical order they are applied. Must stay in sync with
#: ``recipes.attach.RECIPE_ATTACHERS`` (``plan.py`` checks that at import time).
VALID_RECIPES = ("residual_add", "ese", "maxpool")


class Precision(str, Enum):
    """Quantization target precision.

    INT8 is the production path for convolutions (CenterPoint release recipe). FP8 (E4M3,
    per-tensor, max calibration) is validated for Linear layers on PTv3 and BEVFusion
    (2026-09-02/03: within the FP16 band where INT8 Linear lost 6 mIoU) and is the
    precision every Linear should take; conv FP8 is blocked at ONNX export today.
    """

    INT8 = "int8"
    FP8 = "fp8"


@dataclass(frozen=True)
class CalibrationConfig:
    """How activation ``amax`` is computed (``quantization.calibration``).

    The calibration algorithm is part of the recipe — often a bigger accuracy lever than
    the number of samples — so it lives in config and travels inside the checkpoint.

    - ``mse`` (default) / ``entropy`` / ``percentile`` — histogram calibrator on every
      activation quantizer; ``amax`` minimizes MSE / KL divergence, or clips at
      ``percentile``.
    - ``max`` — running max of the observed activations (no histogram); the FP8
      convention and the fastest.
    - ``smoothquant`` — modelopt's SmoothQuant (Xiao et al. 2022): max calibration, then
      every INT8 quantized ``Linear`` migrates activation outliers into its weight through a
      per-input-channel ``pre_quant_scale`` (``alpha`` balances the two sides; 0.5 is the
      paper default). Convolutions are unaffected. Exports as a ``Mul`` before Q/DQ.

    Weight quantizers always use max (their per-channel ``amax`` is exact); FP8 activation
    quantizers always use max — ``percentile``/``mse``/``entropy`` therefore apply to INT8
    activations only, and a config asking for a histogram method on an FP8-only tree
    silently degenerates to max (there is nothing to histogram).
    """

    method: str = "mse"
    percentile: float = 99.99
    smoothquant_alpha: float = 0.5

    METHODS = ("mse", "entropy", "percentile", "max", "smoothquant")
    HISTOGRAM_METHODS = ("mse", "entropy", "percentile")
    KNOWN_KEYS = frozenset({"method", "percentile", "smoothquant_alpha"})
    _KEYS_BY_METHOD: ClassVar[Mapping[str, frozenset[str]]] = {
        "mse": frozenset({"method"}),
        "entropy": frozenset({"method"}),
        "percentile": frozenset({"method", "percentile"}),
        "max": frozenset({"method"}),
        "smoothquant": frozenset({"method", "smoothquant_alpha"}),
    }

    @property
    def activation_calibrator(self) -> str:
        """modelopt calibrator kind the activation quantizers need: ``"histogram"`` or ``"max"``."""
        return "histogram" if self.method in self.HISTOGRAM_METHODS else "max"

    @classmethod
    def from_raw(cls, raw: Any) -> CalibrationConfig:
        """Build from ``None`` (default), a method string, or a ``{method, ...}`` mapping.

        Raises:
            TypeError: If ``raw`` is neither a string nor a mapping.
            ValueError: On an unknown method, a knob that does not belong to the chosen
                method, or an out-of-range value.
        """
        if raw is None:
            return cls()
        if isinstance(raw, str):
            raw = {"method": raw}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"quantization.calibration must be a string or a dict, got {type(raw).__name__}"
            )
        reject_unknown_keys(raw, cls.KNOWN_KEYS, "quantization.calibration")
        method = str(raw.get("method", "mse"))
        if method not in cls.METHODS:
            raise ValueError(
                f"quantization.calibration.method must be one of {list(cls.METHODS)}, got {method!r}."
            )
        foreign = set(raw) - cls._KEYS_BY_METHOD[method]
        if foreign:
            raise ValueError(
                f"quantization.calibration key(s) {sorted(foreign)} do not apply to method "
                f"{method!r} (valid: {sorted(cls._KEYS_BY_METHOD[method] - {'method'})})."
            )
        config = cls(
            method=method,
            percentile=float(raw.get("percentile", 99.99)),
            smoothquant_alpha=float(raw.get("smoothquant_alpha", 0.5)),
        )
        if not (0.0 < config.percentile <= 100.0):
            raise ValueError(
                f"quantization.calibration.percentile must be in (0, 100], got {config.percentile}."
            )
        if not (0.0 <= config.smoothquant_alpha <= 1.0):
            raise ValueError(
                "quantization.calibration.smoothquant_alpha must be in [0, 1], "
                f"got {config.smoothquant_alpha}."
            )
        return config

    def to_dict(self) -> dict[str, Any]:
        """Raw mapping equivalent (round-trips through :meth:`from_raw`)."""
        out: dict[str, Any] = {"method": self.method}
        if self.method == "percentile":
            out["percentile"] = self.percentile
        if self.method == "smoothquant":
            out["smoothquant_alpha"] = self.smoothquant_alpha
        return out

    def describe(self) -> str:
        """One-line human-readable summary for logs."""
        if self.method == "percentile":
            return f"histogram percentile {self.percentile:g}"
        if self.method == "smoothquant":
            return f"smoothquant (alpha={self.smoothquant_alpha:g}, max calibration)"
        if self.method == "max":
            return "max"
        return f"histogram {self.method}"


@dataclass(frozen=True)
class QATScheduleConfig:
    """Learning-rate schedule of the QAT fine-tune (``quantization.qat.schedule``).

    Accepts a plain string (``"cosine"`` / ``"one_cycle"`` / ``"constant"``) or a
    mapping ``{type: ..., <type-specific knobs>}``. ``QATConfig.lr`` is always the
    PEAK of the schedule.

    - ``cosine`` (default) — the NVIDIA integer-quantization recipe (Wu et al. 2020;
      also what modelopt documents): start AT the peak (no warmup — the model is
      already converged), cosine-anneal down to ``lr * final_lr_ratio`` (1/100 by
      default). Recommended peak: 1% of the original training's peak lr, for ~10%
      of the original epochs.
    - ``one_cycle`` — the CUDA-CenterPoint QAT reference: warm up from
      ``lr / div_factor`` to ``lr`` over ``pct_start`` of the run, then anneal to
      ``lr / div_factor / final_div_factor``; optional Adam momentum cycling.
    - ``constant`` — raw optimizer lr, no scheduler. Experiments only: a flat lr at
      QAT scale can collapse a converged model within an epoch.

    Both annealing types are realised with ``torch.optim.lr_scheduler.OneCycleLR``
    stepped per iteration (``total_steps`` is auto-filled by the optimizer builder
    from the trainer's estimated stepping batches, so multi-epoch runs span the
    whole fine-tune).
    """

    type: str = "cosine"
    # cosine
    final_lr_ratio: float = 0.01
    # one_cycle
    div_factor: float = 10.0
    pct_start: float = 0.4
    final_div_factor: float = 1.0e4
    cycle_momentum: bool = False
    base_momentum: float = 0.85
    max_momentum: float = 0.95

    TYPES = ("cosine", "one_cycle", "constant")
    KNOWN_KEYS = frozenset(
        {
            "type",
            "final_lr_ratio",
            "div_factor",
            "pct_start",
            "final_div_factor",
            "cycle_momentum",
            "base_momentum",
            "max_momentum",
        }
    )
    _KEYS_BY_TYPE: ClassVar[Mapping[str, frozenset[str]]] = {
        "cosine": frozenset({"type", "final_lr_ratio"}),
        "one_cycle": frozenset(
            {
                "type",
                "div_factor",
                "pct_start",
                "final_div_factor",
                "cycle_momentum",
                "base_momentum",
                "max_momentum",
            }
        ),
        "constant": frozenset({"type"}),
    }

    @classmethod
    def from_raw(cls, raw: Any) -> QATScheduleConfig:
        """Build from ``None`` (default), a type string, or a ``{type, ...}`` mapping.

        Raises:
            TypeError: If ``raw`` is neither a string nor a mapping.
            ValueError: On an unknown type, a knob that does not belong to the chosen
                type, or an out-of-range value.
        """
        if raw is None:
            return cls()
        if isinstance(raw, str):
            raw = {"type": raw}
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"quantization.qat.schedule must be a string or a dict, got {type(raw).__name__}"
            )
        reject_unknown_keys(raw, cls.KNOWN_KEYS, "quantization.qat.schedule")
        schedule_type = str(raw.get("type", "cosine"))
        if schedule_type not in cls.TYPES:
            raise ValueError(
                f"quantization.qat.schedule.type must be one of {list(cls.TYPES)}, got {schedule_type!r}."
            )
        foreign = set(raw) - cls._KEYS_BY_TYPE[schedule_type]
        if foreign:
            raise ValueError(
                f"quantization.qat.schedule key(s) {sorted(foreign)} do not apply to type "
                f"{schedule_type!r} (valid: {sorted(cls._KEYS_BY_TYPE[schedule_type] - {'type'})})."
            )
        config = cls(
            type=schedule_type,
            final_lr_ratio=float(raw.get("final_lr_ratio", 0.01)),
            div_factor=float(raw.get("div_factor", 10.0)),
            pct_start=float(raw.get("pct_start", 0.4)),
            final_div_factor=float(raw.get("final_div_factor", 1.0e4)),
            cycle_momentum=bool(raw.get("cycle_momentum", False)),
            base_momentum=float(raw.get("base_momentum", 0.85)),
            max_momentum=float(raw.get("max_momentum", 0.95)),
        )
        if not (0.0 < config.final_lr_ratio <= 1.0):
            raise ValueError(
                f"quantization.qat.schedule.final_lr_ratio must be in (0, 1], got {config.final_lr_ratio}."
            )
        if config.div_factor < 1.0:
            raise ValueError(
                f"quantization.qat.schedule.div_factor must be >= 1, got {config.div_factor}."
            )
        if not (0.0 <= config.pct_start <= 1.0):
            raise ValueError(
                f"quantization.qat.schedule.pct_start must be in [0, 1], got {config.pct_start}."
            )
        if config.final_div_factor < 1.0:
            raise ValueError(
                f"quantization.qat.schedule.final_div_factor must be >= 1, got {config.final_div_factor}."
            )
        return config

    def build_lightning_scheduler(self, peak_lr: float) -> tuple[dict | None, dict | None]:
        """Return ``(model.scheduler, model.scheduler_config)`` Hydra nodes for this schedule.

        ``None, None`` for ``constant``. The annealing types return a partial
        ``OneCycleLR`` (``total_steps`` left for the optimizer builder to fill) and
        ``{"interval": "step"}``.
        """
        if self.type == "constant":
            return None, None
        if self.type == "cosine":
            # No warmup (div_factor=1, pct_start=0): OneCycleLR then degenerates to a
            # single cosine phase from peak_lr down to peak_lr / final_div_factor.
            kwargs = {
                "div_factor": 1.0,
                "pct_start": 0.0,
                "final_div_factor": 1.0 / self.final_lr_ratio,
                "cycle_momentum": False,
            }
        else:
            kwargs = {
                "div_factor": self.div_factor,
                "pct_start": self.pct_start,
                "final_div_factor": self.final_div_factor,
                "cycle_momentum": self.cycle_momentum,
                "base_momentum": self.base_momentum,
                "max_momentum": self.max_momentum,
            }
        scheduler = {
            "_target_": "torch.optim.lr_scheduler.OneCycleLR",
            "_partial_": True,
            "max_lr": float(peak_lr),
            "anneal_strategy": "cos",
            **kwargs,
        }
        return scheduler, {"interval": "step"}

    def describe(self) -> str:
        """One-line human-readable summary for logs."""
        if self.type == "constant":
            return "constant lr"
        if self.type == "cosine":
            return f"cosine from peak to peak x {self.final_lr_ratio:g}, no warmup"
        return (
            f"one_cycle: peak/{self.div_factor:g} -> peak at {self.pct_start:.0%} -> "
            f"peak/{self.div_factor * self.final_div_factor:g}"
            + (", momentum cycling" if self.cycle_momentum else "")
        )


@dataclass(frozen=True)
class QATConfig:
    """Typed view of the optional ``quantization.qat`` sub-block.

    Present only when ``quantization.mode == "qat"``. ``epochs`` and ``lr`` are
    required: there is no silent recipe default — they belong in the config,
    visibly. Recommended (Wu et al. 2020 / modelopt): ``epochs`` ≈ 10% of the
    original training and ``lr`` (the schedule PEAK) ≈ 1% of the original
    training's peak lr. See :class:`QATScheduleConfig` for ``schedule``.
    ``calibrate_samples`` counts SAMPLES (never batches — the QAT callback divides
    by the val dataloader's batch size) and defaults to the CUDA-CenterPoint
    reference of 400 samples.
    """

    epochs: int
    lr: float
    schedule: QATScheduleConfig = QATScheduleConfig()
    # The skip_quantize subtrees carry no fake-quant STE masking, so they absorb most
    # of the gradient and drift past the frozen downstream amax; freeze them (they
    # deploy as FP16 anyway).
    freeze_unquantized: bool = True
    # Validate this often WITHIN an epoch (Lightning ``val_check_interval`` fraction).
    # QAT can degrade progressively inside a single epoch, so end-of-epoch-only
    # validation would make ``best.ckpt`` pick the worst model.
    val_check_interval: float = 0.25
    calibrate_samples: int = 400

    # Typo guard — same rationale as QuantizationConfig.KNOWN_KEYS.
    KNOWN_KEYS = frozenset(
        {
            "epochs",
            "lr",
            "schedule",
            "freeze_unquantized",
            "val_check_interval",
            "calibrate_samples",
        }
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> QATConfig:
        """Build QATConfig from ``quantization["qat"]``.

        Raises:
            TypeError: If ``raw`` is not a mapping.
            ValueError: On unknown keys, when ``epochs`` / ``lr`` are missing, or on an
                invalid ``schedule`` / ``val_check_interval``.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"quantization.qat must be a dict, got {type(raw).__name__}")
        reject_unknown_keys(raw, cls.KNOWN_KEYS, "quantization.qat")
        missing = {k for k in ("epochs", "lr") if raw.get(k) is None}
        if missing:
            raise ValueError(
                f"quantization.qat requires {sorted(missing)} — no silent recipe default. "
                "Recommended: epochs ≈ 10% of the original training, lr (schedule peak) ≈ 1% of "
                "the original peak lr (Wu et al. 2020)."
            )
        val_check_interval = float(raw.get("val_check_interval", 0.25))
        if not (0.0 < val_check_interval <= 1.0):
            raise ValueError(
                f"quantization.qat.val_check_interval must be in (0, 1], got {val_check_interval}."
            )
        return cls(
            epochs=int(raw["epochs"]),
            lr=float(raw["lr"]),
            schedule=QATScheduleConfig.from_raw(raw.get("schedule")),
            freeze_unquantized=bool(raw.get("freeze_unquantized", True)),
            val_check_interval=val_check_interval,
            calibrate_samples=int(raw.get("calibrate_samples", 400)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Raw mapping equivalent (round-trips through :meth:`from_dict`)."""
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "schedule": {
                "type": self.schedule.type,
                **(
                    {"final_lr_ratio": self.schedule.final_lr_ratio}
                    if self.schedule.type == "cosine"
                    else {}
                ),
                **(
                    {
                        "div_factor": self.schedule.div_factor,
                        "pct_start": self.schedule.pct_start,
                        "final_div_factor": self.schedule.final_div_factor,
                        "cycle_momentum": self.schedule.cycle_momentum,
                        "base_momentum": self.schedule.base_momentum,
                        "max_momentum": self.schedule.max_momentum,
                    }
                    if self.schedule.type == "one_cycle"
                    else {}
                ),
            },
            "freeze_unquantized": self.freeze_unquantized,
            "val_check_interval": self.val_check_interval,
            "calibrate_samples": self.calibrate_samples,
        }


@dataclass(frozen=True)
class PTQConfig:
    """Typed view of the optional ``quantization.ptq`` sub-block.

    The quantize-stage half of a PTQ run, recorded in the experiment config so one file
    reproduces the produced checkpoint. ``calibrate_samples`` is required: it is
    *the* calibration-recipe knob and gets no silent default — same rationale as
    ``QATConfig.epochs`` / ``lr``.
    """

    calibrate_samples: int
    batch_size: int = 1
    calib_seed: int | None = None
    calib_shuffle: bool = False

    # Typo guard — same rationale as QuantizationConfig.KNOWN_KEYS.
    KNOWN_KEYS = frozenset({"calibrate_samples", "batch_size", "calib_seed", "calib_shuffle"})

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PTQConfig:
        """Build PTQConfig from ``quantization["ptq"]``.

        Raises:
            TypeError: If ``raw`` is not a mapping.
            ValueError: On unknown keys, or when ``calibrate_samples`` is missing.
        """
        if not isinstance(raw, Mapping):
            raise TypeError(f"quantization.ptq must be a dict, got {type(raw).__name__}")
        reject_unknown_keys(raw, cls.KNOWN_KEYS, "quantization.ptq")
        if raw.get("calibrate_samples") is None:
            raise ValueError(
                "quantization.ptq requires calibrate_samples — no silent recipe default. "
                "It is the calibration-recipe knob (release reference: 400 samples @ batch_size=1)."
            )
        return cls(
            calibrate_samples=int(raw["calibrate_samples"]),
            batch_size=int(raw.get("batch_size", 1)),
            calib_seed=raw.get("calib_seed"),
            calib_shuffle=bool(raw.get("calib_shuffle", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Raw mapping equivalent (round-trips through :meth:`from_dict`)."""
        return {
            "calibrate_samples": self.calibrate_samples,
            "batch_size": self.batch_size,
            "calib_seed": self.calib_seed,
            "calib_shuffle": self.calib_shuffle,
        }


@dataclass(frozen=True)
class QuantizationConfig:
    """Typed view of the ``quantization`` config section."""

    enabled: bool = False
    mode: str = "ptq"  # "ptq" | "qat"
    fuse_bn: bool = True
    # Precision placement: INT8 by default, opt out by glob (subtree match). A pattern
    # excludes the matched module and all its descendants from quantization; their
    # runtime precision follows the deploy onnx.precision (FP16 via AutoCast).
    default_precision: Precision = Precision.INT8
    skip_quantize: tuple[str, ...] = ()
    # How activation amax is computed (mse / entropy / percentile / max / smoothquant).
    # Shared by PTQ and the QAT epoch-0 calibration; see CalibrationConfig.
    calibration: CalibrationConfig = CalibrationConfig()
    # Architecture recipes (residual-add / eSE / maxpool) are attached always, gated by
    # module class and scoped to the quantized submodules. List a recipe name here to opt
    # this config out.
    disable_recipes: tuple[str, ...] = ()
    # Quantize-stage only: build the model, prepare the quantized tree, log the full placement
    # record (which module gets which transform and why), and exit WITHOUT calibrating
    # or training. The way to inspect precision placement before spending GPU time.
    dry_run: bool = False
    # Stage blocks (quantize-side) — each present only under its matching mode (a block under the
    # wrong mode is a config lie and from_dict raises; an explicit ``ptq: null`` /
    # ``qat: null`` is fine, so a mode="qat" child config can drop an inherited block).
    # Deploy-load behavior NEVER branches on these: the loader rebuilds the identical
    # tree for PTQ and QAT checkpoints alike.
    ptq: PTQConfig | None = None
    qat: QATConfig | None = None

    # The full key set of the ``quantization`` config section. ``from_dict`` rejects
    # anything else: a misspelled key (``skip_quantizes: ...``) would otherwise silently
    # degrade to "quantize everything INT8" and be visible only as an eval mAP drop.
    KNOWN_KEYS = frozenset(
        {
            "enabled",
            "mode",
            "fuse_bn",
            "default_precision",
            "skip_quantize",
            "calibration",
            "disable_recipes",
            "dry_run",
            "ptq",
            "qat",
        }
    )

    @staticmethod
    def _str_tuple(value: Any) -> tuple[str, ...]:
        return tuple(str(v) for v in value) if value else ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> QuantizationConfig:
        """Build QuantizationConfig from a raw ``quantization`` mapping; empty/None → disabled.

        Raises:
            ValueError: If the dict contains keys outside :attr:`KNOWN_KEYS` (typo guard),
                a ``ptq``/``qat`` block is present under the wrong mode, or
                ``default_precision`` / ``calibration`` hold an unknown value.
        """
        if not raw:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError(f"quantization must be a dict, got {type(raw).__name__}")
        reject_unknown_keys(
            raw,
            cls.KNOWN_KEYS,
            "quantization",
            hint="A misspelled key would silently change what gets quantized.",
        )
        mode = str(raw.get("mode", "ptq"))
        qat_raw = raw.get("qat")
        if qat_raw is not None and mode != "qat":
            raise ValueError(
                f'quantization has a "qat" block but mode="{mode}" — set mode="qat" or drop the block. '
                "A qat block under a non-qat mode is a config lie."
            )
        ptq_raw = raw.get("ptq")
        if ptq_raw is not None and mode != "ptq":
            raise ValueError(
                f'quantization has a "ptq" block but mode="{mode}" — set mode="ptq" or drop the block '
                "(a mode='qat' config inheriting one can set ptq: null). "
                "A ptq block under a non-ptq mode is a config lie."
            )
        raw_precision = str(raw.get("default_precision", Precision.INT8.value))
        try:
            default_precision = Precision(raw_precision)
        except ValueError:
            raise ValueError(
                f"quantization.default_precision={raw_precision!r} — valid values: "
                f"{[p.value for p in Precision]}; skip_quantize opts subtrees out."
            ) from None
        calibration = CalibrationConfig.from_raw(raw.get("calibration"))
        disable_recipes = cls._str_tuple(raw.get("disable_recipes"))
        unknown_recipes = sorted(set(disable_recipes) - set(VALID_RECIPES))
        if unknown_recipes:
            raise ValueError(
                f"quantization.disable_recipes names unknown recipe(s) {unknown_recipes}; "
                f"valid recipes: {list(VALID_RECIPES)}. An unknown name would silently disable nothing."
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=mode,
            fuse_bn=bool(raw.get("fuse_bn", True)),
            default_precision=default_precision,
            skip_quantize=cls._str_tuple(raw.get("skip_quantize")),
            calibration=calibration,
            disable_recipes=disable_recipes,
            dry_run=bool(raw.get("dry_run", False)),
            ptq=PTQConfig.from_dict(ptq_raw) if ptq_raw is not None else None,
            qat=QATConfig.from_dict(qat_raw) if qat_raw is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Raw mapping equivalent (round-trips through :meth:`from_dict`).

        This is what a quantized checkpoint embeds, so a deploy/test run can rebuild the
        identical quantized tree from the checkpoint alone — without the Hydra
        ``quantization`` section that produced it.
        """
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "fuse_bn": self.fuse_bn,
            "default_precision": self.default_precision.value,
            "skip_quantize": list(self.skip_quantize),
            "calibration": self.calibration.to_dict(),
            "disable_recipes": list(self.disable_recipes),
            "dry_run": self.dry_run,
            "ptq": self.ptq.to_dict() if self.ptq is not None else None,
            "qat": self.qat.to_dict() if self.qat is not None else None,
        }
