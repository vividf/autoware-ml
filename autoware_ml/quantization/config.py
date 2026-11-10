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


"""Typed view of the Hydra ``quantization`` config section."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Module kinds a submodule rule may request.
VALID_MODULE_KINDS = ("conv", "linear", "spconv")


class Precision(StrEnum):
    """Quantization target precision.

    INT8 is the production path for convolutions. FP8 (E4M3) is meant for Linear layers: on
    PTv3 and BEVFusion it stays within the FP16 accuracy band, while INT8 Linear cost PTv3
    6 mIoU. Conv FP8 is blocked at ONNX export for now.
    """

    INT8 = "int8"
    FP8 = "fp8"


class Mode(StrEnum):
    """How the quantized checkpoint is produced."""

    PTQ = "ptq"
    QAT = "qat"


class CalibrationMethod(StrEnum):
    """How activation ``amax`` is computed."""

    MSE = "mse"
    ENTROPY = "entropy"
    PERCENTILE = "percentile"
    MAX = "max"
    SMOOTHQUANT = "smoothquant"


class ScheduleType(StrEnum):
    """Learning-rate schedule of the QAT fine-tune."""

    COSINE = "cosine"
    ONE_CYCLE = "one_cycle"
    CONSTANT = "constant"


_HISTOGRAM_METHODS = frozenset(
    {CalibrationMethod.MSE, CalibrationMethod.ENTROPY, CalibrationMethod.PERCENTILE}
)


def _knobs_match(model: BaseModel, section: str, selector: str, allowed: set[str]) -> None:
    """Reject a knob set to a non-default value for another value of ``selector``.

    A knob at its default is accepted whatever the selector: a serialized config carries
    every field, and reading it back must not fail.
    """
    chosen = getattr(model, selector)
    foreign = sorted(
        name
        for name in model.model_fields_set - allowed - {selector}
        if getattr(model, name) != type(model).model_fields[name].default
    )
    if foreign:
        raise ValueError(
            f"{section} key(s) {foreign} do not apply to {selector} {chosen.value!r} "
            f"(valid: {sorted(allowed)})."
        )


class CalibrationConfig(BaseModel):
    """How activation ``amax`` is computed (``quantization.calibration``).

    - ``mse`` (default) / ``entropy`` / ``percentile``: histogram calibration; ``amax``
      minimizes the MSE or the KL divergence, or clips at ``percentile``.
    - ``max``: running maximum of the activations.
    - ``smoothquant``: modelopt's SmoothQuant. Max calibration, then every INT8 ``Linear``
      moves activation outliers into its weight through a per-input-channel
      ``pre_quant_scale``, balanced by ``smoothquant_alpha``. Exports as a ``Mul`` before
      the Q/DQ.

    Weights and FP8 activations always use max calibration, so the histogram methods apply
    to INT8 activations only; on an FP8-only tree they amount to ``max``. A bare method
    string is accepted in place of the mapping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: CalibrationMethod = CalibrationMethod.MSE
    percentile: float = Field(default=99.99, gt=0.0, le=100.0)
    smoothquant_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    _KNOBS = {
        CalibrationMethod.PERCENTILE: {"percentile"},
        CalibrationMethod.SMOOTHQUANT: {"smoothquant_alpha"},
    }

    @model_validator(mode="before")
    @classmethod
    def _method_string(cls, raw: Any) -> Any:
        return {"method": raw} if isinstance(raw, str) else raw

    @model_validator(mode="after")
    def _knobs_apply_to_the_method(self) -> CalibrationConfig:
        _knobs_match(
            self, "quantization.calibration", "method", self._KNOBS.get(self.method, set())
        )
        return self

    @property
    def activation_calibrator(self) -> str:
        """modelopt calibrator kind the activation quantizers need: ``"histogram"`` or ``"max"``."""
        return "histogram" if self.method in _HISTOGRAM_METHODS else "max"

    def describe(self) -> str:
        """One-line summary for logs."""
        if self.method is CalibrationMethod.PERCENTILE:
            return f"histogram percentile {self.percentile:g}"
        if self.method is CalibrationMethod.SMOOTHQUANT:
            return f"smoothquant (alpha={self.smoothquant_alpha:g}, max calibration)"
        if self.method is CalibrationMethod.MAX:
            return "max"
        return f"histogram {self.method.value}"


class QATScheduleConfig(BaseModel):
    """Learning-rate schedule of the QAT fine-tune (``quantization.qat.schedule``).

    A type string or a ``{type: ..., <knobs>}`` mapping. ``QATConfig.lr`` is the peak lr.

    - ``cosine`` (default): the NVIDIA integer-quantization recipe (Wu et al. 2020). No
      warmup, since the model has converged; cosine-anneals to ``lr * final_lr_ratio``.
    - ``one_cycle``: the CUDA-CenterPoint QAT reference. Warms up from ``lr / div_factor``
      over ``pct_start`` of the run, then anneals to ``lr / div_factor / final_div_factor``;
      optional momentum cycling.
    - ``constant``: the optimizer lr, no scheduler. For experiments only: a flat lr at QAT
      scale can collapse a converged model within an epoch.

    The annealing types run as a ``OneCycleLR`` stepped every iteration over the whole
    fine-tune.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ScheduleType = ScheduleType.COSINE
    # cosine
    final_lr_ratio: float = Field(default=0.01, gt=0.0, le=1.0)
    # one_cycle
    div_factor: float = Field(default=10.0, ge=1.0)
    pct_start: float = Field(default=0.4, ge=0.0, le=1.0)
    final_div_factor: float = Field(default=1.0e4, ge=1.0)
    cycle_momentum: bool = False
    base_momentum: float = 0.85
    max_momentum: float = 0.95

    _KNOBS = {
        ScheduleType.COSINE: {"final_lr_ratio"},
        ScheduleType.ONE_CYCLE: {
            "div_factor",
            "pct_start",
            "final_div_factor",
            "cycle_momentum",
            "base_momentum",
            "max_momentum",
        },
        ScheduleType.CONSTANT: set(),
    }

    @model_validator(mode="before")
    @classmethod
    def _type_string(cls, raw: Any) -> Any:
        return {"type": raw} if isinstance(raw, str) else raw

    @model_validator(mode="after")
    def _knobs_apply_to_the_type(self) -> QATScheduleConfig:
        _knobs_match(self, "quantization.qat.schedule", "type", self._KNOBS[self.type])
        return self

    def build_lightning_scheduler(self, peak_lr: float) -> tuple[dict | None, dict | None]:
        """Return the Hydra nodes for ``model.scheduler`` and ``model.scheduler_config``.

        Both are ``None`` for ``constant``. ``OneCycleLR`` is left partial: the optimizer
        builder fills in ``total_steps`` from the trainer's estimated stepping batches.
        """
        if self.type is ScheduleType.CONSTANT:
            return None, None
        if self.type is ScheduleType.COSINE:
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
        """One-line summary for logs."""
        if self.type is ScheduleType.CONSTANT:
            return "constant lr"
        if self.type is ScheduleType.COSINE:
            return f"cosine from peak to peak x {self.final_lr_ratio:g}, no warmup"
        return (
            f"one_cycle: peak/{self.div_factor:g} -> peak at {self.pct_start:.0%} -> "
            f"peak/{self.div_factor * self.final_div_factor:g}"
            + (", momentum cycling" if self.cycle_momentum else "")
        )


class QATConfig(BaseModel):
    """The ``quantization.qat`` block, allowed only under ``mode: qat``.

    ``epochs`` and ``lr`` (the schedule peak) are recipe choices and have no default:
    recommended are epochs of about 10 % of the original training and a peak lr of about
    1 % of the original (Wu et al. 2020). ``calibrate_samples`` counts samples, not batches.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int
    lr: float
    schedule: QATScheduleConfig = QATScheduleConfig()
    # The skip_quantize subtrees carry no fake-quant STE masking, so they absorb most
    # of the gradient and drift past the frozen downstream amax; freeze them (they
    # deploy as FP16 anyway).
    freeze_unquantized: bool = True
    # Validate within the epoch (Lightning ``val_check_interval`` fraction): QAT can degrade
    # inside a single epoch, and ``best.ckpt`` only picks among validated states.
    val_check_interval: float = Field(default=0.25, gt=0.0, le=1.0)
    calibrate_samples: int = 400


class PTQConfig(BaseModel):
    """The ``quantization.ptq`` block, allowed only under ``mode: ptq``.

    The calibration run's settings live in the experiment config, so one file reproduces the
    checkpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    calibrate_samples: int
    batch_size: int = Field(default=1, ge=1)
    calib_seed: int | None = None
    calib_shuffle: bool = False


class QuantizationConfig(BaseModel):
    """Typed view of the ``quantization`` config section."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    mode: Mode = Mode.PTQ
    default_precision: Precision = Precision.INT8
    # Glob patterns over module names. A matched module and its whole subtree stay
    # un-quantized and run at the deploy onnx.precision.
    skip_quantize: tuple[str, ...] = ()
    # Shared by PTQ and the calibration that starts QAT.
    calibration: CalibrationConfig = CalibrationConfig()
    # Quantize stage only: log the placement record and exit before calibrating.
    dry_run: bool = False
    # Each is allowed only under its own mode; ``ptq: null`` lets a qat config drop an
    # inherited block. Loading never reads them: PTQ and QAT checkpoints rebuild the same tree.
    ptq: PTQConfig | None = None
    qat: QATConfig | None = None

    @model_validator(mode="after")
    def _producer_block_matches_the_mode(self) -> QuantizationConfig:
        if self.qat is not None and self.mode is not Mode.QAT:
            raise ValueError(
                f"quantization has a qat block but mode={self.mode.value!r}; set mode: qat or "
                "drop the block."
            )
        if self.ptq is not None and self.mode is not Mode.PTQ:
            raise ValueError(
                f"quantization has a ptq block but mode={self.mode.value!r}; set mode: ptq or "
                "drop the block (a qat config inheriting one can set ptq: null)."
            )
        return self

    @classmethod
    def from_dict(cls, raw: Any) -> QuantizationConfig:
        """Build from the resolved ``quantization`` mapping; an empty or absent one is disabled."""
        return cls.model_validate(raw or {})

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping that round-trips through :meth:`from_dict`; checkpoints embed it."""
        return self.model_dump(mode="json")
