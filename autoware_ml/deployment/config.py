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

"""Typed view of the Hydra ``deploy`` config section — parsed once, typo-guarded.

Layout (mirrors the stage graph):

.. code-block:: yaml

    deploy:
      onnx:      { enabled, dynamo, opset_version, do_constant_folding, precision, modify_graph }  # global
      tensorrt:  { enabled, workspace_size, plugin_libraries }                                     # global
      stages:                                     # per GraphStage, keyed by stage name
        <stage_name>:
          onnx:     { dynamic_axes | dynamic_shapes }
          tensorrt: { input_shapes: { <input>: { min_shape, opt_shape, max_shape } } }
      verification: { enabled, tolerance, num_verify_batches, scenarios }
      evaluation:   { enabled, num_samples, num_warmup, backends: { <backend>: { enabled, device } } }

Every mapping rejects unknown keys: a misspelled option would otherwise silently fall
back to a default (``opset_versoin`` exports with the wrong opset; a stage name that
does not match the model's declaration silently drops its shape profile).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from autoware_ml.deployment.verification.backend_verifier import VerificationScenario
from autoware_ml.types.backend import Backend
from autoware_ml.utils.config_parsing import reject_unknown_keys as _reject_unknown


def _mapping(raw: Any, where: str) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{where} must be a mapping, got {type(raw).__name__}.")
    return raw


class OnnxPrecision(str, Enum):
    """Precision of the exported graphs.

    Engines build strongly typed, so this is where FP16 is decided: ``FP16`` runs
    ModelOpt AutoCast on every exported stage without Q/DQ nodes (quantized stages
    keep the precision their checkpoint bakes in); ``FP32`` exports as traced.
    """

    FP32 = "fp32"
    FP16 = "fp16"


@dataclass(frozen=True)
class OnnxConfig:
    """Global ONNX export options (``deploy.onnx``)."""

    enabled: bool = True
    dynamo: bool = True
    opset_version: int = 21
    do_constant_folding: bool = True
    precision: OnnxPrecision = OnnxPrecision.FP32
    #: Optional Hydra-instantiable graph modifier applied to every exported ONNX.
    modify_graph: Any = None

    KNOWN_KEYS = frozenset(
        {"enabled", "dynamo", "opset_version", "do_constant_folding", "precision", "modify_graph"}
    )

    @classmethod
    def from_dict(cls, raw: Any) -> OnnxConfig:
        raw = _mapping(raw, "deploy.onnx")
        _reject_unknown(raw, cls.KNOWN_KEYS, "deploy.onnx")
        raw_precision = str(raw.get("precision", OnnxPrecision.FP32.value)).lower()
        try:
            precision = OnnxPrecision(raw_precision)
        except ValueError:
            raise ValueError(
                f"Unknown deploy.onnx.precision {raw_precision!r}. "
                f"Valid: {[p.value for p in OnnxPrecision]} "
                "(quantized precisions come from the checkpoint's Q/DQ nodes, not from here)."
            ) from None
        return cls(
            enabled=bool(raw.get("enabled", True)),
            dynamo=bool(raw.get("dynamo", True)),
            opset_version=int(raw.get("opset_version", 21)),
            do_constant_folding=bool(raw.get("do_constant_folding", True)),
            precision=precision,
            modify_graph=raw.get("modify_graph"),
        )


@dataclass(frozen=True)
class TensorRTConfig:
    """Global TensorRT build options (``deploy.tensorrt``).

    There is no precision knob: engines build strongly typed and read precision from
    the ONNX graph (Q/DQ for quantized precisions, tensor types for FP16 — see
    ``deploy.onnx.precision``). TensorRT deprecated the weak-typing precision flags in
    10.12 and removed them in 11.
    """

    enabled: bool = True
    workspace_size: int = 1 << 32
    plugin_libraries: tuple[str, ...] = ()

    KNOWN_KEYS = frozenset({"enabled", "workspace_size", "plugin_libraries"})

    @classmethod
    def from_dict(cls, raw: Any) -> TensorRTConfig:
        raw = _mapping(raw, "deploy.tensorrt")
        _reject_unknown(raw, cls.KNOWN_KEYS, "deploy.tensorrt")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            workspace_size=int(raw.get("workspace_size", 1 << 32)),
            plugin_libraries=tuple(str(p) for p in (raw.get("plugin_libraries") or ())),
        )


@dataclass(frozen=True)
class StageOnnxConfig:
    """Per-stage ONNX options (``deploy.stages.<name>.onnx``).

    Shape declarations plus an optional per-stage precision; input/output *names* are
    never configured — they come from the stage declaration.
    """

    #: Legacy exporter (``dynamo=false``): ``{tensor_name: {dim_index: dim_name}}``.
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None
    #: Dynamo exporter: ``{input_name: {dim_index: dim_name | {name, min, max}}}``.
    dynamic_shapes: Mapping[str, Mapping[int, Any]] | None = None
    #: Overrides ``deploy.onnx.precision`` for this stage only — for a pipeline whose
    #: stages need different precisions (one numerically fragile head kept FP32, say).
    #: ``None`` inherits the global setting.
    precision: OnnxPrecision | None = None

    KNOWN_KEYS = frozenset({"dynamic_axes", "dynamic_shapes", "precision"})

    @classmethod
    def from_dict(cls, raw: Any, stage: str) -> StageOnnxConfig:
        raw = _mapping(raw, f"deploy.stages.{stage}.onnx")
        _reject_unknown(raw, cls.KNOWN_KEYS, f"deploy.stages.{stage}.onnx")
        raw_precision = raw.get("precision")
        try:
            precision = OnnxPrecision(str(raw_precision).lower()) if raw_precision else None
        except ValueError:
            raise ValueError(
                f"deploy.stages.{stage}.onnx.precision={raw_precision!r} — valid values: "
                f"{[p.value for p in OnnxPrecision]}."
            ) from None
        return cls(
            dynamic_axes=raw.get("dynamic_axes"),
            dynamic_shapes=raw.get("dynamic_shapes"),
            precision=precision,
        )


@dataclass(frozen=True)
class ShapeProfile:
    """One TensorRT optimization-profile entry."""

    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]

    KNOWN_KEYS = frozenset({"min_shape", "opt_shape", "max_shape"})

    @classmethod
    def from_dict(cls, raw: Any, where: str) -> ShapeProfile:
        raw = _mapping(raw, where)
        _reject_unknown(raw, cls.KNOWN_KEYS, where)
        missing = cls.KNOWN_KEYS - set(raw)
        if missing:
            raise ValueError(f"{where} is incomplete: missing {sorted(missing)}.")
        return cls(
            min_shape=tuple(int(x) for x in raw["min_shape"]),
            opt_shape=tuple(int(x) for x in raw["opt_shape"]),
            max_shape=tuple(int(x) for x in raw["max_shape"]),
        )


@dataclass(frozen=True)
class StageTensorRTConfig:
    """Per-stage TensorRT options (``deploy.stages.<name>.tensorrt``)."""

    input_shapes: Mapping[str, ShapeProfile] = field(default_factory=dict)

    KNOWN_KEYS = frozenset({"input_shapes"})

    @classmethod
    def from_dict(cls, raw: Any, stage: str) -> StageTensorRTConfig:
        where = f"deploy.stages.{stage}.tensorrt"
        raw = _mapping(raw, where)
        _reject_unknown(raw, cls.KNOWN_KEYS, where)
        shapes = _mapping(raw.get("input_shapes"), f"{where}.input_shapes")
        return cls(
            input_shapes={
                str(name): ShapeProfile.from_dict(profile, f"{where}.input_shapes.{name}")
                for name, profile in shapes.items()
            }
        )


@dataclass(frozen=True)
class StageConfig:
    """Everything configured for one exportable stage."""

    onnx: StageOnnxConfig = StageOnnxConfig()
    tensorrt: StageTensorRTConfig = StageTensorRTConfig()

    KNOWN_KEYS = frozenset({"onnx", "tensorrt"})

    @classmethod
    def from_dict(cls, raw: Any, stage: str) -> StageConfig:
        raw = _mapping(raw, f"deploy.stages.{stage}")
        _reject_unknown(raw, cls.KNOWN_KEYS, f"deploy.stages.{stage}")
        return cls(
            onnx=StageOnnxConfig.from_dict(raw.get("onnx"), stage),
            tensorrt=StageTensorRTConfig.from_dict(raw.get("tensorrt"), stage),
        )


@dataclass(frozen=True)
class VerificationConfig:
    """Cross-backend parity stage (``deploy.verification``)."""

    enabled: bool = False
    #: Default absolute tolerance on raw graph outputs. Lossy backends (fp16 / int8)
    #: set an explicit per-scenario ``tolerance`` instead of loosening this.
    tolerance: float = 0.01
    num_verify_batches: int = 1
    scenarios: tuple[VerificationScenario, ...] = ()

    KNOWN_KEYS = frozenset({"enabled", "tolerance", "num_verify_batches", "scenarios"})

    @classmethod
    def from_dict(cls, raw: Any) -> VerificationConfig:
        raw = _mapping(raw, "deploy.verification")
        _reject_unknown(raw, cls.KNOWN_KEYS, "deploy.verification")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            tolerance=float(raw.get("tolerance", 0.01)),
            num_verify_batches=int(raw.get("num_verify_batches", 1)),
            scenarios=tuple(
                VerificationScenario.from_dict(s) for s in (raw.get("scenarios") or ())
            ),
        )


@dataclass(frozen=True)
class BackendEvaluationConfig:
    """One entry of ``deploy.evaluation.backends``."""

    enabled: bool = True
    device: str = "cuda"

    KNOWN_KEYS = frozenset({"enabled", "device"})

    @classmethod
    def from_dict(cls, raw: Any, backend: str) -> BackendEvaluationConfig:
        where = f"deploy.evaluation.backends.{backend}"
        raw = _mapping(raw, where)
        _reject_unknown(raw, cls.KNOWN_KEYS, where)
        return cls(enabled=bool(raw.get("enabled", True)), device=str(raw.get("device", "cuda")))


@dataclass(frozen=True)
class EvaluationConfig:
    """Per-backend ground-truth evaluation stage (``deploy.evaluation``)."""

    enabled: bool = False
    #: Split the backends are scored on: ``test`` (default, the predict dataloader) or
    #: ``val`` — for when the test split is unavailable or held back. Metric keys carry
    #: the split, so a val evaluation reports under ``val/{backend}/...``.
    split: str = "test"
    #: Samples per backend; -1 = the whole split.
    num_samples: int = -1
    #: Extra re-runs of the first batch that prime the GPU / TensorRT (discarded).
    num_warmup: int = 2
    backends: Mapping[Backend, BackendEvaluationConfig] = field(default_factory=dict)

    KNOWN_KEYS = frozenset({"enabled", "split", "num_samples", "num_warmup", "backends"})

    @classmethod
    def from_dict(cls, raw: Any) -> EvaluationConfig:
        raw = _mapping(raw, "deploy.evaluation")
        _reject_unknown(raw, cls.KNOWN_KEYS, "deploy.evaluation")
        backends = _mapping(raw.get("backends"), "deploy.evaluation.backends")
        split = str(raw.get("split", "test"))
        if split not in ("test", "val"):
            raise ValueError(
                f"deploy.evaluation.split={split!r} — valid values: 'test', 'val'."
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            split=split,
            num_samples=int(raw.get("num_samples", -1)),
            num_warmup=int(raw.get("num_warmup", 2)),
            backends={
                Backend.parse(name): BackendEvaluationConfig.from_dict(cfg, name)
                for name, cfg in backends.items()
            },
        )

    def enabled_backends(self) -> list[tuple[Backend, BackendEvaluationConfig]]:
        """Backends with ``enabled: true``, in configuration order."""
        return [(backend, cfg) for backend, cfg in self.backends.items() if cfg.enabled]


@dataclass(frozen=True)
class DeployConfig:
    """Typed view of the whole ``deploy`` section."""

    onnx: OnnxConfig = OnnxConfig()
    tensorrt: TensorRTConfig = TensorRTConfig()
    stages: Mapping[str, StageConfig] = field(default_factory=dict)
    verification: VerificationConfig = VerificationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    KNOWN_KEYS = frozenset({"onnx", "tensorrt", "stages", "verification", "evaluation"})

    @classmethod
    def from_dict(cls, raw: Any) -> DeployConfig:
        """Parse the resolved ``deploy`` mapping (``OmegaConf.to_container(..., resolve=True)``).

        Raises:
            ValueError: On unknown keys anywhere in the section or an invalid value.
            TypeError: When a sub-section is not a mapping.
        """
        raw = _mapping(raw, "deploy")
        _reject_unknown(raw, cls.KNOWN_KEYS, "deploy")
        stages = _mapping(raw.get("stages"), "deploy.stages")
        return cls(
            onnx=OnnxConfig.from_dict(raw.get("onnx")),
            tensorrt=TensorRTConfig.from_dict(raw.get("tensorrt")),
            stages={
                str(name): StageConfig.from_dict(cfg, str(name)) for name, cfg in stages.items()
            },
            verification=VerificationConfig.from_dict(raw.get("verification")),
            evaluation=EvaluationConfig.from_dict(raw.get("evaluation")),
        )

    def stage(self, name: str) -> StageConfig:
        """Per-stage options, defaulting to empty when the stage has no entry."""
        return self.stages.get(name, StageConfig())

    def check_stage_names(self, declared: Sequence[str]) -> None:
        """Raise when ``deploy.stages`` names a stage the model does not declare.

        A stage name typo would otherwise silently drop that stage's dynamic axes and
        TensorRT shape profile.
        """
        unknown = sorted(set(self.stages) - set(declared))
        if unknown:
            raise ValueError(
                f"deploy.stages configures unknown stage(s) {unknown}; the model declares "
                f"exportable stages {list(declared)}."
            )
