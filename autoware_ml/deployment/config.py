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

"""Typed view of the post-export sections of the Hydra ``deploy`` config.

Export itself (``deploy.onnx``, ``deploy.tensorrt``, ``deploy.onnx.modules.<module>``) is
read by ``scripts/deploy.py`` as before. The two sections parsed here run *after* export
and only for models that declare a stage graph:

.. code-block:: yaml

    deploy:
      verification: { enabled, tolerance, num_verify_batches, scenarios }
      evaluation:   { enabled, split, num_samples, num_warmup,
                      backends: { <backend>: { enabled, device } } }

Every mapping rejects unknown keys: a misspelled option would otherwise silently fall
back to a default (``tolerence`` would leave the gate at its default and pass).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
    #: Split the backends are scored on: ``test`` (default) or ``val`` — for when the
    #: test split is unavailable or held back. Metric keys carry the split.
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
            raise ValueError(f"deploy.evaluation.split={split!r} — valid values: 'test', 'val'.")
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
class PostExportConfig:
    """The ``deploy.verification`` and ``deploy.evaluation`` sections together."""

    verification: VerificationConfig = VerificationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @classmethod
    def from_deploy_cfg(cls, deploy_cfg: Any) -> PostExportConfig:
        """Parse from the resolved ``deploy`` mapping (missing sections are all-defaults)."""
        raw = _mapping(deploy_cfg, "deploy")
        return cls(
            verification=VerificationConfig.from_dict(raw.get("verification")),
            evaluation=EvaluationConfig.from_dict(raw.get("evaluation")),
        )

    @property
    def any_enabled(self) -> bool:
        return self.verification.enabled or self.evaluation.enabled
