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

"""Typed ``deploy.verification`` and ``deploy.evaluation`` config sections.

Both run after export, for models that declare a stage graph. Unknown keys are rejected: a
misspelled ``tolerance`` would otherwise leave the gate at its default.
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
    """``deploy.verification``: cross-backend parity of the raw model outputs."""

    enabled: bool = False
    #: Absolute tolerance on the raw outputs; scenarios of lossy backends (fp16, int8) set
    #: their own.
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
    """``deploy.evaluation``: ground-truth metrics and latency of every backend."""

    enabled: bool = False
    #: ``test``, or ``val`` when the test split is unavailable or held back.
    split: str = "test"
    #: Samples per backend; -1 for the whole split.
    num_samples: int = -1
    #: Discarded re-runs of the first batch that warm up the GPU and TensorRT.
    num_warmup: int = 2
    #: Replaces the split's training-time batch size: the deployed graphs are single-frame
    #: (a model's glue may reject a larger batch) and the latency is per batch.
    batch_size: int = 1
    #: Thread cap of PyTorch's intra-op pool and the BLAS / OpenMP pools threadpoolctl finds;
    #: 0 keeps the process defaults. A TensorRT engine with data-dependent shapes (sparse
    #: convolutions) waits on the host thread at its shape syncs, and the ~30 OpenBLAS
    #: workers a numpy op leaves busy-waiting doubled the measured time (BEVFusion j6gen2:
    #: 10-13 ms vs 5.5 ms with one thread, the standalone engine figure). One thread slows
    #: the CPU metrics down but keeps the latency right.
    cpu_threads: int = 1
    backends: Mapping[Backend, BackendEvaluationConfig] = field(default_factory=dict)

    KNOWN_KEYS = frozenset(
        {"enabled", "split", "num_samples", "num_warmup", "batch_size", "cpu_threads", "backends"}
    )

    @classmethod
    def from_dict(cls, raw: Any) -> EvaluationConfig:
        raw = _mapping(raw, "deploy.evaluation")
        _reject_unknown(raw, cls.KNOWN_KEYS, "deploy.evaluation")
        backends = _mapping(raw.get("backends"), "deploy.evaluation.backends")
        split = str(raw.get("split", "test"))
        if split not in ("test", "val"):
            raise ValueError(f"deploy.evaluation.split={split!r} — valid values: 'test', 'val'.")
        batch_size = int(raw.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError(f"deploy.evaluation.batch_size={batch_size} — must be at least 1.")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            split=split,
            num_samples=int(raw.get("num_samples", -1)),
            num_warmup=int(raw.get("num_warmup", 2)),
            batch_size=batch_size,
            cpu_threads=int(raw.get("cpu_threads", 1)),
            backends={
                Backend(name): BackendEvaluationConfig.from_dict(cfg, name)
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
