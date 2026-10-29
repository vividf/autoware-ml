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
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from autoware_ml.deployment.verification.backend_verifier import VerificationScenario
from autoware_ml.types.backend import Backend


class VerificationConfig(BaseModel):
    """``deploy.verification``: cross-backend parity of the raw model outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    #: Absolute tolerance on the raw outputs; scenarios of lossy backends (fp16, int8) set
    #: their own.
    tolerance: float = 0.01
    num_verify_batches: int = 1
    scenarios: tuple[VerificationScenario, ...] = ()


class BackendEvaluationConfig(BaseModel):
    """One entry of ``deploy.evaluation.backends``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    device: str = "cuda"


class EvaluationConfig(BaseModel):
    """``deploy.evaluation``: ground-truth metrics and latency of every backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    #: ``test``, or ``val`` when the test split is unavailable or held back.
    split: Literal["test", "val"] = "test"
    #: Samples per backend; -1 for the whole split.
    num_samples: int = -1
    #: Discarded re-runs of the first batch that warm up the GPU and TensorRT.
    num_warmup: int = 2
    #: Replaces the split's training-time batch size: the deployed graphs are single-frame
    #: (a model's glue may reject a larger batch) and the latency is per batch.
    batch_size: int = Field(default=1, ge=1)
    #: Thread cap of PyTorch's intra-op pool and the BLAS / OpenMP pools threadpoolctl finds;
    #: 0 keeps the process defaults. A TensorRT engine with data-dependent shapes (sparse
    #: convolutions) waits on the host thread at its shape syncs, and the ~30 OpenBLAS
    #: workers a numpy op leaves busy-waiting doubled the measured time (BEVFusion j6gen2:
    #: 10-13 ms vs 5.5 ms with one thread, the standalone engine figure). One thread slows
    #: the CPU metrics down but keeps the latency right.
    cpu_threads: int = 1
    backends: dict[Backend, BackendEvaluationConfig] = {}

    def enabled_backends(self) -> list[tuple[Backend, BackendEvaluationConfig]]:
        """Backends with ``enabled: true``, in configuration order."""
        return [(backend, cfg) for backend, cfg in self.backends.items() if cfg.enabled]


class PostExportConfig(BaseModel):
    """The ``deploy.verification`` and ``deploy.evaluation`` sections together."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verification: VerificationConfig = VerificationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @classmethod
    def from_deploy_cfg(cls, deploy_cfg: Mapping[str, Any] | DictConfig | None) -> PostExportConfig:
        """Parse the two sections of the ``deploy`` config; a missing section is all defaults."""
        if isinstance(deploy_cfg, DictConfig):
            deploy_cfg = OmegaConf.to_container(deploy_cfg, resolve=True)
        raw = deploy_cfg or {}
        return cls.model_validate(
            {
                name: raw[name]
                for name in ("verification", "evaluation")
                if raw.get(name) is not None
            }
        )

    @property
    def any_enabled(self) -> bool:
        return self.verification.enabled or self.evaluation.enabled
