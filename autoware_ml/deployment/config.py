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

"""Typed ``deploy.verification`` config section.

It runs after export, for models that declare a stage graph. Unknown keys are rejected: a
misspelled ``tolerance`` would otherwise leave the gate at its default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

from autoware_ml.deployment.verification.backend_verifier import VerificationScenario


class VerificationConfig(BaseModel):
    """``deploy.verification``: cross-backend parity of the raw model outputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    #: Absolute tolerance on the raw outputs; scenarios of lossy backends (fp16, int8) set
    #: their own.
    tolerance: float = 0.01
    num_verify_batches: int = 1
    scenarios: tuple[VerificationScenario, ...] = ()


class PostExportConfig(BaseModel):
    """The post-export sections of the ``deploy`` config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verification: VerificationConfig = VerificationConfig()

    @classmethod
    def from_deploy_cfg(cls, deploy_cfg: Mapping[str, Any] | DictConfig | None) -> PostExportConfig:
        """Parse the sections of the ``deploy`` config; a missing section is all defaults."""
        if isinstance(deploy_cfg, DictConfig):
            deploy_cfg = OmegaConf.to_container(deploy_cfg, resolve=True)
        raw = deploy_cfg or {}
        return cls.model_validate(
            {name: raw[name] for name in ("verification",) if raw.get(name) is not None}
        )

    @property
    def any_enabled(self) -> bool:
        return self.verification.enabled
