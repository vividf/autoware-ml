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

"""``quantize`` puts the QAT schedule into the config before the model is instantiated:
Hydra binds the optimizer and scheduler then, so a later override never reaches them."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from autoware_ml.scripts import quantize


class _ModelInstantiated(Exception):
    pass


def test_the_model_is_instantiated_from_the_qat_schedule(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "quantization": {
                "enabled": True,
                "mode": "qat",
                "dry_run": True,
                "qat": {"epochs": 2, "lr": 1.0e-5},
            },
            "model": {
                "_target_": "unused.Model",
                "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1.0e-3},
                "scheduler": None,
                "scheduler_config": None,
            },
            "trainer": {"max_epochs": 30, "devices": 4, "precision": "16-mixed"},
        }
    )
    seen = {}

    def instantiate(node, *args, **kwargs):
        seen["model"] = OmegaConf.to_container(node, resolve=True)
        raise _ModelInstantiated

    monkeypatch.setattr(quantize.hydra.utils, "instantiate", instantiate)
    monkeypatch.setattr(quantize, "log_configuration", lambda cfg: None)
    monkeypatch.setattr(quantize, "resolve_work_dir", lambda: Path("."))
    monkeypatch.setattr(quantize, "get_user_config_name", lambda: "unused")

    with pytest.raises(_ModelInstantiated):
        quantize.main(cfg)

    assert seen["model"]["optimizer"]["lr"] == 1.0e-5
    assert seen["model"]["scheduler"] is not None
    assert seen["model"]["scheduler_config"] is not None
    assert cfg.trainer.max_epochs == 2 and cfg.trainer.devices == 1
