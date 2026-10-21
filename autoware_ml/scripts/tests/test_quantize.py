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

"""``quantize`` puts the QAT schedule into the config before the model is built: Hydra
binds the optimizer and scheduler then, so a later override never reaches them."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from autoware_ml.scripts import quantize


class _ModelBuilt(Exception):
    pass


def test_the_model_is_built_from_the_qat_schedule(monkeypatch) -> None:
    # The experiment config shape: the model's optimizer interpolates the top-level node and
    # every experiment registers the dynamic validation callback.
    cfg = OmegaConf.create(
        {
            "quantization": {
                "enabled": True,
                "mode": "qat",
                "dry_run": True,
                "qat": {"epochs": 2, "lr": 1.0e-5},
            },
            "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1.0e-3},
            "model": {
                "_target_": "unused.Model",
                "optimizer": "${optimizer}",
                "scheduler": None,
                "scheduler_config": None,
            },
            "trainer": {"max_epochs": 30, "devices": 4, "precision": "16-mixed"},
            "callbacks": {"dynamic_validation": {"_target_": "unused.DynamicValidation"}},
        }
    )
    seen = {}

    def build_model(cfg, **kwargs):
        seen["model"] = OmegaConf.to_container(cfg.model, resolve=True)
        raise _ModelBuilt

    monkeypatch.setattr(quantize, "build_model", build_model)
    monkeypatch.setattr(quantize, "build_data_preprocessor", lambda cfg: None)
    monkeypatch.setattr(quantize, "log_configuration", lambda cfg: None)
    monkeypatch.setattr(quantize, "restrict_console_logging_to_rank_zero", lambda: None)
    monkeypatch.setattr(
        quantize.HydraConfig,
        "get",
        staticmethod(lambda: SimpleNamespace(job=SimpleNamespace(config_name="tasks/unused"))),
    )

    with pytest.raises(_ModelBuilt):
        quantize.run_quantize(cfg)

    assert seen["model"]["optimizer"]["lr"] == 1.0e-5
    assert seen["model"]["optimizer"]["_target_"] == "torch.optim.AdamW"
    assert seen["model"]["scheduler"] is not None
    assert seen["model"]["scheduler_config"] is not None
    assert cfg.trainer.max_epochs == 2 and cfg.trainer.devices == 1
    assert cfg.callbacks.dynamic_validation is None
    assert cfg.optimizer.lr == 1.0e-3  # the training node is left alone
