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

"""Unit tests for the ``quantize`` entrypoint."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from autoware_ml.scripts import quantize


class _ModelBuilt(Exception):
    pass


class TestRunQuantize(unittest.TestCase):
    """``run_quantize`` builds the QAT model and trainer from the QAT schedule."""

    def test_the_model_is_built_from_the_qat_schedule(self) -> None:
        # Shaped like the experiment configs: model.optimizer interpolates the top-level node,
        # and the dynamic validation callback is registered.
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

        with (
            patch.object(quantize, "build_model", build_model),
            patch.object(quantize, "build_data_preprocessor", lambda cfg: None),
            patch.object(quantize, "log_configuration", lambda cfg: None),
            patch.object(quantize, "restrict_console_logging_to_rank_zero", lambda: None),
            patch.object(
                quantize.HydraConfig,
                "get",
                staticmethod(
                    lambda: SimpleNamespace(job=SimpleNamespace(config_name="tasks/unused"))
                ),
            ),
            self.assertRaises(_ModelBuilt),
        ):
            quantize.run_quantize(cfg)

        self.assertEqual(seen["model"]["optimizer"]["lr"], 1.0e-5)
        self.assertEqual(seen["model"]["optimizer"]["_target_"], "torch.optim.AdamW")
        self.assertIsNotNone(seen["model"]["scheduler"])
        self.assertIsNotNone(seen["model"]["scheduler_config"])
        self.assertEqual(cfg.trainer.max_epochs, 2)
        self.assertEqual(cfg.trainer.devices, 1)
        self.assertIsNone(cfg.callbacks.dynamic_validation)
        self.assertEqual(cfg.optimizer.lr, 1.0e-3)


if __name__ == "__main__":
    unittest.main()
