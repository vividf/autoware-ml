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

"""QATCallback hard-boundary tests (mode, devices, precision, resume)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.qat_callback import QATCallback

_QAT_CONFIG = QuantizationConfig.from_dict(
    {"enabled": True, "mode": "qat", "qat": {"epochs": 1, "lr": 1e-4}}
)


def _trainer(num_devices=1, world_size=1, precision="32-true", ckpt_path=None):
    return SimpleNamespace(
        num_devices=num_devices,
        world_size=world_size,
        precision=precision,
        ckpt_path=ckpt_path,
    )


class TestQATCallbackBoundaries:
    def test_requires_qat_mode(self):
        ptq_config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "ptq", "ptq": {"calibrate_samples": 4}}
        )
        with pytest.raises(ValueError, match="mode='qat'"):
            QATCallback(ptq_config)

    def test_non_fit_stage_is_noop(self):
        callback = QATCallback(_QAT_CONFIG)
        callback.setup(_trainer(), pl_module=None, stage="validate")
        assert not callback._quantized

    def test_rejects_multi_device(self):
        callback = QATCallback(_QAT_CONFIG)
        with pytest.raises(RuntimeError, match="single-device"):
            callback.setup(_trainer(num_devices=2), pl_module=None, stage="fit")

    def test_rejects_amp(self):
        callback = QATCallback(_QAT_CONFIG)
        with pytest.raises(RuntimeError, match="32-true"):
            callback.setup(_trainer(precision="16-mixed"), pl_module=None, stage="fit")

    def test_rejects_resume(self):
        callback = QATCallback(_QAT_CONFIG)
        with pytest.raises(RuntimeError, match="resume"):
            callback.setup(_trainer(ckpt_path="/tmp/last.ckpt"), pl_module=None, stage="fit")
