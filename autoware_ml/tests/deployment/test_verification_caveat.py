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

"""A model whose raw graph outputs are incomparable across backends declares why,
and the verification gate skips loudly instead of the reason living in a config
comment (or worse, an empirically 'calibrated' tolerance nobody can defend)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from autoware_ml.deployment.config import DeployConfig
from autoware_ml.scripts.deploy import verify


def _deploy_cfg(verification_enabled: bool) -> DeployConfig:
    return DeployConfig.from_dict(
        {
            "onnx": {"enabled": True},
            "tensorrt": {"enabled": False},
            "stages": {},
            "verification": {
                "enabled": verification_enabled,
                "scenarios": [
                    {"ref": {"backend": "pytorch"}, "test": {"backend": "onnx"}},
                ],
            },
        }
    )


class _ExplodingDatamodule:
    """verify() must decide on the caveat BEFORE touching any data."""

    def predict_dataloader(self):
        raise AssertionError("verification should have been skipped before loading data")


def test_declared_caveat_skips_verification_loudly(caplog) -> None:
    model = SimpleNamespace(verification_caveat="raw outputs are stochastic by construction")
    with caplog.at_level(logging.WARNING, logger="autoware_ml.scripts.deploy"):
        verify(
            _deploy_cfg(verification_enabled=True),
            pipelines=None,
            datamodule=_ExplodingDatamodule(),
            model=model,
            device="cpu",
            available=set(),
        )
    assert "Verification SKIPPED" in caplog.text
    assert "stochastic by construction" in caplog.text


def test_declared_models_carry_their_reasons() -> None:
    from autoware_ml.models.detection3d.main_modules.bevfusion.model import (
        BEVFusionLidarDetectionModel,
    )
    from autoware_ml.models.multi_task_base_model import MultiTaskBaseModel
    from autoware_ml.models.segmentation3d.main_modules.ptv3.model import PTv3SegmentationModel

    assert MultiTaskBaseModel.verification_caveat is None
    assert "shuffle_orders" in PTv3SegmentationModel.verification_caveat
    assert "proposals" in BEVFusionLidarDetectionModel.verification_caveat
