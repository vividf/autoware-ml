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

"""Self-describing quantized checkpoints: embed, detect, round-trip, verify."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autoware_ml.quantization.checkpoint import (
    QUANTIZATION_KEY,
    QuantizationDescription,
    attach_quantization,
    find_quantization,
    read_quantization,
    read_quantization_from_file,
    save_quantized_checkpoint,
)
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import PlacementRecord

_CONFIG = QuantizationConfig.from_dict(
    {
        "enabled": True,
        "mode": "ptq",
        "skip_quantize": ["pts_voxel_encoder"],
        "disable_recipes": ["residual_add"],
        "ptq": {"calibrate_samples": 4, "calib_seed": 0},
    }
)


def _record(*modules: str) -> PlacementRecord:
    record = PlacementRecord()
    for module in modules:
        record.add(module, "replace_module", "submodule rule", "Conv2d -> QuantConv2d")
    return record


class TestDescriptionRoundTrip:
    def test_payload_round_trips_config_and_record(self):
        description = QuantizationDescription(
            config=_CONFIG, placement_record=_record("a.conv", "b.conv")
        )
        restored = QuantizationDescription.from_payload(description.to_payload())
        assert restored.config == _CONFIG
        assert restored.placement_record.decisions == description.placement_record.decisions

    def test_qat_config_round_trips(self):
        qat = QuantizationConfig.from_dict(
            {
                "enabled": True,
                "mode": "qat",
                "qat": {
                    "epochs": 2,
                    "lr": 1e-5,
                    "schedule": {"type": "one_cycle", "div_factor": 5},
                },
            }
        )
        assert QuantizationConfig.from_dict(qat.to_dict()) == qat


class TestCheckpointFiles:
    def test_save_read_and_detect(self, tmp_path):
        model = nn.Linear(2, 2)
        description = QuantizationDescription(config=_CONFIG, placement_record=_record("a.conv"))
        path = save_quantized_checkpoint(model, tmp_path / "ptq.ckpt", description)

        checkpoint = torch.load(path, weights_only=False)
        assert set(checkpoint) == {"state_dict", QUANTIZATION_KEY}
        assert set(checkpoint["state_dict"]) == {"weight", "bias"}
        restored = read_quantization(checkpoint)
        assert restored.config == _CONFIG
        assert (
            read_quantization_from_file(path).placement_record.decisions
            == description.placement_record.decisions
        )

    def test_fp_checkpoint_has_no_description(self, tmp_path):
        path = tmp_path / "fp.ckpt"
        torch.save({"state_dict": nn.Linear(2, 2).state_dict()}, path)
        assert read_quantization_from_file(path) is None
        assert find_quantization([path]) is None

    def test_find_returns_the_single_quantized_checkpoint(self, tmp_path):
        fp = tmp_path / "fp.ckpt"
        torch.save({"state_dict": {}}, fp)
        quantized = save_quantized_checkpoint(
            nn.Linear(2, 2),
            tmp_path / "ptq.ckpt",
            QuantizationDescription(config=_CONFIG, placement_record=_record()),
        )
        found = find_quantization([fp, quantized])
        assert found is not None
        assert found[0] == quantized

    def test_two_quantized_checkpoints_rejected(self, tmp_path):
        description = QuantizationDescription(config=_CONFIG, placement_record=_record())
        a = save_quantized_checkpoint(nn.Linear(2, 2), tmp_path / "a.ckpt", description)
        b = save_quantized_checkpoint(nn.Linear(2, 2), tmp_path / "b.ckpt", description)
        with pytest.raises(ValueError, match="exactly one"):
            find_quantization([a, b])

    def test_attach_mimics_lightning_on_save_checkpoint(self):
        checkpoint = {"state_dict": {}, "epoch": 3}
        attach_quantization(
            checkpoint, QuantizationDescription(config=_CONFIG, placement_record=_record("x"))
        )
        assert read_quantization(checkpoint).placement_record.decisions[0].module == "x"
        assert checkpoint["epoch"] == 3


class TestRecordVerify:
    def test_matching_record_passes(self):
        _record("a.conv").verify_matches(_record("a.conv"), source="test")

    def test_drift_raises(self):
        with pytest.raises(RuntimeError, match="drift"):
            _record("b.conv").verify_matches(_record("a.conv"), source="test")


class TestDisableRecipesValidation:
    def test_unknown_recipe_name_is_rejected_instead_of_silently_ignored(self):
        with pytest.raises(ValueError, match="unknown recipe"):
            QuantizationConfig.from_dict({"enabled": True, "disable_recipes": ["add"]})
