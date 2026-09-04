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

"""Tests for the plan layer (rules validation, placement-record round-trip, decision recording).

The one ``prepare`` test uses a model whose submodule rules/recipes match nothing
quantizable, so only the BN-fuse and skip_quantize decisions are exercised.
"""

from __future__ import annotations

import pytest
from torch import nn

from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import (
    PlacementRecord,
    QuantizationPlan,
    QuantRules,
)


def _record(*entries: tuple) -> PlacementRecord:
    record = PlacementRecord()
    for module, transform, reason, detail in entries:
        record.add(module, transform, reason, detail)
    return record


class TestQuantRules:
    def test_valid_rules_pass(self):
        rules = QuantRules(
            quantize_submodules={"pts_backbone": ("conv", "linear")}, recipes=("residual_add",)
        )
        assert rules.quantize_submodules["pts_backbone"] == ("conv", "linear")

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown module kind"):
            QuantRules(quantize_submodules={"pts_backbone": ("conv3d",)})

    def test_unknown_recipe_rejected(self):
        with pytest.raises(ValueError, match="unknown recipe"):
            QuantRules(quantize_submodules={}, recipes=("attention",))


class TestRecordRoundTrip:
    def test_json_dict_round_trip(self):
        record = _record(
            (
                "pts_backbone.conv1",
                "replace_module",
                "submodule rule: pts_backbone",
                "Conv2d -> QuantConv2d",
            ),
            ("pts_neck.deblocks.0", "skip_quantize", "skip_quantize pattern 'pts_neck.*'", ""),
        )
        loaded = PlacementRecord.from_json_dict(record.to_json_dict())
        assert loaded.decisions == record.decisions


class TestRecordDiff:
    def test_identical_records_have_empty_diff(self):
        entry = ("a.conv", "replace_module", "submodule rule: a", "Conv2d -> QuantConv2d")
        only_self, only_other = _record(entry).diff(_record(entry))
        assert only_self == [] and only_other == []

    def test_order_is_irrelevant(self):
        one = ("a.conv", "replace_module", "r", "d")
        two = ("b.conv", "replace_module", "r", "d")
        only_self, only_other = _record(one, two).diff(_record(two, one))
        assert only_self == [] and only_other == []

    def test_drift_is_reported_on_both_sides(self):
        base = ("a.conv", "replace_module", "r", "d")
        only_self, only_other = _record(base, ("extra.conv", "replace_module", "r", "d")).diff(
            _record(base, ("other.conv", "wrap_module", "r", "d"))
        )
        assert [d.module for d in only_self] == ["extra.conv"]
        assert [d.module for d in only_other] == ["other.conv"]


class TestPrepareRecordsDecisions:
    """Backend-free slice of ``prepare``: BN fuse + skip_quantize recording."""

    @staticmethod
    def _model() -> nn.Module:
        model = nn.Module()
        tower = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.ReLU())
        model.some_tower = tower
        return model

    def test_fuse_and_skip_quantize_decisions_recorded(self):
        config = QuantizationConfig.from_dict(
            {
                "enabled": True,
                "mode": "ptq",
                "skip_quantize": ["some_tower"],
                "disable_recipes": ["residual_add", "maxpool"],
                "ptq": {"calibrate_samples": 4},
            }
        )
        # Submodule rules/recipes that match nothing on this model: only the
        # fuse_bn and skip_quantize transforms run.
        plan = QuantizationPlan(
            rules=QuantRules(quantize_submodules={"absent_submodule": ("conv",)}), config=config
        )
        model = self._model()
        plan.prepare(model)

        transforms = {d.transform for d in plan.placement_record.decisions}
        assert transforms == {"fuse_bn", "skip_quantize"}
        fuse = [d for d in plan.placement_record.decisions if d.transform == "fuse_bn"]
        assert fuse[0].module == "some_tower.0"
        assert "some_tower.1" in fuse[0].detail
        # The BN really was folded away.
        assert isinstance(model.some_tower[1], nn.Identity)
        kept = [d for d in plan.placement_record.decisions if d.transform == "skip_quantize"]
        assert kept[0].module == "some_tower"
        assert "skip_quantize pattern" in kept[0].reason

    def test_dry_run_config_key_parses(self):
        config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "ptq", "dry_run": True, "ptq": {"calibrate_samples": 4}}
        )
        assert config.dry_run
        assert not QuantizationConfig.from_dict(None).dry_run
