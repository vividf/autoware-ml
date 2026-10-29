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

"""Unit tests for quantization rules, plans and placement records."""

from __future__ import annotations

import pytest
from torch import nn

from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import PlacementRecord, prepare_quantized_tree
from autoware_ml.quantization.rules import QuantRules


def _record(*entries: tuple) -> PlacementRecord:
    record = PlacementRecord()
    for module, transform, reason, detail in entries:
        record.add(module, transform, reason, detail)
    return record


class TestQuantRules:
    def test_valid_rules_pass(self):
        rules = QuantRules(quantize_submodules={"pts_backbone": ("conv", "linear")})
        assert rules.quantize_submodules["pts_backbone"] == ("conv", "linear")

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown module kind"):
            QuantRules(quantize_submodules={"pts_backbone": ("conv3d",)})

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


class TestVerifyMatches:
    def test_identical_records_match(self):
        entry = ("a.conv", "replace_module", "submodule rule: a", "Conv2d -> QuantConv2d")
        _record(entry).verify_matches(_record(entry), source="test")

    def test_order_is_irrelevant(self):
        one = ("a.conv", "replace_module", "r", "d")
        two = ("b.conv", "replace_module", "r", "d")
        _record(one, two).verify_matches(_record(two, one), source="test")

    def test_drift_is_reported_on_both_sides(self):
        base = ("a.conv", "replace_module", "r", "d")
        rebuilt = _record(base, ("extra.conv", "replace_module", "r", "d"))
        saved = _record(base, ("other.conv", "wrap_module", "r", "d"))
        with pytest.raises(
            RuntimeError, match=r"(?s)rebuilt only: extra.conv.*saved only: other.conv"
        ):
            rebuilt.verify_matches(saved, source="test")


class TestVerifyMatchesIsStructural:
    def test_reworded_reason_and_detail_still_match(self):
        saved = _record(("a.conv", "replace_module", "submodule rule", "Conv2d -> QuantConv2d"))
        # A reworded reason, and a modelopt release that renamed the quantized class.
        rebuilt = _record(("a.conv", "replace_module", "rules[0] matched", "Conv2d -> QConv2d"))
        rebuilt.verify_matches(saved, source="test")

    def test_a_different_module_or_transform_still_raises(self):
        saved = _record(("a.conv", "replace_module", "r", "d"))
        with pytest.raises(RuntimeError, match="tree drift"):
            _record(("b.conv", "replace_module", "r", "d")).verify_matches(saved, source="test")
        with pytest.raises(RuntimeError, match="tree drift"):
            _record(("a.conv", "fuse_bn", "r", "d")).verify_matches(saved, source="test")

    def test_a_duplicated_decision_is_still_a_difference(self):
        saved = _record(("a.conv", "replace_module", "r", "d"))
        twice = _record(
            ("a.conv", "replace_module", "r", "d"), ("a.conv", "replace_module", "r", "d")
        )
        with pytest.raises(RuntimeError, match="tree drift"):
            twice.verify_matches(saved, source="test")


class TestPrepareRecordsDecisions:
    @staticmethod
    def _model() -> nn.Module:
        model = nn.Module()
        tower = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.ReLU())
        model.some_tower = tower
        return model

    def test_a_declared_submodule_the_model_lacks_is_an_error(self):
        config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "ptq", "ptq": {"calibrate_samples": 4}}
        )
        rules = QuantRules(quantize_submodules={"absent_submodule": ("conv",)})
        with pytest.raises(ValueError, match="absent_submodule"):
            prepare_quantized_tree(self._model(), config, rules=rules)

    def test_fuse_and_skip_quantize_decisions_recorded(self):
        config = QuantizationConfig.from_dict(
            {
                "enabled": True,
                "mode": "ptq",
                "skip_quantize": ["some_tower"],
                "ptq": {"calibrate_samples": 4},
            }
        )
        model = self._model()
        prepared = prepare_quantized_tree(
            model, config, rules=QuantRules(quantize_submodules={"some_tower": ("conv",)})
        )

        decisions = prepared.placement_record.decisions
        assert {d.transform for d in decisions} == {"fuse_bn", "skip_quantize"}
        fuse = [d for d in decisions if d.transform == "fuse_bn"]
        assert fuse[0].module == "some_tower.0"
        assert "some_tower.1" in fuse[0].detail
        assert isinstance(model.some_tower[1], nn.Identity)
        kept = [d for d in decisions if d.transform == "skip_quantize"]
        assert kept[0].module == "some_tower"
        assert "skip_quantize pattern" in kept[0].reason
        assert prepared.skip_names == {"some_tower", "some_tower.0", "some_tower.1", "some_tower.2"}

    def test_a_model_without_rules_cannot_be_quantized(self):
        config = QuantizationConfig.from_dict({"enabled": True, "ptq": {"calibrate_samples": 4}})

        class _Model(nn.Module):
            def build_quantization_rules(self):
                return None

        with pytest.raises(NotImplementedError, match="declares no quantization rules"):
            prepare_quantized_tree(_Model(), config)

    def test_dry_run_config_key_parses(self):
        config = QuantizationConfig.from_dict(
            {"enabled": True, "mode": "ptq", "dry_run": True, "ptq": {"calibrate_samples": 4}}
        )
        assert config.dry_run
        assert not QuantizationConfig.from_dict(None).dry_run
