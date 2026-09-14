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

"""Tests for skip_quantize glob expansion (subtree semantics, zero-match warning)."""

from __future__ import annotations

import logging

from torch import nn

from autoware_ml.quantization.core.replace import expand_skip_quantize


def _tiny_model() -> nn.Module:
    model = nn.Module()
    tower = nn.Module()
    tower.conv1 = nn.Conv2d(3, 3, 1)
    tower.block = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU())
    model.pts_backbone = tower
    model.pts_voxel_encoder = nn.Sequential(nn.Linear(4, 4))
    return model


class TestExpandSkipQuantize:
    def test_bare_name_expands_to_subtree(self):
        skip = expand_skip_quantize(_tiny_model(), ["pts_voxel_encoder"], log=False)
        assert "pts_voxel_encoder" in skip
        # Descendants are materialized so tower-root entries actually skip the tower.
        assert "pts_voxel_encoder.0" in skip

    def test_glob_pattern_matches(self):
        skip = expand_skip_quantize(_tiny_model(), ["pts_backbone.block*"], log=False)
        assert "pts_backbone.block" in skip
        assert "pts_backbone.block.0" in skip
        assert "pts_backbone.conv1" not in skip

    def test_zero_match_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            skip = expand_skip_quantize(_tiny_model(), ["does_not_exist*"], log=True)
        assert skip == set()
        assert any("does_not_exist" in record.message for record in caplog.records)

    def test_empty_patterns_yield_empty_set(self):
        assert expand_skip_quantize(_tiny_model(), [], log=False) == set()
