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

"""Unit tests for LatencyStats.from_samples."""

from __future__ import annotations

import statistics

import pytest

from autoware_ml.evaluation.latency import LatencyStats


class TestLatencyStatsFromSamples:
    def test_empty_input_yields_all_zero_stats(self):
        stats = LatencyStats.from_samples([])
        assert stats == LatencyStats(mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0)

    def test_single_sample_has_zero_std(self):
        stats = LatencyStats.from_samples([7.5])
        assert stats.mean == pytest.approx(7.5)
        assert stats.std == 0.0
        assert stats.min == pytest.approx(7.5)
        assert stats.max == pytest.approx(7.5)
        assert stats.median == pytest.approx(7.5)

    def test_multiple_samples_match_statistics_module(self):
        samples = [1.0, 2.0, 3.0, 4.0]
        stats = LatencyStats.from_samples(samples)
        assert stats.mean == pytest.approx(statistics.fmean(samples))
        assert stats.std == pytest.approx(statistics.pstdev(samples))
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(4.0)
        assert stats.median == pytest.approx(statistics.median(samples))

    def test_odd_count_median_is_middle_sample(self):
        stats = LatencyStats.from_samples([9.0, 1.0, 5.0])
        assert stats.median == pytest.approx(5.0)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(9.0)
