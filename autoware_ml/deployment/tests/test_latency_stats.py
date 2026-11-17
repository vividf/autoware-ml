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
import unittest

import numpy as np

from autoware_ml.deployment.evaluation.latency import LatencyStats


class TestLatencyStatsFromSamples(unittest.TestCase):
    """Unit tests for the summary statistics of a list of latency samples."""

    def test_empty_input_yields_all_zero_stats(self):
        stats = LatencyStats.from_samples([])
        self.assertEqual(stats, LatencyStats(mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0))

    def test_single_sample_has_zero_std(self):
        stats = LatencyStats.from_samples([7.5])
        np.testing.assert_allclose(stats.mean, 7.5, rtol=1e-6, atol=1e-12)
        self.assertEqual(stats.std, 0.0)
        np.testing.assert_allclose(stats.min, 7.5, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.max, 7.5, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.median, 7.5, rtol=1e-6, atol=1e-12)

    def test_multiple_samples_match_statistics_module(self):
        samples = [1.0, 2.0, 3.0, 4.0]
        stats = LatencyStats.from_samples(samples)
        np.testing.assert_allclose(stats.mean, statistics.fmean(samples), rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.std, statistics.pstdev(samples), rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.min, 1.0, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.max, 4.0, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.median, statistics.median(samples), rtol=1e-6, atol=1e-12)

    def test_odd_count_median_is_middle_sample(self):
        stats = LatencyStats.from_samples([9.0, 1.0, 5.0])
        np.testing.assert_allclose(stats.median, 5.0, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.min, 1.0, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(stats.max, 9.0, rtol=1e-6, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
