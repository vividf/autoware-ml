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

"""Latency summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
import statistics


@dataclass(frozen=True)
class LatencyStats:
    """Latency summary in milliseconds over the evaluated batches (warmup re-runs excluded)."""

    mean: float
    std: float
    min: float
    max: float
    median: float

    @classmethod
    def from_samples(cls, samples_ms: list[float]) -> "LatencyStats":
        """Compute the summary from raw per-batch timings."""
        if not samples_ms:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0)
        return cls(
            mean=float(statistics.fmean(samples_ms)),
            std=float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
            min=float(min(samples_ms)),
            max=float(max(samples_ms)),
            median=float(statistics.median(samples_ms)),
        )
