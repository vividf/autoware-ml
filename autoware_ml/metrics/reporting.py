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

"""Metric suite helpers shared by the Lightning lifecycle and deployment evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from autoware_ml.metrics.base import EvalStage, MetricSuite


def check_required_keys(
    suites: Iterable[MetricSuite], eval_out: Mapping[str, Any], producer: str
) -> None:
    """Raise when a suite needs a key missing from ``eval_out``; the error names ``producer``."""
    for suite in suites:
        missing = [key for key in suite.required_keys() if key not in eval_out]
        if missing:
            raise ValueError(
                f"Metric {type(suite).__name__!r} needs {missing}, not produced by {producer}."
            )


def collect_suite_results(
    suites: Iterable[MetricSuite],
    stage: EvalStage,
    key_of: Callable[[str, str], str],
) -> dict[str, float]:
    """Compute every suite's ``result`` and key it with ``key_of(prefix, name)``.

    Raises:
        ValueError: When two suites emit the same key (set distinct prefixes).
    """
    report: dict[str, float] = {}
    for suite in suites:
        for name, value in suite.result(stage).items():
            key = key_of(suite.prefix, name)
            if key in report:
                raise ValueError(f"Two metrics log the same key {key!r}. Set a distinct prefix.")
            report[key] = float(value)
    return report
