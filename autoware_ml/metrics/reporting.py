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

"""Feed metric suites and collect their results — shared by Lightning and deployment.

The Lightning lifecycle (:class:`~autoware_ml.metrics.eval_mixin.MetricEvalMixin`) and
per-backend deployment evaluation score the same suites; the required-key check and the
result collection live here so the two paths cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from autoware_ml.metrics.base import EvalStage, MetricSuite


def check_required_keys(
    suites: Iterable[MetricSuite], eval_out: Mapping[str, Any], producer: str
) -> None:
    """Raise when a suite needs an ``eval_out`` key the model did not produce.

    Args:
        suites: Metric suites about to consume ``eval_out``.
        eval_out: The flat dict returned by the model's ``build_eval_output``.
        producer: Name of the model class, for the error message.
    """
    for suite in suites:
        missing = [key for key in suite.required_keys() if key not in eval_out]
        if missing:
            raise ValueError(
                f"Metric {type(suite).__name__!r} needs {missing}, not produced by "
                f"{producer}.build_eval_output."
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
