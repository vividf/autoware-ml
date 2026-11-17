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

from collections.abc import Iterable

from autoware_ml.metrics.base import EvalStage, MetricSuite
from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.metrics.report import MetricReport


def check_requirements(
    suites: Iterable[MetricSuite], eval_output: EvalOutput, producer: str
) -> None:
    """Raise when ``eval_output`` lacks a part a suite needs; the error names ``producer``.

    Args:
        suites: The suites that will read ``eval_output``.
        eval_output: The eval output of one batch.
        producer: The method that built ``eval_output``, for the message.

    Raises:
        ValueError: Naming the suite, the missing parts and ``producer``.
    """
    for suite in suites:
        missing = suite.requirements().check_requirements(eval_output)
        if missing:
            raise ValueError(
                f"Metric {type(suite).__name__!r} needs {list(missing)}, not produced by "
                f"{producer}."
            )


def collect_suite_results(suites: Iterable[MetricSuite], stage: EvalStage) -> MetricReport:
    """Compute every suite's ``result`` into one report, each suite under its prefix.

    Raises:
        ValueError: When two suites emit the same key (set distinct prefixes).
    """
    report = MetricReport()
    for suite in suites:
        report.merge(suite.result(stage), prefix=suite.prefix)
    return report
