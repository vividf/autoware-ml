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

"""Metric-key convention of deployment evaluation.

Lightning's validation / test lifecycle logs ``{stage}/{prefix}/{metric}`` and is left
alone. Deployment evaluation scores several backends on the same split, so its keys carry
the backend as well:

    ``{split}/{backend}/{suite_prefix}/{metric}``   e.g. ``test/tensorrt/det3d/mAP_0m_121m``

Latency lives under its own root so it never collides with a metric:

    ``latency/{backend}/{stage}``                     e.g. ``latency/tensorrt/pts_backbone_neck_head_centerpoint_mean_ms``
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from autoware_ml.metrics.base import EvalStage, MetricSuite
from autoware_ml.types.backend import Backend

LATENCY_ROOT = "latency"

# Suites append the range window to a metric name (``mAP_0m_121m``, ``mIoU_50m_inf``);
# a headline declared as ``mAP`` covers those and nothing else (not ``mAP_car``).
_RANGE_SUFFIX = re.compile(r"_\d[0-9p]*m_(\d[0-9p]*m|inf)$")


def metric_key(split: str, backend: str | Backend, prefix: str, name: str) -> str:
    """Build the canonical metric key ``{split}/{backend}/{prefix}/{name}``.

    An empty ``prefix`` is skipped so a suite without a prefix reports directly under
    ``{split}/{backend}/{name}``.
    """
    backend_name = backend.value if isinstance(backend, Backend) else str(backend)
    parts = [split, backend_name] + ([prefix] if prefix else []) + [name]
    return "/".join(parts)


def latency_key(backend: str | Backend, stage: str) -> str:
    """Build the canonical latency key ``latency/{backend}/{stage}``."""
    backend_name = backend.value if isinstance(backend, Backend) else str(backend)
    return f"{LATENCY_ROOT}/{backend_name}/{stage}"


def is_headline(key: str, headline_metrics: Iterable[str]) -> bool:
    """Whether a metric key names one of the declared headline metrics.

    The comparison is on the unqualified metric name (the last ``/`` component), exact
    up to the suite's range suffix: ``mAP`` selects ``mAP`` and ``mAP_0m_121m``, not the
    per-class ``mAP_car`` and not the sibling ``mAPH``.
    """
    tail = key.rsplit("/", 1)[-1]
    for name in headline_metrics:
        if tail == name:
            return True
        if tail.startswith(name + "_") and _RANGE_SUFFIX.fullmatch(tail[len(name) :]):
            return True
    return False


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
    suites: Iterable[MetricSuite], stage: EvalStage, *, backend: str | Backend
) -> dict[str, float]:
    """Compute every suite's ``result`` and key it canonically.

    Raises:
        ValueError: When two suites emit the same key (set distinct prefixes).
    """
    report: dict[str, float] = {}
    for suite in suites:
        for name, value in suite.result(stage).items():
            key = metric_key(stage.value, backend, suite.prefix, name)
            if key in report:
                raise ValueError(f"Two metrics log the same key {key!r}. Set a distinct prefix.")
            report[key] = float(value)
    return report
