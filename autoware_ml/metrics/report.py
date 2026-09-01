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

"""The one metric-key convention and the one suite-to-report loop.

Every consumer of metric suites — the Lightning validation/test lifecycle
(:class:`~autoware_ml.metrics.eval_mixin.MetricEvalMixin`) and deployment
evaluation (:mod:`autoware_ml.evaluation`) — reports through
:func:`collect_suite_results`, so the same metric computed on the same split lands
under the same key regardless of what ran the forward:

    ``{split}/{backend}/{suite_prefix}/{metric}``   e.g. ``test/tensorrt/detection3d/mAP``

Latency lives under its own root so it never collides with a metric:

    ``latency/{backend}/{stage}``                     e.g. ``latency/tensorrt/pts_backbone_neck_head_mean_ms``
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from autoware_ml.metrics.base import EvalStage, MetricSuite
from autoware_ml.types.backend import Backend

LATENCY_ROOT = "latency"


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
        missing = [key for key in suite._required_keys if key not in eval_out]
        if missing:
            raise ValueError(
                f"Metric {type(suite).__name__!r} needs {missing}, not produced by "
                f"{producer}.build_eval_output."
            )


def collect_suite_results(
    suites: Iterable[MetricSuite], stage: EvalStage, *, backend: str | Backend
) -> dict[str, float]:
    """Compute every suite's ``result`` and key it canonically.

    Args:
        suites: Metric suites with accumulated state.
        stage: Evaluation stage; its value is the ``split`` segment of the key.
        backend: Backend that produced the predictions (``pytorch`` for ``trainer.test``).

    Returns:
        ``{metric_key: value}`` for every metric of every suite.

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
