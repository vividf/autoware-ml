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

"""Metric keys of deployment evaluation.

Lightning logs ``{stage}/{prefix}/{metric}``; deployment evaluation scores several backends
on one split, so its keys add the backend: ``{split}/{backend}/{prefix}/{metric}``. Latency
goes under ``latency/{backend}/{stage}`` so it never collides with a metric.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from autoware_ml.types.backend import Backend

LATENCY_ROOT = "latency"

# Range window that suites append to a metric name: ``mAP_0m_121m``, ``mIoU_50m_inf``.
_RANGE_SUFFIX = re.compile(r"_\d[0-9p]*m_(\d[0-9p]*m|inf)$")


def metric_key(split: str, backend: Backend, prefix: str, name: str) -> str:
    """Return the metric key ``{split}/{backend}/{prefix}/{name}``."""
    return f"{split}/{backend}/{prefix}/{name}"


def latency_key(backend: Backend, stage: str) -> str:
    """Return the latency key ``latency/{backend}/{stage}``."""
    return f"{LATENCY_ROOT}/{backend}/{stage}"


def is_headline(key: str, headline_metrics: Iterable[str]) -> bool:
    """Whether a metric key names one of ``headline_metrics``, with or without a range suffix.

    ``mAP`` matches ``mAP`` and ``mAP_0m_121m``, not ``mAP_car`` or ``mAPH``.
    """
    tail = key.rsplit("/", 1)[-1]
    for name in headline_metrics:
        if tail == name:
            return True
        if tail.startswith(name + "_") and _RANGE_SUFFIX.fullmatch(tail[len(name) :]):
            return True
    return False
