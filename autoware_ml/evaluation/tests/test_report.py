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

"""Metric-key convention and headline selection of deployment evaluation."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from autoware_ml.evaluation.evaluator import EvaluationResult, log_backend_report, log_comparison
from autoware_ml.evaluation.report import is_headline, latency_key, metric_key
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite
from autoware_ml.types.backend import Backend


def test_keys_carry_split_backend_prefix_and_name() -> None:
    assert (
        metric_key("test", Backend.TENSORRT, "det3d", "mAP_0m_121m")
        == "test/tensorrt/det3d/mAP_0m_121m"
    )
    assert metric_key("val", "onnx", "", "loss") == "val/onnx/loss"
    assert latency_key(Backend.ONNX, "head_mean_ms") == "latency/onnx/head_mean_ms"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("test/tensorrt/det3d/mAP", True),
        ("test/tensorrt/det3d/mAP_0m_121m", True),
        ("test/tensorrt/det3d/visible/mAP_50m_inf", True),
        ("test/tensorrt/det3d/mAP_car", False),
        ("test/tensorrt/det3d/mAPH", False),
        ("test/tensorrt/det3d/mAPH_0m_121m", False),
        ("test/tensorrt/seg3d/mIoU", False),
    ],
)
def test_headline_matches_the_declared_name_up_to_its_range_suffix(key, expected) -> None:
    assert is_headline(key, ("mAP", "NDS")) is expected


def test_suites_declare_their_headline_metrics() -> None:
    assert Detection3DMetricSuite.headline_metrics == ("mAP", "NDS")
    assert Segmentation3DConfusionMatrixMetricSuite.headline_metrics == ("mIoU", "fwIoU")


def _result(backend: Backend, headline: tuple[str, ...]) -> EvaluationResult:
    return EvaluationResult(
        backend=backend,
        device="cuda",
        split="test",
        metrics={
            f"test/{backend.value}/seg3d/mIoU": 0.61,
            f"test/{backend.value}/seg3d/iou_car": 0.9,
            f"test/{backend.value}/det3d/mAP_0m_121m": 0.45,
            f"test/{backend.value}/det3d/mAP_car_0m_121m": 0.5,
            f"test/{backend.value}/det3d/mAPH_0m_121m": 0.4,
        },
        latency={},
        num_samples=10,
        headline_metrics=headline,
    )


def test_report_leads_with_the_declared_metrics_only(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="autoware_ml.evaluation.evaluator"):
        log_backend_report(_result(Backend.PYTORCH, ("mAP",)))
    assert "det3d/mAP_0m_121m" in caplog.text
    assert "mAP_car" not in caplog.text and "mAPH" not in caplog.text and "mIoU" not in caplog.text


def test_comparison_table_rows_come_from_the_shared_headlines(caplog) -> None:
    results = [_result(Backend.PYTORCH, ("mIoU",)), _result(Backend.TENSORRT, ("mIoU",))]
    with caplog.at_level(logging.INFO, logger="autoware_ml.evaluation.evaluator"):
        log_comparison(results)
    assert "test/seg3d/mIoU" in caplog.text
    assert "det3d/mAP" not in caplog.text


def test_fallback_stages_are_visible_in_report_and_comparison(caplog) -> None:
    """A backend whose stages ran in torch must say so — never a silent pytorch copy."""
    starred = replace(_result(Backend.ONNX, ("mIoU",)), fallback_stages=("encoder", "head"))
    clean = _result(Backend.TENSORRT, ("mIoU",))
    with caplog.at_level(logging.INFO, logger="autoware_ml.evaluation.evaluator"):
        log_backend_report(starred)
        log_comparison([starred, clean])
    assert "encoder, head" in caplog.text
    assert "onnx*" in caplog.text and "tensorrt*" not in caplog.text
