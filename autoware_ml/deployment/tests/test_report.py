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

"""Unit tests for the evaluation metric keys and headline reporting."""

from __future__ import annotations

import logging
import unittest
from dataclasses import replace

import torch

from autoware_ml.deployment.evaluation.evaluator import (
    EvaluationResult,
    log_backend_report,
    log_comparison,
)
from autoware_ml.deployment.evaluation.report import is_headline, latency_key, metric_key
from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.nds import Nds
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.segmentation3d.iou import IoU
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DConfusionMatrixMetricSuite
from autoware_ml.types.backend import Backend

_EVALUATOR_LOGGER = "autoware_ml.deployment.evaluation.evaluator"


class TestMetricKeys(unittest.TestCase):
    """Unit tests for the names metrics and latencies are logged under."""

    def test_keys_carry_split_backend_prefix_and_name(self) -> None:
        self.assertEqual(
            metric_key("test", Backend.TENSORRT, "det3d", "mAP_0m_121m"),
            "test/tensorrt/det3d/mAP_0m_121m",
        )
        self.assertEqual(metric_key("val", Backend.ONNX, "seg", "loss"), "val/onnx/seg/loss")
        self.assertEqual(latency_key(Backend.ONNX, "head_mean_ms"), "latency/onnx/head_mean_ms")


def _box(x: float) -> list[float]:
    return [x, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]


class TestHeadlineMetrics(unittest.TestCase):
    """Unit tests for matching metric keys against the headline metrics a suite declares."""

    def test_headline_matches_the_declared_name_up_to_its_range_suffix(self) -> None:
        cases = [
            ("test/tensorrt/det3d/mAP", True),
            ("test/tensorrt/det3d/mAP_0m_121m", True),
            ("test/tensorrt/det3d/visible/mAP_50m_inf", True),
            ("test/tensorrt/det3d/mAP_car", False),
            ("test/tensorrt/det3d/mAPH", False),
            ("test/tensorrt/det3d/mAPH_0m_121m", False),
            ("test/tensorrt/det3d/map_based_nds", True),
            ("test/tensorrt/det3d/map_based_nds_0m_50m", True),
            ("test/tensorrt/det3d/mapH_based_nds", False),
            ("test/tensorrt/seg3d/mIoU", False),
        ]
        for key, expected in cases:
            with self.subTest(key=key):
                self.assertIs(is_headline(key, ("mAP", "map_based_nds")), expected)

    def test_suites_declare_their_headline_metrics(self) -> None:
        self.assertEqual(Detection3DMetricSuite.headline_metrics, ("mAP", "map_based_nds"))
        self.assertEqual(
            Segmentation3DConfusionMatrixMetricSuite.headline_metrics, ("mIoU", "fwIoU")
        )

    def test_declared_headlines_are_keys_the_suites_report(self) -> None:
        # A declared "NDS" matched nothing: Nds reports map_based_nds.
        detection = Detection3DMetricSuite(components=[MeanAP(), Nds()], class_names=("car",))
        detection.update(
            {
                "predictions": [
                    {
                        "bboxes_3d": torch.tensor([_box(10.0)]),
                        "scores_3d": torch.tensor([0.9]),
                        "labels_3d": torch.tensor([0]),
                    }
                ],
                "gt_boxes": [torch.tensor([_box(10.0)])],
                "gt_labels": [torch.tensor([0])],
            }
        )
        segmentation = Segmentation3DConfusionMatrixMetricSuite(
            components=[IoU()], num_classes=2, class_names=("road", "car")
        )
        segmentation.update(
            {
                "seg3d_frames": [
                    {
                        "pred": torch.tensor([0, 1]),
                        "target": torch.tensor([0, 1]),
                        "coord": torch.zeros((2, 3)),
                    }
                ]
            }
        )
        for suite in (detection, segmentation):
            keys = suite.result(EvalStage.TEST)
            for name in suite.headline_metrics:
                self.assertTrue(
                    any(is_headline(key, (name,)) for key in keys),
                    f"{type(suite).__name__}: {name}",
                )


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


class TestReportLogging(unittest.TestCase):
    """Unit tests for the per-backend report and the comparison table in the log."""

    def test_report_leads_with_the_declared_metrics_only(self) -> None:
        with self.assertLogs(_EVALUATOR_LOGGER, logging.INFO) as logs:
            log_backend_report(_result(Backend.PYTORCH, ("mAP",)))
        text = "\n".join(logs.output)
        self.assertIn("det3d/mAP_0m_121m", text)
        self.assertNotIn("mAP_car", text)
        self.assertNotIn("mAPH", text)
        self.assertNotIn("mIoU", text)

    def test_comparison_table_rows_come_from_the_shared_headlines(self) -> None:
        results = [_result(Backend.PYTORCH, ("mIoU",)), _result(Backend.TENSORRT, ("mIoU",))]
        with self.assertLogs(_EVALUATOR_LOGGER, logging.INFO) as logs:
            log_comparison(results)
        text = "\n".join(logs.output)
        self.assertIn("test/seg3d/mIoU", text)
        self.assertNotIn("det3d/mAP", text)

    def test_fallback_stages_are_visible_in_report_and_comparison(self) -> None:
        starred = replace(_result(Backend.ONNX, ("mIoU",)), fallback_stages=("encoder", "head"))
        clean = _result(Backend.TENSORRT, ("mIoU",))
        with self.assertLogs(_EVALUATOR_LOGGER, logging.INFO) as logs:
            log_backend_report(starred)
            log_comparison([starred, clean])
        text = "\n".join(logs.output)
        self.assertIn("encoder, head", text)
        self.assertIn("onnx*", text)
        self.assertNotIn("tensorrt*", text)


if __name__ == "__main__":
    unittest.main()
