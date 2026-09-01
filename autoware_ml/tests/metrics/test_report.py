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

"""The one metric-key convention shared by trainer.test and deployment evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.report import (
    check_required_keys,
    collect_suite_results,
    latency_key,
    metric_key,
)
from autoware_ml.types.backend import Backend


class _Suite(SimpleNamespace):
    def result(self, stage):
        return self.results


def _suite(prefix, results, required=()):
    return _Suite(prefix=prefix, results=results, _required_keys=tuple(required))


class TestKeys:
    def test_metric_key_layout(self):
        assert (
            metric_key("test", Backend.TENSORRT, "detection3d", "mAP")
            == "test/tensorrt/detection3d/mAP"
        )
        assert metric_key("val", "pytorch", "", "loss") == "val/pytorch/loss"

    def test_latency_key_layout(self):
        assert (
            latency_key(Backend.ONNX, "pts_voxel_encoder_mean_ms")
            == "latency/onnx/pts_voxel_encoder_mean_ms"
        )


class TestCollect:
    def test_same_suite_same_split_differs_only_by_backend(self):
        suites = [_suite("detection3d", {"mAP": 0.5})]
        trainer = collect_suite_results(suites, EvalStage.TEST, backend=Backend.PYTORCH)
        deploy = collect_suite_results(suites, EvalStage.TEST, backend="tensorrt")
        assert trainer == {"test/pytorch/detection3d/mAP": 0.5}
        assert deploy == {"test/tensorrt/detection3d/mAP": 0.5}

    def test_duplicate_keys_rejected(self):
        suites = [_suite("p", {"m": 1.0}), _suite("p", {"m": 2.0})]
        with pytest.raises(ValueError, match="same key"):
            collect_suite_results(suites, EvalStage.VAL, backend="pytorch")

    def test_required_keys_checked_against_eval_output(self):
        suites = [_suite("p", {}, required=("pred_boxes", "gt_boxes"))]
        check_required_keys(suites, {"pred_boxes": 1, "gt_boxes": 2}, producer="M")
        with pytest.raises(ValueError, match="gt_boxes"):
            check_required_keys(suites, {"pred_boxes": 1}, producer="M")
