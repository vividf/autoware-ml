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

"""Unit tests for compare_outputs."""

from __future__ import annotations

import numpy as np
import torch

from autoware_ml.deployment.verification.output_comparator import compare_outputs


def test_identical_arrays_pass():
    a = [np.zeros((2, 3), dtype=np.float32)]
    summary, details = compare_outputs(["y"], a, [a[0].copy()], tolerance=1e-6)
    assert summary.passed
    assert summary.max_diff == 0.0
    assert [d.name for d in details] == ["y"]


def test_within_tolerance_passes():
    a = [np.zeros((4,), dtype=np.float32)]
    b = [np.full((4,), 0.05, dtype=np.float32)]
    summary, _ = compare_outputs(["y"], a, b, tolerance=0.1)
    assert summary.passed
    assert summary.max_diff <= 0.1


def test_exceeds_tolerance_fails_and_names_the_output():
    a = [np.zeros((4,), dtype=np.float32)]
    b = [np.full((4,), 1.0, dtype=np.float32)]
    summary, details = compare_outputs(["heatmap"], a, b, tolerance=0.1)
    assert not summary.passed
    assert summary.reason.startswith("heatmap: max_diff=1.000000 > tolerance")
    assert details[0].name == "heatmap"


def test_mean_diff_is_over_every_float_element():
    a = [np.zeros((2,), dtype=np.float32), np.zeros((6,), dtype=np.float32)]
    b = [np.full((2,), 1.0, dtype=np.float32), np.zeros((6,), dtype=np.float32)]
    summary, _ = compare_outputs(["p", "q"], a, b, tolerance=2.0)
    assert summary.passed
    assert summary.num_elements == 8
    assert summary.mean_diff == 2 / 8


def test_integer_outputs_report_the_mismatch_ratio_and_pass_under_the_gate():
    labels_ref = np.zeros((40,), dtype=np.int64)
    labels_test = labels_ref.copy()
    labels_test[0] = 15
    probs = torch.zeros((40, 4))
    summary, details = compare_outputs(
        ("pred_labels", "pred_probs"), [labels_ref, probs], [labels_test, probs.clone()], 1e-6
    )
    assert summary.passed
    assert summary.max_diff == 0.0
    label_detail = next(d for d in details if d.name.startswith("pred_labels"))
    assert "mismatch ratio" in label_detail.name
    assert label_detail.max_diff == 1 / 40


def test_integer_outputs_fail_when_most_of_them_differ():
    labels_ref = np.arange(20, dtype=np.int64)
    labels_test = (labels_ref + 1) % 20
    summary, _ = compare_outputs(("pred_labels",), [labels_ref], [labels_test], tolerance=1e-6)
    assert not summary.passed
    assert "integer outputs differ" in summary.reason


def test_shape_mismatch_fails_with_reason():
    a = [np.zeros((2, 3), dtype=np.float32)]
    b = [np.zeros((2, 4), dtype=np.float32)]
    summary, details = compare_outputs(["y"], a, b, tolerance=1.0)
    assert not summary.passed
    assert "shape mismatch" in summary.reason
    assert details[0].max_diff == float("inf")


def test_length_mismatch_fails():
    a = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    b = [np.zeros((2,), dtype=np.float32)]
    summary, details = compare_outputs(["p", "q"], a, b, tolerance=1.0)
    assert not summary.passed
    assert "length mismatch" in summary.reason
    assert details == []


def test_torch_and_numpy_mix():
    summary, _ = compare_outputs(
        ["y"], [torch.zeros(2, 3)], [np.zeros((2, 3), dtype=np.float32)], tolerance=1e-6
    )
    assert summary.passed
