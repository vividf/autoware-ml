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

"""Unit tests for OutputComparator, focused on the shape-mismatch path."""

from __future__ import annotations

import numpy as np
import torch

from autoware_ml.deployment.verification.output_comparator import OutputComparator


class TestOutputComparator:
    def test_identical_arrays_pass(self):
        a = [np.zeros((2, 3), dtype=np.float32)]
        b = [np.zeros((2, 3), dtype=np.float32)]
        summary, details = OutputComparator().compare(a, b, tolerance=1e-6)
        assert summary.passed
        assert summary.max_diff == 0.0
        assert len(details) == 1

    def test_within_tolerance_passes(self):
        a = [np.zeros((4,), dtype=np.float32)]
        b = [np.full((4,), 0.05, dtype=np.float32)]
        summary, _ = OutputComparator().compare(a, b, tolerance=0.1)
        assert summary.passed
        assert summary.max_diff <= 0.1

    def test_exceeds_tolerance_fails(self):
        a = [np.zeros((4,), dtype=np.float32)]
        b = [np.full((4,), 1.0, dtype=np.float32)]
        summary, _ = OutputComparator().compare(a, b, tolerance=0.1)
        assert not summary.passed
        assert "tolerance" in (summary.reason or "")

    def test_integer_outputs_report_mismatch_ratio_without_gating(self):
        # Class labels are decisions over the float outputs: a few flips at near-tie
        # points must not fail the scenario, but the ratio is reported for the log.
        labels_ref = np.zeros((40,), dtype=np.int64)
        labels_test = labels_ref.copy()
        labels_test[0] = 15
        probs = torch.zeros((40, 4))
        summary, details = OutputComparator(("pred_labels", "pred_probs")).compare(
            [labels_ref, probs], [labels_test, probs.clone()], tolerance=1e-6
        )
        assert summary.passed
        assert summary.max_diff == 0.0
        label_detail = next(d for d in details if d.path.startswith("output[pred_labels]"))
        assert "mismatch ratio" in label_detail.path
        assert label_detail.max_diff == 1 / 40

    def test_integer_outputs_fail_when_most_of_them_differ(self):
        labels_ref = np.arange(20, dtype=np.int64)
        labels_test = (labels_ref + 1) % 20  # every label wrong: a wiring mistake
        summary, _ = OutputComparator(("pred_labels",)).compare(
            [labels_ref], [labels_test], tolerance=1e-6
        )
        assert not summary.passed
        assert "integer outputs differ" in (summary.reason or "")

    def test_shape_mismatch_fails_with_reason(self):
        a = [np.zeros((2, 3), dtype=np.float32)]
        b = [np.zeros((2, 4), dtype=np.float32)]
        summary, details = OutputComparator().compare(a, b, tolerance=1.0)
        assert not summary.passed
        assert "shape mismatch" in (summary.reason or "")
        # The mismatched tensor is recorded with infinite diffs, not silently dropped.
        assert details and details[0].max_diff == float("inf")

    def test_length_mismatch_fails(self):
        a = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
        b = [np.zeros((2,), dtype=np.float32)]
        summary, _ = OutputComparator().compare(a, b, tolerance=1.0)
        assert not summary.passed
        assert "length mismatch" in (summary.reason or "")

    def test_named_outputs_label_paths(self):
        a = [np.zeros((2,), dtype=np.float32)]
        b = [np.ones((2,), dtype=np.float32)]
        summary, details = OutputComparator(output_names=["heatmap"]).compare(a, b, tolerance=0.1)
        assert not summary.passed
        assert "heatmap" in details[0].path

    def test_torch_and_numpy_mix(self):
        a = [torch.zeros(2, 3)]
        b = [np.zeros((2, 3), dtype=np.float32)]
        summary, _ = OutputComparator().compare(a, b, tolerance=1e-6)
        assert summary.passed
