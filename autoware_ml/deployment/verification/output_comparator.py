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

"""Element-wise comparison of two backends' model outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

#: Share of an integer output (argmax labels, indices) allowed to differ between two
#: backends. Labels flip at near-ties (measured on PTv3: 0.03 % fp16, 0.3-0.4 % INT8); a
#: wiring or class-order mistake flips most of them.
DECISION_MISMATCH_TOLERANCE = 0.05

#: Margin of the gate a failure suggests over the observed max_diff, which varies a little
#: from run to run (kernel / tactic selection).
SUGGESTED_GATE_HEADROOM = 1.25


@dataclass(frozen=True)
class OutputDiffSummary:
    """Comparison result of all outputs.

    Attributes:
        passed: Whether every output passed.
        max_diff: Largest absolute difference of the float outputs.
        mean_diff: Mean absolute difference of the float outputs, over all their elements.
        num_elements: Number of float elements compared.
        reason: First failure, or ``None``.
    """

    passed: bool
    max_diff: float
    mean_diff: float
    num_elements: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class TensorDiffDetail:
    """Comparison result of one output, for the log.

    Attributes:
        name: Output name, with the gate for an integer output.
        shape: Shape of the reference tensor.
        max_diff: Largest absolute difference; the mismatch ratio for an integer tensor.
        mean_diff: Mean absolute difference; the mismatch ratio for an integer tensor.
    """

    name: str
    shape: tuple[int, ...]
    max_diff: float
    mean_diff: float


def compare_outputs(
    names: Sequence[str],
    reference: Sequence[torch.Tensor | np.ndarray],
    test: Sequence[torch.Tensor | np.ndarray],
    tolerance: float,
) -> tuple[OutputDiffSummary, list[TensorDiffDetail]]:
    """Compare two backends' outputs by position and return the summary and one row per output.

    Float outputs must agree within the absolute ``tolerance``. Integer and boolean outputs
    (labels, indices) are decisions over float outputs: one flips whenever two logits tie
    within backend noise, so they are gated on the share that differ
    (:data:`DECISION_MISMATCH_TOLERANCE`) instead; a mis-wired output differs almost everywhere.
    """
    if len(reference) != len(test):
        return _fail(f"length mismatch {len(reference)} vs {len(test)}"), []
    if len(names) != len(reference):
        return _fail(f"{len(names)} output names for {len(reference)} outputs"), []

    details: list[TensorDiffDetail] = []
    max_diff = 0.0
    total_diff = 0.0
    total_elements = 0
    reason: str | None = None
    for name, ref_value, test_value in zip(names, reference, test):
        ref_np, test_np = _to_numpy(ref_value), _to_numpy(test_value)
        shape = tuple(int(x) for x in ref_np.shape)
        if ref_np.shape != test_np.shape:
            details.append(TensorDiffDetail(name, shape, float("inf"), float("inf")))
            max_diff = float("inf")
            reason = reason or f"{name}: shape mismatch {ref_np.shape} vs {test_np.shape}"
            continue
        if ref_np.dtype.kind in "iub":
            mismatch_ratio = float(np.mean(ref_np != test_np)) if ref_np.size else 0.0
            details.append(
                TensorDiffDetail(
                    f"{name} (decision mismatch ratio, gate {DECISION_MISMATCH_TOLERANCE})",
                    shape,
                    mismatch_ratio,
                    mismatch_ratio,
                )
            )
            if mismatch_ratio > DECISION_MISMATCH_TOLERANCE:
                reason = reason or (
                    f"{name}: {mismatch_ratio:.1%} of the integer outputs differ "
                    f"(gate {DECISION_MISMATCH_TOLERANCE:.0%}, shape={ref_np.shape}). Decisions "
                    "flip at near-ties only in small numbers; this looks like a wiring or "
                    "class-order mismatch."
                )
            continue
        diff = np.abs(ref_np.astype(np.float64) - test_np.astype(np.float64))
        tensor_max = float(np.max(diff)) if diff.size else 0.0
        tensor_mean = float(np.mean(diff)) if diff.size else 0.0
        details.append(TensorDiffDetail(name, shape, tensor_max, tensor_mean))
        max_diff = max(max_diff, tensor_max)
        total_diff += tensor_mean * diff.size
        total_elements += diff.size
        if tensor_max > tolerance:
            reason = reason or (
                f"{name}: max_diff={tensor_max:.6f} > tolerance={tolerance:.6f} "
                f"(shape={ref_np.shape}). Raw-logit diffs between backends are expected for "
                "quantized/FP16 stages; if the per-backend metrics stay equal, recalibrate this "
                f"scenario's gate to ~{tensor_max * SUGGESTED_GATE_HEADROOM:.1f} and record the "
                "observed value in the config comment."
            )
    return (
        OutputDiffSummary(
            passed=reason is None,
            max_diff=max_diff,
            mean_diff=total_diff / total_elements if total_elements else 0.0,
            num_elements=total_elements,
            reason=reason,
        ),
        details,
    )


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fail(reason: str) -> OutputDiffSummary:
    return OutputDiffSummary(
        passed=False, max_diff=float("inf"), mean_diff=float("inf"), reason=reason
    )
