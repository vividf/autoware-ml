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

"""Element-wise comparison of model outputs for verification.

Outputs are tensors / arrays or nested lists and tuples of them; a dict or a bare scalar fails
as a type mismatch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

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
    """Comparison result of a whole output tree.

    Attributes:
        passed: Whether every tensor passed.
        max_diff: Largest absolute difference of the float tensors.
        mean_diff: Mean absolute difference of the float tensors, over all their elements.
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
    """Comparison result of one tensor, for the log.

    Attributes:
        path: Position in the output, e.g. ``output[heatmap]``.
        shape: Shape of the reference tensor.
        max_diff: Largest absolute difference; the mismatch ratio for an integer tensor.
        mean_diff: Mean absolute difference; the mismatch ratio for an integer tensor.
    """

    path: str
    shape: tuple[int, ...]
    max_diff: float
    mean_diff: float


class OutputComparator:
    """Recursively compare nested outputs.

    Args:
        output_names: Labels of the sequence items in the reported paths, by index:
            ``output[heatmap]`` instead of ``output[output_0]``.
    """

    def __init__(self, output_names: Sequence[str] | None = None) -> None:
        self._output_names: tuple[str, ...] | None = tuple(output_names) if output_names else None

    def compare(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str = "output",
    ) -> tuple[OutputDiffSummary, list[TensorDiffDetail]]:
        """Compare two outputs; return the summary and one detail row per tensor."""
        tensor_details: list[TensorDiffDetail] = []
        summary = self._compare_nested(reference, test, tolerance, path, tensor_details)
        return summary, tensor_details

    def _compare_nested(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str,
        tensor_details: list[TensorDiffDetail],
    ) -> OutputDiffSummary:
        if reference is None and test is None:
            return OutputDiffSummary(passed=True, max_diff=0.0, mean_diff=0.0)

        if reference is None or test is None:
            return _fail(path, "one side is None while the other is not")

        if isinstance(reference, (list, tuple)) and isinstance(test, (list, tuple)):
            return self._compare_sequences(reference, test, tolerance, path, tensor_details)

        if self._is_array_like(reference) and self._is_array_like(test):
            return self._compare_arrays(reference, test, tolerance, path, tensor_details)

        return _fail(
            path,
            f"type mismatch {type(reference).__name__} vs {type(test).__name__}",
        )

    def _compare_sequences(
        self,
        reference: list | tuple,
        test: list | tuple,
        tolerance: float,
        path: str,
        tensor_details: list[TensorDiffDetail],
    ) -> OutputDiffSummary:
        if len(reference) != len(test):
            return _fail(path, f"length mismatch {len(reference)} vs {len(test)}")

        names = self._output_names

        def _child_summaries():
            for idx, (ref_item, test_item) in enumerate(zip(reference, test)):
                name = names[idx] if names and idx < len(names) else f"output_{idx}"
                yield self._compare_nested(
                    ref_item, test_item, tolerance, f"{path}[{name}]", tensor_details
                )

        return self._merge_summaries(_child_summaries())

    def _compare_arrays(
        self,
        reference: Any,
        test: Any,
        tolerance: float,
        path: str,
        tensor_details: list[TensorDiffDetail],
    ) -> OutputDiffSummary:
        ref_np = self._to_numpy(reference)
        test_np = self._to_numpy(test)

        if ref_np.shape != test_np.shape:
            tensor_details.append(
                TensorDiffDetail(
                    path=path,
                    shape=tuple(int(x) for x in ref_np.shape),
                    max_diff=float("inf"),
                    mean_diff=float("inf"),
                )
            )
            return _fail(path, f"shape mismatch {ref_np.shape} vs {test_np.shape}")

        if ref_np.dtype.kind in "iub":
            # Integer / boolean outputs (labels, indices) are decisions over float outputs: one
            # flips whenever two logits tie within backend noise, so a max-diff gate would fail
            # on a few near-ties. Gate on the share that differ; a mis-wired output differs
            # almost everywhere.
            mismatch_ratio = float(np.mean(ref_np != test_np)) if ref_np.size else 0.0
            passed = mismatch_ratio <= DECISION_MISMATCH_TOLERANCE
            tensor_details.append(
                TensorDiffDetail(
                    path=f"{path} (decision mismatch ratio, gate {DECISION_MISMATCH_TOLERANCE})",
                    shape=tuple(int(x) for x in ref_np.shape),
                    max_diff=mismatch_ratio,
                    mean_diff=mismatch_ratio,
                )
            )
            reason = (
                None
                if passed
                else (
                    f"{path}: {mismatch_ratio:.1%} of the integer outputs differ "
                    f"(gate {DECISION_MISMATCH_TOLERANCE:.0%}, shape={ref_np.shape}). Decisions "
                    "flip at near-ties only in small numbers; this looks like a wiring or "
                    "class-order mismatch."
                )
            )
            return OutputDiffSummary(
                passed=passed, max_diff=0.0, mean_diff=0.0, num_elements=0, reason=reason
            )

        diff = np.abs(ref_np.astype(np.float64) - test_np.astype(np.float64))
        max_diff = float(np.max(diff)) if diff.size else 0.0
        mean_diff = float(np.mean(diff)) if diff.size else 0.0
        num_elements = int(diff.size)

        passed = max_diff <= tolerance
        reason = (
            None
            if passed
            else (
                f"{path}: max_diff={max_diff:.6f} > tolerance={tolerance:.6f} (shape={ref_np.shape}). "
                "Raw-logit diffs between backends are expected for quantized/FP16 stages; if the "
                f"per-backend metrics stay equal, recalibrate this scenario's gate to "
                f"~{max_diff * SUGGESTED_GATE_HEADROOM:.1f} "
                "and record the observed value in the config comment."
            )
        )
        tensor_details.append(
            TensorDiffDetail(
                path=path,
                shape=tuple(int(x) for x in ref_np.shape),
                max_diff=max_diff,
                mean_diff=mean_diff,
            )
        )
        return OutputDiffSummary(
            passed=passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=num_elements,
            reason=reason,
        )

    @staticmethod
    def _merge_summaries(results: Iterable[OutputDiffSummary]) -> OutputDiffSummary:
        max_diff = 0.0
        total_diff = 0.0
        total_elements = 0
        all_passed = True
        first_reason: str | None = None

        for result in results:
            max_diff = max(max_diff, result.max_diff)
            # A shape or type mismatch has mean_diff=inf and no elements; inf * 0 would turn
            # the mean into nan. The mismatch still fails the summary with max_diff=inf.
            if result.num_elements:
                total_diff += result.mean_diff * result.num_elements
            total_elements += result.num_elements
            if not result.passed and all_passed:
                all_passed = False
                first_reason = result.reason

        mean_diff = total_diff / total_elements if total_elements > 0 else 0.0
        return OutputDiffSummary(
            passed=all_passed,
            max_diff=max_diff,
            mean_diff=mean_diff,
            num_elements=total_elements,
            reason=first_reason,
        )

    @staticmethod
    def _is_array_like(obj: Any) -> bool:
        return isinstance(obj, (torch.Tensor, np.ndarray))

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)


def _fail(path: str, reason: str) -> OutputDiffSummary:
    return OutputDiffSummary(
        passed=False,
        max_diff=float("inf"),
        mean_diff=float("inf"),
        reason=f"{path}: {reason}",
    )
