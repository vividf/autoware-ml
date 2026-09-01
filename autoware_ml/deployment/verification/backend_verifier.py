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

"""Scenario-driven cross-backend verification.

Each scenario names a reference and a test backend (with devices); the verifier
runs both pipelines on the same preprocessed batches and compares the final raw
graph outputs element-wise against an absolute tolerance (the verifier default,
or the scenario's own ``tolerance`` override). No ground truth is involved: this
is numerical parity, a peer of — not a form of — evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.deployment.verification.output_comparator import OutputComparator
from autoware_ml.types.backend import Backend

if TYPE_CHECKING:
    from autoware_ml.deployment.pipeline import PipelineCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationScenario:
    """One reference-vs-test backend comparison.

    Attributes:
        ref_backend: Reference backend name (``pytorch`` / ``onnx`` / ``tensorrt``).
        ref_device: Device string for the reference pipeline (e.g. ``cuda`` / ``cpu``).
        test_backend: Test backend name.
        test_device: Device string for the test pipeline.
        tolerance: Optional per-scenario absolute tolerance overriding the
            verifier default (e.g. relaxed for int8, tight for fp32-vs-onnx).
    """

    ref_backend: str
    ref_device: str
    test_backend: str
    test_device: str
    tolerance: float | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> VerificationScenario:
        """Build a scenario from a ``{ref: {backend, device}, test: {backend, device}}`` mapping.

        An optional top-level ``tolerance`` key overrides the verifier default
        for this scenario only.
        """
        try:
            raw_tolerance = raw.get("tolerance")
            return cls(
                ref_backend=Backend.parse(raw["ref"]["backend"]).value,
                ref_device=str(raw["ref"].get("device", "cuda")),
                test_backend=Backend.parse(raw["test"]["backend"]).value,
                test_device=str(raw["test"].get("device", "cuda")),
                tolerance=float(raw_tolerance) if raw_tolerance is not None else None,
            )
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(
                "A verification scenario must look like "
                "{ref: {backend: ..., device: ...}, test: {backend: ..., device: ...}, "
                "tolerance: <optional float>}, "
                f"got: {raw!r}"
            ) from error

    def describe(self) -> str:
        """Human-readable scenario label."""
        return f"{self.ref_backend}({self.ref_device}) vs {self.test_backend}({self.test_device})"


class BackendVerifier:
    """Run verification scenarios over a set of preprocessed batches.

    Args:
        pipelines: Shared pipeline cache (one pipeline per backend/device, reused by
            evaluation).
        tolerance: Default absolute element-wise tolerance on raw graph outputs;
            a scenario's own ``tolerance`` overrides it for that scenario.
    """

    def __init__(self, pipelines: PipelineCache, tolerance: float) -> None:
        self.pipelines = pipelines
        self.tolerance = float(tolerance)

    def run(
        self,
        batches: Sequence[MultiTaskBatchInputs],
        scenarios: Sequence[VerificationScenario],
        available_backends: set[Backend],
    ) -> bool:
        """Run every applicable scenario; log a per-scenario report.

        Args:
            batches: Preprocessed sample batches shared by all scenarios.
            scenarios: Configured reference-vs-test comparisons.
            available_backends: Backends whose artifacts exist for this run.
                Scenarios touching an unavailable backend are skipped with a warning.

        Returns:
            True when every executed scenario passed on every batch.

        Raises:
            ValueError: If no scenario was executable (misconfiguration).
        """
        if not scenarios:
            logger.warning("Verification enabled but no scenarios configured; nothing to verify.")
            return True

        executed = 0
        all_passed = True
        for scenario in scenarios:
            required = {Backend.parse(scenario.ref_backend), Backend.parse(scenario.test_backend)}
            missing = required - available_backends
            if missing:
                logger.warning(
                    "Skipping verification scenario %s: backend(s) %s not exported in this run.",
                    scenario.describe(),
                    sorted(b.value for b in missing),
                )
                continue
            executed += 1
            all_passed &= self._run_scenario(scenario, batches)

        if executed == 0:
            raise ValueError(
                "No verification scenario was executable — every configured scenario "
                f"references unavailable backends (available: {sorted(b.value for b in available_backends)})."
            )
        return all_passed

    def _run_scenario(
        self, scenario: VerificationScenario, batches: Sequence[MultiTaskBatchInputs]
    ) -> bool:
        ref_pipeline = self.pipelines.get(scenario.ref_backend, scenario.ref_device)
        test_pipeline = self.pipelines.get(scenario.test_backend, scenario.test_device)
        comparator = OutputComparator(output_names=test_pipeline.output_names)

        tolerance = scenario.tolerance if scenario.tolerance is not None else self.tolerance
        tolerance_source = (
            "scenario override" if scenario.tolerance is not None else "verifier default"
        )
        logger.info("=" * 70)
        logger.info(
            "Verification scenario: %s (tolerance=%s [%s], %d batch(es))",
            scenario.describe(),
            tolerance,
            tolerance_source,
            len(batches),
        )
        passed_all = True
        for index, batch_inputs in enumerate(batches):
            ref_result = ref_pipeline.infer(batch_inputs)
            test_result = test_pipeline.infer(batch_inputs)
            summary, details = comparator.compare(
                ref_result.ordered_outputs(),
                test_result.ordered_outputs(),
                tolerance=tolerance,
            )
            status = "PASS" if summary.passed else "FAIL"
            logger.info(
                "  sample %d: %s (max_diff=%.6f, mean_diff=%.6f)",
                index,
                status,
                summary.max_diff,
                summary.mean_diff,
            )
            for detail in details:
                logger.info(
                    "    %-32s shape=%s max=%.6f mean=%.6f",
                    detail.path,
                    detail.shape,
                    detail.max_diff,
                    detail.mean_diff,
                )
            if not summary.passed:
                logger.error("  sample %d failed: %s", index, summary.reason)
                passed_all = False

        logger.info("Scenario %s: %s", scenario.describe(), "PASSED" if passed_all else "FAILED")
        return passed_all
