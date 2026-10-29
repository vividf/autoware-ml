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

"""Cross-backend verification of the raw model outputs.

Each scenario runs a reference and a test pipeline on the same batches. Float outputs must
agree within an absolute tolerance, integer outputs by the share that differ.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, model_validator

from autoware_ml.deployment.verification.output_comparator import compare_outputs
from autoware_ml.types.backend import Backend

if TYPE_CHECKING:
    from autoware_ml.deployment.pipeline import PipelineCache

logger = logging.getLogger(__name__)


class PipelineEndpoint(BaseModel):
    """A ``(backend, device)`` pair a scenario runs on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend
    device: str = "cuda"

    def __str__(self) -> str:
        return f"{self.backend}({self.device})"


class VerificationScenario(BaseModel):
    """One reference-vs-test backend comparison.

    Attributes:
        ref: Reference pipeline.
        test: Tested pipeline.
        tolerance: Absolute tolerance replacing the verifier default, or ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: PipelineEndpoint
    test: PipelineEndpoint
    tolerance: float | None = None

    @model_validator(mode="after")
    def _reject_a_pipeline_against_itself(self) -> VerificationScenario:
        if self.ref == self.test:
            # Both sides would get the same cached pipeline, and the TensorRT runner reuses
            # its output buffers, so the comparison would read one buffer twice and always pass.
            raise ValueError(
                f"Verification scenario compares {self.ref} with itself. A scenario must name "
                "two different (backend, device) pairs; comparing a pipeline against itself "
                "always passes and verifies nothing."
            )
        return self

    def describe(self) -> str:
        """Human-readable scenario label."""
        return f"{self.ref} vs {self.test}"


class BackendVerifier:
    """Run verification scenarios over preprocessed batches.

    Args:
        pipelines: Pipeline cache shared with evaluation.
        tolerance: Absolute tolerance of the scenarios that set none.
    """

    def __init__(self, pipelines: PipelineCache, tolerance: float) -> None:
        self.pipelines = pipelines
        self.tolerance = float(tolerance)

    def run(
        self,
        batches: Sequence[Any],
        scenarios: Sequence[VerificationScenario],
        available_backends: set[Backend],
    ) -> bool:
        """Run the scenarios whose backends are available and log a report per scenario.

        Returns:
            Whether every executed scenario passed on every batch.

        Raises:
            ValueError: If no scenario is configured or none can run, so that verification
                never passes without comparing anything.
        """
        if not scenarios:
            raise ValueError(
                "deploy.verification.enabled=true but deploy.verification.scenarios is empty; "
                "list the reference-vs-test comparisons to run, or disable verification."
            )

        executed = 0
        all_passed = True
        for scenario in scenarios:
            required = {scenario.ref.backend, scenario.test.backend}
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

    def _run_scenario(self, scenario: VerificationScenario, batches: Sequence[Any]) -> bool:
        ref_pipeline = self.pipelines.get(scenario.ref.backend, scenario.ref.device)
        test_pipeline = self.pipelines.get(scenario.test.backend, scenario.test.device)

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
            summary, details = compare_outputs(
                test_pipeline.output_names,
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
                    detail.name,
                    detail.shape,
                    detail.max_diff,
                    detail.mean_diff,
                )
            if not summary.passed:
                logger.error("  sample %d failed: %s", index, summary.reason)
                passed_all = False

        logger.info("Scenario %s: %s", scenario.describe(), "PASSED" if passed_all else "FAILED")
        return passed_all
