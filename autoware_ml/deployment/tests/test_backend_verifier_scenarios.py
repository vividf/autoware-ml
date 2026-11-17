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

"""Unit tests for verification scenarios."""

from __future__ import annotations

import unittest

import numpy as np
from pydantic import ValidationError

from autoware_ml.deployment.verification.backend_verifier import (
    BackendVerifier,
    VerificationScenario,
)


class TestVerificationScenario(unittest.TestCase):
    """Unit tests for parsing a verification scenario from its config mapping."""

    def test_valid_mapping(self):
        scenario = VerificationScenario.model_validate(
            {
                "ref": {"backend": "pytorch", "device": "cpu"},
                "test": {"backend": "onnx", "device": "cuda"},
            }
        )
        self.assertEqual(scenario.ref.backend, "pytorch")
        self.assertEqual(scenario.ref.device, "cpu")
        self.assertEqual(scenario.test.backend, "onnx")
        self.assertEqual(scenario.test.device, "cuda")
        self.assertIsNone(scenario.tolerance)

    def test_device_defaults_to_cuda(self):
        scenario = VerificationScenario.model_validate(
            {"ref": {"backend": "pytorch"}, "test": {"backend": "tensorrt"}}
        )
        self.assertEqual(scenario.ref.device, "cuda")
        self.assertEqual(scenario.test.device, "cuda")

    def test_per_scenario_tolerance_parsed_as_float(self):
        scenario = VerificationScenario.model_validate(
            {
                "ref": {"backend": "pytorch"},
                "test": {"backend": "onnx"},
                "tolerance": "0.001",
            }
        )
        self.assertIsInstance(scenario.tolerance, float)
        np.testing.assert_allclose(scenario.tolerance, 0.001, rtol=1e-6, atol=1e-12)

    def test_missing_tolerance_stays_none(self):
        scenario = VerificationScenario.model_validate(
            {"ref": {"backend": "pytorch"}, "test": {"backend": "onnx"}}
        )
        self.assertIsNone(scenario.tolerance)

    def test_malformed_input_raises_value_error(self):
        cases = [
            {},
            {"ref": {"backend": "pytorch"}},
            {"ref": {"device": "cuda"}, "test": {"backend": "onnx"}},
            {"ref": "pytorch", "test": "onnx"},
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValidationError):
                    VerificationScenario.model_validate(raw)

    def test_scenario_is_frozen(self):
        scenario = VerificationScenario.model_validate(
            {"ref": {"backend": "pytorch"}, "test": {"backend": "onnx"}}
        )
        with self.assertRaises(ValidationError):
            scenario.tolerance = 0.5

    def test_describe_names_backends_and_devices(self):
        scenario = VerificationScenario.model_validate(
            {
                "ref": {"backend": "pytorch", "device": "cpu"},
                "test": {"backend": "onnx", "device": "cuda"},
            }
        )
        self.assertEqual(scenario.describe(), "pytorch(cpu) vs onnx(cuda)")


class TestDegenerateScenariosAreRejected(unittest.TestCase):
    """Unit tests for refusing scenarios and runs that compare nothing."""

    def test_same_backend_and_device_is_refused(self):
        with self.assertRaisesRegex(ValidationError, "with itself"):
            VerificationScenario.model_validate(
                {
                    "ref": {"backend": "tensorrt", "device": "cuda"},
                    "test": {"backend": "tensorrt", "device": "cuda"},
                }
            )

    def test_same_backend_on_different_devices_is_allowed(self):
        scenario = VerificationScenario.model_validate(
            {
                "ref": {"backend": "pytorch", "device": "cpu"},
                "test": {"backend": "pytorch", "device": "cuda"},
            }
        )
        self.assertEqual(scenario.ref.device, "cpu")
        self.assertEqual(scenario.test.device, "cuda")

    def test_verification_enabled_without_scenarios_raises(self):
        verifier = BackendVerifier(pipelines=None, tolerance=0.01)
        with self.assertRaisesRegex(ValueError, "scenarios is empty"):
            verifier.run(batches=[], scenarios=(), available_backends=set())


if __name__ == "__main__":
    unittest.main()
