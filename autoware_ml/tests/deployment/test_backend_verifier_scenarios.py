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

"""Unit tests for VerificationScenario.from_dict parsing."""

from __future__ import annotations

import dataclasses

import pytest

backend_verifier = pytest.importorskip(
    "autoware_ml.deployment.verification.backend_verifier",
    reason="backend_verifier transitively needs the full model stack",
)
VerificationScenario = backend_verifier.VerificationScenario


class TestVerificationScenarioFromDict:
    def test_valid_mapping(self):
        scenario = VerificationScenario.from_dict(
            {
                "ref": {"backend": "pytorch", "device": "cpu"},
                "test": {"backend": "onnx", "device": "cuda"},
            }
        )
        assert scenario.ref_backend == "pytorch"
        assert scenario.ref_device == "cpu"
        assert scenario.test_backend == "onnx"
        assert scenario.test_device == "cuda"
        assert scenario.tolerance is None

    def test_device_defaults_to_cuda(self):
        scenario = VerificationScenario.from_dict(
            {"ref": {"backend": "pytorch"}, "test": {"backend": "tensorrt"}}
        )
        assert scenario.ref_device == "cuda"
        assert scenario.test_device == "cuda"

    def test_per_scenario_tolerance_parsed_as_float(self):
        scenario = VerificationScenario.from_dict(
            {
                "ref": {"backend": "pytorch"},
                "test": {"backend": "onnx"},
                "tolerance": "0.001",
            }
        )
        assert isinstance(scenario.tolerance, float)
        assert scenario.tolerance == pytest.approx(0.001)

    def test_missing_tolerance_stays_none(self):
        scenario = VerificationScenario.from_dict(
            {"ref": {"backend": "pytorch"}, "test": {"backend": "onnx"}}
        )
        assert scenario.tolerance is None

    @pytest.mark.parametrize(
        "raw",
        [
            {},
            {"ref": {"backend": "pytorch"}},
            {"ref": {"device": "cuda"}, "test": {"backend": "onnx"}},
            {"ref": "pytorch", "test": "onnx"},
        ],
    )
    def test_malformed_input_raises_value_error(self, raw):
        with pytest.raises(ValueError, match="verification scenario"):
            VerificationScenario.from_dict(raw)

    def test_scenario_is_frozen(self):
        scenario = VerificationScenario.from_dict(
            {"ref": {"backend": "pytorch"}, "test": {"backend": "onnx"}}
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            scenario.tolerance = 0.5

    def test_describe_names_backends_and_devices(self):
        scenario = VerificationScenario.from_dict(
            {
                "ref": {"backend": "pytorch", "device": "cpu"},
                "test": {"backend": "onnx", "device": "cuda"},
            }
        )
        assert scenario.describe() == "pytorch(cpu) vs onnx(cuda)"
