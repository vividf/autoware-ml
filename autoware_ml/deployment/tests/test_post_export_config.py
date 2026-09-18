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

"""Post-export config: typo guard and defaults for the verification / evaluation sections."""

from __future__ import annotations

import pytest

from autoware_ml.deployment.config import EvaluationConfig, PostExportConfig
from autoware_ml.types.backend import Backend

_RAW = {
    "onnx": {"dynamo": False, "opset_version": 17, "modules": {"m": {"output_names": ["y"]}}},
    "tensorrt": {"enabled": False},
    "verification": {
        "enabled": True,
        "scenarios": [
            {"ref": {"backend": "pytorch"}, "test": {"backend": "onnx"}, "tolerance": 2.0}
        ],
    },
    "evaluation": {
        "enabled": True,
        "num_samples": 10,
        "backends": {
            "pytorch": {"enabled": True},
            "tensorrt": {"enabled": False, "device": "cuda:1"},
        },
    },
}


def test_round_trip_leaves_the_export_sections_alone() -> None:
    cfg = PostExportConfig.from_deploy_cfg(_RAW)
    assert cfg.verification.enabled and cfg.verification.scenarios[0].tolerance == 2.0
    assert cfg.verification.tolerance == 0.01 and cfg.verification.num_verify_batches == 1
    assert cfg.evaluation.num_samples == 10 and cfg.evaluation.split == "test"
    assert [b for b, _ in cfg.evaluation.enabled_backends()] == [Backend.PYTORCH]
    assert cfg.evaluation.backends[Backend.TENSORRT].device == "cuda:1"
    assert cfg.any_enabled


def test_absent_sections_are_all_defaults_and_nothing_is_enabled() -> None:
    cfg = PostExportConfig.from_deploy_cfg({"onnx": {}, "tensorrt": {}})
    assert not cfg.verification.enabled and not cfg.evaluation.enabled
    assert not cfg.any_enabled
    assert PostExportConfig.from_deploy_cfg(None).verification.scenarios == ()


@pytest.mark.parametrize(
    ("raw", "where"),
    [
        ({"verification": {"tolerence": 1.0}}, "deploy.verification"),
        ({"evaluation": {"samples": 3}}, "deploy.evaluation"),
        (
            {"evaluation": {"backends": {"onnx": {"devcie": "cuda"}}}},
            "deploy.evaluation.backends.onnx",
        ),
    ],
)
def test_unknown_keys_are_rejected_with_the_config_path(raw, where) -> None:
    with pytest.raises(ValueError, match=where):
        PostExportConfig.from_deploy_cfg(raw)


def test_evaluation_thread_count_defaults_to_one_and_parses() -> None:
    assert EvaluationConfig.from_dict({}).cpu_threads == 1
    assert EvaluationConfig.from_dict({"cpu_threads": 0}).cpu_threads == 0
    assert EvaluationConfig.from_dict({"cpu_threads": 8}).cpu_threads == 8


def test_unknown_evaluation_backend_and_split_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        PostExportConfig.from_deploy_cfg({"evaluation": {"backends": {"tflite": {}}}})
    with pytest.raises(ValueError, match="evaluation.split"):
        PostExportConfig.from_deploy_cfg({"evaluation": {"split": "train"}})
    assert (
        PostExportConfig.from_deploy_cfg({"evaluation": {"split": "val"}}).evaluation.split == "val"
    )


def test_a_non_mapping_section_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="deploy.verification"):
        PostExportConfig.from_deploy_cfg({"verification": "yes"})
