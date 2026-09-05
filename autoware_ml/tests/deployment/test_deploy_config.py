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

"""DeployConfig: typo guard, per-stage layout, stage-name check."""

from __future__ import annotations

import pytest

from autoware_ml.deployment.config import DeployConfig, OnnxPrecision
from autoware_ml.types.backend import Backend

_RAW = {
    "onnx": {"dynamo": False, "opset_version": 17, "precision": "fp16"},
    "tensorrt": {"enabled": False},
    "stages": {
        "pts_voxel_encoder": {
            "onnx": {"dynamic_axes": {"input_features": {0: "num_voxels"}}},
            "tensorrt": {
                "input_shapes": {
                    "input_features": {
                        "min_shape": [1, 32, 11],
                        "opt_shape": [2, 32, 11],
                        "max_shape": [3, 32, 11],
                    }
                }
            },
        }
    },
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


class TestDeployConfig:
    def test_round_trip(self):
        cfg = DeployConfig.from_dict(_RAW)
        assert cfg.onnx.dynamo is False and cfg.onnx.opset_version == 17
        assert cfg.onnx.precision is OnnxPrecision.FP16
        assert cfg.tensorrt.enabled is False
        stage = cfg.stage("pts_voxel_encoder")
        assert stage.onnx.dynamic_axes == {"input_features": {0: "num_voxels"}}
        assert stage.tensorrt.input_shapes["input_features"].opt_shape == (2, 32, 11)
        assert cfg.verification.scenarios[0].tolerance == 2.0
        assert cfg.evaluation.num_samples == 10
        assert [b for b, _ in cfg.evaluation.enabled_backends()] == [Backend.PYTORCH]
        assert cfg.evaluation.backends[Backend.TENSORRT].device == "cuda:1"

    def test_absent_section_is_all_defaults(self):
        cfg = DeployConfig.from_dict(None)
        assert cfg.onnx.enabled and cfg.tensorrt.enabled
        assert not cfg.verification.enabled and not cfg.evaluation.enabled
        assert cfg.stage("anything").tensorrt.input_shapes == {}

    @pytest.mark.parametrize(
        "raw, where",
        [
            ({"onnx": {"opset_versoin": 17}}, "deploy.onnx"),
            ({"tensorrt": {"workspace": 1}}, "deploy.tensorrt"),
            ({"stages": {"s": {"onnx": {"input_names": ["x"]}}}}, "deploy.stages.s.onnx"),
            ({"stages": {"s": {"trt": {}}}}, "deploy.stages.s"),
            (
                {"evaluation": {"backends": {"onnx": {"devcie": "cuda"}}}},
                "deploy.evaluation.backends.onnx",
            ),
            ({"unknown_top": 1}, "deploy"),
        ],
    )
    def test_unknown_keys_rejected(self, raw, where):
        with pytest.raises(ValueError, match=where):
            DeployConfig.from_dict(raw)

    def test_incomplete_shape_profile_rejected(self):
        raw = {"stages": {"s": {"tensorrt": {"input_shapes": {"x": {"min_shape": [1]}}}}}}
        with pytest.raises(ValueError, match="incomplete"):
            DeployConfig.from_dict(raw)

    def test_unknown_onnx_precision_rejected(self):
        with pytest.raises(ValueError, match="deploy.onnx.precision"):
            DeployConfig.from_dict({"onnx": {"precision": "int8"}})

    def test_removed_precision_policy_rejected_as_unknown_key(self):
        with pytest.raises(ValueError, match="precision_policy"):
            DeployConfig.from_dict({"tensorrt": {"precision_policy": "fp16"}})

    def test_unknown_evaluation_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            DeployConfig.from_dict({"evaluation": {"backends": {"tflite": {}}}})

    def test_stage_name_typo_is_caught_against_the_declaration(self):
        cfg = DeployConfig.from_dict({"stages": {"pts_voxel_encodr": {}}})
        with pytest.raises(ValueError, match="pts_voxel_encodr"):
            cfg.check_stage_names(["pts_voxel_encoder", "pts_backbone_neck_head"])
        cfg = DeployConfig.from_dict(_RAW)
        cfg.check_stage_names(["pts_voxel_encoder"])


def test_stage_onnx_precision_overrides_the_global_setting() -> None:
    """A stage may pin its own precision; unset stages inherit ``deploy.onnx.precision``."""
    from autoware_ml.deployment.config import DeployConfig, OnnxPrecision

    cfg = DeployConfig.from_dict(
        {
            "onnx": {"enabled": True, "precision": "fp16"},
            "tensorrt": {"enabled": False},
            "stages": {"fragile_head": {"onnx": {"precision": "fp32"}}},
        }
    )
    assert cfg.onnx.precision is OnnxPrecision.FP16
    assert cfg.stage("fragile_head").onnx.precision is OnnxPrecision.FP32
    assert cfg.stage("other").onnx.precision is None


def test_stage_onnx_precision_rejects_unknown_values() -> None:
    import pytest

    from autoware_ml.deployment.config import DeployConfig

    with pytest.raises(ValueError, match="fragile_head.onnx.precision"):
        DeployConfig.from_dict(
            {
                "onnx": {"enabled": True},
                "tensorrt": {"enabled": False},
                "stages": {"fragile_head": {"onnx": {"precision": "fp42"}}},
            }
        )


def test_evaluation_split_parses_and_rejects_unknown_values() -> None:
    import pytest

    from autoware_ml.deployment.config import DeployConfig

    base = {"onnx": {"enabled": True}, "tensorrt": {"enabled": False}, "stages": {}}
    cfg = DeployConfig.from_dict({**base, "evaluation": {"enabled": True, "split": "val"}})
    assert cfg.evaluation.split == "val"
    assert DeployConfig.from_dict(base).evaluation.split == "test"
    with pytest.raises(ValueError, match="evaluation.split"):
        DeployConfig.from_dict({**base, "evaluation": {"split": "train"}})
