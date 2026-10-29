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

"""Unit tests for the TensorRT optimization profile."""

from __future__ import annotations

from omegaconf import OmegaConf
import pytest
from pydantic import ValidationError

from autoware_ml.deployment.backends import tensorrt_builder
from autoware_ml.deployment.backends.tensorrt_builder import ShapeProfile


def test_profile_parses_and_keeps_order() -> None:
    profile = ShapeProfile.model_validate(
        {"min_shape": [1, 32, 8, 8], "opt_shape": [1, 32, 16, 16], "max_shape": [1, 32, 32, 32]}
    )
    assert profile.min_shape == (1, 32, 8, 8)
    assert profile.max_shape == (1, 32, 32, 32)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"min_shape": [1], "opt_shape": [1]}, "max_shape"),
        ({"min_shape": [1, 2], "opt_shape": [1], "max_shape": [1]}, "same non-zero rank"),
        ({"min_shape": [2], "opt_shape": [1], "max_shape": [3]}, "min <= opt <= max"),
        ({"min_shape": [0], "opt_shape": [0], "max_shape": [3]}, "opt >= 1"),
        ({"min_shape": [1], "opt_shape": [1], "max_shape": [1], "typo": 1}, "typo"),
    ],
)
def test_bad_profiles_are_config_errors_naming_the_problem(raw, match) -> None:
    with pytest.raises(ValidationError, match=match):
        ShapeProfile.model_validate(raw)


def test_zero_minimum_is_legal() -> None:
    profile = ShapeProfile.model_validate(
        {"min_shape": [0, 11], "opt_shape": [20000, 11], "max_shape": [96000, 11]}
    )
    assert profile.min_shape == (0, 11)


def test_build_engine_from_config_names_the_input_of_a_bad_profile(tmp_path, monkeypatch) -> None:
    deploy_cfg = OmegaConf.create(
        {
            "tensorrt": {
                "input_shapes": {
                    "x": {"min_shape": [2], "opt_shape": [1], "max_shape": [3]},
                }
            }
        }
    )
    monkeypatch.setattr(tensorrt_builder, "build_engine", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match=r"deploy\.tensorrt\.input_shapes\.x"):
        tensorrt_builder.build_engine_from_config(
            tmp_path / "a.onnx", deploy_cfg, tmp_path / "a.engine"
        )


def test_build_engine_from_config_resolves_interpolations(tmp_path, monkeypatch) -> None:
    deploy_cfg = OmegaConf.create(
        {
            "max_voxels": 8,
            "tensorrt": {
                "input_shapes": {
                    "x": {"min_shape": [1], "opt_shape": [4], "max_shape": ["${max_voxels}"]},
                }
            },
        }
    )
    seen = {}
    monkeypatch.setattr(
        tensorrt_builder, "build_engine", lambda *args, **kwargs: seen.update(kwargs)
    )
    tensorrt_builder.build_engine_from_config(
        tmp_path / "a.onnx", deploy_cfg, tmp_path / "a.engine"
    )
    assert seen["input_shapes"] == {
        "x": ShapeProfile(min_shape=(1,), opt_shape=(4,), max_shape=(8,))
    }
