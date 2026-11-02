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

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
from pydantic import ValidationError

from autoware_ml.deployment.backends import tensorrt_builder
from autoware_ml.deployment.backends.tensorrt_builder import ShapeProfile


class TestShapeProfile(unittest.TestCase):
    """A shape profile parses in order and names what is wrong with a bad one."""

    def test_profile_parses_and_keeps_order(self) -> None:
        profile = ShapeProfile.model_validate(
            {"min_shape": [1, 32, 8, 8], "opt_shape": [1, 32, 16, 16], "max_shape": [1, 32, 32, 32]}
        )
        self.assertEqual(profile.min_shape, (1, 32, 8, 8))
        self.assertEqual(profile.max_shape, (1, 32, 32, 32))

    def test_bad_profiles_are_config_errors_naming_the_problem(self) -> None:
        cases = [
            ({"min_shape": [1], "opt_shape": [1]}, "max_shape"),
            ({"min_shape": [1, 2], "opt_shape": [1], "max_shape": [1]}, "same non-zero rank"),
            ({"min_shape": [2], "opt_shape": [1], "max_shape": [3]}, "min <= opt <= max"),
            ({"min_shape": [0], "opt_shape": [0], "max_shape": [3]}, "opt >= 1"),
            ({"min_shape": [1], "opt_shape": [1], "max_shape": [1], "typo": 1}, "typo"),
        ]
        for raw, match in cases:
            with self.subTest(match=match):
                with self.assertRaisesRegex(ValidationError, match):
                    ShapeProfile.model_validate(raw)

    def test_zero_minimum_is_legal(self) -> None:
        profile = ShapeProfile.model_validate(
            {"min_shape": [0, 11], "opt_shape": [20000, 11], "max_shape": [96000, 11]}
        )
        self.assertEqual(profile.min_shape, (0, 11))


class TestBuildEngineFromConfig(unittest.TestCase):
    """Building from the deploy config validates and resolves its input shapes."""

    def setUp(self) -> None:
        """Give each test its own directory for the (never written) ONNX and engine paths."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_build_engine_from_config_names_the_input_of_a_bad_profile(self) -> None:
        deploy_cfg = OmegaConf.create(
            {
                "tensorrt": {
                    "input_shapes": {
                        "x": {"min_shape": [2], "opt_shape": [1], "max_shape": [3]},
                    }
                }
            }
        )
        with patch.object(tensorrt_builder, "build_engine", lambda *args, **kwargs: None):
            with self.assertRaisesRegex(ValueError, r"deploy\.tensorrt\.input_shapes\.x"):
                tensorrt_builder.build_engine_from_config(
                    self.tmp_path / "a.onnx", deploy_cfg, self.tmp_path / "a.engine"
                )

    def test_build_engine_from_config_resolves_interpolations(self) -> None:
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
        with patch.object(
            tensorrt_builder, "build_engine", lambda *args, **kwargs: seen.update(kwargs)
        ):
            tensorrt_builder.build_engine_from_config(
                self.tmp_path / "a.onnx", deploy_cfg, self.tmp_path / "a.engine"
            )
        self.assertEqual(
            seen["input_shapes"],
            {"x": ShapeProfile(min_shape=(1,), opt_shape=(4,), max_shape=(8,))},
        )
