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
from types import SimpleNamespace
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
    """Building from the deploy config reads, validates and resolves the module's input shapes."""

    def setUp(self) -> None:
        """Give each test its own directory for the (never written) ONNX and engine paths."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def _build(self, deploy_cfg: dict, module_name: str = "head") -> dict:
        """Build ``module_name`` with a fake ``build_engine`` and return the keywords it got."""
        seen: dict = {}
        with patch.object(
            tensorrt_builder, "build_engine", lambda *args, **kwargs: seen.update(kwargs)
        ):
            tensorrt_builder.build_engine_from_config(
                self.tmp_path / "a.onnx",
                OmegaConf.create(deploy_cfg),
                self.tmp_path / "a.engine",
                module_name,
            )
        return seen

    def test_build_engine_from_config_names_the_input_of_a_bad_profile(self) -> None:
        """Test that a profile violating min <= opt <= max is refused naming its config key."""
        deploy_cfg = {
            "tensorrt": {
                "modules": {
                    "head": {
                        "input_shapes": {
                            "x": {"min_shape": [2], "opt_shape": [1], "max_shape": [3]},
                        }
                    }
                }
            }
        }
        with self.assertRaisesRegex(
            ValueError, r"deploy\.tensorrt\.modules\.head\.input_shapes\.x"
        ):
            self._build(deploy_cfg)

    def test_build_engine_from_config_reads_the_module_entry_resolved(self) -> None:
        """
        Test that a module's entry builds its profile with interpolations resolved, and that
        a module without an entry builds without a profile.
        """
        deploy_cfg = {
            "max_voxels": 8,
            "tensorrt": {
                "modules": {
                    "head": {
                        "input_shapes": {
                            "x": {
                                "min_shape": [1],
                                "opt_shape": [4],
                                "max_shape": ["${max_voxels}"],
                            },
                        }
                    }
                }
            },
        }
        seen = self._build(deploy_cfg)
        self.assertEqual(
            seen["input_shapes"],
            {"x": ShapeProfile(min_shape=(1,), opt_shape=(4,), max_shape=(8,))},
        )
        self.assertEqual(seen["input_shapes_key"], "deploy.tensorrt.modules.head.input_shapes")
        self.assertIsNone(self._build(deploy_cfg, "encoder")["input_shapes"])

    def test_build_engine_from_config_refuses_the_old_shared_table(self) -> None:
        """Test that ``deploy.tensorrt.input_shapes`` at the top level is refused by key."""
        deploy_cfg = {
            "tensorrt": {
                "input_shapes": {"x": {"min_shape": [1], "opt_shape": [1], "max_shape": [1]}}
            }
        }
        with self.assertRaisesRegex(ValueError, r"deploy\.tensorrt\.modules\.head\.input_shapes"):
            self._build(deploy_cfg)


def _network(**shapes: tuple[int, ...]) -> SimpleNamespace:
    """A stand-in for a parsed TensorRT network with the given input shapes, -1 for dynamic."""
    inputs = [SimpleNamespace(name=name, shape=shape) for name, shape in shapes.items()]
    return SimpleNamespace(num_inputs=len(inputs), get_input=lambda index: inputs[index])


class _FakeProfile:
    """Records the shapes set on it."""

    def __init__(self) -> None:
        self.shapes: dict[str, tuple[list[int], list[int], list[int]]] = {}

    def set_shape(self, name: str, min: list[int], opt: list[int], max: list[int]) -> None:
        self.shapes[name] = (min, opt, max)


class TestOptimizationProfile(unittest.TestCase):
    """Every entry names an input of the network, and a dynamic input needs an entry."""

    def test_an_entry_of_no_input_is_refused_by_key(self) -> None:
        """Test that a misspelled entry is refused naming the key and the graph's inputs."""
        table = {
            "spatial_feature": ShapeProfile(
                min_shape=(1, 8, 4, 4), opt_shape=(1, 8, 4, 4), max_shape=(1, 8, 4, 4)
            ),
        }
        builder = SimpleNamespace(create_optimization_profile=_FakeProfile)
        with self.assertRaisesRegex(
            ValueError, r"k names \['spatial_feature'\], which are not inputs of the graph"
        ):
            tensorrt_builder._create_optimization_profile(
                builder, _network(spatial_features=(1, 8, 4, 4)), table, "k"
            )
        profile = tensorrt_builder._create_optimization_profile(
            builder,
            _network(spatial_features=(1, 8, 4, 4)),
            {"spatial_features": table["spatial_feature"]},
            "k",
        )
        self.assertEqual(
            profile.shapes, {"spatial_features": ([1, 8, 4, 4], [1, 8, 4, 4], [1, 8, 4, 4])}
        )

    def test_dynamic_inputs_without_an_entry_are_named_in_order(self) -> None:
        """Test that only the dynamic inputs missing from the table are reported."""
        network = _network(voxels=(-1, 32, 5), coors=(-1, 3), canvas=(1, 64, 180, 180), mask=(-1,))
        table = {
            "voxels": ShapeProfile(min_shape=(1, 32, 5), opt_shape=(8, 32, 5), max_shape=(9, 32, 5))
        }
        self.assertEqual(
            tensorrt_builder.dynamic_inputs_without_profile(network, table), ["coors", "mask"]
        )
        self.assertEqual(
            tensorrt_builder.dynamic_inputs_without_profile(_network(x=(1, 2)), None), []
        )

    def test_build_engine_names_the_missing_profile_before_building(self) -> None:
        """Test that the build stops with the input and the config key, before TensorRT builds."""
        builder = SimpleNamespace(
            build_serialized_network=lambda network, config: self.fail("the engine was built")
        )
        network = _network(voxels=(-1, 32, 5))
        with (
            patch.object(
                tensorrt_builder, "_create_builder", lambda *args: (builder, network, None, None)
            ),
            patch.object(tensorrt_builder, "_parse_onnx_file", lambda parser, path: None),
        ):
            with self.assertRaisesRegex(
                ValueError, r"model\.onnx has dynamic inputs \['voxels'\] without a k entry"
            ):
                tensorrt_builder.build_engine(
                    Path("model.onnx"), Path("model.engine"), input_shapes_key="k"
                )
