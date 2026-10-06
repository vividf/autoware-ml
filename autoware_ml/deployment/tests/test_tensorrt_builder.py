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

"""TensorRT builder value objects that need no GPU: the optimization-profile validation."""

from __future__ import annotations

import pytest

from autoware_ml.deployment.backends.tensorrt_builder import ShapeProfile


def test_profile_parses_and_keeps_order() -> None:
    profile = ShapeProfile.from_dict(
        {"min_shape": [1, 32, 8, 8], "opt_shape": [1, 32, 16, 16], "max_shape": [1, 32, 32, 32]},
        "deploy.tensorrt.input_shapes.spatial_features",
    )
    assert profile.min_shape == (1, 32, 8, 8)
    assert profile.max_shape == (1, 32, 32, 32)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"min_shape": [1], "opt_shape": [1]}, "incomplete"),
        ({"min_shape": [1, 2], "opt_shape": [1], "max_shape": [1]}, "same non-zero rank"),
        ({"min_shape": [2], "opt_shape": [1], "max_shape": [3]}, "min <= opt <= max"),
        ({"min_shape": [0], "opt_shape": [0], "max_shape": [3]}, "opt >= 1"),
        ({"min_shape": [1], "opt_shape": [1], "max_shape": [1], "typo": 1}, "Unknown"),
    ],
)
def test_bad_profiles_are_config_errors_naming_the_input(raw, match) -> None:
    with pytest.raises(ValueError, match=match):
        ShapeProfile.from_dict(raw, "deploy.tensorrt.input_shapes.x")


def test_zero_minimum_is_legal() -> None:
    profile = ShapeProfile.from_dict(
        {"min_shape": [0, 11], "opt_shape": [20000, 11], "max_shape": [96000, 11]}, "x"
    )
    assert profile.min_shape == (0, 11)
