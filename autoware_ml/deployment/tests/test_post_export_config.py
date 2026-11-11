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

"""Unit tests for the post-export config sections."""

# cspell:ignore tolerence devcie

from __future__ import annotations

import unittest

from omegaconf import OmegaConf
from pydantic import ValidationError

from autoware_ml.deployment.config import PostExportConfig

_RAW = {
    "onnx": {"dynamo": False, "opset_version": 17, "modules": {"m": {"output_names": ["y"]}}},
    "tensorrt": {"enabled": False},
    "verification": {
        "enabled": True,
        "scenarios": [
            {"ref": {"backend": "pytorch"}, "test": {"backend": "onnx"}, "tolerance": 2.0}
        ],
    },
}


class TestPostExportConfig(unittest.TestCase):
    """Unit tests for parsing the verification section of a deploy config."""

    def test_round_trip_leaves_the_export_sections_alone(self) -> None:
        cfg = PostExportConfig.from_deploy_cfg(_RAW)
        self.assertTrue(cfg.verification.enabled)
        self.assertEqual(cfg.verification.scenarios[0].tolerance, 2.0)
        self.assertEqual(cfg.verification.tolerance, 0.01)
        self.assertEqual(cfg.verification.num_verify_batches, 1)
        self.assertTrue(cfg.any_enabled)

    def test_absent_sections_are_all_defaults_and_nothing_is_enabled(self) -> None:
        cfg = PostExportConfig.from_deploy_cfg({"onnx": {}, "tensorrt": {}})
        self.assertFalse(cfg.verification.enabled)
        self.assertFalse(cfg.any_enabled)
        self.assertEqual(PostExportConfig.from_deploy_cfg(None).verification.scenarios, ())

    def test_unknown_keys_are_rejected_with_the_config_path(self) -> None:
        cases = [
            ({"verification": {"tolerence": 1.0}}, "verification.tolerence"),
        ]
        for raw, where in cases:
            with self.subTest(where=where):
                with self.assertRaisesRegex(ValidationError, where):
                    PostExportConfig.from_deploy_cfg(raw)

    def test_a_non_mapping_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "verification"):
            PostExportConfig.from_deploy_cfg({"verification": "yes"})

    def test_the_deploy_config_itself_parses(self) -> None:
        cfg = PostExportConfig.from_deploy_cfg(OmegaConf.create(_RAW))
        self.assertEqual(cfg.verification.scenarios[0].tolerance, 2.0)


if __name__ == "__main__":
    unittest.main()
