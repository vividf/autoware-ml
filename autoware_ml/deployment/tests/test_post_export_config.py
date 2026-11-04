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


class TestPostExportConfig(unittest.TestCase):
    """Unit tests for parsing the verification and evaluation sections of a deploy config."""

    def test_round_trip_leaves_the_export_sections_alone(self) -> None:
        cfg = PostExportConfig.from_deploy_cfg(_RAW)
        self.assertTrue(cfg.verification.enabled)
        self.assertEqual(cfg.verification.scenarios[0].tolerance, 2.0)
        self.assertEqual(cfg.verification.tolerance, 0.01)
        self.assertEqual(cfg.verification.num_verify_batches, 1)
        self.assertEqual(cfg.evaluation.num_samples, 10)
        self.assertEqual(cfg.evaluation.split, "test")
        self.assertEqual([b for b, _ in cfg.evaluation.enabled_backends()], [Backend.PYTORCH])
        self.assertEqual(cfg.evaluation.backends[Backend.TENSORRT].device, "cuda:1")
        self.assertTrue(cfg.any_enabled)

    def test_absent_sections_are_all_defaults_and_nothing_is_enabled(self) -> None:
        cfg = PostExportConfig.from_deploy_cfg({"onnx": {}, "tensorrt": {}})
        self.assertFalse(cfg.verification.enabled)
        self.assertFalse(cfg.evaluation.enabled)
        self.assertFalse(cfg.any_enabled)
        self.assertEqual(PostExportConfig.from_deploy_cfg(None).verification.scenarios, ())

    def test_unknown_keys_are_rejected_with_the_config_path(self) -> None:
        cases = [
            ({"verification": {"tolerence": 1.0}}, "verification.tolerence"),
            ({"evaluation": {"samples": 3}}, "evaluation.samples"),
            (
                {"evaluation": {"backends": {"onnx": {"devcie": "cuda"}}}},
                "evaluation.backends.onnx.devcie",
            ),
        ]
        for raw, where in cases:
            with self.subTest(where=where):
                with self.assertRaisesRegex(ValidationError, where):
                    PostExportConfig.from_deploy_cfg(raw)

    def test_unknown_evaluation_backend_and_split_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "tensorflow"):
            PostExportConfig.from_deploy_cfg({"evaluation": {"backends": {"tensorflow": {}}}})
        with self.assertRaisesRegex(ValidationError, "evaluation.split"):
            PostExportConfig.from_deploy_cfg({"evaluation": {"split": "train"}})
        self.assertEqual(
            PostExportConfig.from_deploy_cfg({"evaluation": {"split": "val"}}).evaluation.split,
            "val",
        )

    def test_a_non_mapping_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "verification"):
            PostExportConfig.from_deploy_cfg({"verification": "yes"})

    def test_the_deploy_config_itself_parses(self) -> None:
        cfg = PostExportConfig.from_deploy_cfg(OmegaConf.create(_RAW))
        self.assertEqual(cfg.evaluation.num_samples, 10)


class TestEvaluationConfig(unittest.TestCase):
    """Unit tests for the defaults and limits of the evaluation section."""

    def test_evaluation_batch_size_defaults_to_one_frame_and_rejects_zero(self) -> None:
        self.assertEqual(EvaluationConfig().batch_size, 1)
        self.assertEqual(EvaluationConfig(batch_size=4).batch_size, 4)
        with self.assertRaisesRegex(ValidationError, "batch_size"):
            EvaluationConfig(batch_size=0)

    def test_evaluation_thread_count_defaults_to_one_and_parses(self) -> None:
        self.assertEqual(EvaluationConfig().cpu_threads, 1)
        self.assertEqual(EvaluationConfig(cpu_threads=0).cpu_threads, 0)
        self.assertEqual(EvaluationConfig(cpu_threads=8).cpu_threads, 8)


if __name__ == "__main__":
    unittest.main()
