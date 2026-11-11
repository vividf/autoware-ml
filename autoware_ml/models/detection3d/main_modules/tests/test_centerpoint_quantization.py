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

"""Unit tests for CenterPoint quantization."""

from __future__ import annotations

import importlib.util
import unittest

import torch

from autoware_ml.quantization.config import QuantizationConfig

MODELOPT_AVAILABLE = importlib.util.find_spec("modelopt") is not None

if MODELOPT_AVAILABLE:
    from modelopt.torch.quantization.nn import TensorQuantizer

    from autoware_ml.models.detection3d.main_modules.centerpoint import CENTERPOINT_QUANT_RULES
    from autoware_ml.quantization.plan import prepare_quantized_tree

_CONFIG = QuantizationConfig.from_dict(
    {
        "enabled": True,
        "mode": "ptq",
        "skip_quantize": ["pts_voxel_encoder"],
        "ptq": {"calibrate_samples": 4},
    }
)


def _build_model():
    # Imported here: at module level the test loader would collect the model tests a second
    # time, and they would run with the ModelOpt module registry the quantization tests leave
    # patched.
    from autoware_ml.models.detection3d.main_modules.tests.test_centerpoint import (
        TestCenterPointDetection3DModel,
    )

    case = TestCenterPointDetection3DModel()
    case.setUp()
    return case.centerpoint.cpu()


def _prepared():
    torch.manual_seed(0)
    model = _build_model().eval()
    return model, prepare_quantized_tree(model, _CONFIG)


@unittest.skipUnless(MODELOPT_AVAILABLE, "could not import 'modelopt'")
class TestCenterPointQuantization(unittest.TestCase):
    """CenterPoint's declared rules build a reproducible quantized tree of the declared towers."""

    def test_rules_are_the_model_declaration(self) -> None:
        model = _build_model()
        self.assertIs(model.build_quantization_rules(), CENTERPOINT_QUANT_RULES)
        self.assertEqual(
            set(CENTERPOINT_QUANT_RULES.quantize_submodules),
            {
                "pts_backbone",
                "pts_neck",
                "bbox_head",
                "pts_voxel_encoder",
            },
        )

    def test_same_plan_builds_identical_trees_and_records(self) -> None:
        model_a, prepared_a = _prepared()
        model_b, prepared_b = _prepared()
        self.assertEqual(set(model_a.state_dict()), set(model_b.state_dict()))
        self.assertEqual(
            prepared_a.placement_record.decisions, prepared_b.placement_record.decisions
        )
        self.assertGreater(len(prepared_a.placement_record), 0)

    def test_producer_state_dict_loads_strict_into_loader_tree(self) -> None:
        producer, _ = _prepared()
        loader, _ = _prepared()
        incompatible = loader.load_state_dict(producer.state_dict(), strict=True)
        self.assertFalse(incompatible.missing_keys)
        self.assertFalse(incompatible.unexpected_keys)

    def test_prepare_folds_bn_and_quantizes_only_the_declared_towers(self) -> None:
        before = set(_build_model().state_dict())
        model, _ = _prepared()
        after = set(model.state_dict())
        self.assertTrue(any("running_mean" in key for key in before))
        self.assertFalse(
            any(key.startswith("pts_backbone") and "running_mean" in key for key in after)
        )
        self.assertTrue(
            any(key.startswith("pts_backbone") and key.endswith("conv.bias") for key in after)
        )

        quantizers = [n for n, m in model.named_modules() if isinstance(m, TensorQuantizer)]
        self.assertFalse(
            any(n.startswith("pts_voxel_encoder") for n in quantizers), "skip_quantize subtree"
        )
        self.assertTrue(
            any(n.startswith("pts_backbone") for n in quantizers),
            "the backbone must carry quantizers",
        )
        self.assertTrue(
            all(
                n.endswith(("input_quantizer", "weight_quantizer", "output_quantizer"))
                for n in quantizers
            )
        )
