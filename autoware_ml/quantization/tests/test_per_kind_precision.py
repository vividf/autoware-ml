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

"""Unit tests for per-kind precision in ``QuantRules``."""

from __future__ import annotations

import unittest

from torch import nn

from autoware_ml.quantization.config import Precision, QuantizationConfig
from autoware_ml.quantization.plan import prepare_quantized_tree
from autoware_ml.quantization.rules import QuantRules


class _Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3)
        self.linear = nn.Linear(4, 4)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = _Body()


def _config() -> QuantizationConfig:
    return QuantizationConfig.from_dict(
        {"enabled": True, "mode": "ptq", "ptq": {"calibrate_samples": 1}}
    )


class TestResolvedKinds(unittest.TestCase):
    """Submodule rules resolve each module kind to a precision, or reject it."""

    def test_tuple_rules_resolve_every_kind_to_the_default_precision(self) -> None:
        rules = QuantRules(quantize_submodules={"body": ("conv", "linear")})
        resolved = rules.resolved_kinds("body", Precision.INT8)
        self.assertEqual(resolved, {"conv": Precision.INT8, "linear": Precision.INT8})

    def test_mapping_rules_carry_per_kind_precision(self) -> None:
        rules = QuantRules(quantize_submodules={"body": {"conv": "int8", "linear": "fp8"}})
        resolved = rules.resolved_kinds("body", Precision.INT8)
        self.assertEqual(resolved, {"conv": Precision.INT8, "linear": Precision.FP8})
        rules = QuantRules(quantize_submodules={"body": {"conv": None}})
        self.assertEqual(rules.resolved_kinds("body", Precision.FP8), {"conv": Precision.FP8})

    def test_unknown_kind_and_unknown_precision_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown module kind"):
            QuantRules(quantize_submodules={"body": {"attention": "int8"}})
        with self.assertRaises(ValueError):
            QuantRules(quantize_submodules={"body": {"conv": "fp42"}})


class TestMixedPrecisionPrepare(unittest.TestCase):
    """Preparing a tree quantizes each declared kind at its own precision."""

    def test_mixed_precision_prepare_quantizes_each_kind_at_its_precision(self) -> None:
        model = _Model().eval()
        rules = QuantRules(quantize_submodules={"body": {"conv": "int8", "linear": "fp8"}})
        prepared = prepare_quantized_tree(model, _config(), rules=rules)

        conv_bits = model.body.conv.weight_quantizer.num_bits
        linear_bits = model.body.linear.weight_quantizer.num_bits
        self.assertEqual(conv_bits, 8)
        self.assertEqual(linear_bits, (4, 3))  # FP8 E4M3

        # The record spells the deviating precision, and only that one.
        details = {d.module: d.detail for d in prepared.placement_record.decisions}
        self.assertIn("@fp8", details["body.linear"])
        self.assertNotIn("@", details["body.conv"])

    def test_attention_out_proj_is_never_replaced(self) -> None:
        """The attention forward reads ``out_proj.weight`` directly; a quantizer would never run."""

        class _AttnBody(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
                self.linear = nn.Linear(8, 8)

        class _AttnModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = _AttnBody()

        model = _AttnModel().eval()
        rules = QuantRules(quantize_submodules={"body": {"linear": "fp8"}})
        prepared = prepare_quantized_tree(model, _config(), rules=rules)

        self.assertEqual(type(model.body.attn.out_proj).__name__, "NonDynamicallyQuantizableLinear")
        self.assertFalse(hasattr(model.body.attn.out_proj, "weight_quantizer"))
        self.assertNotIn("weight_quantizer", "".join(model.body.attn.state_dict().keys()))
        self.assertEqual(model.body.linear.weight_quantizer.num_bits, (4, 3))
        replaced = [
            d.module for d in prepared.placement_record.decisions if d.transform == "replace_module"
        ]
        self.assertEqual(replaced, ["body.linear"])

    def test_default_precision_records_stay_identical_to_the_pre_feature_format(self) -> None:
        model = _Model().eval()
        rules = QuantRules(quantize_submodules={"body": ("conv", "linear")})
        prepared = prepare_quantized_tree(model, _config(), rules=rules)
        details = [
            d.detail for d in prepared.placement_record.decisions if d.transform == "replace_module"
        ]
        self.assertEqual(details, ["Conv2d -> QuantConv2d", "Linear -> QuantLinear"])
