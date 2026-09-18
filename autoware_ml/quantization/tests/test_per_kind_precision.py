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

"""Per-kind precision in QuantRules: mixed INT8/FP8 trees, and record compatibility
with the single-precision checkpoints that predate the feature."""

from __future__ import annotations

import pytest
from torch import nn

from autoware_ml.quantization.config import Precision, QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan, QuantRules


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


def test_tuple_rules_resolve_every_kind_to_the_default_precision() -> None:
    rules = QuantRules(quantize_submodules={"body": ("conv", "linear")}, recipes=())
    resolved = rules.resolved_kinds("body", Precision.INT8)
    assert resolved == {"conv": Precision.INT8, "linear": Precision.INT8}


def test_mapping_rules_carry_per_kind_precision() -> None:
    rules = QuantRules(quantize_submodules={"body": {"conv": "int8", "linear": "fp8"}}, recipes=())
    resolved = rules.resolved_kinds("body", Precision.INT8)
    assert resolved == {"conv": Precision.INT8, "linear": Precision.FP8}
    # None follows the default.
    rules = QuantRules(quantize_submodules={"body": {"conv": None}}, recipes=())
    assert rules.resolved_kinds("body", Precision.FP8) == {"conv": Precision.FP8}


def test_unknown_kind_and_unknown_precision_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown module kind"):
        QuantRules(quantize_submodules={"body": {"attention": "int8"}}, recipes=())
    with pytest.raises(ValueError):
        QuantRules(quantize_submodules={"body": {"conv": "fp42"}}, recipes=())


def test_mixed_precision_prepare_quantizes_each_kind_at_its_precision() -> None:
    model = _Model().eval()
    rules = QuantRules(quantize_submodules={"body": {"conv": "int8", "linear": "fp8"}}, recipes=())
    plan = QuantizationPlan(rules=rules, config=_config())
    plan.prepare(model)

    conv_bits = model.body.conv.weight_quantizer.num_bits
    linear_bits = model.body.linear.weight_quantizer.num_bits
    assert conv_bits == 8
    assert linear_bits == (4, 3)  # FP8 E4M3

    # The record spells the deviating precision, and only that one.
    details = {d.module: d.detail for d in plan.placement_record.decisions}
    assert "@fp8" in details["body.linear"]
    assert "@" not in details["body.conv"]


def test_attention_out_proj_is_never_replaced() -> None:
    """``nn.MultiheadAttention.out_proj`` must not become a QuantLinear.

    Its forward is bypassed by the attention fast path (``F.multi_head_attention_forward``
    reads ``.weight`` directly), so a quantizer there never collects calibration data and
    silently vanishes from any export that rebuilds the attention module. A plain Linear
    sibling in the same subtree still quantizes.
    """

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
    rules = QuantRules(quantize_submodules={"body": {"linear": "fp8"}}, recipes=())
    plan = QuantizationPlan(rules=rules, config=_config())
    plan.prepare(model)

    assert type(model.body.attn.out_proj).__name__ == "NonDynamicallyQuantizableLinear"
    assert not hasattr(model.body.attn.out_proj, "weight_quantizer")
    assert "weight_quantizer" not in "".join(model.body.attn.state_dict().keys())
    assert model.body.linear.weight_quantizer.num_bits == (4, 3)
    replaced = [
        d.module for d in plan.placement_record.decisions if d.transform == "replace_module"
    ]
    assert replaced == ["body.linear"]


def test_default_precision_records_stay_identical_to_the_pre_feature_format() -> None:
    """Single-precision trees must produce the exact detail strings existing
    checkpoints embed, or every deployed PTQ checkpoint would fail verify_matches."""
    model = _Model().eval()
    rules = QuantRules(quantize_submodules={"body": ("conv", "linear")}, recipes=())
    plan = QuantizationPlan(rules=rules, config=_config())
    plan.prepare(model)
    details = [d.detail for d in plan.placement_record.decisions if d.transform == "replace_module"]
    assert details == ["Conv2d -> QuantConv2d", "Linear -> QuantLinear"]
