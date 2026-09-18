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

"""Same-plan-same-tree parity of CenterPoint quantization.

The invariant every quantization stage relies on: the PTQ producer and the deploy / test
loader build the quantized module tree by calling the model's one
``build_quantization_plan`` — so two independently prepared models must have identical
state_dict key sets, and a producer state_dict must load into a loader-prepared tree with
``strict=True``. Requires nvidia-modelopt (skipped otherwise).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("modelopt")

from modelopt.torch.quantization.nn import TensorQuantizer  # noqa: E402

from autoware_ml.models.detection3d.centerpoint import CENTERPOINT_QUANT_RULES  # noqa: E402
from autoware_ml.models.detection3d.tests.test_centerpoint import _build_model  # noqa: E402
from autoware_ml.quantization.config import QuantizationConfig  # noqa: E402

_CONFIG = QuantizationConfig.from_dict(
    {
        "enabled": True,
        "mode": "ptq",
        "skip_quantize": ["pts_voxel_encoder"],
        "ptq": {"calibrate_samples": 4},
    }
)


def _prepared():
    torch.manual_seed(0)
    model = _build_model().eval()
    plan = model.build_quantization_plan(_CONFIG)
    plan.prepare(model)
    return model, plan


def test_rules_are_the_model_declaration() -> None:
    model = _build_model()
    assert model.build_quantization_rules() is CENTERPOINT_QUANT_RULES
    assert set(CENTERPOINT_QUANT_RULES.quantize_submodules) == {
        "pts_backbone",
        "pts_neck",
        "bbox_head",
        "pts_voxel_encoder",
    }


def test_same_plan_builds_identical_trees_and_records() -> None:
    model_a, plan_a = _prepared()
    model_b, plan_b = _prepared()
    assert set(model_a.state_dict()) == set(model_b.state_dict())
    only_a, only_b = plan_a.placement_record.diff(plan_b.placement_record)
    assert only_a == [] and only_b == []
    assert len(plan_a.placement_record) > 0


def test_producer_state_dict_loads_strict_into_loader_tree() -> None:
    producer, _ = _prepared()
    loader, _ = _prepared()
    incompatible = loader.load_state_dict(producer.state_dict(), strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys


def test_prepare_folds_bn_and_quantizes_only_the_declared_towers() -> None:
    before = set(_build_model().state_dict())
    model, _ = _prepared()
    after = set(model.state_dict())
    assert any("running_mean" in key for key in before)
    assert not any(key.startswith("pts_backbone") and "running_mean" in key for key in after)
    assert any(key.startswith("pts_backbone") and key.endswith("conv.bias") for key in after)

    quantizers = [n for n, m in model.named_modules() if isinstance(m, TensorQuantizer)]
    assert not any(n.startswith("pts_voxel_encoder") for n in quantizers), "skip_quantize subtree"
    assert any(n.startswith("pts_backbone") for n in quantizers), (
        "the backbone must carry quantizers"
    )
    assert all(
        n.endswith(("input_quantizer", "weight_quantizer", "output_quantizer")) for n in quantizers
    )
