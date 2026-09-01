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

"""TransFusion head-dict <-> typed-outputs adapter: lossless round trip."""

from __future__ import annotations

import torch

from autoware_ml.models.detection3d.main_modules.bevfusion.model import (
    head_dict_to_outputs,
    outputs_to_head_dict,
)


def _head_dict(with_vel: bool) -> dict[str, torch.Tensor]:
    batch, classes, proposals, layers = 1, 5, 6, 2
    predictions = proposals * layers
    outputs = {
        "center": torch.randn(batch, 2, predictions),
        "height": torch.randn(batch, 1, predictions),
        "dim": torch.randn(batch, 3, predictions),
        "rot": torch.randn(batch, 2, predictions),
        "heatmap": torch.randn(batch, classes, predictions),
        "dense_heatmap": torch.randn(batch, classes, 8, 8),
        "query_heatmap_score": torch.randn(batch, classes, proposals),
        "query_labels": torch.randint(0, classes, (batch, proposals)),
    }
    if with_vel:
        outputs["vel"] = torch.randn(batch, 2, predictions)
    return outputs


def test_round_trip_is_lossless_with_velocity() -> None:
    head_dict = _head_dict(with_vel=True)
    restored = outputs_to_head_dict(head_dict_to_outputs(head_dict))
    assert set(restored) == set(head_dict)
    for key, value in head_dict.items():
        assert restored[key] is value  # same tensors, no copies


def test_round_trip_drops_absent_velocity() -> None:
    head_dict = _head_dict(with_vel=False)
    restored = outputs_to_head_dict(head_dict_to_outputs(head_dict))
    assert "vel" not in restored
    assert set(restored) == set(head_dict)
