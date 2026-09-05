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

"""Channel table: record, apply (rebuild), round-trip, verify — on a SECOND-shaped tree."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.pruning.channels import ChannelTable, apply_channel_table


class _Dense(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pts_backbone = SECONDBackbone(
            in_channels=8, out_channels=[16, 32], layer_nums=[1, 1], layer_strides=[1, 2]
        )
        self.pts_neck = SECONDFPN(
            in_channels=[16, 32], out_channels=[16, 16], upsample_strides=[1, 2]
        )
        self.head = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        return self.head(self.pts_neck(self.pts_backbone(x)))


def _narrowed() -> _Dense:
    """The same architecture family with hand-pruned widths (what FastNAS would produce)."""
    torch.manual_seed(0)
    model = _Dense()
    # blocks.0: 8->16->16 becomes 8->16->8 ; blocks.1: 16->32->32 becomes 8->32->32 (input follows)
    b0 = model.pts_backbone.blocks[0]
    b0[1].conv = nn.Conv2d(16, 8, 3, padding=1, bias=False)
    b0[1].norm = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
    b1 = model.pts_backbone.blocks[1]
    b1[0].conv = nn.Conv2d(8, 32, 3, stride=2, padding=1, bias=False)
    model.pts_neck.blocks[0].conv = nn.ConvTranspose2d(8, 16, 1, stride=1, bias=False)
    return model


class TestRecord:
    def test_records_every_channel_shaped_layer_under_the_roots(self):
        table = ChannelTable.record(_Dense(), ("pts_backbone", "pts_neck"))
        names = list(table.entries)
        assert "pts_backbone.blocks.0.0.conv" in names
        assert "pts_backbone.blocks.0.0.norm" in names
        assert "pts_neck.blocks.1.conv" in names
        assert not any(n.startswith("head") for n in names)
        assert table.entries["pts_backbone.blocks.1.0.conv"] == {
            "in_channels": 16,
            "out_channels": 32,
        }
        assert table.entries["pts_backbone.blocks.0.0.norm"] == {"num_features": 16}

    def test_whole_model_when_no_roots(self):
        table = ChannelTable.record(_Dense())
        assert "head" in table.entries


class TestApply:
    def test_rebuilds_only_changed_layers_and_state_dict_loads(self):
        pruned = _narrowed()
        table = ChannelTable.record(pruned, ("pts_backbone", "pts_neck"))
        fresh = _Dense()
        untouched_before = fresh.pts_backbone.blocks[1][1].conv.weight.clone()

        rebuilt = apply_channel_table(fresh, table)

        assert rebuilt == 4  # blocks.0.1 conv+norm, blocks.1.0 conv, neck.blocks.0 deconv
        assert torch.equal(fresh.pts_backbone.blocks[1][1].conv.weight, untouched_before)
        table.verify_matches(
            ChannelTable.record(fresh, ("pts_backbone", "pts_neck")), source="test"
        )
        fresh.load_state_dict(pruned.state_dict(), strict=True)
        x = torch.randn(1, 8, 16, 16)
        fresh.eval(), pruned.eval()
        assert torch.allclose(fresh(x), pruned(x))

    def test_preserves_hyperparameters_and_device_dtype(self):
        pruned = _narrowed()
        table = ChannelTable.record(pruned, ("pts_backbone",))
        fresh = _Dense()
        apply_channel_table(fresh, table)
        conv = fresh.pts_backbone.blocks[1][0].conv
        assert (conv.stride, conv.padding, conv.bias) == ((2, 2), (1, 1), None)
        norm = fresh.pts_backbone.blocks[0][1].norm
        assert (norm.eps, norm.momentum) == (1e-3, 0.01)

    def test_idempotent(self):
        pruned = _narrowed()
        table = ChannelTable.record(pruned, ("pts_backbone", "pts_neck"))
        assert apply_channel_table(pruned, table) == 0

    def test_unknown_module_is_an_error(self):
        table = ChannelTable(
            {"pts_backbone.blocks.9.0.conv": {"in_channels": 1, "out_channels": 1}}
        )
        with pytest.raises(AttributeError):
            apply_channel_table(_Dense(), table)


class TestRoundTripAndVerify:
    def test_json_round_trip(self):
        table = ChannelTable.record(_narrowed(), ("pts_backbone",))
        restored = ChannelTable.from_json_dict(table.to_json_dict())
        assert restored.entries == table.entries
        assert restored.diff(table) == []

    def test_verify_matches_raises_with_the_differing_layers(self):
        base = ChannelTable.record(_Dense(), ("pts_backbone",))
        pruned = ChannelTable.record(_narrowed(), ("pts_backbone",))
        with pytest.raises(RuntimeError, match="pts_backbone.blocks.0.1.conv"):
            pruned.verify_matches(base, source="test")
        assert "pts_backbone.blocks.0.1.conv" in pruned.changed_from(base)

    def test_restricted_to(self):
        full = ChannelTable.record(_Dense())
        part = ChannelTable.record(_Dense(), ("pts_neck",))
        assert list(full.restricted_to(part).entries) == list(part.entries)
