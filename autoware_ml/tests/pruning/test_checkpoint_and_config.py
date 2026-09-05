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

"""Pruning config parsing and self-describing pruned checkpoints."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autoware_ml.pruning import (
    PRUNING_DESCRIPTION_ATTR,
    PRUNING_KEY,
    ChannelTable,
    PruningConfig,
    PruningDescription,
    attach_pruning_from_model,
    find_pruning,
    read_pruning,
    save_pruned_checkpoint,
)
from autoware_ml.pruning.checkpoint import read_pruning_from_file
from autoware_ml.quantization.checkpoint import QuantizationDescription, save_quantized_checkpoint
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import PlacementRecord

_SEARCH = {"enabled": True, "mode": "search", "flops": "60%", "score": "proxy"}
_FINETUNE = {
    "enabled": True,
    "mode": "finetune",
    "flops": "40%",
    "finetune": {"epochs": 2, "lr": 1e-4, "schedule": {"type": "one_cycle", "div_factor": 5}},
}


class TestPruningConfig:
    def test_absent_is_disabled(self):
        assert PruningConfig.from_dict(None).enabled is False

    def test_round_trip(self):
        for raw in (_SEARCH, _FINETUNE):
            config = PruningConfig.from_dict(raw)
            assert PruningConfig.from_dict(config.to_dict()) == config

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown pruning key"):
            PruningConfig.from_dict({**_SEARCH, "channel_divisors": 16})

    def test_mode_block_consistency(self):
        with pytest.raises(ValueError, match="requires a pruning.finetune"):
            PruningConfig.from_dict({"enabled": True, "mode": "finetune"})
        with pytest.raises(ValueError, match="config lie"):
            PruningConfig.from_dict({**_SEARCH, "finetune": {"epochs": 1, "lr": 1e-4}})

    def test_finetune_requires_epochs_and_lr(self):
        with pytest.raises(ValueError, match="epochs"):
            PruningConfig.from_dict({"enabled": True, "mode": "finetune", "finetune": {"lr": 1e-4}})

    def test_bad_values(self):
        with pytest.raises(ValueError, match="channels_ratio"):
            PruningConfig.from_dict({**_SEARCH, "channels_ratio": [0.5, 1.5]})
        with pytest.raises(ValueError, match="score"):
            PruningConfig.from_dict({**_SEARCH, "score": "loss"})


def _table() -> ChannelTable:
    return ChannelTable(
        {"a.conv": {"in_channels": 8, "out_channels": 4}, "a.norm": {"num_features": 4}}
    )


class TestCheckpoint:
    def test_payload_round_trip(self):
        description = PruningDescription(
            config=PruningConfig.from_dict(_FINETUNE), channel_table=_table()
        )
        restored = PruningDescription.from_payload(description.to_payload())
        assert restored.config == description.config
        assert restored.channel_table.entries == description.channel_table.entries

    def test_save_read_find(self, tmp_path):
        description = PruningDescription(
            config=PruningConfig.from_dict(_SEARCH), channel_table=_table()
        )
        path = save_pruned_checkpoint(nn.Linear(2, 2), tmp_path / "pruned.ckpt", description)
        loaded = torch.load(path, weights_only=False)
        assert PRUNING_KEY in loaded and "state_dict" in loaded
        assert read_pruning(loaded).channel_table.entries == _table().entries
        assert read_pruning_from_file(path) is not None

        fp = tmp_path / "fp.ckpt"
        torch.save({"state_dict": nn.Linear(2, 2).state_dict()}, fp)
        assert read_pruning_from_file(fp) is None
        found = find_pruning([fp, path])
        assert found is not None and found[0] == path
        with pytest.raises(ValueError, match="More than one"):
            find_pruning([path, path])

    def test_quantized_checkpoint_keeps_the_pruning_payload(self, tmp_path):
        """A pruned model carries its description; the PTQ writer embeds it too."""
        model = nn.Linear(2, 2)
        description = PruningDescription(
            config=PruningConfig.from_dict(_SEARCH), channel_table=_table()
        )
        setattr(model, PRUNING_DESCRIPTION_ATTR, description)
        checkpoint: dict = {}
        assert attach_pruning_from_model(model, checkpoint) is True
        assert PRUNING_KEY in checkpoint
        assert attach_pruning_from_model(nn.Linear(2, 2), {}) is False

        path = save_quantized_checkpoint(
            model,
            tmp_path / "ptq.ckpt",
            QuantizationDescription(
                config=QuantizationConfig.from_dict(
                    {"enabled": True, "mode": "ptq", "ptq": {"calibrate_samples": 4}}
                ),
                placement_record=PlacementRecord(),
            ),
        )
        loaded = torch.load(path, weights_only=False)
        assert "quantization" in loaded and PRUNING_KEY in loaded
