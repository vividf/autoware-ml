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

"""What a model declares to be prunable (the pruning counterpart of ``QuantRules``)."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class PruningSpec:
    """A model's pruning declaration — an architecture fact, written in code.

    Attributes:
        subtree: A ``torch.fx``-traceable module whose ``forward`` takes the ONE stage-input
            tensor and returns a tuple of tensors, and which holds the model's OWN
            submodules (no copies): FastNAS patches modules in place, and the full model's
            pipeline must see the pruned widths. Container forwards that ``torch.fx``
            cannot trace (a neck iterating a list input, a head building a dataclass) are
            inlined here — a module traced as a leaf is un-prunable.
        stage_input: Name of the stage-graph context tensor the subtree consumes
            (e.g. ``"spatial_features"``); cached from the pytorch pipeline for the
            search's BN re-calibration and proxy score.
        submodules: Top-level attribute names whose layers the channel table records.
    """

    subtree: nn.Module
    stage_input: str
    submodules: tuple[str, ...]
