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

"""Structured (channel) pruning: FastNAS search, self-describing checkpoints, KD fine-tune.

See ``README.md`` in this package. The design mirrors :mod:`autoware_ml.quantization`:
the model declares what is prunable (:class:`PruningSpec`), the ``pruning`` config only
sets the budget and the score, the produced checkpoint carries the architecture it needs
(:class:`ChannelTable`), and ``build_model`` rebuilds that architecture before loading.
"""

from .channels import ChannelTable, apply_channel_table
from .checkpoint import (
    PRUNING_DESCRIPTION_ATTR,
    PRUNING_KEY,
    PruningDescription,
    attach_pruning,
    attach_pruning_from_model,
    find_pruning,
    read_pruning,
    save_pruned_checkpoint,
)
from .config import FinetuneConfig, PruningConfig
from .spec import PruningSpec

__all__ = [
    "PRUNING_DESCRIPTION_ATTR",
    "PRUNING_KEY",
    "ChannelTable",
    "FinetuneConfig",
    "PruningConfig",
    "PruningDescription",
    "PruningSpec",
    "apply_channel_table",
    "attach_pruning",
    "attach_pruning_from_model",
    "find_pruning",
    "read_pruning",
    "save_pruned_checkpoint",
]
