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

"""Dict batch for the toy stage-graph models of the deployment tests."""

from __future__ import annotations

import torch


class ToyBatch(dict):
    """``{name: tensor}`` that moves like ``ModelGTBatch``."""

    def to_device(self, device: torch.device) -> ToyBatch:
        return ToyBatch({name: tensor.to(device) for name, tensor in self.items()})
