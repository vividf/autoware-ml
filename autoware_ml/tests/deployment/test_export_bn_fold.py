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

"""Export-time BN folding: identity outputs, no BN node, no-op without BN."""

from __future__ import annotations

import torch
from torch import nn

from autoware_ml.deployment.export import _bn_folded_for_export
from autoware_ml.deployment.stages import GraphStage


def _stage(module: nn.Module) -> GraphStage:
    return GraphStage("stage", module=module, inputs=("x",), outputs=("y",))


def test_fold_removes_bn_and_preserves_outputs() -> None:
    torch.manual_seed(0)
    module = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU()).eval()
    # Non-trivial BN statistics so the fold actually rewrites the conv weights.
    module[1].running_mean.uniform_(-1, 1)
    module[1].running_var.uniform_(0.5, 2.0)

    folded = _bn_folded_for_export(_stage(module))

    assert folded is not module
    assert not any(isinstance(m, nn.BatchNorm2d) for m in folded.modules())
    # The shared model keeps its BN — the fold runs on a copy.
    assert isinstance(module[1], nn.BatchNorm2d)
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(folded(x), module(x), rtol=1e-4, atol=1e-5)


def test_nothing_to_fold_returns_the_original_module() -> None:
    module = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU()).eval()

    assert _bn_folded_for_export(_stage(module)) is module
