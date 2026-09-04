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

"""Numerical tests for Conv+BN fusion — the highest-risk math in the package.

Asserts ``fused(x) ≈ bn(conv(x))`` for every fusion ``core/fusion.py`` implements
(Conv1d+BN1d, Conv2d+BN2d, ConvTranspose2d+BN2d, Linear+BN1d), with the
ConvTranspose2d dim-1 scaling and the bias formula covered explicitly.

``core.fusion`` is pure torch — these tests run WITHOUT nvidia-modelopt and must
never skip (the module is imported directly, bypassing any backend-gated API).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autoware_ml.quantization.core.fusion import fuse_model_bn

_ATOL = 1e-5


def _randomized_bn(bn: nn.Module, seed: int) -> nn.Module:
    """Give the BN non-trivial affine parameters and running statistics."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        bn.weight.copy_(0.5 + torch.rand(bn.num_features, generator=generator))
        bn.bias.copy_(torch.randn(bn.num_features, generator=generator))
        bn.running_mean.copy_(torch.randn(bn.num_features, generator=generator))
        bn.running_var.copy_(0.1 + torch.rand(bn.num_features, generator=generator))
    return bn


def _assert_fusion_matches(model: nn.Sequential, x: torch.Tensor) -> None:
    """Assert the fused model reproduces the original conv→bn output on ``x``."""
    model.eval()
    with torch.no_grad():
        reference = model(x)
        fuse_model_bn(model)
        assert isinstance(model[1], nn.Identity), "BN must be replaced by Identity"
        fused = model(x)
    assert torch.allclose(fused, reference, atol=_ATOL), (
        f"max abs error {(fused - reference).abs().max().item():.3e} exceeds {_ATOL}"
    )


class TestFusionNumerics:
    @pytest.mark.parametrize("bias", [True, False])
    def test_conv2d_bn2d(self, bias: bool):
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=bias),
            _randomized_bn(nn.BatchNorm2d(8), seed=1),
        )
        _assert_fusion_matches(model, torch.randn(2, 3, 16, 16))

    @pytest.mark.parametrize("bias", [True, False])
    def test_conv_transpose2d_bn2d(self, bias: bool):
        # ConvTranspose2d weights are [in, out, H, W]: the BN scale applies to dim 1.
        torch.manual_seed(2)
        model = nn.Sequential(
            nn.ConvTranspose2d(4, 6, kernel_size=2, stride=2, bias=bias),
            _randomized_bn(nn.BatchNorm2d(6), seed=3),
        )
        _assert_fusion_matches(model, torch.randn(2, 4, 8, 8))

    def test_conv1d_bn1d(self):
        torch.manual_seed(4)
        model = nn.Sequential(
            nn.Conv1d(3, 5, kernel_size=3, padding=1),
            _randomized_bn(nn.BatchNorm1d(5), seed=5),
        )
        _assert_fusion_matches(model, torch.randn(2, 3, 12))

    def test_linear_bn1d(self):
        torch.manual_seed(6)
        model = nn.Sequential(
            nn.Linear(7, 9),
            _randomized_bn(nn.BatchNorm1d(9), seed=7),
        )
        _assert_fusion_matches(model, torch.randn(4, 7))

    def test_grouped_conv2d_bn2d(self):
        torch.manual_seed(8)
        model = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=4),
            _randomized_bn(nn.BatchNorm2d(8), seed=9),
        )
        _assert_fusion_matches(model, torch.randn(2, 8, 10, 10))


def test_bn_replacement_works_in_a_container_without_item_assignment() -> None:
    """BN folding must reach models built from custom containers.

    PTv3 composes its blocks in ``PointSequential``, which registers children under
    numeric names but implements no ``__setitem__``; folding used to crash there.
    """
    import torch
    from torch import nn

    from autoware_ml.quantization.core.fusion import find_conv_bn_pairs, fuse_model_bn

    class NumericContainer(nn.Module):
        """A container with numeric child names and no ``__getitem__``/``__setitem__``."""

        def __init__(self, *children: nn.Module) -> None:
            super().__init__()
            for index, child in enumerate(children):
                self.add_module(str(index), child)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for child in self._modules.values():
                x = child(x)
            return x

    torch.manual_seed(0)
    linear = nn.Linear(4, 4)
    norm = nn.BatchNorm1d(4)
    norm.running_mean.normal_()
    norm.running_var.uniform_(0.5, 1.5)
    model = NumericContainer(linear, norm).eval()

    assert find_conv_bn_pairs(model) == [("0", "1")]
    x = torch.randn(3, 4)
    expected = model(x)

    fuse_model_bn(model)

    assert isinstance(model.get_submodule("1"), nn.Identity)
    assert torch.allclose(model(x), expected, atol=1e-5)
