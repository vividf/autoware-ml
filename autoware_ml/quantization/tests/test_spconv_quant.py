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

"""Unit tests for quantizing sparse convolutions on modelopt's registry."""

from __future__ import annotations

import copy
import unittest

import torch

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.quantization.config import Precision

CUDA = torch.cuda.is_available()

if IS_SPCONV_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch.core import SparseConvTensor

    from autoware_ml.ops.submanifold_conv_spconv import SpconvSubMConv3d
    from autoware_ml.quantization.core.replace import replace_quantizable_modules


def _sparse_tensor(channels: int, device: torch.device) -> SparseConvTensor:
    """Return a two-sample sparse tensor of random features on a 32^3 grid."""
    generator = torch.Generator().manual_seed(0)
    coords = torch.unique(torch.randint(0, 32, (2000, 3), generator=generator), dim=0)
    batch = (torch.arange(coords.shape[0]) >= coords.shape[0] // 2).int().unsqueeze(1)
    indices = torch.cat([batch, coords.int()], dim=1).to(device)
    features = torch.randn(coords.shape[0], channels, generator=generator).to(device)
    return SparseConvTensor(features, indices, [32, 32, 32], 2)


def _quantized(layer: torch.nn.Module) -> torch.nn.Module:
    """A converted copy of ``layer`` with its quantizers off, so it must equal ``layer``."""
    model = torch.nn.ModuleDict({"layer": copy.deepcopy(layer)})
    replace_quantizable_modules(model, kinds=("spconv",), precision=Precision.INT8)
    model["layer"].input_quantizer.disable()
    model["layer"].weight_quantizer.disable()
    return model["layer"]


@unittest.skipUnless(IS_SPCONV_AVAILABLE, "needs spconv")
class TestSparseConvolutionConversion(unittest.TestCase):
    """Both submanifold convolutions a model can hold become quantized modules."""

    def test_spconv_and_the_autocast_safe_flavour_convert(self) -> None:
        """
        Input: spconv's SubMConv3d and SpconvSubMConv3d, which overrides forward.
        Expected: the spconv kind converts both; modelopt only matches a subclass that inherits
        the registered forward, so SpconvSubMConv3d needs its own registration.
        Check: both carry quantizers and keep their class.
        """
        model = torch.nn.ModuleDict(
            {
                "plain": spconv.SubMConv3d(8, 8, 3, indice_key="a"),
                "autocast_safe": SpconvSubMConv3d(8, 8, 3, indice_key="b"),
            }
        )

        replace_quantizable_modules(model, kinds=("spconv",), precision=Precision.INT8)

        for name, cls in (("plain", spconv.SubMConv3d), ("autocast_safe", SpconvSubMConv3d)):
            self.assertIsInstance(model[name], cls, name)
            self.assertTrue(hasattr(model[name], "input_quantizer"), name)
            self.assertTrue(hasattr(model[name], "weight_quantizer"), name)


@unittest.skipUnless(IS_SPCONV_AVAILABLE and CUDA, "spconv's kernels need a CUDA device")
class TestQuantizedForward(unittest.TestCase):
    """The quantized module runs the forward of the class it was converted from."""

    def setUp(self) -> None:
        """Build an SpconvSubMConv3d and a random sparse tensor on the GPU."""
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.layer = SpconvSubMConv3d(16, 16, 3, indice_key="k").to(self.device).eval()
        self.x = _sparse_tensor(16, self.device)

    def test_halved_features_are_cast_to_the_weight_dtype(self) -> None:
        """
        Input: fp16 features against fp32 weights.
        Expected: SpconvSubMConv3d's forward still casts the features; running
        SparseConvolution.forward directly would hand spconv mismatched dtypes.
        Check: the call succeeds and returns fp32 features.
        """
        halved = self.x.replace_feature(self.x.features.half())

        with torch.no_grad():
            y = _quantized(self.layer)(halved)

        self.assertEqual(y.features.dtype, torch.float32)

    def test_quantizers_off_reproduce_the_layer_under_autocast(self) -> None:
        """
        Input: the same sparse tensor under fp16 autocast, through the layer and through its
        converted copy with the quantizers disabled.
        Expected: the converted copy runs outside autocast in fp32 like the layer.
        Check: the outputs match.
        """
        quantized = _quantized(self.layer)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            expected = self.layer(self.x).features
            actual = quantized(self.x).features

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
