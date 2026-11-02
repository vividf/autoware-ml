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

"""Unit tests for converting modules in place to modelopt's quantized classes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from autoware_ml.quantization.config import Precision

MODELOPT_AVAILABLE = importlib.util.find_spec("modelopt") is not None
ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None

if MODELOPT_AVAILABLE:
    from modelopt.torch.quantization.nn import TensorQuantizer
    from modelopt.torch.quantization.tensor_quant import fake_tensor_quant

    from autoware_ml.quantization.core.replace import replace_quantizable_modules

if ONNX_AVAILABLE:
    import onnx


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.up = nn.ConvTranspose2d(4, 2, 2, stride=2)
        self.linear = nn.Linear(2, 5)

    def forward(self, x):
        y = self.up(self.conv(x))
        return self.linear(y.mean((2, 3)))


def _calibrate_max(model: nn.Module, x: torch.Tensor) -> None:
    quantizers = [m for m in model.modules() if isinstance(m, TensorQuantizer) and not m._disabled]
    for q in quantizers:
        q.disable_quant()
        q.enable_calib()
    with torch.no_grad():
        model(x)
    for q in quantizers:
        if q._calibrator.__class__.__name__ == "HistogramCalibrator":
            q.load_calib_amax("percentile", percentile=100.0)
        else:
            q.load_calib_amax()
        q.enable_quant()
        q.disable_calib()


def _fq(t: torch.Tensor, q: TensorQuantizer) -> torch.Tensor:
    return fake_tensor_quant(t, q._amax, None, 8, 0, False)


@unittest.skipUnless(MODELOPT_AVAILABLE, "could not import 'modelopt'")
class TestReplaceOnRegistry(unittest.TestCase):
    """Modules convert in place to modelopt's quantized classes and quantize as by hand."""

    def setUp(self) -> None:
        """Give each test its own directory for the ONNX file."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_conversion_is_in_place_and_keeps_type_identity(self):
        torch.manual_seed(0)
        net = _Net().eval()
        conv, up, linear = net.conv, net.up, net.linear
        replace_quantizable_modules(net, kinds=("conv", "linear"), precision=Precision.INT8)
        self.assertIs(net.conv, conv)
        self.assertIs(net.up, up)
        self.assertIs(net.linear, linear)
        self.assertIsInstance(net.conv, nn.Conv2d)
        self.assertIsInstance(net.linear, nn.Linear)
        self.assertEqual(type(net.conv).__name__, "QuantConv2d")
        self.assertEqual(type(net.up).__name__, "QuantConvTranspose2d")
        self.assertEqual(type(net.linear).__name__, "QuantLinear")

    def test_descriptors_follow_the_framework_tables(self):
        net = _Net().eval()
        replace_quantizable_modules(net, kinds=("conv", "linear"), precision=Precision.INT8)
        self.assertEqual(
            net.conv.input_quantizer._calibrator.__class__.__name__, "HistogramCalibrator"
        )
        self.assertIsNone(net.conv.input_quantizer.axis)
        self.assertEqual(net.conv.weight_quantizer.axis, 0)
        self.assertIsNone(net.up.weight_quantizer.axis)
        self.assertEqual(net.linear.weight_quantizer.axis, 0)
        net2 = _Net().eval()
        replace_quantizable_modules(
            net2, kinds=("conv",), precision=Precision.INT8, calibrator="max"
        )
        self.assertEqual(net2.conv.input_quantizer._calibrator.__class__.__name__, "MaxCalibrator")
        # FP8 (E4M3) always calibrates with max, whatever the request.
        net3 = _Net().eval()
        replace_quantizable_modules(
            net3, kinds=("linear",), precision=Precision.FP8, calibrator="histogram"
        )
        self.assertEqual(net3.linear.weight_quantizer.num_bits, (4, 3))
        self.assertEqual(
            net3.linear.input_quantizer._calibrator.__class__.__name__, "MaxCalibrator"
        )

    def test_state_dict_uses_modelopt_names_and_only_after_calibration(self):
        torch.manual_seed(0)
        net = _Net().eval()
        replace_quantizable_modules(net, kinds=("conv", "linear"), precision=Precision.INT8)
        self.assertFalse([k for k in net.state_dict() if "quantizer" in k])
        _calibrate_max(net, torch.randn(2, 3, 8, 8))
        keys = sorted(k for k in net.state_dict() if "quantizer" in k)
        self.assertEqual(
            keys,
            [
                "conv.input_quantizer._amax",
                "conv.weight_quantizer._amax",
                "linear.input_quantizer._amax",
                "linear.weight_quantizer._amax",
                "up.input_quantizer._amax",
                "up.weight_quantizer._amax",
            ],
        )

    def test_forward_equals_hand_fake_quant_and_weights_stay_clean(self):
        torch.manual_seed(0)
        net = _Net().eval()
        ref = _Net().eval()
        ref.load_state_dict(net.state_dict())
        replace_quantizable_modules(net, kinds=("conv", "linear"), precision=Precision.INT8)
        x = torch.randn(2, 3, 8, 8)
        _calibrate_max(net, x)
        with torch.no_grad():
            got = net(x)
            h = F.conv2d(
                _fq(x, net.conv.input_quantizer),
                _fq(ref.conv.weight, net.conv.weight_quantizer),
                ref.conv.bias,
                padding=1,
            )
            h = F.conv_transpose2d(
                _fq(h, net.up.input_quantizer),
                _fq(ref.up.weight, net.up.weight_quantizer),
                ref.up.bias,
                stride=2,
            )
            h = _fq(h.mean((2, 3)), net.linear.input_quantizer)
            want = F.linear(h, _fq(ref.linear.weight, net.linear.weight_quantizer), ref.linear.bias)
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(torch.equal(net.conv.weight, ref.conv.weight))

    def test_skip_names_and_kinds_are_honored(self):
        net = _Net().eval()
        replace_quantizable_modules(
            net, kinds=("conv",), skip_names={"up"}, precision=Precision.INT8
        )
        self.assertTrue(hasattr(net.conv, "weight_quantizer"))
        self.assertFalse(hasattr(net.up, "weight_quantizer"))
        self.assertFalse(hasattr(net.linear, "weight_quantizer"))

    def test_second_pass_is_a_noop(self):
        net = _Net().eval()
        seen = []
        replace_quantizable_modules(
            net,
            kinds=("conv", "linear"),
            precision=Precision.INT8,
            on_replace=lambda n, o, m: seen.append((n, o)),
        )
        self.assertEqual(
            seen, [("conv", "Conv2d"), ("up", "ConvTranspose2d"), ("linear", "Linear")]
        )
        again = []
        replace_quantizable_modules(
            net,
            kinds=("conv", "linear"),
            precision=Precision.INT8,
            on_replace=lambda n, o, m: again.append(n),
        )
        self.assertEqual(again, [])

    @unittest.skipUnless(ONNX_AVAILABLE, "could not import 'onnx'")
    def test_onnx_export_emits_qdq(self):
        torch.manual_seed(0)
        net = _Net().eval()
        replace_quantizable_modules(net, kinds=("conv", "linear"), precision=Precision.INT8)
        x = torch.randn(1, 3, 8, 8)
        _calibrate_max(net, x)
        path = self.tmp_path / "net.onnx"
        torch.onnx.export(net, x, str(path), opset_version=17, dynamo=False)
        ops = [node.op_type for node in onnx.load(str(path)).graph.node]
        self.assertEqual(ops.count("QuantizeLinear"), 6)
        self.assertEqual(ops.count("DequantizeLinear"), 6)

    def test_calibrated_state_dict_loads_into_a_fresh_tree(self):
        torch.manual_seed(0)
        net = _Net().eval()
        replace_quantizable_modules(net, kinds=("conv", "linear"), precision=Precision.INT8)
        _calibrate_max(net, torch.randn(2, 3, 8, 8))
        fresh = _Net().eval()
        replace_quantizable_modules(fresh, kinds=("conv", "linear"), precision=Precision.INT8)
        result = fresh.load_state_dict(net.state_dict(), strict=False)
        self.assertEqual(result.unexpected_keys, [])
        self.assertEqual(result.missing_keys, [])
        self.assertTrue(
            torch.equal(fresh.conv.input_quantizer._amax, net.conv.input_quantizer._amax)
        )
