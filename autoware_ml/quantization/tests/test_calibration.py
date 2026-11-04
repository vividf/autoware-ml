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

"""Unit tests for quantizer calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from autoware_ml.quantization.config import CalibrationConfig, Precision

MODELOPT_AVAILABLE = importlib.util.find_spec("modelopt") is not None
ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None

if MODELOPT_AVAILABLE:
    from modelopt.torch.quantization.nn import TensorQuantizer

    from autoware_ml.quantization.core.calibration import Calibrator
    from autoware_ml.quantization.core.quantizer_state import validate_quantizer_amax
    from autoware_ml.quantization.core.replace import replace_quantizable_modules

if ONNX_AVAILABLE:
    import onnx


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.linear = nn.Linear(4, 6)

    def forward(self, x):
        return self.linear(self.conv(x).mean((2, 3)))


def _forward(model, batch):
    model(batch)


def _prepared(calibration: CalibrationConfig) -> _Net:
    torch.manual_seed(0)
    net = _Net().eval()
    replace_quantizable_modules(
        net,
        kinds=("conv", "linear"),
        precision=Precision.INT8,
        calibrator=calibration.activation_calibrator,
    )
    return net


def _batches(n: int = 4):
    torch.manual_seed(1)
    # An outlier channel makes SmoothQuant's per-channel migration observable.
    return [
        torch.randn(2, 3, 8, 8) * torch.tensor([1.0, 8.0, 1.0]).view(1, 3, 1, 1) for _ in range(n)
    ]


@unittest.skipUnless(MODELOPT_AVAILABLE, "could not import 'modelopt'")
class TestCalibrationMethods(unittest.TestCase):
    """Every calibration method leaves validated amax, and percentile clips below max."""

    def test_every_method_yields_validated_amax(self):
        for method in ("mse", "entropy", "percentile", "max"):
            with self.subTest(method=method):
                calibration = CalibrationConfig.model_validate(method)
                net = _prepared(calibration)
                Calibrator(net).calibrate(
                    _batches(), num_batches=4, calibration=calibration, forward_fn=_forward
                )
                validate_quantizer_amax(net)
                for q in net.modules():
                    if isinstance(q, TensorQuantizer) and not q._disabled:
                        self.assertIsNotNone(q._amax)
                        self.assertTrue(torch.isfinite(q._amax).all())
                        self.assertTrue(q._if_quant)
                        self.assertFalse(q._if_calib)
                # Weight amax is exact max either way; the activation amax depends on the method.
                self.assertTrue(
                    torch.allclose(
                        net.linear.weight_quantizer._amax.squeeze(),
                        net.linear.weight.abs().amax(dim=1),
                    )
                )

    def test_percentile_clips_below_max(self):
        hi = _prepared(CalibrationConfig.model_validate("max"))
        Calibrator(hi).calibrate(
            _batches(), 4, CalibrationConfig.model_validate("max"), forward_fn=_forward
        )
        lo_cfg = CalibrationConfig.model_validate({"method": "percentile", "percentile": 90.0})
        lo = _prepared(lo_cfg)
        Calibrator(lo).calibrate(_batches(), 4, lo_cfg, forward_fn=_forward)
        self.assertLess(lo.conv.input_quantizer._amax, hi.conv.input_quantizer._amax)


@unittest.skipUnless(MODELOPT_AVAILABLE, "could not import 'modelopt'")
class TestSmoothQuant(unittest.TestCase):
    """SmoothQuant moves activation outliers into the weights and survives save, load and export."""

    def setUp(self) -> None:
        """Give each test its own directory for the ONNX file."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_smoothquant_migrates_outliers_and_round_trips(self):
        calibration = CalibrationConfig.model_validate(
            {"method": "smoothquant", "smoothquant_alpha": 0.5}
        )
        net = _prepared(calibration)
        weight_before = net.linear.weight.detach().clone()
        Calibrator(net).calibrate(_batches(), 4, calibration, forward_fn=_forward)
        validate_quantizer_amax(net)
        # SmoothQuant only applies to Linear layers; the conv is plain max-calibrated.
        scale = net.linear.input_quantizer.pre_quant_scale
        self.assertIsNotNone(scale)
        self.assertEqual(scale.shape, (4,))
        self.assertFalse(torch.equal(net.linear.weight, weight_before))
        self.assertIsNone(net.linear.input_quantizer.axis)  # back to per-tensor after smoothing
        self.assertIsNone(net.conv.input_quantizer.pre_quant_scale)
        keys = [k for k in net.state_dict() if "pre_quant_scale" in k]
        self.assertEqual(keys, ["linear.input_quantizer._pre_quant_scale"])

        # The patched TensorQuantizer._load_from_state_dict creates pre_quant_scale on load too.
        fresh = _prepared(calibration)
        result = fresh.load_state_dict(net.state_dict(), strict=False)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        x = _batches(1)[0]
        with torch.no_grad():
            self.assertTrue(torch.equal(fresh(x), net(x)))

        # The scale exports as a Mul in front of the Q/DQ pair.
        if not ONNX_AVAILABLE:
            self.skipTest("could not import 'onnx'")
        path = self.tmp_path / "sq.onnx"
        torch.onnx.export(net, x, str(path), opset_version=17, dynamo=False)
        ops = [n.op_type for n in onnx.load(str(path)).graph.node]
        self.assertEqual(ops.count("QuantizeLinear"), 4)
        self.assertIn("Mul", ops)
