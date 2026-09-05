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

"""Calibrator on a tiny model: every ``calibration.method`` yields usable amax, SmoothQuant
produces a ``pre_quant_scale`` that survives save/load and ONNX export."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("modelopt")

from modelopt.torch.quantization.nn import TensorQuantizer  # noqa: E402
from torch import nn  # noqa: E402

from autoware_ml.quantization.config import CalibrationConfig, Precision  # noqa: E402
from autoware_ml.quantization.core.calibration import Calibrator  # noqa: E402
from autoware_ml.quantization.core.quantizer_state import validate_quantizer_amax  # noqa: E402
from autoware_ml.quantization.core.replace import replace_quantizable_modules  # noqa: E402


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


@pytest.mark.parametrize("method", ["mse", "entropy", "percentile", "max"])
def test_every_method_yields_validated_amax(method):
    calibration = CalibrationConfig.from_raw(method)
    net = _prepared(calibration)
    Calibrator(net).calibrate(
        _batches(), num_batches=4, calibration=calibration, forward_fn=_forward
    )
    validate_quantizer_amax(net)
    for q in net.modules():
        if isinstance(q, TensorQuantizer) and not q._disabled:
            assert q._amax is not None and torch.isfinite(q._amax).all()
            assert q._if_quant and not q._if_calib
    # Weight amax is exact max either way; the activation amax depends on the method.
    assert torch.allclose(
        net.linear.weight_quantizer._amax.squeeze(), net.linear.weight.abs().amax(dim=1)
    )


def test_percentile_clips_below_max():
    hi = _prepared(CalibrationConfig.from_raw("max"))
    Calibrator(hi).calibrate(_batches(), 4, CalibrationConfig.from_raw("max"), forward_fn=_forward)
    lo_cfg = CalibrationConfig.from_raw({"method": "percentile", "percentile": 90.0})
    lo = _prepared(lo_cfg)
    Calibrator(lo).calibrate(_batches(), 4, lo_cfg, forward_fn=_forward)
    assert lo.conv.input_quantizer._amax < hi.conv.input_quantizer._amax


def test_smoothquant_migrates_outliers_and_round_trips(tmp_path):
    calibration = CalibrationConfig.from_raw({"method": "smoothquant", "smoothquant_alpha": 0.5})
    net = _prepared(calibration)
    weight_before = net.linear.weight.detach().clone()
    Calibrator(net).calibrate(_batches(), 4, calibration, forward_fn=_forward)
    validate_quantizer_amax(net)
    # The INT8 Linear got a per-input-channel pre_quant_scale and a rescaled weight; the
    # conv (not a Linear) is plain max-calibrated.
    scale = net.linear.input_quantizer.pre_quant_scale
    assert scale is not None and scale.shape == (4,)
    assert not torch.equal(net.linear.weight, weight_before)
    assert net.linear.input_quantizer.axis is None  # back to per-tensor after smoothing
    assert net.conv.input_quantizer.pre_quant_scale is None
    keys = [k for k in net.state_dict() if "pre_quant_scale" in k]
    assert keys == ["linear.input_quantizer._pre_quant_scale"]

    # Loads into a fresh tree (the lazy-buffer patch covers pre_quant_scale too) ...
    fresh = _prepared(calibration)
    result = fresh.load_state_dict(net.state_dict(), strict=False)
    assert result.missing_keys == [] and result.unexpected_keys == []
    x = _batches(1)[0]
    with torch.no_grad():
        assert torch.equal(fresh(x), net(x))

    # ... and exports: the scale becomes a Mul feeding the Q/DQ pair.
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "sq.onnx"
    torch.onnx.export(net, x, str(path), opset_version=17, dynamo=False)
    ops = [n.op_type for n in onnx.load(str(path)).graph.node]
    assert ops.count("QuantizeLinear") == 4 and "Mul" in ops
