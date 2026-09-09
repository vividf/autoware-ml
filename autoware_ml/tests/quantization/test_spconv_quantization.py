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

"""The ``spconv`` module kind: what the plan builds on a sparse encoder, and what export drops."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.ops.spconv.onnx_int8 import collect_sparse_layer_scales
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan, QuantRules

pytestmark = pytest.mark.skipif(not IS_SPCONV_AVAILABLE, reason="spconv is not installed")

if IS_SPCONV_AVAILABLE:
    from spconv.pytorch.conv import SparseConvolution
    from torch import nn

    from autoware_ml.models.detection3d.encoders.sparse import SparseEncoder


def _encoder() -> "SparseEncoder":
    """A two-stage encoder — small, but the same block shapes as the deployed one."""
    return SparseEncoder(
        in_channels=4,
        sparse_shape=(16, 16, 8),
        base_channels=4,
        encoder_channels=((4, 4), (8, 8)),
        encoder_paddings=((1, 1), (1, 1)),
        output_channels=8,
        dense_output_shapes=(16, 16, 2),
    ).eval()


def _model_with(encoder: "SparseEncoder") -> "nn.Module":
    model = nn.Module()
    model.pts_middle_encoder = encoder
    return model.eval()


def _plan(**config: object) -> QuantizationPlan:
    return QuantizationPlan(
        rules=QuantRules(quantize_submodules={"pts_middle_encoder": ("spconv",)}),
        config=QuantizationConfig(enabled=True, fuse_bn=True, **config),
    )


def test_plan_quantizes_every_sparse_convolution_and_folds_their_norms():
    model = _model_with(_encoder())

    _plan().prepare(model)

    convolutions = [m for m in model.modules() if isinstance(m, SparseConvolution)]
    assert convolutions, "the fixture encoder has sparse convolutions"
    assert all(
        hasattr(m, "input_quantizer") and hasattr(m, "weight_quantizer") for m in convolutions
    )
    # BN folding is what makes the calibrated weight the one the exported graph carries.
    assert not [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert all(m.bias is not None for m in convolutions), "the fold gives each conv a bias"


def test_skip_quantize_leaves_a_sparse_layer_untouched():
    model = _model_with(_encoder())

    _plan(skip_quantize=["pts_middle_encoder.conv_input"]).prepare(model)

    assert not hasattr(model.pts_middle_encoder.conv_input[0], "input_quantizer")
    assert hasattr(model.pts_middle_encoder.conv_out[0], "input_quantizer")


def test_weight_amax_is_one_scale_per_output_channel():
    model = _model_with(_encoder())
    _plan().prepare(model)
    convolution = model.pts_middle_encoder.conv_out[0]

    # A forward under stats collection is the calibration the framework runs; here the
    # weight quantizer's own maximum is enough to pin the axis.
    convolution.weight_quantizer._calibrator.collect(convolution.weight)
    convolution.weight_quantizer.load_calib_amax()

    assert convolution.weight_quantizer.amax.numel() == convolution.out_channels


def test_export_copy_carries_no_quantizers():
    """The exported sparse graph must be pure float — its scales travel as plugin inputs."""
    model = _model_with(_encoder())
    _plan().prepare(model)

    exported = model.pts_middle_encoder.prepare_for_export()

    assert not [
        name
        for name, _ in exported.named_modules()
        if name.endswith(("input_quantizer", "weight_quantizer"))
    ]


def test_collect_scales_reports_only_calibrated_layers():
    model = _model_with(_encoder())
    _plan(skip_quantize=["pts_middle_encoder.conv_input"]).prepare(model)
    for module in model.modules():
        if isinstance(module, SparseConvolution) and hasattr(module, "input_quantizer"):
            module.input_quantizer.amax = torch.tensor(2.54)
            module.weight_quantizer.amax = torch.full(
                (module.out_channels, 1, 1, 1, 1), 1.27, dtype=torch.float32
            )

    scales = collect_sparse_layer_scales(model.pts_middle_encoder)

    assert scales, "the quantized layers report scales"
    assert not [stem for stem in scales if stem.startswith("conv_input")]
    layer = scales["conv_out.0"]
    assert layer.input_scale == pytest.approx(0.02)
    assert layer.channel_scale.tolist() == pytest.approx([0.02 * 0.01] * 8)


def test_an_uncalibrated_quantizer_is_an_error_not_a_silent_fp16_tower():
    model = _model_with(_encoder())
    _plan().prepare(model)

    with pytest.raises(ValueError, match="un-calibrated"):
        collect_sparse_layer_scales(model.pts_middle_encoder)
