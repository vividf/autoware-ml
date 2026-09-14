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

"""nvidia-modelopt bug workarounds, applied once when this package is imported.

modelopt (``modelopt.torch.quantization``) is the fake-quant library the framework builds
on: ``TensorQuantizer`` is the Q/DQ leaf, ``QuantModuleRegistry`` converts ``nn.Conv2d`` /
``nn.ConvTranspose2d`` / ``nn.Linear`` in place, and its ONNX symbolics emit
QuantizeLinear/DequantizeLinear (INT8) or the ``trt::TRT_FP8*`` ops (FP8). The framework
imports those names from modelopt directly; this module only carries the patches every
other core module must see before it touches a quantizer, so importing
:mod:`autoware_ml.quantization.core` applies them.

Both patches are behavior-probed so a fixed upstream is left untouched (as of 0.46.0 both
still apply):

1. **Histogram-MSE calibration**: ``modelopt...calib.histogram._compute_amax_mse`` calls
   ``fake_tensor_quant(centers, amax, num_bits, unsigned)`` with pytorch-quantization's
   positional signature, but modelopt's ``FakeTensorQuantFunction.forward`` takes ``bias``
   as the third argument. ``num_bits=8`` therefore lands in ``bias`` and ``unsigned=False``
   in ``num_bits``, and the MSE search degenerates to near-histogram-max amax
   (entropy/percentile are unaffected). Replaced with a corrected copy.

2. **Checkpoint load into an uncalibrated tree**: ``TensorQuantizer`` registers its
   ``_amax`` (and SmoothQuant's ``_pre_quant_scale``) buffers lazily, at calibration time,
   and has no ``_load_from_state_dict`` override — so loading a calibrated checkpoint into a
   freshly built tree silently drops every scale as an "unexpected key". The override below
   creates the buffers from the incoming state_dict first.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from modelopt.torch.quantization.calib import histogram as _mo_histogram
from modelopt.torch.quantization.nn import TensorQuantizer as _MoTensorQuantizer
from modelopt.torch.quantization.tensor_quant import fake_tensor_quant as _mo_ftq

logger = logging.getLogger(__name__)

#: Buffers a calibrated ``TensorQuantizer`` may carry that a fresh one does not have yet.
_LAZY_BUFFERS = ("_amax", "_pre_quant_scale")


def _fixed_compute_amax_mse(
    calib_hist, calib_bin_edges, num_bits, unsigned, stride=1, start_bin=128
):
    """modelopt's ``_compute_amax_mse`` with the ``fake_tensor_quant`` call corrected."""
    if calib_bin_edges is None and calib_hist is None:
        return None
    if not (isinstance(num_bits, int) and num_bits >= 0):
        raise TypeError("Invalid num_bits. num_bits must be a positive integer.")
    counts = torch.from_numpy(calib_hist[:]).float()
    edges = torch.from_numpy(calib_bin_edges[:]).float()
    device = None
    if torch.cuda.is_available():
        device = counts.device
        counts = counts.cuda()
        edges = edges.cuda()
    centers = (edges[1:] + edges[:-1]) / 2
    mses = []
    arguments = []
    for i in range(start_bin, len(centers), stride):
        amax = centers[i]
        # Positional: (inputs, amax, bias, num_bits, exponent_bits, unsigned)
        quant_centers = _mo_ftq(centers, amax, None, num_bits, 0, unsigned)
        mses.append(((quant_centers - centers) ** 2 * counts).mean().cpu())
        arguments.append(i)
    argmin = int(np.argmin(mses))
    calib_amax = centers[arguments[argmin]]
    if device is not None:
        calib_amax = calib_amax.to(device)
    return calib_amax


def _patch_histogram_mse() -> None:
    # Probe with the exact call shape histogram.py uses; identity-range inputs must survive.
    probe = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    try:
        out = _mo_histogram.fake_tensor_quant(probe, torch.tensor(1.0), 8, False)
        if torch.allclose(out, probe, atol=0.05):
            return  # upstream fixed — nothing to do
    except Exception as error:  # noqa: BLE001 — broken in a louder way; patch below
        logger.debug("modelopt fake_tensor_quant probe raised %r; patching", error)
    _mo_histogram._compute_amax_mse = _fixed_compute_amax_mse
    logger.warning(
        "Patched modelopt histogram MSE calibration (upstream fake_tensor_quant signature bug)"
    )


def _patch_state_dict_load() -> None:
    probe = _MoTensorQuantizer()
    probe._load_from_state_dict({"_amax": torch.tensor(1.5)}, "", {}, True, [], [], [])
    if getattr(probe, "_amax", None) is not None:
        return  # upstream creates the buffer itself now

    original = _MoTensorQuantizer._load_from_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in _LAZY_BUFFERS:
            value = state_dict.get(prefix + name)
            if value is not None and name not in self._buffers:
                self.register_buffer(name, value.data.clone())
        original(self, state_dict, prefix, *args, **kwargs)

    _MoTensorQuantizer._load_from_state_dict = _load_from_state_dict
    logger.warning(
        "Patched modelopt TensorQuantizer._load_from_state_dict to create amax/pre_quant_scale on load"
    )


_patch_histogram_mse()
_patch_state_dict_load()
