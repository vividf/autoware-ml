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

"""Quantization backend: nvidia-modelopt (``modelopt.torch.quantization``).

Single seam through which the framework reaches the fake-quant library. The framework was
originally built on ``pytorch-quantization``; modelopt descends from the same NVIDIA codebase
and kept the calibration surface the framework relies on (``TensorQuantizer`` with
``disable_quant``/``enable_calib``/``load_calib_amax``, ``_amax`` buffers with identical
state_dict keys, ``MaxCalibrator``/``HistogramCalibrator``), so the rest of
``autoware_ml.quantization`` imports everything quantizer-related from here instead of from the
library directly. pytorch-quantization support has been dropped (it is deprecated upstream);
PTQ checkpoints produced with it remain loadable — the ``_amax`` state_dict layout is identical.

Two places the framework must know modelopt differs from the pytorch-quantization heritage,
both wrapped here:

1. **Descriptors**: modelopt's ``config.QuantizerAttributeConfig`` spells the calibrator field
   ``calibrator`` (pytorch-quantization said ``calib_method``) and takes ``axis`` as an int.
   :func:`make_quant_desc` keeps accepting the framework's historical vocabulary and translates.
2. **ONNX export**: modelopt's ``TensorQuantizer`` traces to QuantizeLinear/DequantizeLinear
   natively — no global ``use_fb_fake_quant`` switch, and the quantizers must NOT be bypassed
   during tracing (:func:`exports_qdq_natively` → :func:`setup_onnx_export` is a no-op).

It also carries the modelopt bug workarounds in :func:`_ensure_modelopt_patches`.

Leaf module: imports only stdlib + modelopt (lazily); safe for every core submodule.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ENV_VAR = "AUTOWARE_ML_QUANT_BACKEND"
MODELOPT = "modelopt"

_resolved: Optional[str] = None
_resolve_done = False


def resolve() -> Optional[str]:
    """Return ``"modelopt"`` when nvidia-modelopt is installed, else ``None``.

    Resolution is cached for the life of the process (the quantized modules bind their
    ``TensorQuantizer`` class at attach time). ``AUTOWARE_ML_QUANT_BACKEND`` survives only as a
    guard: selecting the removed ``pytorch-quantization`` backend raises instead of silently
    running on modelopt.
    """
    global _resolved, _resolve_done
    if _resolve_done:
        return _resolved

    requested = os.environ.get(_ENV_VAR, "auto").strip().lower() or "auto"
    if requested not in ("auto", MODELOPT):
        raise ValueError(
            f"{_ENV_VAR}={requested!r} — the pytorch-quantization backend has been removed; "
            f"only nvidia-modelopt is supported (unset {_ENV_VAR} or set it to 'modelopt')."
        )

    _resolved = MODELOPT if importlib.util.find_spec("modelopt") is not None else None
    _resolve_done = True
    if _resolved is not None:
        logger.info("Quantization backend: %s", _resolved)
    return _resolved


def available() -> bool:
    """Whether the quantization backend (nvidia-modelopt) is installed."""
    return resolve() is not None


def install_hint(purpose: str = "quantization support") -> str:
    """Standard install message for the quantization backend."""
    return (
        f"nvidia-modelopt is required for {purpose}. Install it with: pip install nvidia-modelopt"
    )


def require(purpose: str = "quantization support") -> str:
    """Return the active backend name, raising the install hint when it is not available."""
    name = resolve()
    if name is None:
        raise ImportError(install_hint(purpose))
    return name


# ---------------------------------------------------------------------------
# modelopt bug workarounds
# ---------------------------------------------------------------------------

_modelopt_patched = False


def _ensure_modelopt_patches() -> None:
    """Apply behavior-probed fixes for modelopt bugs (as of 0.46.0).

    Two seams, both detected behaviorally so a fixed upstream is left untouched, both applied
    once per process:

    1. **Histogram-MSE calibration**: ``modelopt...calib.histogram._compute_amax_mse`` still
       calls ``fake_tensor_quant(centers, amax, num_bits, unsigned)`` with pytorch-quantization's
       positional signature, but modelopt's ``FakeTensorQuantFunction.forward`` takes ``bias``
       as the third argument. ``num_bits=8`` therefore lands in ``bias`` and ``unsigned=False``
       in ``num_bits``, and the MSE search degenerates to near-histogram-max amax
       (entropy/percentile are unaffected). Replace the module-level function with a
       corrected copy.

    2. **Checkpoint amax load**: pytorch-quantization's ``TensorQuantizer._load_from_state_dict``
       creates the ``_amax`` buffer on the fly when the incoming state_dict carries one; modelopt's
       ``TensorQuantizer`` has no such override, so loading a PTQ checkpoint into a freshly built
       (uncalibrated) quantizer tree silently drops every ``_amax`` as an "unexpected key". Add an
       equivalent override so existing PTQ checkpoints keep loading.
    """
    global _modelopt_patched
    if _modelopt_patched:
        return
    _modelopt_patched = True

    import numpy as np
    import torch
    from modelopt.torch.quantization.calib import histogram as _mo_histogram
    from modelopt.torch.quantization.nn import TensorQuantizer as _MoTensorQuantizer
    from modelopt.torch.quantization.tensor_quant import fake_tensor_quant as _mo_ftq

    # Probe with the exact call shape histogram.py uses; identity-range inputs must survive.
    probe = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    try:
        out = _mo_histogram.fake_tensor_quant(probe, torch.tensor(1.0), 8, False)
        if torch.allclose(out, probe, atol=0.05):
            return  # upstream fixed — nothing to do
    except Exception:
        pass  # broken in a louder way; patch below

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

    _mo_histogram._compute_amax_mse = _fixed_compute_amax_mse
    logger.warning(
        "Patched modelopt histogram MSE calibration (upstream fake_tensor_quant signature bug)"
    )

    # --- 2. checkpoint amax load -------------------------------------------------------------
    probe_q = _MoTensorQuantizer()
    probe_q._load_from_state_dict({"_amax": torch.tensor(1.5)}, "", {}, True, [], [], [])
    if getattr(probe_q, "_amax", None) is None:
        _orig_load = _MoTensorQuantizer._load_from_state_dict

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            amax = state_dict.get(prefix + "_amax")
            if amax is not None and "_amax" not in self._buffers:
                self.register_buffer("_amax", amax.data.clone())
            _orig_load(self, state_dict, prefix, *args, **kwargs)

        _MoTensorQuantizer._load_from_state_dict = _load_from_state_dict
        logger.warning(
            "Patched modelopt TensorQuantizer._load_from_state_dict to create _amax on load"
        )


# ---------------------------------------------------------------------------
# Class / module accessors
# ---------------------------------------------------------------------------


def get_tensor_quantizer_cls() -> Any:
    """The backend's ``TensorQuantizer`` class (raises when the backend is not installed)."""
    require("TensorQuantizer")
    from modelopt.torch.quantization.nn import TensorQuantizer

    _ensure_modelopt_patches()
    return TensorQuantizer


def get_calib() -> Any:
    """The backend's ``calib`` module (``MaxCalibrator`` / ``HistogramCalibrator``)."""
    require("calibrators")
    from modelopt.torch.quantization import calib

    _ensure_modelopt_patches()
    return calib


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def make_quant_desc(**kwargs: Any) -> Any:
    """Build a ``QuantizerAttributeConfig`` from the framework's descriptor vocabulary.

    Accepts the historical pytorch-quantization spelling (``num_bits``, ``axis`` tuple,
    ``calib_method``) that the framework's descriptor helpers speak, and translates to
    modelopt's (``calibrator``, int ``axis``).
    """
    require("quantization descriptors")
    from modelopt.torch.quantization.config import QuantizerAttributeConfig

    translated = dict(kwargs)
    if "calib_method" in translated:
        translated["calibrator"] = translated.pop("calib_method")
    axis = translated.get("axis")
    if isinstance(axis, tuple) and len(axis) == 1:
        translated["axis"] = axis[0]
    return QuantizerAttributeConfig(**translated)


def get_preset_desc(preset: str) -> Any:
    """A named ``QUANT_DESC_*`` preset from ``modelopt.torch.quantization.tensor_quant``.

    (modelopt kept pytorch-quantization's preset names.)
    """
    require("quantization descriptors")
    from modelopt.torch.quantization import tensor_quant

    return getattr(tensor_quant, preset)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------


def exports_qdq_natively() -> bool:
    """Whether ``TensorQuantizer`` traces to Q/DQ during ONNX export on its own.

    modelopt registers ONNX symbolics for its fake-quant autograd functions, so a quantized model
    exports QuantizeLinear/DequantizeLinear without any global switch — and the quantizers must
    NOT be bypassed during tracing. (pytorch-quantization needed ``use_fb_fake_quant=True``.)
    """
    return resolve() == MODELOPT


def setup_onnx_export() -> None:
    """Put the backend into ONNX-export mode (idempotent; no-op — see :func:`exports_qdq_natively`)."""
    if resolve() is not None:
        logger.info("modelopt backend: TensorQuantizer exports Q/DQ natively (no global switch)")
