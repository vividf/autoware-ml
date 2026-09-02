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

"""PTQ calibration: feed data, collect activation statistics, compute amax."""

import logging
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

from autoware_ml.quantization.core import modelopt as quant_backend

# Resolved once at import: the framework does not support switching backends mid-process
# (see autoware_ml.quantization.core.modelopt.resolve).
TensorQuantizer = quant_backend.get_tensor_quantizer_cls()
calib = quant_backend.get_calib()


@contextmanager
def _allow_nondeterministic_algorithms():
    """Temporarily lift ``torch.use_deterministic_algorithms`` around the calibration forward pass.

    Under QAT the training config's ``randomness = dict(..., deterministic=True)`` makes mmengine
    call ``torch.use_deterministic_algorithms(True)``, but the backend's
    ``HistogramCalibrator.collect`` uses ``torch.histc``, which has no deterministic CUDA kernel —
    every calibration batch would raise ``RuntimeError: _histc_cuda ... does not have a
    deterministic implementation`` and calibration would end with ``amax=None`` on every quantizer.
    Statistics collection has no bearing on training reproducibility, so the flag is lifted only
    for the collection loop and restored exactly (including ``warn_only``) afterwards.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


class Calibrator:
    """
    Manages PTQ calibration for a quantized model (model-agnostic).

    This class handles the complete calibration workflow:
    1. Enable calibration mode on all quantizers
    2. Feed calibration data through the model
    3. Compute optimal amax values from collected statistics
    4. Enable quantization mode with computed amax values

    Args:
        model: PyTorch model with quantization modules

    Example:
        >>> model = CenterPoint(...)
        >>> model.build_quantization_plan(config).prepare(model)  # Insert Q/DQ nodes
        >>> calibrator = Calibrator(model)
        >>> calibrator.calibrate(dataloader, num_batches=100, forward_fn=default_calib_forward)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = self._get_device()

    def _get_device(self) -> torch.device:
        """Get the device of the model."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def enable_torch_histogram(self):
        """Switch every histogram calibrator to torch-native histogram collection.

        This significantly speeds up calibration by using PyTorch's native histogram
        implementation instead of numpy (``HistogramCalibrator._torch_hist``; modelopt
        defaults it to True, set explicitly so the behavior never depends on the
        library default).
        """
        for _name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if hasattr(module, "_calibrator") and module._calibrator is not None:
                    if isinstance(module._calibrator, calib.HistogramCalibrator):
                        module._calibrator._torch_hist = True

    def _enable_calibration_mode(self):
        """Enable calibration mode on all TensorQuantizers."""
        for _name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()  # Disable fake quantization
                    module.enable_calib()  # Enable statistics collection
                else:
                    module.disable()

    def _disable_calibration_mode(self):
        """Disable calibration mode and enable quantization."""
        for _name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()  # Enable fake quantization
                    module.disable_calib()  # Disable statistics collection
                else:
                    module.enable()

    def collect_stats(
        self,
        dataloader: Any,
        num_batches: int,
        forward_fn: Callable[[nn.Module, Any], None],
    ):
        """Collect activation statistics for calibration.

        Feeds calibration data through the model while collecting statistics
        (min, max, histogram) for each quantizer. A failing batch raises
        immediately: silently skipping batches would shrink the calibration
        statistics without any visible signal (the resulting amax would still
        validate, just on less data than the recipe asked for).

        Args:
            dataloader: DataLoader providing calibration samples.
            num_batches: Number of batches to feed.
            forward_fn: ``forward_fn(model, batch)`` — owns device transfer and any
                runtime preprocessing (e.g. ``default_calib_forward``).
        """
        self.model.eval()
        self._enable_calibration_mode()

        # histc-based histogram collection is non-deterministic on CUDA; see the context manager.
        with torch.no_grad(), _allow_nondeterministic_algorithms():
            for i, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Calibrating"):
                if i >= num_batches:
                    break
                forward_fn(self.model, batch)

        self._disable_calibration_mode()

    def compute_amax(self, method: str = "mse"):
        """
        Compute amax values from collected statistics.

        The amax value determines the quantization scale. Different methods
        trade off between clipping error and rounding error:
        - "max": Use maximum observed value (no clipping, higher rounding error)
        - "mse": Minimize mean squared error (balanced)
        - "entropy": Minimize KL divergence (preserve distribution)
        - "percentile": Use percentile of distribution (robust to outliers)

        Args:
            method: Method for computing amax. One of:
                    "max", "mse", "entropy", "percentile"
        """
        for _name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(method=method, strict=False)

                    # Move amax to model device
                    if module._amax is not None:
                        module._amax = module._amax.to(self.device)

    def calibrate(
        self,
        dataloader: Any,
        num_batches: int,
        method: str = "mse",
        *,
        forward_fn: Callable[[nn.Module, Any], None],
    ):
        """Run the full calibration pipeline.

        This is the main entry point for calibration. It:
        1. Enables torch-native histogram collection
        2. Collects statistics from calibration data
        3. Computes optimal amax values

        Args:
            dataloader: DataLoader providing calibration samples.
            num_batches: Number of batches to feed.
            method: Method for computing amax ("max", "mse", "entropy", "percentile").
            forward_fn: ``forward_fn(model, batch)`` — owns device transfer and any
                runtime preprocessing.

        Example:
            >>> calibrator = Calibrator(model)
            >>> calibrator.calibrate(val_dataloader, num_batches=100, method="mse",
            ...                      forward_fn=default_calib_forward)
        """
        logger.info("Starting calibration with %d batches, method=%s", num_batches, method)

        self.enable_torch_histogram()
        self.collect_stats(dataloader, num_batches, forward_fn)
        self.compute_amax(method)

        num_quantizers = sum(
            1 for _, m in self.model.named_modules() if isinstance(m, TensorQuantizer)
        )
        logger.info("Calibration complete. %d quantizers calibrated.", num_quantizers)
