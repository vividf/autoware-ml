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

"""PTQ calibration: modelopt's calibrators driven by the framework's own forward loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from modelopt.torch.quantization import calib
from modelopt.torch.quantization.model_calib import enable_stats_collection, smoothquant
from modelopt.torch.quantization.nn import TensorQuantizer
from torch import nn
from tqdm import tqdm

from autoware_ml.quantization.config import CalibrationConfig

logger = logging.getLogger(__name__)


@contextmanager
def _allow_nondeterministic_algorithms():
    """Lift ``torch.use_deterministic_algorithms`` for the calibration loop.

    ``HistogramCalibrator.collect`` uses ``torch.histc``, which has no deterministic CUDA
    kernel and raises when a deterministic training run turned the flag on. Statistics
    collection does not affect reproducibility.
    """
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


class Calibrator:
    """PTQ calibration of a model whose tree the quantization plan has prepared."""

    def __init__(self, model: nn.Module):
        self.model = model

    def _quantizers(self):
        for _name, module in self.model.named_modules():
            if isinstance(module, TensorQuantizer) and not module._disabled:
                yield module

    def _forward_loop(
        self, dataloader: Any, num_batches: int, forward_fn: Callable[[nn.Module, Any], None]
    ) -> Callable[[nn.Module], None]:
        """Build the ``forward_loop(model)`` callable modelopt's calibrators expect.

        A failing batch raises: skipping it would silently calibrate on less data than
        configured, and the resulting amax would still pass validation.
        """

        def forward_loop(model: nn.Module) -> None:
            model.eval()
            with torch.no_grad(), _allow_nondeterministic_algorithms():
                for i, batch in tqdm(enumerate(dataloader), total=num_batches, desc="Calibrating"):
                    if i >= num_batches:
                        break
                    forward_fn(model, batch)

        return forward_loop

    def collect_stats(
        self,
        dataloader: Any,
        num_batches: int,
        forward_fn: Callable[[nn.Module, Any], None],
    ) -> None:
        """Collect activation statistics with the calibrators on and fake-quant off; no amax yet."""
        enable_stats_collection(self.model)
        self._forward_loop(dataloader, num_batches, forward_fn)(self.model)

    def compute_amax(self, calibration: CalibrationConfig) -> None:
        """Load ``amax`` from every enabled quantizer's calibrator and switch fake-quant back on."""
        for quantizer in self._quantizers():
            if quantizer._calibrator is None:
                continue
            if isinstance(quantizer._calibrator, calib.HistogramCalibrator):
                kwargs = (
                    {"percentile": calibration.percentile}
                    if calibration.method == "percentile"
                    else {}
                )
                quantizer.load_calib_amax(calibration.method, strict=False, **kwargs)
            else:
                quantizer.load_calib_amax(strict=False)
            quantizer.enable_quant()
            quantizer.disable_calib()

    def calibrate(
        self,
        dataloader: Any,
        num_batches: int,
        calibration: CalibrationConfig,
        *,
        forward_fn: Callable[[nn.Module, Any], None],
    ) -> None:
        """Calibrate on ``num_batches`` batches with ``calibration.method``.

        ``forward_fn(model, batch)`` moves the batch to the model's device and preprocesses it.
        """
        logger.info("Starting calibration with %d batches, %s", num_batches, calibration.describe())
        if calibration.method == "smoothquant":
            # modelopt rescales the INT8 Linear weights in place (input_quantizer.pre_quant_scale).
            smoothquant(
                self.model,
                self._forward_loop(dataloader, num_batches, forward_fn),
                alpha=calibration.smoothquant_alpha,
            )
        else:
            self.collect_stats(dataloader, num_batches, forward_fn)
            self.compute_amax(calibration)

        num_quantizers = sum(1 for _ in self._quantizers())
        logger.info("Calibration complete. %d quantizers calibrated.", num_quantizers)


def default_calib_forward(model: nn.Module, batch: Any) -> None:
    """Run ``on_after_batch_transfer`` and the forward like Lightning's steps, without the loss."""
    device = next(model.parameters()).device
    batch_inputs = model.on_after_batch_transfer(batch.to_device(device), dataloader_idx=0)
    model(batch_inputs)
