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

"""PTQ calibration: feed data, collect activation statistics, compute amax.

A thin layer over modelopt's calibration primitives: statistics collection is modelopt's
``enable_stats_collection`` (calibrators on, fake-quant off) around a forward loop the
framework owns (progress bar, fail-loud batches, the ``torch.histc`` determinism guard);
``amax`` is then loaded per quantizer with the configured method
(:class:`~autoware_ml.quantization.config.CalibrationConfig`). SmoothQuant delegates to
modelopt's ``smoothquant`` (max calibration + per-channel activation-to-weight migration
for every INT8 quantized Linear).
"""

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
    """PTQ calibration for a quantized model (model-agnostic).

    Args:
        model: Model whose tree the quantization plan already prepared.

    Example:
        >>> model.build_quantization_plan(config).prepare(model)  # Insert Q/DQ nodes
        >>> Calibrator(model).calibrate(dataloader, num_batches=100,
        ...                             calibration=config.calibration,
        ...                             forward_fn=default_calib_forward)
    """

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

        A failing batch raises immediately: silently skipping batches would shrink the
        calibration statistics without any visible signal (the resulting amax would still
        validate, just on less data than the recipe asked for).
        """

        def forward_loop(model: nn.Module) -> None:
            model.eval()
            # histc-based histogram collection is non-deterministic on CUDA; see the context manager.
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
        """Collect activation statistics (calibrators on, fake-quant off) — no amax yet.

        Args:
            dataloader: DataLoader providing calibration samples.
            num_batches: Number of batches to feed.
            forward_fn: ``forward_fn(model, batch)`` — owns device transfer and any
                runtime preprocessing (e.g. ``default_calib_forward``).
        """
        enable_stats_collection(self.model)
        self._forward_loop(dataloader, num_batches, forward_fn)(self.model)

    def compute_amax(self, calibration: CalibrationConfig) -> None:
        """Load ``amax`` from every enabled quantizer's calibrator and switch fake-quant back on.

        Histogram calibrators take the configured method (``mse`` / ``entropy`` /
        ``percentile``); max calibrators (all weights, FP8 activations, and INT8
        activations under ``method: max``) have exactly one answer.
        """
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
        """Run the full calibration pipeline for ``calibration.method``.

        Args:
            dataloader: DataLoader providing calibration samples.
            num_batches: Number of batches to feed.
            calibration: The ``quantization.calibration`` block.
            forward_fn: ``forward_fn(model, batch)`` — owns device transfer and any
                runtime preprocessing.
        """
        logger.info("Starting calibration with %d batches, %s", num_batches, calibration.describe())
        if calibration.method == "smoothquant":
            # modelopt: max-calibrate everything (per-channel on the Linear inputs), then
            # migrate each INT8 Linear's activation outliers into its weight
            # (input_quantizer.pre_quant_scale, weight rescaled in place).
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
