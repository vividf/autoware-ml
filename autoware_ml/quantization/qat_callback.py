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

"""QAT training callback (Lightning) — the one home for the QAT training-loop logic.

``QATCallback`` holds everything model-agnostic about QAT: plan-prepare before
training, epoch-0 calibration, skip_quantize quantizer disable, embedding the
quantization description into every saved checkpoint, and the end-of-training
status log. The plan itself comes from the model's own
``build_quantization_plan`` — the same plan every other stage builds, so the
QAT tree is identical by construction.

The QAT method is frozen-amax STE fine-tuning: calibrated scales stay fixed
buffers and only the weights train — the production method in both
CUDA-CenterPoint and modelopt. There is deliberately no learnable-amax
machinery here.

Hard boundaries (fail loud instead of training wrong):

- **single device** — the tree mutation happens around the strategy wrap; multi-GPU
  DDP reducer buckets would desynchronize.
- **full precision** — AMP interacts with fake-quant; trainer precision must be
  ``32-true``.
- **no resume** — Lightning would restore optimizer/module state into a tree that
  is rebuilt from config, not from the checkpoint's exact structure.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import lightning as L
from modelopt.torch.quantization.nn import TensorQuantizer

from autoware_ml.quantization.checkpoint import QuantizationDescription, attach_quantization
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.core.calibration import Calibrator
from autoware_ml.quantization.core.quantizer_state import (
    count_quantizers,
    disable_quantizers_in,
    validate_quantizer_amax,
)
from autoware_ml.quantization.core.replace import expand_skip_quantize

logger = logging.getLogger(__name__)


class QATCallback(L.Callback):
    """Turn a training run into frozen-amax QAT fine-tuning.

    Args:
        quantization_config: Parsed ``quantization`` config (mode must be ``qat``); its
            ``calibration`` block drives the epoch-0 calibration.
        calib_forward_fn: Optional ``fn(model, batch)`` overriding the default
            calibration forward (``model.preprocess_batch`` + ``forward``).
    """

    def __init__(
        self,
        quantization_config: QuantizationConfig,
        calib_forward_fn: Callable | None = None,
    ) -> None:
        if quantization_config.mode != "qat" or quantization_config.qat is None:
            raise ValueError(
                "QATCallback requires quantization.mode='qat' with a qat block "
                f"(got mode={quantization_config.mode!r})."
            )
        self.config = quantization_config
        self.calib_forward_fn = calib_forward_fn
        self._quantized = False
        self._calibrated = False
        #: Placement record of the prepared tree (recorded by ``plan.prepare`` in
        #: :meth:`setup`). Embedded into every checkpoint Lightning saves so the
        #: result is self-describing.
        self.placement_record = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Fuse BN + insert Q/DQ via the model's shared plan before the strategy wrap."""
        if stage != "fit":
            return
        if trainer.num_devices > 1 or trainer.world_size > 1:
            raise RuntimeError(
                "QAT supports single-device training only (v1): the callback mutates the "
                "module tree, which desynchronizes DDP reducer buckets. Run on one GPU."
            )
        if str(trainer.precision) != "32-true":
            raise RuntimeError(
                f"QAT requires full precision (trainer.precision='32-true'), got "
                f"'{trainer.precision}': AMP interacts with fake-quant and desyncs the "
                "calibrated scales."
            )
        if getattr(trainer, "ckpt_path", None):
            raise RuntimeError(
                "QAT does not support resume: Lightning would restore state into the "
                "unquantized module tree. Start from --weights (an FP checkpoint) instead."
            )

        if any(isinstance(m, TensorQuantizer) for m in pl_module.modules()):
            raise RuntimeError(
                "QATCallback: the module tree is already quantized. QAT starts from an FP "
                "checkpoint and prepares the tree itself; preparing again would double-insert "
                "quantizers."
            )

        logger.info("QATCallback: fusing BatchNorm + inserting Q/DQ via the model plan...")
        plan = pl_module.build_quantization_plan(self.config)
        plan.prepare(pl_module)
        self.placement_record = plan.placement_record
        pl_module.train()
        self._quantized = True
        logger.info("QATCallback: quantization modules inserted")

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Calibrate quantizers at epoch 0."""
        if not self._quantized:
            logger.warning("QATCallback: model not quantized, skipping calibration")
            return
        if trainer.current_epoch != 0 or self._calibrated:
            return

        qat = self.config.qat
        # Calibrate on the CLEAN (val, un-augmented) dataloader, not the augmented
        # train loader: this is exactly what PTQ calibrates on, so the QAT amax
        # matches the proven-good PTQ amax, and it avoids augmentation feeding
        # degenerate inputs that can poison a histogram with Inf.
        dataloader = self._calibration_dataloader(trainer)
        # calibrate_samples counts SAMPLES (same unit as PTQ); the dataloader runs at
        # its native batch size, so convert to the number of batches to feed.
        batch_size = getattr(dataloader, "batch_size", None) or 1
        num_batches = math.ceil(qat.calibrate_samples / batch_size)
        logger.info(
            "QATCallback: calibrating on the val dataloader with %d samples "
            "in %d batches (batch_size=%d)...",
            qat.calibrate_samples,
            num_batches,
            batch_size,
        )
        Calibrator(pl_module).calibrate(
            dataloader,
            num_batches=num_batches,
            calibration=self.config.calibration,
            forward_fn=self.calib_forward_fn or default_calib_forward,
        )
        # Calibration switches the module to eval; hand it back to the training loop hot.
        pl_module.train()

        skip_names = expand_skip_quantize(pl_module, self.config.skip_quantize, log=False)
        if skip_names:
            logger.info(
                "QATCallback: disabling quantizers in %d skip_quantize module(s)...",
                len(skip_names),
            )
            disable_quantizers_in(pl_module, skip_names)

        validate_quantizer_amax(pl_module)
        if qat.freeze_unquantized:
            modules = dict(pl_module.named_modules())
            frozen = 0
            for name in skip_names:
                module = modules.get(name)
                if module is None:
                    continue
                for param in module.parameters():
                    if param.requires_grad:
                        param.requires_grad_(False)
                        frozen += 1
            logger.info(
                "QATCallback: froze %d parameter tensor(s) in %d skip_quantize module(s) "
                "(un-quantized layers take no STE gradient masking and otherwise drift past "
                "the frozen downstream amax)",
                frozen,
                len(skip_names),
            )
        self._calibrated = True
        logger.info("QATCallback: calibration complete")

    def _calibration_dataloader(self, trainer: L.Trainer):
        """Return the clean val dataloader for calibration (train loader as a warned fallback)."""
        datamodule = trainer.datamodule
        if datamodule is not None:
            try:
                return datamodule.val_dataloader()
            except Exception as error:  # noqa: BLE001 — any failure falls back to train
                logger.warning(
                    "QATCallback: could not use the val dataloader for calibration (%s); "
                    "falling back to the train dataloader (augmented — amax may drift from PTQ).",
                    error,
                )
        return trainer.train_dataloader

    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict
    ) -> None:
        """Embed the quantization description so the saved checkpoint is self-describing."""
        if not self._quantized:
            return
        if self.placement_record is None:
            raise RuntimeError(
                "QATCallback: no placement record to embed — the tree was never prepared."
            )
        attach_quantization(
            checkpoint,
            QuantizationDescription(config=self.config, placement_record=self.placement_record),
        )

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log quantizer status after training."""
        if self._quantized:
            counts = count_quantizers(pl_module)
            logger.info(
                "QATCallback: training complete. Quantizers: %d enabled, %d disabled, %d total",
                counts["enabled"],
                counts["disabled"],
                counts["total"],
            )


def default_calib_forward(model: L.LightningModule, batch) -> None:
    """Default calibration forward: ``preprocess_batch`` + ``forward``.

    No loss, no logging — statistics collection only needs the activations. The single
    definition shared by the QAT callback and the PTQ path (``scripts/quantize.py``).
    """
    device = next(model.parameters()).device
    model(model.preprocess_batch(batch, device))
