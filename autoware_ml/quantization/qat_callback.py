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


"""Lightning callback for quantization-aware training."""

from __future__ import annotations

import logging
import math

import lightning as L
from modelopt.torch.quantization.nn import TensorQuantizer

from autoware_ml.quantization.checkpoint import QuantizationDescription, attach_quantization
from autoware_ml.quantization.config import Mode, QuantizationConfig
from autoware_ml.quantization.core.calibration import Calibrator, default_calib_forward
from autoware_ml.quantization.core.quantizer_state import (
    count_quantizers,
    disable_quantizers_in,
    validate_quantizer_amax,
)
from autoware_ml.quantization.plan import PreparedTree, prepare_quantized_tree

logger = logging.getLogger(__name__)


class QATCallback(L.Callback):
    """Turn a training run into frozen-amax QAT fine-tuning.

    The calibrated scales stay fixed and only the weights train (STE), as in CUDA-CenterPoint
    and modelopt; there is deliberately no learnable amax. The quantized tree comes from
    :func:`prepare_quantized_tree`, as in every other stage.

    Args:
        quantization_config: Parsed ``quantization`` config with ``mode: qat``.
    """

    def __init__(self, quantization_config: QuantizationConfig) -> None:
        if quantization_config.mode is not Mode.QAT or quantization_config.qat is None:
            raise ValueError(
                "QATCallback requires quantization.mode='qat' with a qat block "
                f"(got mode={quantization_config.mode.value!r})."
            )
        self.config = quantization_config
        #: Set by :meth:`setup`; its record is embedded in every checkpoint Lightning saves.
        self._prepared: PreparedTree | None = None
        self._calibrated = False

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Prepare the quantized tree before the strategy wraps the model."""
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

        self._prepared = prepare_quantized_tree(pl_module, self.config)
        pl_module.train()
        if self.config.qat.freeze_unquantized:
            # Before the strategy builds the optimizer: frozen later, the parameters would
            # still be in the optimizer, and weight decay could still move them.
            modules = dict(pl_module.named_modules())
            frozen = 0
            for name in self._prepared.skip_names:
                for param in modules[name].parameters():
                    if param.requires_grad:
                        param.requires_grad_(False)
                        frozen += 1
            logger.info(
                "QATCallback: froze %d parameter tensor(s) in %d skip_quantize module(s) "
                "(un-quantized layers take no STE gradient masking and otherwise drift past "
                "the frozen downstream amax)",
                frozen,
                len(self._prepared.skip_names),
            )

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Calibrate once, at the start of epoch 0, before any weight update."""
        if self._prepared is None:
            raise RuntimeError("QATCallback: setup('fit') did not run, the tree is not prepared.")
        if trainer.current_epoch != 0 or self._calibrated:
            return

        qat = self.config.qat
        # The un-augmented val loader, as in PTQ: the amax then matches PTQ's, and train
        # augmentation can feed degenerate inputs that poison a histogram with Inf.
        dataloader = trainer.datamodule.val_dataloader()
        # calibrate_samples counts samples, as in PTQ; the loader keeps its own batch size.
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
            forward_fn=default_calib_forward,
        )
        # Calibration leaves the module in eval mode.
        pl_module.train()
        disable_quantizers_in(pl_module, self._prepared.skip_names)
        validate_quantizer_amax(pl_module)
        self._calibrated = True
        logger.info("QATCallback: calibration complete")

    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict
    ) -> None:
        """Embed the quantization description so the saved checkpoint is self-describing."""
        if self._prepared is None:
            return
        attach_quantization(
            checkpoint,
            QuantizationDescription(
                config=self.config, placement_record=self._prepared.placement_record
            ),
        )

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log quantizer status after training."""
        if self._prepared is not None:
            counts = count_quantizers(pl_module)
            logger.info(
                "QATCallback: training complete. Quantizers: %d enabled, %d disabled, %d total",
                counts.enabled,
                counts.disabled,
                counts.total,
            )
