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
from collections.abc import Callable

import lightning as L
from modelopt.torch.quantization.nn import TensorQuantizer

from autoware_ml.quantization.checkpoint import QuantizationDescription, attach_quantization
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.core.calibration import Calibrator, default_calib_forward
from autoware_ml.quantization.core.quantizer_state import (
    count_quantizers,
    disable_quantizers_in,
    validate_quantizer_amax,
)
from autoware_ml.quantization.core.replace import expand_skip_quantize
from autoware_ml.quantization.plan import QuantizationPlan

logger = logging.getLogger(__name__)


class QATCallback(L.Callback):
    """Turn a training run into frozen-amax QAT fine-tuning.

    The calibrated scales stay fixed and only the weights train (STE), as in CUDA-CenterPoint
    and modelopt; there is deliberately no learnable amax. The quantized tree comes from
    ``QuantizationPlan.for_model``, as in every other stage.

    Args:
        quantization_config: Parsed ``quantization`` config with ``mode: qat``.
        calib_forward_fn: ``fn(model, batch)`` replacing the default calibration forward
            (``on_after_batch_transfer`` + ``forward``).
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
        #: Set by :meth:`setup`; embedded in every checkpoint Lightning saves.
        self.placement_record = None

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

        logger.info("QATCallback: fusing BatchNorm + inserting Q/DQ via the model plan...")
        plan = QuantizationPlan.for_model(pl_module, self.config)
        plan.prepare(pl_module)
        self.placement_record = plan.placement_record
        pl_module.train()
        self._quantized = True
        logger.info("QATCallback: quantization modules inserted")
        # Freeze before the strategy builds the optimizer: frozen later, the parameters would
        # still be in the optimizer, and weight decay could still move them.
        qat = self.config.qat
        skip_names = expand_skip_quantize(pl_module, self.config.skip_quantize, log=False)
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

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Calibrate once, at the start of epoch 0, before any weight update."""
        if not self._quantized:
            logger.warning("QATCallback: model not quantized, skipping calibration")
            return
        if trainer.current_epoch != 0 or self._calibrated:
            return

        qat = self.config.qat
        # The un-augmented val loader, as in PTQ: the amax then matches PTQ's, and train
        # augmentation can feed degenerate inputs that poison a histogram with Inf.
        dataloader = self._calibration_dataloader(trainer)
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
            forward_fn=self.calib_forward_fn or default_calib_forward,
        )
        # Calibration leaves the module in eval mode.
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
            self._warn_unmasked_trainables(pl_module)
        self._calibrated = True
        logger.info("QATCallback: calibration complete")

    @staticmethod
    def _warn_unmasked_trainables(pl_module: L.LightningModule) -> None:
        """Warn about trainable modules that are neither quantized nor frozen.

        A module in neither ``quantize_submodules`` nor ``skip_quantize`` escapes
        ``freeze_unquantized``, and its weights can drift past the frozen amax downstream
        just like those of the ``skip_quantize`` modules it freezes. The current models have
        none.
        """
        quantized_parents = {
            name.rsplit(".", 1)[0]
            for name, module in pl_module.named_modules()
            if isinstance(module, TensorQuantizer) and "." in name
        }
        unmasked = sorted(
            {
                name.rsplit(".", 1)[0]
                for name, param in pl_module.named_parameters()
                if param.requires_grad
                and not any(
                    name.startswith(parent + ".") or name.rsplit(".", 1)[0] == parent
                    for parent in quantized_parents
                )
            }
        )
        if unmasked:
            logger.warning(
                "QATCallback: %d trainable module(s) are neither quantized nor frozen "
                "(no STE gradient mask, no freeze): %s%s. Add them to "
                "quantization.skip_quantize if they should be frozen.",
                len(unmasked),
                ", ".join(unmasked[:10]),
                ", ..." if len(unmasked) > 10 else "",
            )

    def _calibration_dataloader(self, trainer: L.Trainer):
        """Return the val dataloader, or the train one when the trainer has no datamodule.

        Errors from ``val_dataloader()`` propagate: falling back to augmented training data
        would let a broken validation setup degrade the amax silently.
        """
        datamodule = trainer.datamodule
        if datamodule is None:
            logger.warning(
                "QATCallback: the trainer has no datamodule, so there is no val dataloader; "
                "calibrating on the train dataloader (augmented — amax may drift from PTQ)."
            )
            return trainer.train_dataloader
        return datamodule.val_dataloader()

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
