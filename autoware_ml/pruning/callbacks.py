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

"""Lightning callbacks of the pruning fine-tune stage.

- :class:`PruningCallback` makes every checkpoint Lightning saves self-describing
  (the pruning counterpart of ``QATCallback.on_save_checkpoint``).
- :class:`DistillationCallback` adds a knowledge-distillation term to the task loss
  through the base model's auxiliary-loss registry: the frozen FP teacher (the model
  as it was before the search) runs on the same preprocessed batch, and the model's own
  ``distillation_loss`` hook compares the two output dataclasses.
"""

from __future__ import annotations

import logging

import lightning as L
import torch
from torch import nn

from autoware_ml.pruning.checkpoint import (
    PRUNING_DESCRIPTION_ATTR,
    PruningDescription,
    attach_pruning,
)

logger = logging.getLogger(__name__)

AUXILIARY_LOSS_NAME = "kd"


class PruningCallback(L.Callback):
    """Embed the pruning description into every saved checkpoint."""

    def __init__(self, description: PruningDescription) -> None:
        self.description = description

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Remember the description on the model too (later stages read it from there)."""
        setattr(pl_module, PRUNING_DESCRIPTION_ATTR, self.description)

    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict
    ) -> None:
        """Make the checkpoint self-describing."""
        attach_pruning(checkpoint, self.description)


class DistillationCallback(L.Callback):
    """Add ``weight * pl_module.distillation_loss(student, teacher)`` to the training loss.

    Args:
        teacher: The un-pruned FP model. Frozen and kept in eval mode; moved to the
            training device at fit start.
        weight: Scale of the distillation term (``0`` disables it while keeping the
            teacher forward out of the step).
    """

    def __init__(self, teacher: nn.Module, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError(f"distillation weight must be >= 0, got {weight}")
        self.teacher = teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.weight = weight

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Register the distillation term on the student."""
        if stage != "fit" or self.weight == 0:
            return
        if not hasattr(pl_module, "distillation_loss"):
            raise RuntimeError(
                f"{type(pl_module).__name__} has no distillation_loss(student, teacher) hook; "
                "knowledge distillation needs the model to say how its outputs compare."
            )
        if trainer.num_devices > 1 or trainer.world_size > 1:
            raise RuntimeError(
                "Distillation fine-tune supports single-device training only (v1): the "
                "teacher is a plain module outside the strategy wrap."
            )

        def term(inputs, outputs) -> dict[str, torch.Tensor]:
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
            return {"loss_kd": self.weight * pl_module.distillation_loss(outputs, teacher_outputs)}

        pl_module.register_auxiliary_loss(AUXILIARY_LOSS_NAME, term)
        logger.info("DistillationCallback: KD term registered (weight=%.3g)", self.weight)

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Teacher follows the student's device."""
        self.teacher.to(pl_module.device).eval()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Lightning's ``train()`` on the student never reaches the teacher; keep it eval."""
        self.teacher.eval()

    def teardown(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Leave the student as it was."""
        if stage == "fit" and self.weight != 0:
            pl_module.unregister_auxiliary_loss(AUXILIARY_LOSS_NAME)
