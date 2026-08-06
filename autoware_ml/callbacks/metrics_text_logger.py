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

"""Epoch-end metric summaries in the text log.

The experiment tracker (MLflow) owns the full metric history, but the hydra
``train.log`` should stay readable on its own: the progress bar renders only to
the live terminal and leaves no usable trace in the log file. This callback
writes one summary line per epoch for training and validation metrics, so
``grep "metrics:" train.log`` reconstructs the training curve without opening
the tracker UI.
"""

from __future__ import annotations

import logging

import lightning as L
from lightning.pytorch.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


class MetricsTextLogger(L.Callback):
    """Log epoch-end metric summaries through the standard logging module."""

    def _log_metrics(self, trainer: L.Trainer, prefix: str, stage: str) -> None:
        metrics = {
            key: float(value)
            for key, value in trainer.callback_metrics.items()
            if key.startswith(prefix)
        }
        if not metrics:
            return
        formatted = "  ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        logger.info("Epoch %d %s metrics: %s", trainer.current_epoch, stage, formatted)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the validation metric summary for the finished epoch."""
        if trainer.sanity_checking:
            return
        self._log_metrics(trainer, "val", "validation")

    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the training metric summary for the finished epoch."""
        self._log_metrics(trainer, "train", "train")

    @rank_zero_only
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Write the test metric summary."""
        self._log_metrics(trainer, "test", "test")
