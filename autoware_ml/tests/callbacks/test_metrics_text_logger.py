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

"""Tests for the epoch-end text-log metric summaries."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from autoware_ml.callbacks.metrics_text_logger import MetricsTextLogger


class _RangeDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index)])


class _TinyModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.layer(batch).mean()
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.layer(batch).mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/det3d/mAP", 0.5, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_metrics_text_logger_writes_epoch_summaries(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=1,
        callbacks=[MetricsTextLogger()],
        default_root_dir=str(tmp_path),
    )
    loader = DataLoader(_RangeDataset(), batch_size=2)
    trainer.fit(_TinyModule(), train_dataloaders=loader, val_dataloaders=loader)

    summaries = [record.message for record in caplog.records if "metrics:" in record.message]
    validation_lines = [line for line in summaries if "validation metrics" in line]
    train_lines = [line for line in summaries if " train metrics" in line]
    # One line per epoch per stage; the sanity check must not produce a line.
    assert len(validation_lines) == 2
    assert len(train_lines) == 2
    assert validation_lines[0].startswith("Epoch 0 validation metrics: ")
    assert "val/det3d/mAP=0.5000" in validation_lines[0]
    assert "val/loss=" in validation_lines[0]
    assert "train/loss=" in train_lines[0]
