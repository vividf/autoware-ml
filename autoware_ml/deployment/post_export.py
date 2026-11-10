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

"""Post-export verification over a model's stage graph."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch

from autoware_ml.deployment.config import PostExportConfig, VerificationConfig
from autoware_ml.deployment.pipeline import PipelineCache, available_backends
from autoware_ml.deployment.verification.backend_verifier import BackendVerifier
from autoware_ml.types.backend import Backend

logger = logging.getLogger(__name__)


def predict_batches(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
    *,
    limit: int,
) -> list[Any]:
    """Load up to ``limit`` predict batches, all if negative, preprocessed as Lightning does."""
    # A cache hit after the export; a stand-alone call builds the record tables here.
    datamodule.prepare_data()
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()
    batches: list[Any] = []
    for index, batch in enumerate(dataloader):
        if 0 <= limit <= index:
            break
        batches.append(model.on_after_batch_transfer(batch.to_device(device), dataloader_idx=0))
    return batches


def run_verification(
    cfg: VerificationConfig,
    model: L.LightningModule,
    pipelines: PipelineCache,
    available: set[Backend],
    datamodule: L.LightningDataModule,
    device: torch.device,
) -> None:
    """Compare the backends' raw model outputs on the first predict batches; raise on failure."""
    batches = predict_batches(datamodule, model, device, limit=cfg.num_verify_batches)
    if not batches:
        raise ValueError("Verification produced zero batches from the predict dataloader.")

    verifier = BackendVerifier(pipelines, tolerance=cfg.tolerance)
    if not verifier.run(batches, cfg.scenarios, available):
        raise RuntimeError(
            "Backend verification FAILED — exported artifacts diverge from the reference "
            "beyond tolerance. See the scenario logs above."
        )
    logger.info("Backend verification passed.")


def run_post_export(
    cfg: PostExportConfig,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    output_dir: str | Path,
    device: torch.device,
    plugin_libraries: Sequence[str] = (),
) -> None:
    """Run the enabled ``deploy.verification`` over ``output_dir``.

    ``plugin_libraries`` are the TensorRT plugins the engines were built with. Enabling the
    step on a model without ``build_stages()`` is an error.
    """
    if not cfg.any_enabled:
        return
    stages = model.build_stages()
    if stages is None:
        raise ValueError(
            "deploy.verification needs a stage graph, but "
            f"{type(model).__name__} does not implement build_stages(). Declare the stage "
            "graph (see autoware_ml.deployment.stages) or disable the section."
        )
    output_dir = Path(output_dir)
    pipelines = PipelineCache(stages, output_dir, plugin_libraries=plugin_libraries)
    available = available_backends(stages, output_dir)
    if cfg.verification.enabled:
        run_verification(cfg.verification, model, pipelines, available, datamodule, device)
