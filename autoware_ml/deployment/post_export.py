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

"""Post-export steps over a model's stage graph: verification (and, next, evaluation).

Both need the same things — the exported artifacts, pipelines per backend, batches from
the datamodule preprocessed the way the model's ``on_after_batch_transfer`` does — so
they are wired here rather than in ``scripts/deploy.py``, which only decides whether to
call in.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import torch

from autoware_ml.deployment.config import PostExportConfig, VerificationConfig
from autoware_ml.deployment.pipeline import PipelineCache, available_backends
from autoware_ml.deployment.stages import Stage
from autoware_ml.deployment.verification.backend_verifier import BackendVerifier
from autoware_ml.types.backend import Backend
from autoware_ml.utils.deploy import move_to_device

logger = logging.getLogger(__name__)


def preprocessed_batches(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
    *,
    split: str,
    limit: int,
) -> list[Any]:
    """Load up to ``limit`` batches of ``split`` and preprocess them like Lightning would.

    ``limit < 0`` loads the whole split. The batch goes through
    ``model.on_after_batch_transfer`` — the one place preprocessing runs — so a stage
    graph sees exactly what ``forward()`` sees.
    """
    datamodule.setup({"test": "test", "val": "validate", "predict": "predict"}[split])
    dataloader = getattr(datamodule, f"{split}_dataloader")()
    batches: list[Any] = []
    for index, batch in enumerate(dataloader):
        if 0 <= limit <= index:
            break
        batch = move_to_device(batch, device)
        batches.append(model.on_after_batch_transfer(batch, dataloader_idx=0))
    return batches


def run_verification(
    cfg: VerificationConfig,
    model: L.LightningModule,
    stages: Sequence[Stage],
    pipelines: PipelineCache,
    available: set[Backend],
    datamodule: L.LightningDataModule,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Compare the backends' raw graph outputs on the first predict batches; raise on failure.

    A model may declare ``verification_caveat`` (a sentence) when its raw outputs are
    incomparable across backends by construction; verification is then skipped loudly
    and per-backend evaluation is the meaningful gate.
    """
    del output_dir
    caveat = getattr(model, "verification_caveat", None)
    if caveat:
        logger.warning(
            "Verification SKIPPED: %s declares its raw graph outputs incomparable across "
            "backends — %s Per-backend metrics (deploy.evaluation) are the meaningful gate.",
            type(model).__name__,
            caveat,
        )
        return

    batches = preprocessed_batches(
        datamodule, model, device, split="predict", limit=cfg.num_verify_batches
    )
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
    deploy_cfg: Any,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
    output_dir: str | Path,
    device: torch.device,
) -> None:
    """Run the enabled post-export steps of ``deploy_cfg`` over the artifacts in ``output_dir``.

    Nothing runs when both ``deploy.verification`` and ``deploy.evaluation`` are disabled
    (the default). Enabling either on a model without ``build_stages()`` is an error: the
    steps execute the stage graph, there is nothing else that could run the artifacts
    together with the model's glue.
    """
    cfg = PostExportConfig.from_deploy_cfg(deploy_cfg)
    if not cfg.any_enabled:
        return
    stages = model.build_stages()
    if stages is None:
        raise ValueError(
            "deploy.verification / deploy.evaluation need a stage graph, but "
            f"{type(model).__name__} does not implement build_stages(). Declare the stage "
            "graph (see autoware_ml.deployment.stages) or disable these sections."
        )
    output_dir = Path(output_dir)
    pipelines = PipelineCache(stages, output_dir)
    available = available_backends(stages, output_dir)
    if cfg.verification.enabled:
        run_verification(
            cfg.verification, model, stages, pipelines, available, datamodule, device, output_dir
        )
