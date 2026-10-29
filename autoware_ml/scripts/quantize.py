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

"""Quantization entrypoint: FP checkpoint in, self-describing quantized checkpoint out.

``quantization.mode: ptq`` folds BatchNorm and inserts Q/DQ through the model's quantization
plan, calibrates on the validation split and saves ``ptq.ckpt``. The checkpoint embeds its
:class:`~autoware_ml.quantization.QuantizationDescription` (config and placement record), so
``deploy`` and ``test`` rebuild the same quantized tree from the checkpoint alone.

``quantization.dry_run: true`` only logs the placement table, on CPU without weights or data.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from types import MappingProxyType

import hydra
from hydra.core.hydra_config import HydraConfig
import lightning as L
import numpy as np
import torch
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from autoware_ml.quantization.checkpoint import QuantizationDescription, save_quantized_checkpoint
from autoware_ml.quantization.config import Mode, QuantizationConfig
from autoware_ml.quantization.core.calibration import Calibrator, default_calib_forward
from autoware_ml.quantization.core.quantizer_state import (
    disable_quantizers_in,
    print_quantizer_status,
    validate_quantizer_amax,
)
from autoware_ml.quantization.plan import prepare_quantized_tree
from autoware_ml.utils.deploy import validate_cuda_available
from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.mlflow_builder import build_mlflow_run_context
from autoware_ml.builders.model_builder import (
    build_data_preprocessor,
    build_model,
    build_weight_checkpoint_paths,
)
from autoware_ml.utils.mlflow_helpers import (
    MlflowRunContext,
    build_run_metadata,
    log_config_params,
    resolve_deploy_lineage,
    write_run_config_artifacts,
    write_run_metadata,
)
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    get_config_path,
    log_configuration,
    restrict_console_logging_to_rank_zero,
    set_seed,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = get_config_path()
CONFIG_NAME_PREFIX = "tasks/"

_MAX_CALIB_WORKERS = 4


def build_calibration_dataloader(
    datamodule: L.LightningDataModule, batch_size: int, seed: int | None, shuffle: bool
) -> DataLoader:
    """Build calibration batches from the validation split, through the test-time pipeline.

    The test split stays out of every model-producing step, and train augmentation
    (rotation / flip / paste) would skew the calibration histograms.
    """
    datamodule.prepare_data()
    datamodule.setup("validate")
    val_loader = datamodule.val_dataloader()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed if seed is not None else 0)
    return DataLoader(
        dataset=val_loader.dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(int(val_loader.num_workers), _MAX_CALIB_WORKERS),
        collate_fn=val_loader.collate_fn,
        generator=generator,
    )


def run_ptq(
    model: L.LightningModule,
    quantization_config: QuantizationConfig,
    datamodule: L.LightningDataModule,
    checkpoints_dir: Path,
) -> Path:
    """Prepare the quantized tree, calibrate it and save the self-describing checkpoint.

    ``model`` holds the un-fused FP training weights; ``prepare`` folds BN into them.
    """
    ptq = quantization_config.ptq
    if ptq is None:
        raise ValueError("quantization.mode='ptq' requires a quantization.ptq block.")

    prepared = prepare_quantized_tree(model, quantization_config)

    calibrate_batches = math.ceil(ptq.calibrate_samples / ptq.batch_size)
    dataloader = build_calibration_dataloader(
        datamodule, batch_size=ptq.batch_size, seed=ptq.calib_seed, shuffle=ptq.calib_shuffle
    )
    logger.info(
        "PTQ calibration: %d samples in %d batches (batch_size=%d, seed=%s, shuffle=%s, %s)",
        ptq.calibrate_samples,
        calibrate_batches,
        ptq.batch_size,
        ptq.calib_seed,
        ptq.calib_shuffle,
        quantization_config.calibration.describe(),
    )
    Calibrator(model).calibrate(
        dataloader,
        num_batches=calibrate_batches,
        calibration=quantization_config.calibration,
        forward_fn=default_calib_forward,
    )

    disable_quantizers_in(model, prepared.skip_names)
    # Runs after the skipped quantizers are disabled, which it ignores; a failed calibration
    # must fail here and not at deploy.
    validate_quantizer_amax(model)
    print_quantizer_status(model)

    return save_quantized_checkpoint(
        model,
        checkpoints_dir / "ptq.ckpt",
        QuantizationDescription(
            config=quantization_config, placement_record=prepared.placement_record
        ),
    )


def log_placement_dry_run(
    model: L.LightningModule, quantization_config: QuantizationConfig
) -> None:
    """Prepare the quantized tree on the (weightless, CPU) model and log the placement table."""
    prepare_quantized_tree(model, quantization_config).placement_record.log_table()
    logger.info("quantization.dry_run=true — exiting before calibration.")


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint of the quantize stage.

    Args:
        cfg: Hydra configuration
    """
    run_quantize(cfg)


def _parse_quantization_config(cfg: DictConfig) -> QuantizationConfig:
    quantization_raw = cfg.get("quantization")
    if quantization_raw is None:
        raise ValueError("Config must define a 'quantization' section for quantize.")
    quantization_config = QuantizationConfig.from_dict(
        OmegaConf.to_container(quantization_raw, resolve=True)
    )
    if not quantization_config.enabled:
        raise ValueError("quantization.enabled must be true for quantize.")
    if quantization_config.mode is not Mode.PTQ:
        raise NotImplementedError(
            f"quantization.mode={quantization_config.mode.value!r} is not available yet; this "
            "entrypoint runs post-training quantization (mode: ptq)."
        )
    return quantization_config


def record_quantize_run(
    cfg: DictConfig,
    run_context: MlflowRunContext,
    config_name: str,
    checkpoint_path: Path,
) -> None:
    """Persist what a quantize run was made from into its MLflow artifacts (as deploy does)."""
    write_run_config_artifacts(cfg, run_context.artifact_dir)
    write_run_metadata(
        run_context.artifact_dir,
        build_run_metadata(
            run_context,
            config_name,
            run_context.hydra_dir,
            "quantize",
            extra_metadata={"checkpoint_path": str(checkpoint_path)},
        ),
    )
    log_config_params(
        MlflowClient(tracking_uri=run_context.tracking_uri),
        run_context.run_id,
        OmegaConf.to_container(cfg, resolve=True),
    )


def run_quantize(cfg: DictConfig) -> None:
    """Quantize the configured model checkpoint under its MLflow quantize run.

    The MLflow run, when logging is enabled, is marked FAILED if anything after its creation
    raises and FINISHED otherwise, so a crashed quantize never stays RUNNING.

    Args:
        cfg: Hydra configuration
    """
    # First, so that every rank after rank zero stays silent from the config dump onwards.
    restrict_console_logging_to_rank_zero()
    quantization_config = _parse_quantization_config(cfg)

    log_configuration(cfg)
    config_name = HydraConfig.get().job.config_name
    if config_name is None:
        raise ValueError("Hydra config name is not available.")
    config_name = config_name.removeprefix(CONFIG_NAME_PREFIX)

    if quantization_config.dry_run:
        # The placement table needs the module tree only: no weights, no data, no GPU.
        model = build_model(
            cfg,
            data_preprocessor=build_data_preprocessor(cfg),
            weights_path=None,
            resume_checkpoint_path=None,
            device=torch.device("cpu"),
            set_eval=True,
        )
        log_placement_dry_run(model.eval(), quantization_config)
        return

    weight_paths, checkpoint_path = build_weight_checkpoint_paths(cfg)

    # Nest the quantize run under the source training run, when there is exactly one
    parent_run_id, source_checkpoints = resolve_deploy_lineage(weight_paths)
    logger_enabled = cfg.get("logger") is not None
    run_context = build_mlflow_run_context(
        cfg,
        stage="quantize",
        config_name=config_name,
        logger_enabled=logger_enabled,
        parent_run_id=parent_run_id,
        extra_tags=MappingProxyType(
            {
                "checkpoint_path": str(checkpoint_path),
                "quantization_mode": quantization_config.mode.value,
                "source_run_id": parent_run_id or "",
                "source_checkpoint_count": str(len(source_checkpoints)),
            }
        ),
    )

    if run_context is None:
        _quantize(cfg, quantization_config, weight_paths, run_context=None)
        return

    try:
        record_quantize_run(cfg, run_context, config_name, checkpoint_path=checkpoint_path)
        _quantize(cfg, quantization_config, weight_paths, run_context=run_context)
    except Exception:
        MlflowClient(tracking_uri=run_context.tracking_uri).set_terminated(
            run_context.run_id, status=RunStatus.to_string(RunStatus.FAILED)
        )
        raise
    MlflowClient(tracking_uri=run_context.tracking_uri).set_terminated(
        run_context.run_id, status=RunStatus.to_string(RunStatus.FINISHED)
    )


def _quantize(
    cfg: DictConfig,
    quantization_config: QuantizationConfig,
    weight_paths: list[Path],
    run_context: MlflowRunContext | None,
) -> None:
    """Build the data module and the FP model, then calibrate and write the checkpoint.

    Args:
        cfg: Hydra configuration.
        quantization_config: The parsed ``quantization`` section.
        weight_paths: FP checkpoints the model is built from.
        run_context: The quantize run, whose ``checkpoints`` directory receives
            ``ptq.ckpt``. Without one it lands in ``experiment_run_dir`` of the config.
    """
    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)
    device = torch.device("cuda")

    if run_context is not None:
        checkpoints_dir = run_context.checkpoints_dir
    elif cfg.get("experiment_run_dir") is not None:
        checkpoints_dir = Path(cfg.experiment_run_dir)
    else:
        raise ValueError(
            "Output directory must be specified in the configuration or obtained from MLflow run context."
        )
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoint directory for the quantized model: %s", checkpoints_dir)

    datamodule = build_datamodule(cfg)
    data_preprocessor = build_data_preprocessor(cfg)
    # A quantized checkpoint is refused: it would be quantized twice.
    model = build_model(
        cfg,
        data_preprocessor=data_preprocessor,
        weights_path=weight_paths,
        resume_checkpoint_path=None,
        device=device,
        set_eval=True,
        enforce_full_coverage=True,
        allow_quantized_weights=False,
    )
    result_path = run_ptq(model, quantization_config, datamodule, checkpoints_dir)
    logger.info("Quantization complete. Deploy with: autoware-ml deploy --weights %s", result_path)


if __name__ == "__main__":
    main()
