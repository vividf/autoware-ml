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

- ``quantization.mode: ptq`` folds BatchNorm and inserts Q/DQ through the model's quantization
  plan, calibrates on the validation split and saves ``ptq.ckpt``.
- ``quantization.mode: qat`` fine-tunes with frozen amax: a short training run with
  :class:`~autoware_ml.quantization.qat_callback.QATCallback`, from which Lightning saves
  ``best.ckpt`` / ``last.ckpt``.

Both checkpoints embed their :class:`~autoware_ml.quantization.QuantizationDescription` (config
and placement record), so ``deploy`` and ``test`` rebuild the same quantized tree from the
checkpoint alone.

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
from omegaconf import DictConfig, OmegaConf, open_dict
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
from autoware_ml.quantization.qat_callback import QATCallback
from autoware_ml.utils.deploy import validate_cuda_available
from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.logger_builder import build_trainer_logger
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
    instantiate_callbacks,
    instantiate_trainer,
    log_configuration,
    log_hyperparameters,
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


def apply_qat_trainer_overrides(cfg: DictConfig, quantization_config: QuantizationConfig) -> None:
    """Turn the training config into the short QAT fine-tune schedule, in place.

    From config (``quantization.qat``): ``epochs``, ``lr`` + ``schedule`` (the fine-tune lr
    curve) and ``val_check_interval``. Enforced and deliberately not configurable (the
    hard boundaries ``QATCallback`` fails loud on): ``devices=1`` (the callback mutates the
    module tree; DDP buckets would desync), ``precision="32-true"`` (AMP interacts with
    fake-quant), ``num_sanity_val_steps=0`` (sanity-val would run on enabled but not yet
    calibrated quantizers) and ``check_val_every_n_epoch=1`` (a training config validating
    every N epochs would never produce best.ckpt over a few QAT epochs). The experiments'
    ``DynamicValidation`` callback is dropped for the same reason: it would restore the
    training config's validation cadence at every epoch start and refuses a fractional
    ``val_check_interval``.

    Must run before anything instantiates from ``cfg``: Hydra binds the model's optimizer
    and scheduler when the model is instantiated, so a later override would silently leave
    QAT on the full-training lr and schedule.

    Raises:
        ValueError: If ``quantization.qat`` is missing.
    """
    qat = quantization_config.qat
    if qat is None:
        raise ValueError("quantization.mode='qat' requires a quantization.qat block.")
    with open_dict(cfg):
        cfg.trainer.max_epochs = qat.epochs
        cfg.trainer.devices = 1
        cfg.trainer.precision = "32-true"
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.val_check_interval = qat.val_check_interval
        if "callbacks" in cfg and "dynamic_validation" in cfg.callbacks:
            cfg.callbacks.dynamic_validation = None
        # The experiment bases interpolate `model.optimizer` from the top-level `optimizer`
        # node; resolve it to its own node first, or the lr override would replace the
        # interpolation with a bare {lr} and lose the optimizer target.
        optimizer = OmegaConf.to_container(cfg.model.optimizer, resolve=True)
        optimizer["lr"] = qat.lr
        cfg.model.optimizer = optimizer
        # The full-training cyclic / warmup schedule makes no sense over a short fine-tune;
        # the QAT schedule is stepped per iteration and the optimizer builder fills
        # total_steps so it spans all QAT epochs.
        scheduler, scheduler_config = qat.schedule.build_lightning_scheduler(qat.lr)
        cfg.model.scheduler = scheduler
        cfg.model.scheduler_config = scheduler_config
    logger.info(
        "QAT config: epochs=%d, peak lr=%g (%s), freeze_unquantized=%s, val every %.2f epoch, "
        "calibrate_samples=%d (single device, no sanity-val)",
        qat.epochs,
        qat.lr,
        qat.schedule.describe(),
        qat.freeze_unquantized,
        qat.val_check_interval,
        qat.calibrate_samples,
    )


def run_qat(
    cfg: DictConfig,
    model: L.LightningModule,
    quantization_config: QuantizationConfig,
    datamodule: L.LightningDataModule,
    checkpoints_dir: Path,
    run_context: MlflowRunContext | None,
    config_name: str,
) -> Path:
    """Frozen-amax QAT fine-tuning; Lightning saves the self-describing checkpoints.

    ``cfg`` already carries the QAT schedule (:func:`apply_qat_trainer_overrides`, applied
    before ``model`` was built). ``model`` carries the FP weights;
    :class:`QATCallback` prepares the quantized tree in its ``setup`` and calibrates at
    epoch 0 on the validation dataloader. The trainer is assembled like ``train`` does.
    """
    logger_enabled = run_context is not None
    callbacks = instantiate_callbacks(
        cfg, logger_enabled=logger_enabled, checkpoint_dir=checkpoints_dir
    )
    callbacks.append(QATCallback(quantization_config))
    trainer_logger = build_trainer_logger(
        cfg,
        ml_flow_run_context=run_context,
        stage="quantize",
        config_name=config_name,
        logger_enabled=logger_enabled,
    )
    trainer_root_dir = (
        run_context.artifact_dir if run_context is not None else cfg.experiment_run_dir
    )
    trainer: L.Trainer = instantiate_trainer(cfg, callbacks, trainer_logger, trainer_root_dir)
    log_hyperparameters(cfg, trainer_logger)
    model.train()
    trainer.fit(model, datamodule=datamodule)

    # Prefer the best checkpoint (measured on the quantized model); any produced checkpoint
    # stays valid deploy input.
    best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
    best_path = Path(best_model_path) if best_model_path else None
    last_path = checkpoints_dir / "last.ckpt"
    if best_path is not None and best_path.exists():
        return best_path
    if last_path.exists():
        logger.warning(
            "QAT produced no best checkpoint — validation never ran (or the checkpoint "
            "callback tracks no monitored metric). Falling back to last.ckpt."
        )
        return last_path
    raise FileNotFoundError(
        f"QAT training produced no checkpoint under {checkpoints_dir} (expected best/last)."
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
    if quantization_config.mode == "qat":
        # First, so the logged config, the run's params and the model all see the QAT
        # schedule.
        apply_qat_trainer_overrides(cfg, quantization_config)

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
        _quantize(cfg, quantization_config, weight_paths, config_name, run_context=None)
        return

    try:
        record_quantize_run(cfg, run_context, config_name, checkpoint_path=checkpoint_path)
        _quantize(cfg, quantization_config, weight_paths, config_name, run_context=run_context)
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
    config_name: str,
    run_context: MlflowRunContext | None,
) -> None:
    """Build the data module and the FP model, then quantize it (PTQ or QAT).

    Args:
        cfg: Hydra configuration.
        quantization_config: The parsed ``quantization`` section.
        weight_paths: FP checkpoints the model is built from.
        config_name: User-facing config name, for the QAT trainer logger.
        run_context: The quantize run, whose ``checkpoints`` directory receives the
            quantized checkpoint. Without one it lands in ``experiment_run_dir`` of the
            config.
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
    if quantization_config.mode == "ptq":
        result_path = run_ptq(model, quantization_config, datamodule, checkpoints_dir)
    else:
        result_path = run_qat(
            cfg, model, quantization_config, datamodule, checkpoints_dir, run_context, config_name
        )
    logger.info("Quantization complete. Deploy with: autoware-ml deploy --weights %s", result_path)


if __name__ == "__main__":
    main()
