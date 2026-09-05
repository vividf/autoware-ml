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

"""Quantization entrypoint (PTQ and QAT): FP checkpoint in, self-describing quantized checkpoint out.

- **PTQ** (``quantization.mode: ptq``): rebuild the quantized tree via the model's plan,
  calibrate on the validation split, and save ``ptq.ckpt``.
- **QAT** (``quantization.mode: qat``): frozen-amax STE fine-tuning — a short training
  run with :class:`~autoware_ml.quantization.qat_callback.QATCallback` injected (single
  device, full precision, lr schedule from ``quantization.qat.schedule``, no resume);
  Lightning saves ``best.ckpt`` / ``last.ckpt``.

Both outputs embed their :class:`~autoware_ml.quantization.QuantizationDescription`
(config + placement record) next to the ``state_dict``, so ``deploy`` and ``test``
rebuild the identical tree from the checkpoint alone — no ``quantization`` config, no
mode branch, no sidecar files.
Reference PTQ recipe: 400 samples @ batch_size=1, seed 0, histogram + MSE amax
(``quantization.calibration`` picks the amax method; see ``CalibrationConfig``).
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from autoware_ml.builders.database_builder import build_database, build_datamodule
from autoware_ml.builders.logger_builder import build_trainer_logger
from autoware_ml.builders.mlflow_builder import build_mlflow_run_context, mlflow_run_scope
from autoware_ml.builders.model_builder import (
    build_data_preprocessor,
    build_model,
    build_weight_checkpoint_paths,
)
from autoware_ml.datamodule.multi_task.multi_task_data_module import MultiTaskDataModule
from autoware_ml.quantization import (
    Calibrator,
    QuantizationDescription,
    disable_quantizers_in,
    expand_skip_quantize,
    print_quantizer_status,
    save_quantized_checkpoint,
    validate_quantizer_amax,
)
from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.qat_callback import QATCallback, default_calib_forward
from autoware_ml.utils.mlflow_helpers import resolve_deploy_lineage
from autoware_ml.utils.runtime import (
    EXPERIMENT_CONFIG_NAME_PREFIX,
    configure_torch_runtime,
    get_config_path,
    instantiate_callbacks,
    instantiate_trainer,
    log_configuration,
    log_hyperparameters,
    set_seed,
    validate_cuda_available,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()

# Deliberately a fixed constant, not a config key: the reference calibration method
# (CUDA-CenterPoint parity) is histogram + MSE amax, and the release numbers depend on it.
_MAX_CALIB_WORKERS = 4


def build_calibration_dataloader(
    datamodule: MultiTaskDataModule,
    batch_size: int,
    seed: int | None,
    shuffle: bool,
) -> DataLoader:
    """Build the calibration dataloader from the *validation* split.

    Calibration uses validation data through the clean test-time
    pipeline: the test split stays out of every model-producing step, and train
    augmentation (rot/flip/paste) would feed the histograms degenerate inputs.
    """
    datamodule.setup("validate")
    dataset = datamodule.validation_dataset
    if dataset is None:
        raise ValueError("Calibration requires a validation dataset; none is configured.")

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

    num_workers = 0
    val_config = datamodule.validation_dataloader_config
    if val_config is not None:
        num_workers = min(int(val_config.num_workers), _MAX_CALIB_WORKERS)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        generator=generator,
    )


def build_quantized_model(
    cfg: DictConfig,
    quantization_config: QuantizationConfig,
    weights_path,
    device: torch.device,
    set_eval: bool,
):
    """Build the model, load the FP weights, and prepare the quantized module tree.

    The FP input must be a normal (un-fused) training checkpoint: weights load into
    the plain model first, then ``prepare`` fuses BN and inserts Q/DQ.

    Returns:
        ``(model, plan)`` — the plan carries the placement record of the prepared tree.
    """
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        weights_path=weights_path,
        resume_checkpoint_path=None,
        device=device,
        set_eval=set_eval,
        enforce_full_coverage=True,
    )
    plan = model.build_quantization_plan(quantization_config)
    plan.prepare(model)
    model.eval() if set_eval else model.train()
    return model, plan


def run_ptq(
    cfg: DictConfig,
    quantization_config: QuantizationConfig,
    weights_path,
    datamodule: MultiTaskDataModule,
    device: torch.device,
    checkpoints_dir: Path,
) -> Path:
    """Run post-training quantization and save the calibrated, self-describing checkpoint."""
    ptq = quantization_config.ptq
    if ptq is None:
        raise ValueError("quantization.mode='ptq' requires a quantization.ptq block.")

    model, plan = build_quantized_model(
        cfg, quantization_config, weights_path, device, set_eval=True
    )
    # Expand skip_quantize AFTER prepare — prepare mutates the tree (e.g. Pooling ->
    # .pool/.quantizer), so a glob can resolve differently before and after; the loader
    # expands after prepare too.
    skip_layers = expand_skip_quantize(model, quantization_config.skip_quantize)

    calibrate_batches = math.ceil(ptq.calibrate_samples / ptq.batch_size)
    dataloader = build_calibration_dataloader(
        datamodule,
        batch_size=ptq.batch_size,
        seed=ptq.calib_seed,
        shuffle=ptq.calib_shuffle,
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

    disable_quantizers_in(model, skip_layers)
    # Validate AFTER disabling skip_quantize quantizers (the validator skips disabled ones):
    # a failed calibration must not produce a checkpoint that only explodes at deploy.
    validate_quantizer_amax(model)
    print_quantizer_status(model)

    return save_quantized_checkpoint(
        model,
        checkpoints_dir / "ptq.ckpt",
        QuantizationDescription(config=quantization_config, placement_record=plan.placement_record),
    )


def run_qat(
    cfg: DictConfig,
    quantization_config: QuantizationConfig,
    weights_path,
    datamodule: MultiTaskDataModule,
    checkpoints_dir: Path,
    run_context,
    logger_enabled: bool,
    config_name: str,
) -> Path:
    """Run frozen-amax QAT fine-tuning; Lightning saves the self-describing checkpoints."""
    qat = quantization_config.qat
    if qat is None:
        raise ValueError("quantization.mode='qat' requires a quantization.qat block.")

    apply_qat_trainer_overrides(cfg, quantization_config)

    # QATCallback prepares the quantized tree itself in setup(); here the FP weights just
    # load into the plain model.
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        weights_path=weights_path,
        resume_checkpoint_path=None,
        device=torch.device("cpu"),
        set_eval=False,
        enforce_full_coverage=False,
    )

    trainer_logger = build_trainer_logger(
        cfg,
        ml_flow_run_context=run_context,
        stage="quantize",
        config_name=config_name,
        logger_enabled=logger_enabled,
    )
    callbacks = instantiate_callbacks(
        cfg, logger_enabled=logger_enabled, checkpoint_dir=checkpoints_dir
    )
    callbacks.append(QATCallback(quantization_config))

    trainer_root_dir = (
        run_context.artifact_dir if run_context is not None else cfg.experiment_run_dir
    )
    trainer: L.Trainer = instantiate_trainer(cfg, callbacks, trainer_logger, trainer_root_dir)
    log_hyperparameters(cfg, trainer_logger)

    trainer.fit(model, datamodule)

    # Prefer the best checkpoint (measured on the *quantized* model). The filename comes
    # from the ModelCheckpoint callback config, so ask Lightning instead of hardcoding it.
    # Any produced checkpoint (best, last, or an epoch save) stays valid deploy input:
    # `autoware-ml deploy --weights <path>` takes whichever the user picks.
    best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
    best_path = Path(best_model_path) if best_model_path else None
    last_path = checkpoints_dir / "last.ckpt"
    if best_path is not None and best_path.exists():
        result_path = best_path
    elif last_path.exists():
        result_path = last_path
        logger.warning(
            "QAT produced no best checkpoint — validation never ran (or the checkpoint "
            "callback tracks no monitored metric), so 'best on the quantized model' was "
            "never measured. Falling back to last.ckpt (the final training step)."
        )
    else:
        raise FileNotFoundError(
            f"QAT training produced no checkpoint under {checkpoints_dir} (expected best/last)."
        )
    logger.info(
        "QAT checkpoints ready: best=%s, last=%s — deploying suggestion uses %s",
        best_path if best_path is not None and best_path.exists() else "(none)",
        last_path if last_path.exists() else "(none)",
        result_path,
    )
    return result_path


def apply_qat_trainer_overrides(cfg: DictConfig, quantization_config: QuantizationConfig) -> None:
    """Turn the training config into the short QAT fine-tune schedule, in place.

    From config (``quantization.qat``): ``epochs``, ``lr`` + ``schedule`` (the fine-tune
    lr curve), and ``val_check_interval``.

    Enforced, deliberately NOT configurable (QAT correctness constraints, matching the
    hard boundaries ``QATCallback`` fails loud on):

    - ``devices=1`` — the callback mutates the module tree; DDP buckets would desync.
    - ``precision="32-true"`` — AMP interacts with fake-quant.
    - ``num_sanity_val_steps=0`` — sanity-val would run on enabled but uncalibrated
      quantizers (amax=None) and crash.
    - ``check_val_every_n_epoch=1`` — a training config validating every N epochs would
      never produce best.ckpt over a few QAT epochs.
    """
    qat = quantization_config.qat
    with open_dict(cfg):
        cfg.trainer.max_epochs = qat.epochs
        cfg.trainer.devices = 1
        cfg.trainer.precision = "32-true"
        # Sanity-val runs BEFORE on_train_epoch_start, i.e. on enabled but not yet
        # calibrated quantizers (amax=None) — it must not run at all.
        cfg.trainer.num_sanity_val_steps = 0
        # Training configs often validate every N epochs; over a few QAT epochs that
        # would mean validation never runs and best.ckpt is never produced.
        cfg.trainer.check_val_every_n_epoch = 1
        # QAT degrades progressively within an epoch; validate several times per epoch
        # so the best.ckpt selection can catch the peak instead of the end-of-epoch tail.
        cfg.trainer.val_check_interval = qat.val_check_interval
        OmegaConf.update(cfg, "optimizer.lr", qat.lr, merge=False)
        # The full-training cyclic/warmup schedule makes no sense over a short QAT
        # fine-tune. The schedule shape comes from the config (QATScheduleConfig);
        # it is stepped per iteration and total_steps is auto-filled by the
        # optimizer builder, so it spans all QAT epochs.
        cfg.model.scheduler, cfg.model.scheduler_config = qat.schedule.build_lightning_scheduler(
            qat.lr
        )
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


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main quantize entrypoint.

    Args:
        cfg: Hydra configuration.
    """
    quantization_raw = cfg.get("quantization")
    if quantization_raw is None:
        raise ValueError("Config must define a 'quantization' section for quantize.")
    quantization_config = QuantizationConfig.from_dict(
        OmegaConf.to_container(quantization_raw, resolve=True)
    )
    if not quantization_config.enabled:
        raise ValueError("quantization.enabled must be true for quantize.")

    log_configuration(cfg)
    config_name = HydraConfig.get().job.config_name
    if config_name is None:
        raise ValueError("Hydra config name is not available.")
    config_name = config_name.removeprefix(EXPERIMENT_CONFIG_NAME_PREFIX)

    weights_path, checkpoint_path = build_weight_checkpoint_paths(cfg)
    experiment_name, parent_run_id, source_checkpoints = resolve_deploy_lineage(
        config_name,
        weights_path,
    )
    source_run_ids = [
        source["run_id"] for source in source_checkpoints if source["run_id"] is not None
    ]
    logger_enabled = cfg.get("logger") is not None
    run_context = build_mlflow_run_context(
        cfg,
        stage="quantize",
        experiment_name=experiment_name,
        config_name=config_name,
        experiment_uid=cfg.experiment_uid,
        logger_enabled=logger_enabled,
        parent_run_id=parent_run_id,
        extra_tags={
            "checkpoint_path": str(checkpoint_path),
            "quantization_mode": quantization_config.mode,
            "source_run_id": parent_run_id or "",
            "source_checkpoint_count": str(len(source_checkpoints)),
            "source_run_ids": ",".join(source_run_ids),
        },
    )
    with mlflow_run_scope(run_context):
        result_path = _run_quantization(
            cfg,
            quantization_config=quantization_config,
            weights_path=weights_path,
            run_context=run_context,
            logger_enabled=logger_enabled,
            config_name=config_name,
        )

    if result_path is None:
        logger.info("Dry run complete — no checkpoint produced.")
    else:
        logger.info(
            "Quantization complete. Deploy with: autoware-ml deploy --weights %s", result_path
        )


def _log_placement_dry_run(cfg: DictConfig, quantization_config: QuantizationConfig) -> None:
    """Build the model on CPU, prepare the quantized tree, and log the placement table.

    ``quantization.dry_run=true``: the way to inspect precision placement (which module
    gets which transform and why) before spending GPU time — no weights, no data, no
    calibration, no artifact.
    """
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        weights_path=None,
        resume_checkpoint_path=None,
        device=torch.device("cpu"),
        set_eval=True,
        enforce_full_coverage=False,
    )
    plan = model.build_quantization_plan(quantization_config)
    plan.prepare(model)
    plan.placement_record.log_table()
    logger.info("quantization.dry_run=true — exiting before calibration/training.")


def _run_quantization(
    cfg: DictConfig,
    *,
    quantization_config: QuantizationConfig,
    weights_path,
    run_context,
    logger_enabled: bool,
    config_name: str,
) -> Path | None:
    """Run the PTQ or QAT stage inside the MLflow run scope (``None`` on a dry run)."""
    if quantization_config.dry_run:
        _log_placement_dry_run(cfg, quantization_config)
        return None

    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)
    device = torch.device("cuda")

    database = build_database(cfg)
    datamodule = build_datamodule(cfg, database=database)
    datamodule.prepare_data()

    if run_context is not None:
        checkpoints_dir = Path(run_context.checkpoints_dir)
    else:
        checkpoints_dir = Path(cfg.experiment_run_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if quantization_config.mode == "ptq":
        if run_context is not None:
            trainer_logger = build_trainer_logger(
                cfg,
                ml_flow_run_context=run_context,
                stage="quantize",
                config_name=config_name,
                logger_enabled=logger_enabled,
            )
            log_hyperparameters(cfg, trainer_logger)
        return run_ptq(cfg, quantization_config, weights_path, datamodule, device, checkpoints_dir)
    if quantization_config.mode == "qat":
        return run_qat(
            cfg,
            quantization_config,
            weights_path,
            datamodule,
            checkpoints_dir,
            run_context,
            logger_enabled,
            config_name,
        )
    raise ValueError(f"Unknown quantization.mode: {quantization_config.mode!r}")


if __name__ == "__main__":
    main()
