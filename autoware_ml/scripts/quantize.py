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

- ``quantization.mode: ptq`` rebuilds the quantized module tree through the model's own
  plan (BN fold + Q/DQ insertion), calibrates on the validation split through the clean
  test-time pipeline, and saves ``ptq.ckpt``.
- ``quantization.mode: qat`` runs frozen-amax STE fine-tuning: a short training run with
  :class:`~autoware_ml.quantization.qat_callback.QATCallback` injected (single device,
  full precision, lr schedule from ``quantization.qat.schedule``, no resume); Lightning
  saves ``best.ckpt`` / ``last.ckpt``.

Both outputs embed their
:class:`~autoware_ml.quantization.QuantizationDescription` (config + placement record)
next to the ``state_dict``, so ``deploy`` and ``test`` rebuild the identical tree from
the checkpoint alone — no ``quantization`` section, no mode branch, no sidecar files.

``quantization.dry_run: true`` builds the model on CPU, prepares the tree and logs the
placement table (which module gets which transform and why), then exits: the way to
inspect precision placement before spending GPU time.
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path

import hydra
import lightning as L
import numpy as np
import torch
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

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
from autoware_ml.quantization.core.calibration import default_calib_forward
from autoware_ml.quantization.qat_callback import QATCallback
from autoware_ml.utils.checkpoints import apply_matching_weights
from autoware_ml.utils.deploy import validate_cuda_available
from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_RUN_ID_ENV,
    build_run_metadata,
    configure_logger,
    get_user_config_name,
    load_run_context,
    log_config_params,
    prepare_run_context,
    resolve_deploy_lineage,
    should_enable_logger,
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
    resolve_work_dir,
    set_seed,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = get_config_path()

_MAX_CALIB_WORKERS = 4


def build_calibration_dataloader(
    datamodule: L.LightningDataModule, batch_size: int, seed: int | None, shuffle: bool
) -> DataLoader:
    """Calibration batches from the *validation* split through the test-time pipeline.

    The test split stays out of every model-producing step, and train augmentation
    (rotation / flip / paste) would feed the histograms degenerate inputs. The
    datamodule's own validation loader supplies the dataset and collation; only the batch
    size, worker count and sampling order are the recipe's.
    """
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
    """Prepare the quantized tree, calibrate, and save the self-describing checkpoint.

    ``model`` carries the FP weights already (a normal, un-fused training checkpoint):
    ``prepare`` folds BN and inserts Q/DQ on top of them.
    """
    ptq = quantization_config.ptq
    if ptq is None:
        raise ValueError("quantization.mode='ptq' requires a quantization.ptq block.")

    plan = model.build_quantization_plan(quantization_config)
    plan.prepare(model)
    model.eval()
    # Expand skip_quantize AFTER prepare — prepare mutates the tree (e.g. Pooling ->
    # .pool/.quantizer), so a glob can resolve differently before and after; the loader
    # expands after prepare too.
    skip_layers = expand_skip_quantize(model, quantization_config.skip_quantize)

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


def apply_qat_trainer_overrides(cfg: DictConfig, quantization_config: QuantizationConfig) -> None:
    """Turn the training config into the short QAT fine-tune schedule, in place.

    From config (``quantization.qat``): ``epochs``, ``lr`` + ``schedule`` (the fine-tune lr
    curve) and ``val_check_interval``. Enforced and deliberately not configurable (the
    hard boundaries ``QATCallback`` fails loud on): ``devices=1`` (the callback mutates the
    module tree; DDP buckets would desync), ``precision="32-true"`` (AMP interacts with
    fake-quant), ``num_sanity_val_steps=0`` (sanity-val would run on enabled but not yet
    calibrated quantizers) and ``check_val_every_n_epoch=1`` (a training config validating
    every N epochs would never produce best.ckpt over a few QAT epochs).
    """
    qat = quantization_config.qat
    with open_dict(cfg):
        cfg.trainer.max_epochs = qat.epochs
        cfg.trainer.devices = 1
        cfg.trainer.precision = "32-true"
        cfg.trainer.num_sanity_val_steps = 0
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.val_check_interval = qat.val_check_interval
        OmegaConf.update(cfg, "model.optimizer.lr", qat.lr, merge=False)
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
    run_context,
    logger_enabled: bool,
    work_dir: Path,
) -> Path:
    """Frozen-amax QAT fine-tuning; Lightning saves the self-describing checkpoints.

    ``model`` carries the FP weights; :class:`QATCallback` prepares the quantized tree in
    its ``setup`` and calibrates at epoch 0 on the validation dataloader.
    """
    if quantization_config.qat is None:
        raise ValueError("quantization.mode='qat' requires a quantization.qat block.")
    apply_qat_trainer_overrides(cfg, quantization_config)

    callbacks = instantiate_callbacks(
        cfg, logger_enabled=logger_enabled, checkpoint_dir=checkpoints_dir
    )
    callbacks.append(QATCallback(quantization_config))
    trainer_logger = None
    if logger_enabled and run_context is not None:
        configure_logger(
            cfg.logger,
            run_context.experiment_name,
            run_context.run_name,
            run_context.tags,
            run_id=run_context.run_id,
        )
        trainer_logger = hydra.utils.instantiate(cfg.logger)
    trainer: L.Trainer = instantiate_trainer(
        cfg, callbacks, trainer_logger, run_context.artifact_dir if run_context else work_dir
    )
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
    plan = model.build_quantization_plan(quantization_config)
    plan.prepare(model)
    plan.placement_record.log_table()
    logger.info("quantization.dry_run=true — exiting before calibration.")


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Quantize a configured model checkpoint."""
    quantization_raw = cfg.get("quantization")
    if quantization_raw is None:
        raise ValueError("Config must define a 'quantization' section for quantize.")
    quantization_config = QuantizationConfig.from_dict(
        OmegaConf.to_container(quantization_raw, resolve=True)
    )
    if not quantization_config.enabled:
        raise ValueError("quantization.enabled must be true for quantize.")
    log_configuration(cfg)
    work_dir = resolve_work_dir()
    config_name = get_user_config_name()

    if quantization_config.dry_run:
        model: L.LightningModule = hydra.utils.instantiate(cfg.model)
        model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))
        log_placement_dry_run(model.eval(), quantization_config)
        return

    weights_arg = cfg.get("weights", None)
    if weights_arg is None:
        raise ValueError("--weights <path> (repeatable) must be specified.")
    weight_paths = (
        [Path(weights_arg)] if isinstance(weights_arg, str) else [Path(p) for p in weights_arg]
    )
    for path in weight_paths:
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
    checkpoint_path = weight_paths[-1]

    logger_enabled = should_enable_logger(cfg)
    mlflow_client: MlflowClient | None = None
    run_id: str | None = None
    run_context = None
    if logger_enabled:
        experiment_name, parent_run_id, source_checkpoints = resolve_deploy_lineage(
            config_name, weight_paths
        )
        pre_created_run_id = os.environ.get(AUTOWARE_ML_RUN_ID_ENV)
        if pre_created_run_id is not None:
            run_context = load_run_context(cfg.logger.tracking_uri, pre_created_run_id)
        else:
            run_context = prepare_run_context(
                cfg.logger.tracking_uri,
                config_name,
                hydra_dir=work_dir,
                stage="quantize",
                parent_run_id=parent_run_id,
                experiment_name=experiment_name,
                extra_tags={
                    "checkpoint_path": str(checkpoint_path),
                    "quantization_mode": quantization_config.mode,
                    "source_run_id": parent_run_id or "",
                    "source_checkpoint_count": str(len(source_checkpoints)),
                },
            )
        mlflow_client = MlflowClient(tracking_uri=run_context.tracking_uri)
        run_id = run_context.run_id

    try:
        if run_context is not None:
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
            log_config_params(mlflow_client, run_id, OmegaConf.to_container(cfg, resolve=True))

        validate_cuda_available()
        configure_torch_runtime()
        set_seed(cfg)
        device = torch.device("cuda")

        configured_output_dir = cfg.get("output_dir", None)
        if run_context is not None:
            checkpoints_dir = run_context.checkpoints_dir
        elif configured_output_dir is not None:
            checkpoints_dir = Path(configured_output_dir)
        else:
            checkpoints_dir = checkpoint_path.parent / "quantized"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Instantiating datamodule...")
        datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
        logger.info("Instantiating model...")
        model = hydra.utils.instantiate(cfg.model)
        model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))
        # The FP input must be a normal (un-fused) training checkpoint: weights load into
        # the plain model first, then the plan folds BN and inserts Q/DQ.
        apply_matching_weights(
            model,
            weight_paths,
            map_location=device,
            device=device,
            set_eval=True,
            enforce_full_coverage=True,
            logger=logger,
        )
        if quantization_config.mode == "ptq":
            result_path = run_ptq(model, quantization_config, datamodule, checkpoints_dir)
        elif quantization_config.mode == "qat":
            result_path = run_qat(
                cfg,
                model,
                quantization_config,
                datamodule,
                checkpoints_dir,
                run_context,
                logger_enabled,
                work_dir,
            )
        else:
            raise ValueError(f"Unknown quantization.mode: {quantization_config.mode!r}")
    except Exception:
        if mlflow_client is not None and run_id is not None:
            mlflow_client.set_terminated(run_id, status=RunStatus.to_string(RunStatus.FAILED))
        raise
    if mlflow_client is not None and run_id is not None:
        mlflow_client.set_terminated(run_id, status=RunStatus.to_string(RunStatus.FINISHED))

    logger.info("Quantization complete. Deploy with: autoware-ml deploy --weights %s", result_path)


if __name__ == "__main__":
    main()
