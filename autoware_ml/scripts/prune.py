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

"""Pruning entrypoint (search and KD fine-tune): FP checkpoint in, pruned checkpoint out.

Two modes, selected by ``pruning.mode`` (the quantize stage's PTQ / QAT split):

- **search**: FastNAS width search on the model's declared subtree
  (``model.build_pruning_spec()``), the un-fine-tuned subnet saved as ``pruned.ckpt``.
  Enough to measure latency; its accuracy is not meaningful.
- **finetune**: the search, then a short knowledge-distillation fine-tune (teacher = the
  FP model as it was before the search) with :class:`PruningCallback` (self-describing
  checkpoints) and :class:`DistillationCallback` injected. ``--weights`` may also be an
  already pruned checkpoint, in which case the search is skipped and
  ``pruning.finetune.teacher_weights`` names the FP teacher.

Either output embeds its channel table: ``quantize`` / ``deploy`` / ``test`` rebuild the
narrowed architecture from it and need no ``pruning`` section.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from autoware_ml.builders.database_builder import build_database, build_datamodule
from autoware_ml.builders.logger_builder import build_trainer_logger
from autoware_ml.builders.mlflow_builder import build_mlflow_run_context, mlflow_run_scope
from autoware_ml.builders.model_builder import (
    build_data_preprocessor,
    build_model,
    build_weight_checkpoint_paths,
)
from autoware_ml.pruning import (
    PRUNING_DESCRIPTION_ATTR,
    PruningConfig,
    PruningDescription,
    find_pruning,
    save_pruned_checkpoint,
)
from autoware_ml.pruning.callbacks import DistillationCallback, PruningCallback
from autoware_ml.pruning.search import search_channel_table
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


def _build_fp_model(cfg: DictConfig, weights_path, device: torch.device):
    return build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        weights_path=weights_path,
        resume_checkpoint_path=None,
        device=device,
        set_eval=True,
        enforce_full_coverage=True,
    )


def run_search(
    cfg, pruning_config: PruningConfig, weights_path, datamodule, device, checkpoints_dir: Path
):
    """Search the subnet on the FP model and save ``pruned.ckpt``."""
    model = _build_fp_model(cfg, weights_path, device)
    if getattr(model, PRUNING_DESCRIPTION_ATTR, None) is not None:
        raise ValueError(
            "--weights is already a pruned checkpoint; the search starts from FP weights."
        )
    result = search_channel_table(
        model, model.build_pruning_spec(), datamodule, pruning_config, device
    )
    description = PruningDescription(config=pruning_config, channel_table=result.channel_table)
    setattr(model, PRUNING_DESCRIPTION_ATTR, description)
    return save_pruned_checkpoint(model, checkpoints_dir / "pruned.ckpt", description)


def run_finetune(
    cfg,
    pruning_config: PruningConfig,
    weights_path,
    datamodule,
    device,
    checkpoints_dir: Path,
    run_context,
    logger_enabled: bool,
    config_name: str,
) -> Path:
    """Search (unless ``--weights`` is pruned already), then KD fine-tune the subnet."""
    finetune = pruning_config.finetune
    if finetune is None:
        raise ValueError("pruning.mode='finetune' requires a pruning.finetune block.")

    weight_paths = [weights_path] if isinstance(weights_path, (str, Path)) else list(weights_path)
    already_pruned = find_pruning(weight_paths)
    if already_pruned is None:
        model = _build_fp_model(cfg, weights_path, device)
        # The teacher is the model as it is now — FP weights, un-pruned — kept aside
        # before the search narrows the student in place.
        teacher = copy.deepcopy(model).eval()
        result = search_channel_table(
            model, model.build_pruning_spec(), datamodule, pruning_config, device
        )
        description = PruningDescription(config=pruning_config, channel_table=result.channel_table)
        setattr(model, PRUNING_DESCRIPTION_ATTR, description)
        save_pruned_checkpoint(model, checkpoints_dir / "pruned.ckpt", description)
    else:
        if finetune.teacher_weights is None:
            raise ValueError(
                "--weights is a pruned checkpoint: the FP teacher must come from "
                "pruning.finetune.teacher_weights."
            )
        model = _build_fp_model(cfg, weights_path, device)  # build_model applies the table
        description = PruningDescription(
            config=pruning_config, channel_table=already_pruned[1].channel_table
        )
        setattr(model, PRUNING_DESCRIPTION_ATTR, description)
        teacher = _build_fp_model(cfg, finetune.teacher_weights, device)
        logger.info(
            "Skipping the search: %s is already pruned; teacher = %s",
            already_pruned[0],
            finetune.teacher_weights,
        )

    apply_finetune_trainer_overrides(cfg, pruning_config)
    model.train()

    trainer_logger = build_trainer_logger(
        cfg,
        ml_flow_run_context=run_context,
        stage="prune",
        config_name=config_name,
        logger_enabled=logger_enabled,
    )
    callbacks = instantiate_callbacks(
        cfg, logger_enabled=logger_enabled, checkpoint_dir=checkpoints_dir
    )
    callbacks.append(PruningCallback(description))
    callbacks.append(DistillationCallback(teacher, weight=finetune.kd_weight))

    trainer_root_dir = (
        run_context.artifact_dir if run_context is not None else cfg.experiment_run_dir
    )
    trainer: L.Trainer = instantiate_trainer(cfg, callbacks, trainer_logger, trainer_root_dir)
    log_hyperparameters(cfg, trainer_logger)
    trainer.fit(model, datamodule)

    best_model_path = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
    best_path = Path(best_model_path) if best_model_path else None
    last_path = checkpoints_dir / "last.ckpt"
    if best_path is not None and best_path.exists():
        result_path = best_path
    elif last_path.exists():
        result_path = last_path
        logger.warning(
            "Fine-tune produced no best checkpoint (validation never ran); using last.ckpt."
        )
    else:
        raise FileNotFoundError(f"Fine-tune produced no checkpoint under {checkpoints_dir}.")
    logger.info("Pruned + fine-tuned checkpoints ready: best=%s, last=%s", best_path, last_path)
    return result_path


def apply_finetune_trainer_overrides(cfg: DictConfig, pruning_config: PruningConfig) -> None:
    """Turn the training config into the short KD fine-tune schedule, in place.

    Same shape as the QAT overrides: ``epochs``, peak ``lr`` + ``schedule`` from
    ``pruning.finetune``; validate ``val_check_interval`` times per epoch so ``best.ckpt``
    tracks the recovery; no resume.
    """
    finetune = pruning_config.finetune
    assert finetune is not None
    with open_dict(cfg):
        cfg.trainer.max_epochs = finetune.epochs
        cfg.trainer.check_val_every_n_epoch = 1
        cfg.trainer.val_check_interval = finetune.val_check_interval
        OmegaConf.update(cfg, "optimizer.lr", finetune.lr, merge=False)
        cfg.model.scheduler, cfg.model.scheduler_config = (
            finetune.schedule.build_lightning_scheduler(finetune.lr)
        )
    logger.info(
        "Fine-tune schedule: %d epoch(s), peak lr %.2e, %s, kd_weight=%.3g, val every %.2f epoch",
        finetune.epochs,
        finetune.lr,
        finetune.schedule.describe(),
        finetune.kd_weight,
        finetune.val_check_interval,
    )


def _run_pruning(
    cfg: DictConfig,
    *,
    pruning_config: PruningConfig,
    weights_path,
    run_context,
    logger_enabled: bool,
    config_name: str,
) -> Path:
    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)
    device = torch.device("cuda")

    database = build_database(cfg)
    datamodule = build_datamodule(cfg, database=database)
    datamodule.prepare_data()
    datamodule.setup("predict")

    checkpoints_dir = (
        Path(run_context.checkpoints_dir)
        if run_context is not None
        else Path(cfg.experiment_run_dir) / "checkpoints"
    )
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if pruning_config.mode == "search":
        if run_context is not None:
            trainer_logger = build_trainer_logger(
                cfg,
                ml_flow_run_context=run_context,
                stage="prune",
                config_name=config_name,
                logger_enabled=logger_enabled,
            )
            log_hyperparameters(cfg, trainer_logger)
        return run_search(cfg, pruning_config, weights_path, datamodule, device, checkpoints_dir)
    return run_finetune(
        cfg,
        pruning_config,
        weights_path,
        datamodule,
        device,
        checkpoints_dir,
        run_context,
        logger_enabled,
        config_name,
    )


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main prune entrypoint."""
    pruning_raw = cfg.get("pruning")
    if pruning_raw is None:
        raise ValueError("Config must define a 'pruning' section for prune.")
    pruning_config = PruningConfig.from_dict(OmegaConf.to_container(pruning_raw, resolve=True))
    if not pruning_config.enabled:
        raise ValueError("pruning.enabled must be true for prune.")

    log_configuration(cfg)
    config_name = HydraConfig.get().job.config_name
    if config_name is None:
        raise ValueError("Hydra config name is not available.")
    config_name = config_name.removeprefix(EXPERIMENT_CONFIG_NAME_PREFIX)

    weights_path, checkpoint_path = build_weight_checkpoint_paths(cfg)
    experiment_name, parent_run_id, source_checkpoints = resolve_deploy_lineage(
        config_name, weights_path
    )
    logger_enabled = cfg.get("logger") is not None
    run_context = build_mlflow_run_context(
        cfg,
        stage="prune",
        experiment_name=experiment_name,
        config_name=config_name,
        experiment_uid=cfg.experiment_uid,
        logger_enabled=logger_enabled,
        parent_run_id=parent_run_id,
        extra_tags={
            "checkpoint_path": str(checkpoint_path),
            "pruning_mode": pruning_config.mode,
            "pruning_flops": pruning_config.flops,
            "source_run_id": parent_run_id or "",
            "source_checkpoint_count": str(len(source_checkpoints)),
        },
    )
    with mlflow_run_scope(run_context):
        result_path = _run_pruning(
            cfg,
            pruning_config=pruning_config,
            weights_path=weights_path,
            run_context=run_context,
            logger_enabled=logger_enabled,
            config_name=config_name,
        )
    logger.info("Pruning complete. Next: autoware-ml quantize/deploy --weights %s", result_path)


if __name__ == "__main__":
    main()
