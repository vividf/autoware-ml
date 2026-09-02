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

"""Deployment entrypoint: export -> verify -> evaluate, all derived from the model's stage graph.

A deployment run is three peer stages sharing one exported-artifact directory and
one set of backend pipelines:

1. **Export** — one ONNX (and optionally TensorRT) artifact per exportable stage.
2. **Verification** — cross-backend numerical parity on the final raw graph outputs.
3. **Evaluation** — per-backend ground-truth metrics and latency, under the same
   ``{split}/{backend}/{metric}`` keys as ``trainer.test``.

A quantized (PTQ / QAT) checkpoint describes itself, so this script never reads a
``quantization`` config section: ``build_model`` rebuilds the quantized tree from the
checkpoint and the exported ONNX carries Q/DQ nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import torch

from autoware_ml.builders.database_builder import build_database, build_datamodule
from autoware_ml.builders.logger_builder import build_trainer_logger
from autoware_ml.builders.mlflow_builder import build_mlflow_run_context, mlflow_run_scope
from autoware_ml.builders.model_builder import (
    build_data_preprocessor,
    build_model,
    build_weight_checkpoint_paths,
)
from autoware_ml.datamodule.multi_task.multi_task_data_module import MultiTaskDataModule
from autoware_ml.deployment.config import DeployConfig
from autoware_ml.deployment.export import available_backends, export_stages
from autoware_ml.deployment.pipeline import PipelineCache
from autoware_ml.deployment.stages import Stage
from autoware_ml.deployment.verification import BackendVerifier
from autoware_ml.evaluation import (
    EvaluationResult,
    evaluate_backend,
    log_comparison,
    log_results_to_mlflow,
)
from autoware_ml.metrics.base import EvalStage
from autoware_ml.models.multi_task_base_model import MultiTaskBaseModel
from autoware_ml.utils.deploy import (
    build_tensorrt_engine,
    export_to_onnx,
    merge_module_onnx_cfg,
    modify_onnx_graph,
    resolve_export_specs,
    should_modify_graph,
)
from autoware_ml.utils.mlflow_helpers import resolve_deploy_lineage
from autoware_ml.utils.runtime import (
    EXPERIMENT_CONFIG_NAME_PREFIX,
    configure_torch_runtime,
    get_config_path,
    log_configuration,
    log_hyperparameters,
    set_seed,
    validate_cuda_available,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


def export(
    deploy_cfg: DeployConfig,
    stages: Sequence[Stage],
    output_dir: Path,
    datamodule: MultiTaskDataModule,
    model: MultiTaskBaseModel,
    device: torch.device,
) -> None:
    """Stage 1: export every exportable stage (skipped when both exporters are disabled)."""
    if not (deploy_cfg.onnx.enabled or deploy_cfg.tensorrt.enabled):
        logger.info(
            "Export disabled (deploy.onnx and deploy.tensorrt); reusing artifacts in %s.",
            output_dir,
        )
        return
    batch = next(iter(datamodule.predict_dataloader()))
    batch_inputs = model.preprocess_batch(batch, device)
    artifacts = export_stages(stages, batch_inputs, deploy_cfg, output_dir, device)
    for stage, path in artifacts.onnx.items():
        logger.info("ONNX  [%s]: %s", stage, path)
    for stage, path in artifacts.engines.items():
        logger.info("TensorRT [%s]: %s", stage, path)


def verify(
    deploy_cfg: DeployConfig,
    pipelines: PipelineCache,
    datamodule: MultiTaskDataModule,
    model: MultiTaskBaseModel,
    device: torch.device,
    available: set,
) -> None:
    """Stage 2: cross-backend numerical verification; raise when any scenario fails."""
    cfg = deploy_cfg.verification
    if not cfg.enabled:
        logger.info("Verification disabled; skipping.")
        return
    caveat = getattr(model, "verification_caveat", None)
    if caveat:
        logger.warning(
            "Verification SKIPPED: %s declares its raw graph outputs incomparable across "
            "backends — %s Per-backend metrics (deploy.evaluation) are the meaningful gate.",
            type(model).__name__,
            caveat,
        )
        return

    batches = []
    for index, batch in enumerate(datamodule.predict_dataloader()):
        if index >= cfg.num_verify_batches:
            break
        batches.append(model.preprocess_batch(batch, device))
    if not batches:
        raise ValueError("Verification produced zero batches from the predict dataloader.")

    verifier = BackendVerifier(pipelines, tolerance=cfg.tolerance)
    if not verifier.run(batches, cfg.scenarios, available):
        raise RuntimeError(
            "Backend verification FAILED — exported artifacts diverge from the reference "
            "beyond tolerance. See the scenario logs above."
        )
    logger.info("Backend verification passed.")


def evaluate(
    deploy_cfg: DeployConfig,
    pipelines: PipelineCache,
    datamodule: MultiTaskDataModule,
    model: MultiTaskBaseModel,
    device: torch.device,
    available: set,
) -> list[EvaluationResult]:
    """Stage 3: per-backend ground-truth evaluation and latency."""
    cfg = deploy_cfg.evaluation
    if not cfg.enabled:
        logger.info("Evaluation disabled; skipping.")
        return []

    # The predict dataloader carries the test split; a val evaluation scores the
    # validation dataloader instead and its metric keys report under val/... .
    if cfg.split == "val":
        datamodule.setup("validate")
        make_dataloader = datamodule.val_dataloader
        eval_stage = EvalStage.VAL
        logger.info("Evaluation split: val (metric keys report under val/...).")
    else:
        make_dataloader = datamodule.predict_dataloader
        eval_stage = EvalStage.TEST

    results: list[EvaluationResult] = []
    for backend, backend_cfg in cfg.enabled_backends():
        if backend not in available:
            logger.warning(
                "Skipping evaluation of backend '%s': artifacts not available in this run.",
                backend.value,
            )
            continue
        logger.info("=" * 70)
        logger.info(
            "Evaluating backend '%s' on %s (num_samples=%d, num_warmup=%d)",
            backend.value,
            backend_cfg.device,
            cfg.num_samples,
            cfg.num_warmup,
        )
        results.append(
            evaluate_backend(
                model,
                make_dataloader(),
                pipelines.get(backend, backend_cfg.device),
                device,
                num_samples=cfg.num_samples,
                num_warmup=cfg.num_warmup,
                stage=eval_stage,
            )
        )
    log_comparison(results)
    return results


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main deployment function.

    Args:
        cfg: Hydra configuration
    """
    if "deploy" not in cfg:
        raise ValueError("Config must define a 'deploy' section.")
    # TODO(vividf): legacy ExportSpec fallback — remove once every model implements
    # build_stages(). The legacy schema is recognised by its `deploy.onnx.modules`
    # key, which the strict DeployConfig parser would reject.
    deploy_cfg = None
    if not is_legacy_deploy_config(cfg.deploy):
        deploy_cfg = DeployConfig.from_dict(OmegaConf.to_container(cfg.deploy, resolve=True))

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
        stage="deploy",
        experiment_name=experiment_name,
        config_name=config_name,
        experiment_uid=cfg.experiment_uid,
        logger_enabled=logger_enabled,
        parent_run_id=parent_run_id,
        extra_tags={
            "checkpoint_path": str(checkpoint_path),
            "source_run_id": parent_run_id or "",
            "source_checkpoint_count": str(len(source_checkpoints)),
            "source_run_ids": ",".join(source_run_ids),
        },
    )
    with mlflow_run_scope(run_context) as mlflow_client:
        _run_deployment(
            cfg,
            deploy_cfg,
            config_name=config_name,
            weights_path=weights_path,
            checkpoint_path=checkpoint_path,
            parent_run_id=parent_run_id,
            source_checkpoints=source_checkpoints,
            run_context=run_context,
            logger_enabled=logger_enabled,
            mlflow_client=mlflow_client,
        )


def is_legacy_deploy_config(deploy_cfg: DictConfig) -> bool:
    """True when the ``deploy`` section uses the legacy ``onnx.modules`` schema."""
    # TODO(vividf): legacy ExportSpec fallback — remove once every model implements
    # build_stages() and the last `deploy.onnx.modules` config is migrated to
    # `deploy.stages.<stage_name>`.
    onnx_cfg = deploy_cfg.get("onnx")
    return onnx_cfg is not None and "modules" in onnx_cfg


# TODO(vividf): legacy ExportSpec export path (export only) — delete once every
# BaseModel migrates to MultiTaskBaseModel.build_stages() (stage-graph export).
def run_legacy_export(
    deploy_cfg: DictConfig,
    output_dir: Path,
    datamodule: MultiTaskDataModule,
    model,
    device: torch.device,
) -> None:
    """Export a legacy ``BaseModel`` (ExportSpec contract) to ONNX / TensorRT.

    Verification and evaluation are stage-graph features and are skipped here.
    """
    logger.warning(
        "%s uses the legacy ExportSpec deploy path (deploy.onnx.modules schema): "
        "running export only — verification/evaluation require build_stages().",
        type(model).__name__,
    )
    export_specs = resolve_export_specs(datamodule, model, device)
    onnx_enabled = bool(deploy_cfg.onnx.get("enabled", True))
    tensorrt_enabled = bool(deploy_cfg.get("tensorrt", {}).get("enabled", False))
    for module_name, export_spec in export_specs.items():
        onnx_path = output_dir / f"{module_name}.onnx"
        engine_path = output_dir / f"{module_name}.engine"
        if onnx_enabled:
            if "onnx" not in export_spec.supported_stages:
                raise RuntimeError(
                    f"Module '{module_name}' does not support ONNX export but "
                    "deploy.onnx.enabled=true."
                )
            module_onnx_cfg = merge_module_onnx_cfg(deploy_cfg.onnx, module_name)
            export_to_onnx(
                export_spec.module,
                export_spec.args,
                module_onnx_cfg,
                export_spec.input_param_names,
                export_spec.output_names,
                export_spec.dynamic_axes,
                onnx_path,
            )
            logger.info("ONNX module: %s", onnx_path)
            modify_graph_cfg = module_onnx_cfg.get("modify_graph", None)
            if should_modify_graph(modify_graph_cfg):
                onnx_path = modify_onnx_graph(onnx_path, modify_graph_cfg)
        if tensorrt_enabled:
            if "tensorrt" not in export_spec.supported_stages:
                raise RuntimeError(
                    f"Module '{module_name}' does not support TensorRT export but "
                    "deploy.tensorrt.enabled=true."
                )
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX file not found: {onnx_path}. TensorRT export requires a valid ONNX model."
                )
            build_tensorrt_engine(onnx_path, deploy_cfg, engine_path)
            logger.info("TensorRT engine: %s", engine_path)


def _run_deployment(
    cfg: DictConfig,
    deploy_cfg: DeployConfig | None,
    *,
    config_name: str,
    weights_path,
    checkpoint_path,
    parent_run_id,
    source_checkpoints,
    run_context,
    logger_enabled: bool,
    mlflow_client: MlflowClient | None,
) -> None:
    """Run export, verification, and evaluation inside the MLflow run scope."""
    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)

    device = torch.device("cuda")
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))

    output_dir = cfg.get("experiment_run_dir", None)
    if run_context is not None:
        output_dir = str(run_context.exports_dir)
    if output_dir is None:
        raise ValueError(
            "Output directory must be specified in the configuration or obtained from MLflow run context."
        )
    output_dir = Path(output_dir)
    logger.info("Output directory for deployment: %s", output_dir)

    database = build_database(cfg)
    datamodule = build_datamodule(cfg, database=database)
    datamodule.prepare_data()
    datamodule.setup("predict")

    # FP or quantized: the checkpoint decides, build_model handles both.
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        weights_path=weights_path,
        resume_checkpoint_path=None,
        device=device,
        set_eval=True,
        enforce_full_coverage=True,
    )
    if deploy_cfg is None:
        # TODO(vividf): delete this branch with the last legacy BaseModel migration.
        run_legacy_export(cfg.deploy, output_dir, datamodule, model, device)
        return
    stages = model.build_stages()

    if run_context is not None:
        trainer_logger = build_trainer_logger(
            cfg,
            ml_flow_run_context=run_context,
            stage="deploy",
            config_name=config_name,
            logger_enabled=logger_enabled,
            extra_metadata={
                "source_run_id": parent_run_id,
                "checkpoint_path": str(checkpoint_path),
                "source_checkpoints": source_checkpoints,
            },
        )
        log_hyperparameters(cfg, trainer_logger)

    export(deploy_cfg, stages, output_dir, datamodule, model, device)

    available = available_backends(stages, output_dir)
    logger.info("Available backends: %s", sorted(b.value for b in available))
    pipelines = PipelineCache(stages, output_dir, assemble=model.assemble_predictions)

    verify(deploy_cfg, pipelines, datamodule, model, device, available)

    results = evaluate(deploy_cfg, pipelines, datamodule, model, device, available)
    if results and mlflow_client is not None and run_context is not None:
        log_results_to_mlflow(mlflow_client, run_context.run_id, results)


if __name__ == "__main__":
    main()
