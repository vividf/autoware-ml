import logging
from pathlib import Path
from typing import Sequence
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict
from omegaconf import DictConfig

from autoware_ml.datamodule.multi_task.multi_task_data_module import MultiTaskDataModule
from autoware_ml.models.multi_task_base_model import MultiTaskBaseModel
from autoware_ml.utils.deploy import (
    resolve_export_specs,
    merge_module_onnx_cfg,
    should_export_stage,
    supports_export_stage,
    export_to_onnx,
    should_modify_graph,
    modify_onnx_graph,
    build_tensorrt_engine,
)

logger = logging.getLogger(__name__)


class DeploymentExportOutputs(BaseModel):
    """Pydantic model for deployment export outputs."""

    model_config: ConfigDict = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    onnx_exported_paths: Sequence[Path]
    tensorrt_exported_paths: Sequence[Path]


class DeploymentExport:
    """Class for exporting a trained model for deployment."""

    def __init__(
        self,
        deploy_cfg: DictConfig,
        output_dir: Path,
        datamodule: MultiTaskDataModule,
        model: MultiTaskBaseModel,
    ) -> None:
        """ """
        self.deploy_cfg = deploy_cfg
        self.output_dir = output_dir
        self.datamodule = datamodule
        self.model = model
        self.device = self.model.device
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_export_specs(self) -> MappingProxyType[str, dict]:
        """Prepare export specifications for the model."""
        export_specs = resolve_export_specs(
            datamodule=self.datamodule, model=self.model, device=self.device
        )
        return MappingProxyType(export_specs)

    def export(self) -> DeploymentExportOutputs:
        """Export the trained model for deployment."""
        logger.info("Starting model export...")

        export_specs = self.prepare_export_specs()

        onnx_exported_paths: Sequence[Path] = []
        tensorrt_exported_paths: Sequence[Path] = []

        for module_name, export_spec in export_specs.items():
            module_onnx_cfg = merge_module_onnx_cfg(self.deploy_cfg.onnx, module_name)
            module_onnx_path = self.output_dir / f"{module_name}.onnx"
            module_engine_path = self.output_dir / f"{module_name}.engine"

            if should_export_stage(self.deploy_cfg.onnx):
                if not supports_export_stage(export_spec, "onnx"):
                    raise RuntimeError(
                        f"Module '{module_name}' does not support ONNX export but "
                        "deploy.onnx.enabled=true. Disable the stage or use a supported model."
                    )
                else:
                    export_to_onnx(
                        export_spec.module,
                        export_spec.args,
                        module_onnx_cfg,
                        export_spec.input_param_names,
                        export_spec.output_names,
                        export_spec.dynamic_axes,
                        module_onnx_path,
                    )
                    onnx_exported_paths.append(module_onnx_path)

                    modify_graph_cfg = module_onnx_cfg.get("modify_graph", None)
                    if should_modify_graph(modify_graph_cfg):
                        module_onnx_path = modify_onnx_graph(module_onnx_path, modify_graph_cfg)

            if should_export_stage(self.deploy_cfg.tensorrt):
                if not supports_export_stage(export_spec, "tensorrt"):
                    raise RuntimeError(
                        f"Module '{module_name}' does not support TensorRT export but "
                        "deploy.tensorrt.enabled=true. Disable the stage or use a supported model."
                    )
                else:
                    if not module_onnx_path.exists():
                        raise FileNotFoundError(
                            f"ONNX file not found: {module_onnx_path}. "
                            "TensorRT export requires a valid ONNX model."
                        )
                    build_tensorrt_engine(module_onnx_path, self.deploy_cfg, module_engine_path)
                    tensorrt_exported_paths.append(module_engine_path)

        # Implement the export logic here
        logger.info("Model export completed.")
        return DeploymentExportOutputs(
            onnx_exported_paths=onnx_exported_paths,
            tensorrt_exported_paths=tensorrt_exported_paths,
        )
