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

"""LEGACY single-module export contract (``ExportSpec``) and its DictConfig adapters.

TODO(vividf): delete this whole module once every legacy ``BaseModel`` (ptv3 / frnet /
transfusion / bevfusion / calibration_status) migrates to
``MultiTaskBaseModel.build_stages()`` (design doc Q5). Everything current lives in
:mod:`autoware_ml.deployment.onnx` (ONNX primitives) and
:mod:`autoware_ml.deployment.backends.tensorrt_builder` (engine build); the wrappers
here only adapt the legacy ``deploy.onnx.modules`` DictConfig schema onto them.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from typing import Any

import lightning as L
from omegaconf import DictConfig, OmegaConf
import torch

from autoware_ml.deployment.backends.tensorrt_builder import build_engine
from autoware_ml.deployment.config import ShapeProfile
from autoware_ml.deployment.onnx.export import export_to_onnx as _export_to_onnx
from autoware_ml.deployment.onnx.modify import (  # noqa: F401  (legacy re-exports)
    modify_onnx_graph,
    should_modify_graph,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportSpec:
    """Describe the module and tensor inputs used for legacy single-module export.

    Attributes:
        module: Module instance exported to ONNX.
        args: Example positional inputs supplied during export.
        input_param_names: Names associated with the positional input tensors.
        output_names: Optional names associated with exported output tensors.
        dynamic_axes: Optional legacy ONNX dynamic-axis mapping generated with
            the export arguments. Used only when exporting with ``dynamo=False``.
        supported_stages: Export stages supported by this specification.
    """

    module: torch.nn.Module
    args: tuple[Any, ...]
    input_param_names: list[str]
    output_names: list[str] | None = None
    dynamic_axes: dict[str, dict[int, str]] | None = None
    supported_stages: frozenset[str] = frozenset({"onnx", "tensorrt"})


def get_forward_signature(model: L.LightningModule) -> inspect.Signature:
    """Return the cached forward signature from BaseModel, or compute it."""
    return getattr(model, "forward_signature", inspect.signature(model.forward))


def get_export_parameter_names(model: L.LightningModule) -> list[str]:
    """Return concrete forward parameter names used for export."""
    signature = get_forward_signature(model)
    return [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def extract_input_from_batch(batch: dict[str, Any], param_name: str) -> Any:
    """Extract one export input from a batch dictionary."""
    if param_name not in batch:
        raise ValueError(
            f"Parameter '{param_name}' not found in batch. Available keys: {list(batch.keys())}"
        )

    input_value = batch[param_name]
    if isinstance(input_value, (list, tuple)):
        input_value = input_value[0]
    return input_value


def infer_export_spec(model: L.LightningModule, batch: dict[str, Any]) -> ExportSpec:
    """Infer an export specification directly from the model forward signature."""
    forward_params = get_export_parameter_names(model)
    if not forward_params:
        raise ValueError("Model forward signature has no parameters.")

    if (
        isinstance(batch, dict)
        and len(forward_params) == 1
        and forward_params[0] == "batch_inputs_dict"
    ):
        return ExportSpec(module=model, args=(batch,), input_param_names=forward_params)

    input_args = tuple(extract_input_from_batch(batch, param_name) for param_name in forward_params)
    return ExportSpec(module=model, args=input_args, input_param_names=forward_params)


def get_predict_batch(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
) -> dict[str, Any]:
    """Load one prediction batch and apply transfer-time preprocessing."""
    datamodule.setup("predict")
    predict_dataloader = datamodule.predict_dataloader()
    batch = next(iter(predict_dataloader))
    batch = batch.to_device(device)
    return model.on_after_batch_transfer(batch, dataloader_idx=0)


def resolve_export_specs(
    datamodule: L.LightningDataModule,
    model: L.LightningModule,
    device: torch.device,
) -> dict[str, ExportSpec]:
    """Resolve per-module export specifications for a legacy ``BaseModel``."""
    batch = get_predict_batch(datamodule, model, device)
    return model.build_export_specs(batch)


def merge_module_onnx_cfg(onnx_cfg: DictConfig, module_name: str) -> DictConfig:
    """Merge shared ONNX settings with per-module overrides (legacy ``onnx.modules`` schema).

    Raises:
        KeyError: If ``module_name`` is not found in ``onnx_cfg.modules``.
    """
    if "modules" not in onnx_cfg or module_name not in onnx_cfg.modules:
        raise KeyError(
            f"Module '{module_name}' not found in deploy.onnx.modules. "
            f"Available: {list(onnx_cfg.get('modules', {}).keys())}"
        )
    shared = {
        k: v for k, v in OmegaConf.to_container(onnx_cfg, resolve=True).items() if k != "modules"
    }
    module_overrides = OmegaConf.to_container(onnx_cfg.modules[module_name], resolve=True)
    return OmegaConf.create({**shared, **(module_overrides or {})})


def export_to_onnx(
    model: torch.nn.Module,
    input_sample: tuple[Any, ...],
    onnx_cfg: DictConfig,
    input_param_names: list[str],
    output_names_override: list[str] | None,
    dynamic_axes_override: dict[str, dict[int, str]] | None,
    output_path: Path,
) -> None:
    """Legacy DictConfig adapter over :func:`autoware_ml.deployment.onnx.export.export_to_onnx`."""
    if not input_param_names:
        raise ValueError("Model forward signature has no parameters.")
    _export_to_onnx(
        model,
        tuple(input_sample),
        output_path,
        input_names=list(onnx_cfg.get("input_names", input_param_names)),
        output_names=list(output_names_override or onnx_cfg.get("output_names", ["output"])),
        opset_version=int(onnx_cfg.opset_version),
        dynamo=bool(onnx_cfg.get("dynamo", True)),
        do_constant_folding=bool(onnx_cfg.get("do_constant_folding", True)),
        dynamic_shapes=onnx_cfg.get("dynamic_shapes"),
        dynamic_axes=dynamic_axes_override or onnx_cfg.get("dynamic_axes"),
    )


def build_tensorrt_engine(
    onnx_path: Path,
    deploy_cfg: DictConfig,
    output_path: Path,
) -> None:
    """Legacy DictConfig adapter over :func:`...backends.tensorrt_builder.build_engine`."""
    tensorrt_cfg = deploy_cfg.tensorrt
    policy = tensorrt_cfg.get("precision_policy")
    if policy is not None and str(policy).lower() != "strongly_typed":
        logger.warning(
            "Ignoring legacy deploy.tensorrt.precision_policy=%r: engines always build "
            "strongly typed now (precision lives in the ONNX graph).",
            policy,
        )
    raw_shapes = tensorrt_cfg.get("input_shapes") or {}
    input_shapes = {
        str(name): ShapeProfile.from_dict(profile, f"deploy.tensorrt.input_shapes.{name}")
        for name, profile in raw_shapes.items()
    }
    plugin_libraries = tensorrt_cfg.get("plugin_libraries", None)
    build_engine(
        onnx_path,
        output_path,
        workspace_size=int(tensorrt_cfg.get("workspace_size", 1 << 30)),
        plugin_libraries=list(plugin_libraries) if plugin_libraries is not None else (),
        input_shapes=input_shapes,
    )


def supports_export_stage(export_spec: ExportSpec, stage_name: str) -> bool:
    """Return whether an export specification supports a stage."""
    return stage_name in export_spec.supported_stages
