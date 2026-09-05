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

"""Export every exportable stage of a model to ONNX (and optionally TensorRT).

The export units, their trace inputs, and their input/output names are all
*derived* from the stage graph: the pytorch pipeline runs once on a tracing batch,
and each :class:`~autoware_ml.deployment.stages.GraphStage` is traced with the
context tensors it declares as inputs. Nothing about the model is spelled out here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.deployment.backends.tensorrt_builder import build_engine
from autoware_ml.deployment.config import DeployConfig, OnnxPrecision
from autoware_ml.deployment.onnx.export import export_to_onnx
from autoware_ml.deployment.onnx.modify import modify_onnx_graph, should_modify_graph
from autoware_ml.deployment.onnx.autocast import autocast_to_fp16
from autoware_ml.deployment.onnx.precision import (
    cast_graph_to_fp16,
    onnx_custom_op_domains,
    onnx_has_qdq,
)
from autoware_ml.deployment.pipeline import StagedPipeline
from autoware_ml.deployment.stages import GraphStage, Stage, artifact_path, graph_stages
from autoware_ml.quantization.core.fusion import find_conv_bn_pairs, fuse_model_bn
from autoware_ml.types.backend import Backend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportedArtifacts:
    """Artifacts written by :func:`export_stages`, keyed by stage name."""

    onnx: dict[str, Path] = field(default_factory=dict)
    engines: dict[str, Path] = field(default_factory=dict)


def _bn_folded_for_export(stage: GraphStage) -> nn.Module:
    """Return the stage module for export, with BatchNorm folded on a deep copy.

    Deployed graphs carry no BatchNormalization node (an inference identity; ONNX
    Runtime's CUDA kernel also rejects the PFN's flattened-2D BN). Quantized trees
    fold during ``prepare`` and come back unchanged.
    """
    pairs = find_conv_bn_pairs(stage.module)
    if not pairs:
        return stage.module
    folded = copy.deepcopy(stage.module).eval()
    fuse_model_bn(folded)
    logger.info("Folded %d BN layer(s) into the export copy of stage %r", len(pairs), stage.name)
    return folded


def export_stages(
    stages: Sequence[Stage],
    batch_inputs: MultiTaskBatchInputs,
    deploy_cfg: DeployConfig,
    output_dir: str | Path,
    device: torch.device,
) -> ExportedArtifacts:
    """Export every ``GraphStage`` of ``stages`` according to ``deploy_cfg``.

    Args:
        stages: The model's stage declaration.
        batch_inputs: One preprocessed batch used to trace the graph stages.
        deploy_cfg: Parsed ``deploy`` section.
        output_dir: Directory receiving ``<stage>.onnx`` / ``<stage>.engine``.
        device: Device the tracing run executes on.

    Returns:
        The written artifact paths.

    Raises:
        FileNotFoundError: When TensorRT is enabled but a stage's ONNX is missing (ONNX
            export disabled and no earlier export in ``output_dir``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exportable = graph_stages(stages)
    deploy_cfg.check_stage_names([stage.name for stage in exportable])

    # One pytorch run gives every graph stage its trace inputs (the context tensors it reads).
    pipeline = StagedPipeline(stages, backend=Backend.PYTORCH, device=device)
    _, context = pipeline.run(batch_inputs)

    artifacts = ExportedArtifacts()
    for stage in exportable:
        onnx_path = artifact_path(output_dir, stage.name, Backend.ONNX)
        engine_path = artifact_path(output_dir, stage.name, Backend.TENSORRT)

        stage_cfg = deploy_cfg.stage(stage.name)

        if deploy_cfg.onnx.enabled:
            logger.info("Exporting stage %r -> %s", stage.name, onnx_path)
            export_to_onnx(
                _bn_folded_for_export(stage),
                tuple(context[name] for name in stage.inputs),
                onnx_path,
                input_names=stage.inputs,
                output_names=stage.outputs,
                opset_version=deploy_cfg.onnx.opset_version,
                dynamo=deploy_cfg.onnx.dynamo,
                do_constant_folding=deploy_cfg.onnx.do_constant_folding,
                dynamic_shapes=stage_cfg.onnx.dynamic_shapes,
                # The config wins when it says something; otherwise the graph's own
                # intrinsic axes apply (a point graph has no static point count).
                dynamic_axes=stage_cfg.onnx.dynamic_axes or (stage.onnx_dynamic_axes or None),
            )
            stage_precision = stage_cfg.onnx.precision or deploy_cfg.onnx.precision
            if stage_precision is OnnxPrecision.FP16:
                custom_domains = onnx_custom_op_domains(onnx_path)
                has_qdq = onnx_has_qdq(onnx_path)
                if custom_domains:
                    # AutoCast cannot type a plugin op (it infers types with TensorRT's
                    # parser), so plugin graphs take the whole-graph FP16 cast — which is
                    # Q/DQ-aware: the quantization islands stay FP32 exactly as the
                    # checkpoint calibrated them, everything around them stops running
                    # FP32 (the point for a model whose plugin ops dominate its runtime).
                    logger.info(
                        "Stage %r uses runtime plugin ops (%s) — applying the whole-graph "
                        "FP16 cast%s.",
                        stage.name,
                        ", ".join(custom_domains),
                        " around its Q/DQ islands" if has_qdq else "",
                    )
                    cast_graph_to_fp16(onnx_path)
                elif has_qdq:
                    # AutoCast does not support Q/DQ models, but leaving the rest of the
                    # graph FP32 makes every *un*-quantized region pay full price: with
                    # CenterPoint's release recipe (backbone stage 0 deliberately kept out
                    # of INT8) the as-is build ran that stage FP32 and cost the whole INT8
                    # speedup back (backbone_neck_head 5.91 ms vs 2.50 all-INT8). So a
                    # plugin-free Q/DQ graph takes the same island-aware whole-graph FP16
                    # cast as plugin graphs: islands stay FP32 exactly as calibrated,
                    # everything else runs FP16 (the pass PTv3 measured 25.7 -> 7.05 ms
                    # with unchanged mIoU).
                    logger.info(
                        "Stage %r carries Q/DQ nodes — applying the whole-graph FP16 cast "
                        "around its Q/DQ islands (AutoCast rejects quantized graphs).",
                        stage.name,
                    )
                    cast_graph_to_fp16(onnx_path)
                else:
                    autocast_to_fp16(onnx_path, {name: context[name] for name in stage.inputs})
            for transform in stage.onnx_transforms:
                onnx_path = Path(transform(onnx_path))
            if should_modify_graph(deploy_cfg.onnx.modify_graph):
                onnx_path = modify_onnx_graph(onnx_path, deploy_cfg.onnx.modify_graph)
            artifacts.onnx[stage.name] = onnx_path

        if deploy_cfg.tensorrt.enabled:
            if Backend.TENSORRT in stage.torch_fallback_backends:
                # The stage declares that TensorRT cannot execute it (a missing runtime
                # plugin, say), and the pipeline runs it in PyTorch on that backend, so
                # an engine would fail to build and would never be used. Drop the
                # backend from the stage's fallbacks to start building one.
                logger.info(
                    "Stage %r falls back to torch on tensorrt — skipping its engine build.",
                    stage.name,
                )
                continue
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX for stage {stage.name!r} not found at {onnx_path}; TensorRT needs it "
                    "(enable deploy.onnx or export into a directory holding an earlier export)."
                )
            logger.info("Building TensorRT engine for stage %r -> %s", stage.name, engine_path)
            build_engine(
                onnx_path,
                engine_path,
                workspace_size=deploy_cfg.tensorrt.workspace_size,
                plugin_libraries=deploy_cfg.tensorrt.plugin_libraries,
                input_shapes=stage_cfg.tensorrt.input_shapes,
            )
            artifacts.engines[stage.name] = engine_path

    return artifacts


def available_backends(stages: Sequence[Stage], output_dir: str | Path) -> set[Backend]:
    """Backends whose artifacts exist for every exportable stage.

    PyTorch is always available. A ``.engine`` older than its sibling ``.onnx`` still
    counts (artifact reuse is a feature) but is flagged loudly: its verification and
    evaluation results may not reflect the current weights.
    """
    output_dir = Path(output_dir)
    exportable = graph_stages(stages)
    available = {Backend.PYTORCH}
    for backend in (Backend.ONNX, Backend.TENSORRT):
        if all(
            backend in stage.torch_fallback_backends
            or artifact_path(output_dir, stage.name, backend).exists()
            for stage in exportable
        ):
            available.add(backend)
    if Backend.TENSORRT in available:
        for stage in exportable:
            onnx_path = artifact_path(output_dir, stage.name, Backend.ONNX)
            engine_path = artifact_path(output_dir, stage.name, Backend.TENSORRT)
            if (
                onnx_path.exists()
                and engine_path.exists()
                and engine_path.stat().st_mtime < onnx_path.stat().st_mtime
            ):
                logger.warning(
                    "STALE TENSORRT ENGINE: %s is older than its sibling ONNX %s — it likely "
                    "predates the current export. Re-enable tensorrt export (or delete the "
                    ".engine) to rebuild it.",
                    engine_path,
                    onnx_path,
                )
    return available
