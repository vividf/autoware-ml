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

"""Export specs derived from a stage graph."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from autoware_ml.deployment.pipeline import StagedPipeline
from autoware_ml.deployment.stages import GraphStage, RunState, Stage, graph_stages
from autoware_ml.types.backend import Backend
from autoware_ml.utils.deploy import ExportSpec


def export_spec_for_stage(stage: GraphStage, state: RunState) -> ExportSpec:
    """Build one graph stage's spec from a completed run of the stage graph.

    The trace inputs are the tensors the stage ran on (``state.graph_inputs``). The spec takes
    the stage's declared input and output names and its ``onnx_transforms``. Its
    ``dynamic_axes`` are the stage's
    ``onnx_dynamic_axes``, or ``None`` so that the module's config entry applies.
    """
    dynamic_axes = (
        {name: dict(axes) for name, axes in stage.onnx_dynamic_axes.items()}
        if stage.onnx_dynamic_axes
        else None
    )
    return ExportSpec(
        # BatchNorm stays a graph node (see utils/bn_fusion.py); a path that needs folded
        # weights folds its own copy before it gets here.
        module=stage.module,
        args=state.graph_inputs[stage.name],
        input_names=list(stage.inputs),
        output_names=list(stage.outputs),
        dynamic_axes=dynamic_axes,
        onnx_transforms=tuple(stage.onnx_transforms),
    )


def build_export_specs_from_stages(
    stages: Sequence[Stage], batch: Any, device: torch.device
) -> dict[str, ExportSpec]:
    """Trace the stage graph once in PyTorch and return ``{stage name: ExportSpec}`` per graph stage.

    The graph modules are switched to eval first. An export wrapper starts in training
    mode, and ``torch.onnx.export`` restores the module's mode recursively when it is done:
    a wrapper left in training mode would switch the submodules it shares with the model,
    BatchNorm included, to training mode.

    Args:
        stages: The model's ``build_stages()``.
        batch: One preprocessed example batch.
        device: Device of the tracing run.
    """
    pipeline = StagedPipeline(stages, Backend.PYTORCH, device)
    for stage in graph_stages(pipeline.stages):
        stage.module.eval()
    state = pipeline.trace(batch)
    return {
        stage.name: export_spec_for_stage(stage, state) for stage in graph_stages(pipeline.stages)
    }
