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

"""Export specs derived from a deployment declaration."""

from __future__ import annotations

from typing import Any

import torch

from autoware_ml.deployment.stages import DeployedModel, GraphStage, run_deployment_in_torch
from autoware_ml.utils.deploy import ExportSpec


def export_spec_for_graph(stage: GraphStage, args: tuple[torch.Tensor, ...]) -> ExportSpec:
    """Build one graph's spec from the tensors it ran on.

    The trace inputs are ``args``, the tensors the deployed function passed the graph. The spec
    takes the graph's declared input and output names. Its ``dynamic_axes`` are the graph's
    ``onnx_dynamic_axes``, or ``None`` when the graph declares none.
    """
    dynamic_axes = (
        {name: dict(axes) for name, axes in stage.onnx_dynamic_axes.items()}
        if stage.onnx_dynamic_axes
        else None
    )
    return ExportSpec(
        module=stage.module,
        args=args,
        input_names=list(stage.inputs),
        output_names=list(stage.outputs),
        dynamic_axes=dynamic_axes,
    )


def build_export_specs_from_deployment(
    deployment: DeployedModel, batch: Any, device: torch.device
) -> dict[str, ExportSpec]:
    """Run the deployed function once and return ``{graph name: ExportSpec}`` per graph.

    The graph modules are switched to eval first. An export wrapper starts in training
    mode, and ``torch.onnx.export`` restores the module's mode recursively when it is done:
    a wrapper left in training mode would switch the submodules it shares with the model,
    BatchNorm included, to training mode.

    Args:
        deployment: The model's graphs and deployed function.
        batch: One preprocessed example batch.
        device: Device of the tracing run.
    """
    for stage in deployment.graphs:
        stage.module.eval()
    run = run_deployment_in_torch(deployment, batch, device)
    return {
        stage.name: export_spec_for_graph(stage, run.graph_inputs[stage.name])
        for stage in deployment.graphs
    }
