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

"""Run a model's deployed function on one backend.

Graphs run their module or their exported artifact; the deployed function, the glue, runs in
PyTorch on every backend, so backends differ only in the graphs. A run starts from a
preprocessed batch and ends at the model outputs the function returns, with the latency of
each graph and of the glue.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from torch import nn

from autoware_ml.deployment.stages import (
    DeployedModel,
    DeploymentRun,
    GraphRunner,
    GraphStage,
    artifact_path,
    call_deployed_forward,
    run_graph_stage,
)
from autoware_ml.types.backend import Backend

if TYPE_CHECKING:
    from autoware_ml.deployment.backends.onnx_runner import OnnxModuleRunner
    from autoware_ml.deployment.backends.tensorrt_runner import TensorRTModuleRunner

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Raw outputs and latency of one pipeline run.

    Attributes:
        outputs: Model outputs keyed by name, as the deployed function returned them.
        output_names: Model output names, in the order the deployed function returned them.
        stage_times_ms: Latency per stage name.
        graph_stage_names: The graphs :attr:`model_ms` sums.
    """

    outputs: dict[str, torch.Tensor]
    output_names: tuple[str, ...]
    stage_times_ms: dict[str, float] = field(default_factory=dict)
    graph_stage_names: tuple[str, ...] = ()

    @property
    def model_ms(self) -> float:
        """Summed latency of the graphs (pure GPU time on TensorRT)."""
        return sum(self.stage_times_ms.get(name, 0.0) for name in self.graph_stage_names)

    def ordered_outputs(self) -> tuple[torch.Tensor, ...]:
        """Return the outputs in ``output_names`` order."""
        return tuple(self.outputs[name] for name in self.output_names)


@contextmanager
def cuda_synced_timer(
    times_ms: dict[str, float], stage: str, device: torch.device
) -> Iterator[None]:
    """Time a block with wall clock, synchronizing CUDA so async kernels are included."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    times_ms[stage] = times_ms.get(stage, 0.0) + (time.perf_counter() - start) * 1000.0


class _ModuleRunner:
    """Run a graph's own module (the pytorch backend), timed like an artifact runner."""

    def __init__(self, stage: GraphStage, device: torch.device) -> None:
        module = stage.module
        module_device = next(module.parameters(), torch.empty(0)).device
        if module_device != device:
            # Never move the shared model's modules: this pipeline gets its own copy.
            module = copy.deepcopy(module).to(device)
        self.module: nn.Module = module.eval()
        self.device = device
        self.stage = stage

    def run(self, inputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        args = tuple(inputs[name].to(self.device) for name in self.stage.inputs)
        times: dict[str, float] = {}
        with cuda_synced_timer(times, "run", self.device):
            outputs = run_graph_stage(self.stage, args)
        return outputs, times["run"]


#: What runs one graph: ``run(inputs) -> (outputs, ms)``.
Runner: TypeAlias = "_ModuleRunner | OnnxModuleRunner | TensorRTModuleRunner"


def normalize_device(device: str | torch.device) -> torch.device:
    """Return ``device`` with an explicit CUDA index, so ``cuda`` and ``cuda:0`` compare equal.

    Parameters report an indexed device: a bare ``cuda`` would make the pytorch backend
    deep-copy every stage module and the cache build a second pipeline for the same device.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device("cuda", index)
    return device


def _artifact_runner(
    stage: GraphStage,
    backend: Backend,
    device: torch.device,
    artifacts_dir: Path,
    plugin_libraries: Sequence[str] = (),
) -> Runner:
    path = artifact_path(artifacts_dir, stage.name, backend)
    if backend is Backend.ONNX:
        # Lazy imports: a pytorch pipeline needs neither onnxruntime nor tensorrt.
        from autoware_ml.deployment.backends.onnx_runner import OnnxModuleRunner

        return OnnxModuleRunner(path, device)
    if backend is Backend.TENSORRT:
        from autoware_ml.deployment.backends.tensorrt_runner import TensorRTModuleRunner

        return TensorRTModuleRunner(path, device, plugin_libraries)
    raise ValueError(f"No artifact runner for backend {backend}.")


#: Latency key of everything the deployed function does outside the graphs.
GLUE_STAGE = "glue"


class DeployedPipeline:
    """Run a model's deployed function on one backend.

    Args:
        deployment: The model's graphs and deployed function (``deployment_declaration()``).
        backend: Backend that runs the graphs.
        device: Device the graphs run on.
        artifacts_dir: Directory of the ``<graph>.onnx`` / ``<graph>.engine`` artifacts,
            required by the ``onnx`` and ``tensorrt`` backends.
        plugin_libraries: TensorRT plugin libraries the engines were built with.
    """

    def __init__(
        self,
        deployment: DeployedModel,
        backend: Backend,
        device: torch.device,
        artifacts_dir: str | Path | None = None,
        plugin_libraries: Sequence[str] = (),
    ) -> None:
        self.deployment = deployment
        self.graphs = deployment.graphs
        self.backend = backend
        self.plugin_libraries = tuple(plugin_libraries)
        self.device = normalize_device(device)
        self.graph_stage_names = tuple(graph.name for graph in self.graphs)

        self._runners: dict[str, Runner] = {}
        #: Graphs that fall back to PyTorch on this backend. Reports must name them, or a
        #: backend's numbers could silently be PyTorch's.
        self.fallback_stage_names: tuple[str, ...] = tuple(
            graph.name
            for graph in self.graphs
            if self.backend is not Backend.PYTORCH and self.backend in graph.torch_fallback_backends
        )
        for graph in self.graphs:
            if self.backend is Backend.PYTORCH or self.backend in graph.torch_fallback_backends:
                self._runners[graph.name] = _ModuleRunner(graph, self.device)
            else:
                if artifacts_dir is None:
                    raise ValueError(f"artifacts_dir is required for the {self.backend} backend.")
                self._runners[graph.name] = _artifact_runner(
                    graph, self.backend, self.device, Path(artifacts_dir), self.plugin_libraries
                )

    def _run(self, batch: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, float]]:
        """Run the deployed function once on this backend's runners.

        Returns:
            The model outputs, the call of each graph by name, and the latency of each graph
            and of the glue (:data:`GLUE_STAGE`: the function's time outside the graphs).
        """
        times: dict[str, float] = {}

        def runner_for(stage: GraphStage) -> GraphRunner:
            runner = self._runners[stage.name]

            def run(args: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
                outputs, elapsed_ms = runner.run(dict(zip(stage.inputs, args)))
                times[stage.name] = times.get(stage.name, 0.0) + elapsed_ms
                return outputs

            return run

        total: dict[str, float] = {}
        with cuda_synced_timer(total, GLUE_STAGE, self.device):
            outputs, calls = call_deployed_forward(self.deployment, runner_for, batch, self.device)
        graph_ms = sum(times.get(name, 0.0) for name in self.graph_stage_names)
        times[GLUE_STAGE] = max(total[GLUE_STAGE] - graph_ms, 0.0)
        return outputs, calls, times

    def infer(self, batch: Any) -> PipelineResult:
        """Run the deployed function; return the model outputs with the latency per graph."""
        outputs, _, times = self._run(batch)
        return PipelineResult(
            outputs=outputs,
            output_names=tuple(outputs),
            stage_times_ms=times,
            graph_stage_names=self.graph_stage_names,
        )

    def trace(self, batch: Any) -> DeploymentRun:
        """Run the deployed function; return its outputs and the inputs each graph ran on.

        Export derives its trace arguments from ``graph_inputs``; a ``pytorch`` pipeline runs
        the graph modules themselves, so the recorded inputs are what the graphs are traced on.

        Raises:
            ValueError: When the function modified a graph's input in place after the graph
                ran; export traces the graph on that tensor.
        """
        outputs, calls, _ = self._run(batch)
        modified = {name: call.modified_inputs() for name, call in calls.items()}
        modified = {name: names for name, names in modified.items() if names}
        if modified:
            raise ValueError(
                f"deployed_forward modified graph inputs in place after the graphs ran: "
                f"{modified}; export traces each graph on the tensors it ran on."
            )
        return DeploymentRun(outputs, {name: call.args for name, call in calls.items()})

    def assemble(
        self, result: PipelineResult, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        """Return the model outputs as ``build_deployed_eval_output`` takes them.

        The tensors are owned copies, floating-point ones widened to fp32.

        Args:
            result: Output of :meth:`infer`.
            device: Device to move the outputs to, e.g. the metrics device.
        """
        fields: dict[str, torch.Tensor] = {}
        for name in result.output_names:
            tensor = result.outputs[name]
            if device is not None:
                tensor = tensor.to(device)
            # Artifact runners reuse their output buffers on the next run, and a metric suite
            # may hold the tensor past that.
            tensor = tensor.detach().clone()
            if tensor.is_floating_point():
                tensor = tensor.float()
            fields[name] = tensor.contiguous()
        return fields


class PipelineCache:
    """Build each ``(backend, device)`` pipeline once and share it across consumers.

    There is no invalidation: use a new cache when the artifacts on disk change.
    """

    def __init__(
        self,
        deployment: DeployedModel,
        artifacts_dir: str | Path,
        plugin_libraries: Sequence[str] = (),
    ) -> None:
        self.deployment = deployment
        self.artifacts_dir = Path(artifacts_dir)
        self.plugin_libraries = tuple(plugin_libraries)
        self._pipelines: dict[tuple[Backend, str], DeployedPipeline] = {}

    def get(self, backend: Backend, device: str | torch.device) -> DeployedPipeline:
        """Return the cached pipeline for ``(backend, device)``, building it on first use."""
        device = normalize_device(device)
        key = (backend, str(device))
        if key not in self._pipelines:
            self._pipelines[key] = DeployedPipeline(
                self.deployment,
                backend=backend,
                device=device,
                artifacts_dir=self.artifacts_dir,
                plugin_libraries=self.plugin_libraries,
            )
        return self._pipelines[key]


def available_backends(deployment: DeployedModel, artifacts_dir: str | Path) -> set[Backend]:
    """Return the backends whose artifacts exist for every graph; PyTorch always counts.

    An engine older than its ONNX still counts, since artifacts may be reused, but is logged
    as stale: it may not reflect the current weights.
    """
    artifacts_dir = Path(artifacts_dir)

    def path_of(stage: GraphStage, backend: Backend) -> Path:
        return artifact_path(artifacts_dir, stage.name, backend)

    exportable = deployment.graphs
    available = {Backend.PYTORCH}
    for backend in (Backend.ONNX, Backend.TENSORRT):
        if all(
            backend in stage.torch_fallback_backends or path_of(stage, backend).exists()
            for stage in exportable
        ):
            available.add(backend)
    if Backend.TENSORRT in available:
        for stage in exportable:
            onnx_path = path_of(stage, Backend.ONNX)
            engine_path = artifact_path(artifacts_dir, stage.name, Backend.TENSORRT)
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
