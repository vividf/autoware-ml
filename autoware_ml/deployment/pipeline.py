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

"""Run a model's stage graph on one backend.

Graph stages run their module or their exported artifact; glue stages run in PyTorch on every
backend, so backends differ only in the graphs. A run starts from a preprocessed batch and
ends at the model outputs, with a per-stage latency breakdown.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from autoware_ml.deployment.stages import (
    GraphStage,
    RunState,
    Stage,
    TorchStage,
    artifact_path,
    graph_input_tensors,
    graph_stages,
    model_output_names,
    run_graph_stage,
    run_torch_stage,
    validate_stages,
)
from autoware_ml.types.backend import Backend

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Raw outputs and latency of one pipeline run.

    Attributes:
        outputs: Model outputs keyed by ONNX output name.
        output_names: Model output names in declaration order.
        stage_times_ms: Latency per stage name.
        graph_stage_names: The stages :attr:`model_ms` sums.
    """

    outputs: dict[str, torch.Tensor]
    output_names: list[str]
    stage_times_ms: dict[str, float] = field(default_factory=dict)
    graph_stage_names: tuple[str, ...] = ()

    @property
    def model_ms(self) -> float:
        """Summed latency of the graph stages (pure GPU time on TensorRT)."""
        return sum(self.stage_times_ms.get(name, 0.0) for name in self.graph_stage_names)

    def ordered_outputs(self) -> list[torch.Tensor]:
        """Return the outputs in ``output_names`` order."""
        return [self.outputs[name] for name in self.output_names]


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
    """Run a GraphStage's own module (the pytorch backend), timed like an artifact runner."""

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
    onnx_paths: Mapping[str, Path] | None = None,
):
    path = artifact_path(artifacts_dir, stage.name, backend)
    if backend is Backend.ONNX and onnx_paths and stage.name in onnx_paths:
        path = Path(onnx_paths[stage.name])  # the graph the export loop actually shipped
    if backend is Backend.ONNX:
        # Lazy imports: a pytorch pipeline needs neither onnxruntime nor tensorrt.
        from autoware_ml.deployment.backends.onnx_runner import OnnxModuleRunner

        return OnnxModuleRunner(path, device)
    if backend is Backend.TENSORRT:
        from autoware_ml.deployment.backends.tensorrt_runner import TensorRTModuleRunner

        return TensorRTModuleRunner(path, device, plugin_libraries)
    raise ValueError(f"No artifact runner for backend {backend}.")


class StagedPipeline:
    """Execute a stage graph on one backend.

    Args:
        stages: The model's ``build_stages()``.
        backend: Backend that runs the graph stages.
        device: Device the graph stages run on.
        artifacts_dir: Directory of the ``<stage>.onnx`` / ``<stage>.engine`` artifacts,
            required by the ``onnx`` and ``tensorrt`` backends.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        backend: Backend,
        device: torch.device,
        artifacts_dir: str | Path | None = None,
        plugin_libraries: Sequence[str] = (),
        onnx_paths: Mapping[str, Path] | None = None,
    ) -> None:
        self.stages = validate_stages(stages)
        self.backend = backend
        self.plugin_libraries = tuple(plugin_libraries)
        self.onnx_paths = dict(onnx_paths or {})
        self.device = normalize_device(device)
        self.output_names = list(model_output_names(self.stages))
        self.graph_stage_names = tuple(stage.name for stage in graph_stages(self.stages))

        self._runners: dict[str, Any] = {}
        #: Graph stages that fall back to PyTorch on this backend. Reports must name them, or
        #: a backend's numbers could silently be PyTorch's.
        self.fallback_stage_names: tuple[str, ...] = tuple(
            stage.name
            for stage in graph_stages(self.stages)
            if self.backend is not Backend.PYTORCH and self.backend in stage.torch_fallback_backends
        )
        for stage in graph_stages(self.stages):
            if self.backend is Backend.PYTORCH or self.backend in stage.torch_fallback_backends:
                self._runners[stage.name] = _ModuleRunner(stage, self.device)
            else:
                if artifacts_dir is None:
                    raise ValueError(f"artifacts_dir is required for the {self.backend} backend.")
                self._runners[stage.name] = _artifact_runner(
                    stage,
                    self.backend,
                    self.device,
                    Path(artifacts_dir),
                    self.plugin_libraries,
                    self.onnx_paths,
                )

    def run(self, batch: Any) -> tuple[PipelineResult, RunState]:
        """Run every stage; return the result and the run state holding every named tensor."""
        state = RunState(batch=batch, device=self.device)
        times: dict[str, float] = {}
        with torch.no_grad():
            for stage in self.stages:
                if isinstance(stage, TorchStage):
                    with cuda_synced_timer(times, stage.name, self.device):
                        produced = run_torch_stage(stage, state)
                    state.add(produced)
                else:
                    inputs = dict(zip(stage.inputs, graph_input_tensors(stage, state)))
                    outputs, elapsed_ms = self._runners[stage.name].run(inputs)
                    times[stage.name] = times.get(stage.name, 0.0) + elapsed_ms
                    state.add({name: outputs[name] for name in stage.outputs})
        result = PipelineResult(
            outputs={name: state[name] for name in self.output_names},
            output_names=list(self.output_names),
            stage_times_ms=times,
            graph_stage_names=self.graph_stage_names,
        )
        return result, state

    def infer(self, batch: Any) -> PipelineResult:
        """Run every stage and return the final raw outputs plus timing."""
        result, _ = self.run(batch)
        return result

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
        for name in self.output_names:
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
        stages: Sequence[Stage],
        artifacts_dir: str | Path,
        plugin_libraries: Sequence[str] = (),
        onnx_paths: Mapping[str, Path] | None = None,
    ) -> None:
        self.stages = validate_stages(stages)
        self.artifacts_dir = Path(artifacts_dir)
        self.plugin_libraries = tuple(plugin_libraries)
        #: Stage name -> the ``.onnx`` the export loop shipped, where a graph modifier wrote it
        #: under a name other than ``<stage>.onnx``.
        self.onnx_paths = dict(onnx_paths or {})
        self._pipelines: dict[tuple[Backend, str], StagedPipeline] = {}

    def get(self, backend: Backend, device: str | torch.device) -> StagedPipeline:
        """Return the cached pipeline for ``(backend, device)``, building it on first use."""
        device = normalize_device(device)
        key = (backend, str(device))
        if key not in self._pipelines:
            self._pipelines[key] = StagedPipeline(
                self.stages,
                backend=backend,
                device=device,
                artifacts_dir=self.artifacts_dir,
                plugin_libraries=self.plugin_libraries,
                onnx_paths=self.onnx_paths,
            )
        return self._pipelines[key]


def available_backends(
    stages: Sequence[Stage],
    artifacts_dir: str | Path,
    onnx_paths: Mapping[str, Path] | None = None,
) -> set[Backend]:
    """Return the backends whose artifacts exist for every graph stage; PyTorch always counts.

    An engine older than its ONNX still counts, since artifacts may be reused, but is logged
    as stale: it may not reflect the current weights.
    """
    artifacts_dir = Path(artifacts_dir)
    onnx_paths = dict(onnx_paths or {})

    def path_of(stage: GraphStage, backend: Backend) -> Path:
        if backend is Backend.ONNX and stage.name in onnx_paths:
            return Path(onnx_paths[stage.name])
        return artifact_path(artifacts_dir, stage.name, backend)

    exportable = graph_stages(stages)
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
