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

:class:`StagedPipeline` is the generic executor of a :mod:`~autoware_ml.deployment.stages`
declaration. Glue stages always run their PyTorch callable; exportable stages run either
their module (``pytorch`` backend) or the runner of their exported artifact
(``onnx`` / ``tensorrt``). Every backend therefore consumes and produces exactly the
same named tensors, so cross-backend differences are pure backend differences.

Preprocessing (``model.preprocess_batch``) and decoding/metrics stay outside the
pipeline: it starts from preprocessed inputs and ends at the final graph stage's raw
outputs plus a per-stage latency breakdown.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Iterator, Sequence

import torch
from torch import nn

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.deployment.stages import (
    GraphStage,
    Stage,
    StageContext,
    TorchStage,
    artifact_path,
    final_stage,
    graph_stages,
    validate_stages,
)
from autoware_ml.types.backend import Backend

AssembleFn = Callable[[Mapping[str, torch.Tensor]], MultiTaskPredictions]


@dataclass
class PipelineResult:
    """Raw outputs of one pipeline run.

    Attributes:
        outputs: Final graph stage's output tensors keyed by their ONNX output name.
        output_names: Output names in the declared (frozen ABI) order.
        stage_times_ms: Per-stage latency in milliseconds.
        graph_stage_names: Names of the exportable stages; :attr:`model_ms` sums these
            (pure GPU time for TensorRT).
    """

    outputs: dict[str, torch.Tensor]
    output_names: list[str]
    stage_times_ms: dict[str, float] = field(default_factory=dict)
    graph_stage_names: tuple[str, ...] = ()

    @property
    def model_ms(self) -> float:
        """Summed latency of the exportable stages."""
        return sum(self.stage_times_ms.get(name, 0.0) for name in self.graph_stage_names)

    def ordered_outputs(self) -> list[torch.Tensor]:
        """Return the outputs in the frozen ABI order."""
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
            raw = self.module(*args)
        if isinstance(raw, torch.Tensor):
            raw = (raw,)
        if len(raw) != len(self.stage.outputs):
            raise ValueError(
                f"GraphStage {self.stage.name!r} returned {len(raw)} tensor(s) but declares "
                f"outputs {list(self.stage.outputs)}."
            )
        return dict(zip(self.stage.outputs, raw)), times["run"]


def _artifact_runner(
    stage: GraphStage, backend: Backend, device: torch.device, artifacts_dir: Path
):
    path = artifact_path(artifacts_dir, stage.name, backend)
    if backend is Backend.ONNX:
        from autoware_ml.deployment.backends.onnx_runner import OnnxModuleRunner

        return OnnxModuleRunner(path, device)
    if backend is Backend.TENSORRT:
        from autoware_ml.deployment.backends.tensorrt_runner import TensorRTModuleRunner

        return TensorRTModuleRunner(path, device)
    raise ValueError(f"No artifact runner for backend {backend}.")


class StagedPipeline:
    """Execute a stage graph on one backend.

    Args:
        stages: The model's stage declaration (``model.build_stages()``).
        backend: Backend that runs the exportable stages.
        device: Device the exportable stages execute on; glue stages hand their results
            over on this device.
        artifacts_dir: Directory holding ``<stage>.onnx`` / ``.engine`` (non-pytorch backends).
        assemble: The model's ``assemble_predictions`` hook, used by :meth:`assemble`.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        backend: str | Backend,
        device: torch.device,
        artifacts_dir: str | Path | None = None,
        assemble: AssembleFn | None = None,
    ) -> None:
        self.stages = validate_stages(stages)
        self.backend = Backend.parse(backend)
        self.device = torch.device(device)
        self._assemble = assemble
        self.final_stage = final_stage(self.stages)
        self.output_names = list(self.final_stage.outputs)
        self.graph_stage_names = tuple(stage.name for stage in graph_stages(self.stages))

        self._runners: dict[str, Any] = {}
        #: Graph stages that run their PyTorch module on this (non-pytorch) backend
        #: because they declare it as a fallback — reports must say so, or a backend
        #: column can silently be the pytorch numbers under another name.
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
                    raise ValueError(
                        f"artifacts_dir is required for the {self.backend.value} backend."
                    )
                self._runners[stage.name] = _artifact_runner(
                    stage, self.backend, self.device, Path(artifacts_dir)
                )

    def run(self, batch_inputs: MultiTaskBatchInputs) -> tuple[PipelineResult, StageContext]:
        """Run every stage and return the result together with the full context.

        The context (every named tensor produced along the way) is what export uses
        as trace inputs for each graph stage.
        """
        context = StageContext(batch_inputs=batch_inputs, device=self.device)
        times: dict[str, float] = {}
        with torch.no_grad():
            for stage in self.stages:
                if isinstance(stage, TorchStage):
                    with cuda_synced_timer(times, stage.name, self.device):
                        produced = stage.run(context)
                    context.tensors.update(produced)
                else:
                    inputs = {name: context[name] for name in stage.inputs}
                    outputs, elapsed_ms = self._runners[stage.name].run(inputs)
                    times[stage.name] = times.get(stage.name, 0.0) + elapsed_ms
                    context.tensors.update({name: outputs[name] for name in stage.outputs})
        result = PipelineResult(
            outputs={name: context[name] for name in self.output_names},
            output_names=list(self.output_names),
            stage_times_ms=times,
            graph_stage_names=self.graph_stage_names,
        )
        return result, context

    def infer(self, batch_inputs: MultiTaskBatchInputs) -> PipelineResult:
        """Run every stage and return the final raw outputs plus timing."""
        result, _ = self.run(batch_inputs)
        return result

    def assemble(
        self, result: PipelineResult, device: torch.device | None = None
    ) -> MultiTaskPredictions:
        """Turn a result into the model's predictions (what metrics consume).

        Args:
            result: Output of :meth:`infer`.
            device: Optional device to move the tensors to first (e.g. the metrics device).
        """
        if self._assemble is None:
            raise RuntimeError(
                "StagedPipeline was built without the model's assemble_predictions hook."
            )
        fields: dict[str, torch.Tensor] = {}
        for onnx_name, field_name in self.final_stage.output_fields:
            tensor = result.outputs[onnx_name]
            if device is not None:
                tensor = tensor.to(device)
            fields[field_name] = tensor.float().contiguous()
        return self._assemble(fields)


class PipelineCache:
    """Build each ``(backend, device)`` pipeline once and share it across stages.

    Verification and evaluation both need pipelines for the same backends; loading
    an ONNX session or deserializing a TensorRT engine twice per deploy run is waste.
    One instance lives for one deploy run over one set of artifacts — there is no
    invalidation: if an artifact on disk changes, use a fresh cache.
    """

    def __init__(
        self,
        stages: Sequence[Stage],
        artifacts_dir: str | Path,
        assemble: AssembleFn,
    ) -> None:
        self.stages = validate_stages(stages)
        self.artifacts_dir = Path(artifacts_dir)
        self._assemble = assemble
        self._pipelines: dict[tuple[Backend, str], StagedPipeline] = {}

    def get(self, backend: str | Backend, device: str | torch.device) -> StagedPipeline:
        """Return the cached pipeline for ``(backend, device)``, building it on first use."""
        backend = Backend.parse(backend)
        device = torch.device(device)
        key = (backend, str(device))
        if key not in self._pipelines:
            self._pipelines[key] = StagedPipeline(
                self.stages,
                backend=backend,
                device=device,
                artifacts_dir=self.artifacts_dir,
                assemble=self._assemble,
            )
        return self._pipelines[key]
