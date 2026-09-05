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

"""Stage graph: the one declaration deployment derives everything from.

A model describes its inference as an ordered list of *stages* over a shared
bag of named tensors (the :class:`StageContext`):

- :class:`GraphStage` — an exportable sub-graph: an ``nn.Module`` plus the
  context names it reads (its ONNX input names) and writes (its ONNX output
  names). One ``GraphStage`` = one ``<name>.onnx`` / ``<name>.engine`` artifact,
  and on a non-PyTorch backend it is replaced by that artifact's runner.
- :class:`TorchStage` — glue that is not exported (pillar decoration, BEV
  scatter, shape bookkeeping ...). It always runs in PyTorch, on every backend.

From the declaration, generic code derives the export units and their trace
inputs (:mod:`.export`), the per-backend inference pipeline (:mod:`.pipeline`),
the artifact names (:func:`artifact_path`), the verification outputs, and the
latency breakdown. Nothing model-specific lives outside the model's own
``build_stages``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs
from autoware_ml.types.backend import Backend


@dataclass
class StageContext:
    """The named-tensor bag stages read from and write to.

    Attributes:
        batch_inputs: The preprocessed model inputs the run started from.
        device: Device exportable stages execute on; glue stages place their outputs
            here so the next graph stage finds its inputs in place.
        tensors: Name -> value produced so far (graph inputs/outputs and glue results).
    """

    batch_inputs: MultiTaskBatchInputs
    device: torch.device
    tensors: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Any:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(
                f"Stage context has no tensor {name!r}; available: {sorted(self.tensors)}. "
                "A stage reads a name no earlier stage produced."
            ) from error


@dataclass(frozen=True)
class TorchStage:
    """A non-exportable glue stage that always runs in PyTorch.

    Attributes:
        name: Unique stage name (latency breakdown key).
        run: ``fn(context) -> {name: value}``; the returned mapping is merged into the
            context. Reads earlier results via ``context[name]`` and the raw batch via
            ``context.batch_inputs``.
    """

    name: str
    run: Callable[[StageContext], Mapping[str, Any]]

    @property
    def exportable(self) -> bool:
        return False


@dataclass(frozen=True)
class GraphStage:
    """An exportable sub-graph: one ONNX / TensorRT artifact.

    Attributes:
        name: Unique stage name; also the artifact stem (``<name>.onnx``).
        module: The traced module. Its positional forward arguments are the context
            tensors named by ``inputs``, in order.
        inputs: Context names fed to the module — these ARE the ONNX input names.
        outputs: Names the module's outputs are written under — the ONNX output names,
            in the module's return order (a single tensor return maps to one name).
        output_fields: Only on the final stage: ``(output_name, key)`` pairs naming what
            the model's ``assemble_predictions`` receives each ONNX output under. Empty on
            intermediate stages.
        torch_fallback_backends: Backends on which this stage runs its PyTorch module
            instead of an artifact (and needs no artifact for availability) — for graphs
            a backend cannot execute, e.g. a spconv graph on ONNX Runtime.
        onnx_dynamic_axes: Axes this graph makes dynamic *by construction*
            (``{tensor_name: {dim_index: dim_name}}``), for graphs whose dynamic axes are
            a property of the declaration rather than a choice — a point model where every
            tensor is indexed by a point count that no configuration can pin down, say.
            ``deploy.stages.<name>.onnx.dynamic_axes`` overrides this when set.
        onnx_transforms: Rewrites applied to this stage's exported ``.onnx``, in order,
            each taking and returning the file path. For fusions intrinsic to the
            deployed form of this graph — folding a bias and an activation into a
            runtime plugin node, say — not for user-configurable graph surgery, which
            belongs in ``deploy.onnx.modify_graph``.
    """

    name: str
    module: nn.Module
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    output_fields: tuple[tuple[str, str], ...] = ()
    torch_fallback_backends: tuple[Backend, ...] = ()
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)
    onnx_transforms: tuple[Callable[[Path], Path], ...] = ()

    @property
    def exportable(self) -> bool:
        return True

    def __post_init__(self) -> None:
        if not self.inputs or not self.outputs:
            raise ValueError(f"GraphStage {self.name!r} must declare inputs and outputs.")
        declared = {onnx_name for onnx_name, _ in self.output_fields}
        unknown = declared - set(self.outputs)
        if unknown:
            raise ValueError(
                f"GraphStage {self.name!r} maps output_fields for {sorted(unknown)}, "
                f"which are not among its outputs {list(self.outputs)}."
            )


Stage = TorchStage | GraphStage


def validate_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """Check a stage declaration and return it as a tuple.

    Raises:
        ValueError: On duplicate names, no exportable stage, a graph stage reading a
            name no earlier graph stage wrote and no glue stage precedes it, or a final
            graph stage without ``output_fields``.
    """
    stages = tuple(stages)
    names = [stage.name for stage in stages]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate stage names: {duplicates}.")
    graph = graph_stages(stages)
    if not graph:
        raise ValueError("A stage graph needs at least one exportable GraphStage.")
    if not graph[-1].output_fields:
        raise ValueError(
            f"The final GraphStage {graph[-1].name!r} must declare output_fields so the "
            "pipeline can hand its outputs to the model's assemble_predictions."
        )
    for stage in graph[:-1]:
        if stage.output_fields:
            raise ValueError(
                f"Only the final GraphStage may declare output_fields (got them on {stage.name!r})."
            )
    return stages


def graph_stages(stages: Sequence[Stage]) -> tuple[GraphStage, ...]:
    """Return the exportable stages in order."""
    return tuple(stage for stage in stages if isinstance(stage, GraphStage))


def final_stage(stages: Sequence[Stage]) -> GraphStage:
    """Return the last exportable stage (the one whose outputs are the model outputs)."""
    return graph_stages(stages)[-1]


def artifact_path(output_dir: str | Path, stage_name: str, backend: Backend) -> Path:
    """Path of a graph stage's exported artifact: ``<output_dir>/<stage_name><suffix>``."""
    backend = Backend.parse(backend)
    if backend is Backend.PYTORCH:
        raise ValueError("The pytorch backend has no exported artifact.")
    return Path(output_dir) / f"{stage_name}{backend.artifact_suffix}"
