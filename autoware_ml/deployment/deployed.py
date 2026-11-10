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

"""PROTOTYPE: deployment declared as graphs plus one function.

The alternative to a stage list (:mod:`autoware_ml.deployment.stages`): a model declares its
exported graphs and writes the deployed inference as an ordinary function that calls them.
The framework hands that function the graphs as callables, backed by the PyTorch modules or
by the exported artifacts, so the function is the same on every backend.

    def deploy_graphs(self):
        return (GraphStage("encoder", module=..., inputs=("x",), outputs=("feat",)), ...)

    def deployed_forward(self, batch, graphs):
        feat = graphs["encoder"](x=prepare(batch))["feat"]
        return graphs["head"](feat=regroup(feat, batch))
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from autoware_ml.deployment.pipeline import (
    PipelineResult,
    Runner,
    _artifact_runner,
    _ModuleRunner,
    assemble_outputs,
    cuda_synced_timer,
    normalize_device,
)
from autoware_ml.deployment.stages import GraphStage
from autoware_ml.types.backend import Backend

#: Latency key of everything ``deployed_forward`` does outside the graphs.
GLUE_STAGE = "glue"


class GraphCall:
    """One declared graph as ``deployed_forward`` calls it: tensors in by name, tensors out by name.

    Attributes:
        inputs: The graph's input names, in the declared order.
        outputs: The graph's output names, in the declared order.
    """

    def __init__(self, graph: GraphStage, runner: Runner, device: torch.device) -> None:
        self.graph = graph
        self.inputs = graph.inputs
        self.outputs = graph.outputs
        self._runner = runner
        self._device = device
        self.args: tuple[torch.Tensor, ...] | None = None
        self.elapsed_ms = 0.0

    def __call__(self, **tensors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the graph on exactly its declared inputs and return its declared outputs.

        Raises:
            RuntimeError: On a second call in one run; a graph is one exported artifact.
            TypeError: When an input is not a tensor, or the names are not the declared inputs.
        """
        name = self.graph.name
        if self.args is not None:
            raise RuntimeError(f"Graph {name!r} was called twice in one run.")
        if set(tensors) != set(self.inputs):
            raise TypeError(
                f"Graph {name!r} takes {list(self.inputs)}, got {sorted(tensors)}: missing "
                f"{sorted(set(self.inputs) - set(tensors))}, unknown "
                f"{sorted(set(tensors) - set(self.inputs))}."
            )
        others = sorted(n for n, value in tensors.items() if not torch.is_tensor(value))
        if others:
            raise TypeError(f"Graph {name!r} binds tensors only; {others} are not tensors.")
        self.args = tuple(tensors[n].to(self._device) for n in self.inputs)
        self._versions = tuple(arg._version for arg in self.args)
        produced, self.elapsed_ms = self._runner.run(dict(zip(self.inputs, self.args)))
        return {n: produced[n] for n in self.outputs}

    def modified_inputs(self) -> list[str]:
        """Names of the inputs modified in place since the call (export traces on them)."""
        if self.args is None:
            return []
        return [
            n
            for n, arg, version in zip(self.inputs, self.args, self._versions)
            if arg._version != version
        ]


class Graphs(Mapping[str, GraphCall]):
    """The graphs of one run, by name, as handed to ``deployed_forward``.

    Attributes:
        device: Device the graphs run on; glue moves its own tensors there.
    """

    def __init__(self, calls: Mapping[str, GraphCall], device: torch.device) -> None:
        self._calls = dict(calls)
        self.device = device

    def __getitem__(self, name: str) -> GraphCall:
        try:
            return self._calls[name]
        except KeyError:
            raise KeyError(
                f"No graph {name!r}; the model declares {sorted(self._calls)}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._calls)

    def __len__(self) -> int:
        return len(self._calls)


#: ``deployed_forward(batch, graphs) -> {output name: tensor}``.
DeployedForward = Callable[[Any, Graphs], Mapping[str, torch.Tensor]]


@dataclass(frozen=True)
class DeployedModel:
    """A model's deployment in the function form.

    Attributes:
        graphs: The exported graphs (:meth:`deploy_graphs`), each a :class:`GraphStage`.
        forward: The deployed inference (:meth:`deployed_forward`).
    """

    graphs: tuple[GraphStage, ...]
    forward: DeployedForward

    def __post_init__(self) -> None:
        if not self.graphs:
            raise ValueError("A deployed model declares at least one graph.")
        others = [type(g).__name__ for g in self.graphs if not isinstance(g, GraphStage)]
        if others:
            raise TypeError(f"deploy_graphs() returns GraphStage entries, got {others}.")
        names = [graph.name for graph in self.graphs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate graph names: {duplicates}.")


class DeployedPipeline:
    """Run a :class:`DeployedModel` on one backend; the interface of ``StagedPipeline``.

    Args:
        declaration: The model's graphs and deployed forward.
        backend: Backend that runs the graphs.
        device: Device the graphs run on.
        artifacts_dir: Directory of the ``<graph>.onnx`` / ``<graph>.engine`` artifacts,
            required by the ``onnx`` and ``tensorrt`` backends.
        plugin_libraries: TensorRT plugin libraries the engines were built with.
    """

    def __init__(
        self,
        declaration: DeployedModel,
        backend: Backend,
        device: torch.device,
        artifacts_dir: str | Path | None = None,
        plugin_libraries: Sequence[str] = (),
    ) -> None:
        self.declaration = declaration
        self.graphs = declaration.graphs
        self.backend = backend
        self.device = normalize_device(device)
        self.graph_stage_names = tuple(graph.name for graph in self.graphs)
        self.fallback_stage_names = tuple(
            graph.name
            for graph in self.graphs
            if backend is not Backend.PYTORCH and backend in graph.torch_fallback_backends
        )
        self._runners: dict[str, Runner] = {}
        for graph in self.graphs:
            if backend is Backend.PYTORCH or backend in graph.torch_fallback_backends:
                self._runners[graph.name] = _ModuleRunner(graph, self.device)
            else:
                if artifacts_dir is None:
                    raise ValueError(f"artifacts_dir is required for the {backend} backend.")
                self._runners[graph.name] = _artifact_runner(
                    graph, backend, self.device, Path(artifacts_dir), tuple(plugin_libraries)
                )

    def _run(self, batch: Any) -> tuple[dict[str, torch.Tensor], Graphs, dict[str, float]]:
        calls = {
            graph.name: GraphCall(graph, self._runners[graph.name], self.device)
            for graph in self.graphs
        }
        graphs = Graphs(calls, self.device)
        total: dict[str, float] = {}
        with torch.no_grad(), cuda_synced_timer(total, "total", self.device):
            outputs = self.declaration.forward(batch, graphs)
        if not isinstance(outputs, Mapping) or not outputs:
            raise TypeError(
                "deployed_forward returns the model outputs as a non-empty mapping of name -> "
                f"tensor, got {type(outputs).__name__}."
            )
        others = sorted(name for name, value in outputs.items() if not torch.is_tensor(value))
        if others:
            raise TypeError(f"deployed_forward returned non-tensor model outputs: {others}.")
        uncalled = [name for name, call in calls.items() if call.args is None]
        if uncalled:
            raise RuntimeError(
                f"deployed_forward did not call the declared graphs {uncalled}; every declared "
                "graph is exported, so each has to run."
            )
        times = {name: call.elapsed_ms for name, call in calls.items()}
        times[GLUE_STAGE] = max(total["total"] - sum(times.values()), 0.0)
        return dict(outputs), graphs, times

    def infer(self, batch: Any) -> PipelineResult:
        """Run the deployed forward and return the model outputs with the latency per graph."""
        outputs, _, times = self._run(batch)
        return PipelineResult(
            outputs=outputs,
            output_names=tuple(outputs),
            stage_times_ms=times,
            graph_stage_names=self.graph_stage_names,
        )

    def trace(self, batch: Any) -> dict[str, tuple[torch.Tensor, ...]]:
        """Run the deployed forward and return the positional inputs each graph ran on.

        Raises:
            ValueError: When a graph's input was modified in place after the graph ran;
                export traces the graph on that tensor.
        """
        _, graphs, _ = self._run(batch)
        modified = {name: call.modified_inputs() for name, call in graphs.items()}
        modified = {name: names for name, names in modified.items() if names}
        if modified:
            raise ValueError(
                f"deployed_forward modified graph inputs in place after they ran: {modified}; "
                "export traces each graph on the tensors it ran on."
            )
        return {name: call.args for name, call in graphs.items()}

    def assemble(
        self, result: PipelineResult, device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        """Return the model outputs as ``build_deployed_eval_output`` takes them."""
        return assemble_outputs(result, device)
