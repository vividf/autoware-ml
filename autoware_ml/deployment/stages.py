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

"""Deployment declared as graphs plus one function.

A model declares the graphs it exports (:class:`GraphStage`) and writes its deployed inference
as one function, ``deployed_forward(batch, graphs)``: the glue is plain code in that function,
and each graph is called by name. The framework hands the function its graphs as callables
(:class:`GraphCall`), so one function serves export and every later run, and the model outputs
are what the function returns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, NamedTuple

import torch
from torch import nn

from autoware_ml.types.backend import Backend


def _reject_bare_string(stage: GraphStage, kind: str, names: Sequence[str]) -> None:
    """Refuse a ``str`` for a tuple of names: iterated, it would declare one-letter names."""
    if isinstance(names, str):
        raise TypeError(
            f"{type(stage).__name__} {stage.name!r} declares its {kind} as the string "
            f"{names!r}; declare a tuple of names, e.g. ({names!r},)."
        )


@dataclass(frozen=True)
class GraphStage:
    """One exported ONNX / TensorRT graph.

    Attributes:
        name: Unique graph name, also the ``deploy.onnx.modules`` key and the artifact stem.
        module: The traced module, export-ready as declared (the live submodule or an
            export copy); its positional arguments are the ``inputs`` tensors. The framework
            runs and traces it and never alters it.
        inputs: The names the deployed function passes the tensors under, in the module's
            argument order; they are the ONNX input names.
        outputs: ONNX output names, in the module's return order; a call of the graph returns
            its tensors under these names.
        torch_fallback_backends: Backends that run the PyTorch module instead of the
            artifact, for graphs they cannot execute (plugin ops on ONNX Runtime).
        onnx_dynamic_axes: Axes dynamic by construction, ``{tensor: {dim: name}}``. They
            become the spec's ``dynamic_axes``, the one declaration both ONNX exporters
            derive from.
    """

    name: str
    module: nn.Module
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    torch_fallback_backends: tuple[Backend, ...] = ()
    onnx_dynamic_axes: Mapping[str, Mapping[int, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _reject_bare_string(self, "inputs", self.inputs)
        _reject_bare_string(self, "outputs", self.outputs)
        if not self.inputs or not self.outputs:
            raise ValueError(f"GraphStage {self.name!r} must declare inputs and outputs.")
        # The exporter renames a repeated name (``x`` -> ``x.1``), so the artifact would not
        # carry the declared names.
        for kind, names in (("inputs", self.inputs), ("outputs", self.outputs)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"GraphStage {self.name!r} lists {repeated} more than once in its {kind}; "
                    "each is one ONNX tensor name."
                )
        # The TorchScript exporter only warns about an axis of an unknown tensor and exports
        # it static.
        unknown = sorted(set(self.onnx_dynamic_axes) - set(self.inputs) - set(self.outputs))
        if unknown:
            raise ValueError(
                f"GraphStage {self.name!r} declares onnx_dynamic_axes for {unknown}, which are "
                f"not among its inputs {list(self.inputs)} or outputs {list(self.outputs)}."
            )


def validate_graphs(graphs: Sequence[GraphStage]) -> tuple[GraphStage, ...]:
    """Check a model's graph declaration and return it as a tuple.

    Whether the deployed function calls each graph with its declared inputs is checked when
    it runs.

    Raises:
        TypeError: On an entry that is not a :class:`GraphStage`.
        ValueError: On no graph, or on two graphs with one name (the artifact's file name).
    """
    graphs = tuple(graphs)
    others = [type(graph).__name__ for graph in graphs if not isinstance(graph, GraphStage)]
    if others:
        raise TypeError(f"A deployment declares GraphStage entries, got {others}.")
    if not graphs:
        raise ValueError("A deployment needs at least one exportable GraphStage.")
    names = [graph.name for graph in graphs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate graph names: {duplicates}.")
    return graphs


#: File suffix of the artifact each backend runs; pytorch runs the modules themselves.
_ARTIFACT_SUFFIX: Final = {Backend.ONNX: ".onnx", Backend.TENSORRT: ".engine"}


def artifact_path(output_dir: str | Path, stage_name: str, backend: Backend) -> Path:
    """Path of a graph's exported artifact: ``<output_dir>/<stage_name><suffix>``."""
    suffix = _ARTIFACT_SUFFIX.get(backend)
    if suffix is None:
        raise ValueError(f"The {backend} backend has no exported artifact.")
    return Path(output_dir) / f"{stage_name}{suffix}"


def check_inputs_unmodified(
    stage: GraphStage, inputs: Mapping[str, Any], versions: Mapping[str, int]
) -> None:
    """Reject a graph module that modified an input tensor in place.

    ``versions`` are the tensors' ``_version`` counters before the module ran; every in-place
    operation increments the counter. The deployed function may read the same tensors again,
    and export traces a graph on the tensors it ran on, so no graph may modify what it reads.

    Raises:
        ValueError: Naming the graph and the modified inputs.
    """
    modified = sorted(
        name for name, version in versions.items() if inputs[name]._version != version
    )
    if modified:
        raise ValueError(
            f"{type(stage).__name__} {stage.name!r} modified its inputs {modified} in place; a "
            "graph reads its inputs and returns new tensors, since the deployed function may "
            "read the same tensors again and export traces them."
        )


def _tensor_versions(inputs: Mapping[str, Any]) -> dict[str, int]:
    return {name: value._version for name, value in inputs.items() if torch.is_tensor(value)}


def run_graph_stage(stage: GraphStage, args: Sequence[Any]) -> dict[str, Any]:
    """Call a graph's module on its positional inputs and check the return against ``outputs``.

    Raises:
        TypeError: When the module returns a mapping or a non-tensor.
        ValueError: When the number of tensors differs from ``outputs``, or the module
            modifies an input in place.
    """
    inputs = dict(zip(stage.inputs, args))
    versions = _tensor_versions(inputs)
    raw = stage.module(*args)
    check_inputs_unmodified(stage, inputs, versions)
    if isinstance(raw, Mapping):
        # len() and zip() would accept a dict and name the outputs by its keys.
        raise TypeError(
            f"GraphStage {stage.name!r} returned a mapping; a graph returns its tensors "
            f"positionally, in the order of outputs {list(stage.outputs)}."
        )
    if isinstance(raw, torch.Tensor):
        raw = (raw,)
    if len(raw) != len(stage.outputs):
        raise ValueError(
            f"GraphStage {stage.name!r} returned {len(raw)} tensor(s) but declares "
            f"outputs {list(stage.outputs)}."
        )
    # The exporter flattens a nested tuple into extra outputs and drops Python values,
    # so neither would match the declared names.
    others = sorted({type(value).__name__ for value in raw if not torch.is_tensor(value)})
    if others:
        raise TypeError(
            f"GraphStage {stage.name!r} returned {others} among its outputs; a graph "
            f"returns one tensor per declared output {list(stage.outputs)}."
        )
    return dict(zip(stage.outputs, raw))


#: Runs one graph on its positional inputs and returns its outputs by name.
GraphRunner = Callable[[tuple[torch.Tensor, ...]], Mapping[str, torch.Tensor]]


class GraphCall:
    """One declared graph as the deployed function calls it: tensors in and out by name.

    Attributes:
        stage: The graph's declaration.
        inputs: The graph's input names, in declaration order.
        outputs: The graph's output names, in declaration order.
        args: The positional inputs of the call, in declaration order, on the run device;
            ``None`` until the graph is called. Export traces the graph on them.
    """

    def __init__(self, stage: GraphStage, run: GraphRunner, device: torch.device) -> None:
        self.stage = stage
        self.inputs = stage.inputs
        self.outputs = stage.outputs
        self.args: tuple[torch.Tensor, ...] | None = None
        self._run = run
        self._device = device
        self._versions: tuple[int, ...] = ()

    def __call__(self, **tensors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the graph on exactly its declared inputs and return its declared outputs.

        Raises:
            RuntimeError: On a second call in one run; a graph is one artifact, run once.
            TypeError: When the keyword names are not the declared inputs, or a value is not
                a tensor.
        """
        name = self.stage.name
        if self.args is not None:
            raise RuntimeError(f"Graph {name!r} was called twice in one run.")
        if set(tensors) != set(self.inputs):
            raise TypeError(
                f"Graph {name!r} takes {list(self.inputs)}, got {sorted(tensors)}: missing "
                f"{sorted(set(self.inputs) - set(tensors))}, unknown "
                f"{sorted(set(tensors) - set(self.inputs))}."
            )
        others = {
            key: type(value).__name__
            for key, value in tensors.items()
            if not torch.is_tensor(value)
        }
        if others:
            raise TypeError(
                f"Graph {name!r} binds tensors only; got {others}. The deployed function has "
                "to pass each input as a tensor."
            )
        self.args = tuple(tensors[key].to(self._device) for key in self.inputs)
        self._versions = tuple(arg._version for arg in self.args)
        produced = self._run(self.args)
        return {key: produced[key] for key in self.outputs}

    def modified_inputs(self) -> list[str]:
        """Names of the inputs modified in place since the call; export traces on them."""
        if self.args is None:
            return []
        return [
            key
            for key, arg, version in zip(self.inputs, self.args, self._versions)
            if arg._version != version
        ]


class Graphs(Mapping[str, GraphCall]):
    """The graphs of one run, by name, as the deployed function receives them.

    Attributes:
        device: Device the graphs run on; the function moves the tensors it makes there.
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
    """A model's deployment: the graphs it exports and the function that runs them.

    Attributes:
        graphs: The exported graphs, checked by :func:`validate_graphs`.
        forward: The deployed inference, ``forward(batch, graphs) -> model outputs``.
    """

    graphs: tuple[GraphStage, ...]
    forward: DeployedForward

    def __post_init__(self) -> None:
        object.__setattr__(self, "graphs", validate_graphs(self.graphs))


def call_deployed_forward(
    deployment: DeployedModel,
    runner_for: Callable[[GraphStage], GraphRunner],
    batch: Any,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, GraphCall]]:
    """Run the deployed function once, each graph on ``runner_for(graph)``; check what it did.

    Returns:
        The model outputs, and the call of each graph by name.

    Raises:
        TypeError: When the function returns something other than a non-empty mapping of
            tensors.
        RuntimeError: When the function did not call a declared graph.
    """
    calls = {stage.name: GraphCall(stage, runner_for(stage), device) for stage in deployment.graphs}
    with torch.no_grad():
        outputs = deployment.forward(batch, Graphs(calls, device))
    if not isinstance(outputs, Mapping) or not outputs:
        raise TypeError(
            "deployed_forward returns the model outputs as a non-empty mapping of name -> "
            f"tensor, got {type(outputs).__name__}."
        )
    others = sorted(name for name, value in outputs.items() if not torch.is_tensor(value))
    if others:
        raise TypeError(f"deployed_forward returned model outputs that are not tensors: {others}.")
    uncalled = [name for name, call in calls.items() if call.args is None]
    if uncalled:
        raise RuntimeError(
            f"deployed_forward did not call the declared graphs {uncalled}; every declared "
            "graph is exported, so each has to run."
        )
    return dict(outputs), calls


class DeploymentRun(NamedTuple):
    """One tracing run of a deployment.

    Attributes:
        outputs: The model outputs the deployed function returned, by name.
        graph_inputs: The positional inputs each graph ran on, by graph name, filled by a
            tracing run (:meth:`DeployedPipeline.trace`): the tensors the function passed, which
            export traces each graph on.
    """

    outputs: dict[str, torch.Tensor]
    graph_inputs: dict[str, tuple[torch.Tensor, ...]]
