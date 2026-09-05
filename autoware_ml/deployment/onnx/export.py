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


"""The ONNX export primitive: the one place a ``torch.onnx.export`` call is spelled.

The stage-graph exporter (:mod:`autoware_ml.deployment.export`) drives
:func:`export_to_onnx` with values straight from the typed
:class:`~autoware_ml.deployment.config.DeployConfig`.

.. todo:: TODO(vividf): the legacy ``ExportSpec`` path (``autoware_ml.utils.deploy``)
   also adapts its DictConfig schema onto this primitive — that adapter disappears
   with utils/deploy.py at Q5 (legacy BaseModel migration).

Dynamic-shape declarations arrive as the plain mappings the config carries:

- ``dynamic_shapes`` (dynamo exporter): ``{input_name: {dim_index: name | {name, min, max}}}``
- ``dynamic_axes`` (legacy exporter):   ``{tensor_name: {dim_index: name}}`` — when absent,
  a ``dynamic_shapes`` declaration is down-converted (names only).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.export import Dim

from autoware_ml.ops.segment.scatter_reduce import register_scatter_reduce_onnx_symbolic

logger = logging.getLogger(__name__)


def build_dynamic_shapes(
    spec: Mapping[str, Any] | None,
    input_names: Sequence[str],
) -> tuple[dict[int, Dim] | None, ...] | None:
    """Build the dynamo-exporter dynamic-shape structure from a config mapping.

    Args:
        spec: ``{input_name: {dim_index: name | {name, min, max}}}`` or ``None``.
        input_names: Positional input names, in export order; every key of ``spec``
            must be one of them.

    Raises:
        ValueError: When ``spec`` names an input not in ``input_names`` or a dim
            entry lacks a ``name``.
    """
    if spec is None:
        return None

    unknown_params = [name for name in spec if name not in input_names]
    if unknown_params:
        raise ValueError(
            f"Dynamic shape parameters {unknown_params} not found in export inputs. "
            f"Available inputs: {list(input_names)}."
        )

    dynamic_shapes: list[dict[int, Dim] | None] = []
    for param_name in input_names:
        dim_mapping = spec.get(param_name)
        if dim_mapping is None:
            dynamic_shapes.append(None)
            continue

        param_dynamic_shapes: dict[int, Dim] = {}
        for dim_idx, dim_spec in dim_mapping.items():
            if isinstance(dim_spec, str):
                param_dynamic_shapes[int(dim_idx)] = Dim(dim_spec)
                continue

            dim_name = dim_spec.get("name")
            if dim_name is None:
                raise ValueError(
                    f"Dynamic shape spec for '{param_name}[{dim_idx}]' must define 'name'."
                )
            dim_kwargs = {key: dim_spec[key] for key in ("min", "max") if key in dim_spec}
            param_dynamic_shapes[int(dim_idx)] = Dim(dim_name, **dim_kwargs)

        dynamic_shapes.append(param_dynamic_shapes or None)

    if all(param_dynamic_shapes is None for param_dynamic_shapes in dynamic_shapes):
        return None
    return tuple(dynamic_shapes)


def normalize_dynamic_shapes_for_model(
    model: torch.nn.Module,
    dynamic_shapes: tuple[dict[int, Dim] | None, ...] | None,
) -> tuple[Any, ...] | None:
    """Adapt dynamic-shape structure to the model forward signature.

    ``torch.export`` requires ``dynamic_shapes`` to mirror the positional input
    pytree passed to the model. Wrappers that expose ``forward(*args)`` receive
    one tuple-valued positional argument, so their dynamic-shape structure must
    be wrapped one level deeper.
    """
    if dynamic_shapes is None:
        return None

    signature = inspect.signature(model.forward)
    parameters = [parameter for parameter in signature.parameters.values()]
    if len(parameters) == 1 and parameters[0].kind == inspect.Parameter.VAR_POSITIONAL:
        return (dynamic_shapes,)
    return dynamic_shapes


def build_dynamic_axes(spec: Mapping[str, Any] | None) -> dict[str, dict[int, str]] | None:
    """Build the legacy-exporter (``dynamo=False``) dynamic-axes mapping.

    Accepts either an axes mapping (``{tensor: {dim: name}}``) or a
    ``dynamic_shapes``-style mapping whose ``{name, min, max}`` entries are
    down-converted to their names.
    """
    if spec is None:
        return None

    dynamic_axes: dict[str, dict[int, str]] = {}
    for tensor_name, dim_mapping in spec.items():
        tensor_dynamic_axes: dict[int, str] = {}
        for dim_idx, dim_spec in dim_mapping.items():
            if isinstance(dim_spec, str):
                tensor_dynamic_axes[int(dim_idx)] = dim_spec
                continue

            dim_name = dim_spec.get("name")
            if dim_name is None:
                raise ValueError(
                    f"Dynamic axis/shape spec for '{tensor_name}[{dim_idx}]' must define 'name'."
                )
            tensor_dynamic_axes[int(dim_idx)] = dim_name

        if tensor_dynamic_axes:
            dynamic_axes[tensor_name] = tensor_dynamic_axes

    return dynamic_axes or None


def _log_export_inputs(args: Sequence[Any], input_names: Sequence[str]) -> None:
    for input_name, input_value in zip(input_names, args):
        if isinstance(input_value, torch.Tensor):
            logger.info(
                "Input '%s': shape=%s, dtype=%s",
                input_name,
                tuple(input_value.shape),
                input_value.dtype,
            )
        else:
            logger.info("Input '%s': type=%s", input_name, type(input_value).__name__)


def _merge_onnx_external_data(onnx_path: Path) -> None:
    """Merge ONNX external data shards back into a single file."""
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    onnx_model = onnx.load(str(onnx_path), load_external_data=True)
    convert_model_from_external_data(onnx_model)
    onnx.save_model(onnx_model, str(onnx_path))


def export_to_onnx(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    output_path: Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    opset_version: int,
    dynamo: bool,
    do_constant_folding: bool = True,
    dynamic_shapes: Mapping[str, Any] | None = None,
    dynamic_axes: Mapping[str, Any] | None = None,
) -> None:
    """Export one module to ONNX.

    Args:
        module: Module to export.
        args: Example positional inputs, in ``input_names`` order.
        output_path: Destination ``.onnx`` path.
        input_names: ONNX input names (one per positional argument).
        output_names: ONNX output names, in the module's return order.
        opset_version: ONNX opset.
        dynamo: Use the dynamo exporter (``dynamic_shapes``) instead of the legacy
            tracer (``dynamic_axes``).
        do_constant_folding: Fold constants during export.
        dynamic_shapes: Dynamo dynamic-shape declaration (see module docstring).
        dynamic_axes: Legacy dynamic-axes declaration; when ``None`` under
            ``dynamo=False``, ``dynamic_shapes`` is down-converted instead.
    """
    logger.info("Exporting model to ONNX...")
    if not input_names:
        raise ValueError("ONNX export needs at least one input name.")

    shapes = None
    axes = None
    if dynamo:
        shapes = normalize_dynamic_shapes_for_model(
            module, build_dynamic_shapes(dynamic_shapes, list(input_names))
        )
    else:
        axes = build_dynamic_axes(dynamic_axes if dynamic_axes is not None else dynamic_shapes)

    logger.info("Dynamic shapes: %s", shapes)
    logger.info("Dynamic axes: %s", axes)
    logger.info("ONNX opset version: %s", opset_version)
    logger.info("Input names: %s", list(input_names))
    logger.info("Output names: %s", list(output_names))
    _log_export_inputs(args, input_names)

    # Register shared ONNX symbolics needed by export-aware ops packages.
    register_scatter_reduce_onnx_symbolic(opset_version=int(opset_version))

    export_kwargs: dict[str, Any] = {
        "model": module,
        "args": args,
        "f": str(output_path),
        "input_names": list(input_names),
        "output_names": list(output_names),
        "opset_version": int(opset_version),
        "dynamo": dynamo,
        "do_constant_folding": do_constant_folding,
    }
    if dynamo:
        export_kwargs["dynamic_shapes"] = shapes
    else:
        export_kwargs["dynamic_axes"] = axes

    torch.onnx.export(**export_kwargs)

    logger.info("Successfully exported ONNX model to %s", output_path)

    data_path = output_path.with_suffix(output_path.suffix + ".data")
    if data_path.exists():
        logger.info("Found external data file %s. Merging into the ONNX file...", data_path)
        _merge_onnx_external_data(output_path)
        data_path.unlink()
        logger.info("Successfully merged external data into the ONNX file")

