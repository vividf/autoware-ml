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


"""ONNX export primitive and graph modification.

The one place a ``torch.onnx.export`` call is spelled. The stage-graph exporter
(:mod:`.export`) drives :func:`export_to_onnx` with values straight from the typed
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


# ---------------------------------------------------------------------------
# ONNX graph modification (Hydra-instantiable modifier)
# ---------------------------------------------------------------------------


def _instantiate_modifier(modify_graph_cfg: Any) -> Any:
    import hydra

    modifier = hydra.utils.instantiate(modify_graph_cfg)
    if callable(modifier):
        return modifier
    if hasattr(modifier, "modify"):
        return modifier
    raise ValueError(f"Modifier {modifier} must be callable or have a 'modify' method.")


def _apply_modifier(modifier: Any, onnx_path: Path) -> Path:
    modified_path = modifier(onnx_path) if callable(modifier) else modifier.modify(onnx_path)
    if modified_path is None:
        raise ValueError("Modifier returned None. Must return Path or str.")
    return Path(modified_path)


def should_modify_graph(modify_graph_cfg: Any) -> bool:
    """Return whether graph modification is enabled (a non-None modifier config)."""
    if modify_graph_cfg is None:
        return False
    from omegaconf import DictConfig, OmegaConf

    if isinstance(modify_graph_cfg, DictConfig):
        return OmegaConf.to_container(modify_graph_cfg, resolve=False) is not None
    return True


def modify_onnx_graph(onnx_path: Path, modify_graph_cfg: Any) -> Path:
    """Apply the configured (Hydra-instantiable) modifier to an exported ONNX file."""
    logger.info("Modifying ONNX graph...")
    modifier = _instantiate_modifier(modify_graph_cfg)
    modified_path = _apply_modifier(modifier, onnx_path)
    logger.info("Successfully modified ONNX graph: %s", modified_path)
    return modified_path


# ---------------------------------------------------------------------------
# FP16 (ModelOpt AutoCast)
# ---------------------------------------------------------------------------


def onnx_has_qdq(onnx_path: Path) -> bool:
    """Whether the ONNX graph contains QuantizeLinear/DequantizeLinear nodes."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    return any(node.op_type in ("QuantizeLinear", "DequantizeLinear") for node in model.graph.node)


def onnx_custom_op_domains(onnx_path: Path) -> tuple[str, ...]:
    """Non-standard operator domains used by the graph's nodes.

    Nodes outside the default ONNX domain (and ``ai.onnx.*``) are runtime plugins —
    ``autoware::ImplicitGemm`` and friends. AutoCast cannot process such a graph:
    it infers types with TensorRT's ONNX parser, which rejects an op whose plugin
    is not registered in the exporting process.
    """
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    domains = {
        node.domain
        for node in model.graph.node
        if node.domain and not node.domain.startswith("ai.onnx")
    }
    return tuple(sorted(domains))


def cast_graph_to_fp16(onnx_path: Path) -> None:
    """Convert a whole graph to FP16 in place, keeping the I/O tensors FP32.

    The FP16 path for graphs AutoCast cannot process: AutoCast types the graph with
    TensorRT's parser and calibrates per node, which needs every operator implemented in
    the exporting process, while a plugin graph's compute lives almost entirely in its
    plugin nodes anyway — per-node selection has nothing meaningful to keep in FP32. So
    such graphs get the blunt conversion: every float initializer and internal tensor
    becomes FP16 (the plugins run FP16 when their tensors are — filters and bias follow
    the feature dtype), engines still build strongly typed, and ``keep_io_types`` holds
    the artifact ABI at FP32.
    """
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16

    model = onnx.load(str(onnx_path))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)

    # The converter rewrites float tensors and initializers but leaves pre-existing
    # int-to-FLOAT Cast nodes at FLOAT, which then meet FP16 tensors downstream
    # ("DIV must have same input types"). After a whole-graph conversion the only
    # legitimate FLOAT casts are the boundary ones feeding the kept-FP32 graph outputs.
    graph_outputs = {output.name for output in converted.graph.output}
    for node in converted.graph.node:
        if node.op_type != "Cast" or node.output[0] in graph_outputs:
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == TensorProto.FLOAT:
                attribute.i = TensorProto.FLOAT16

    onnx.save(converted, str(onnx_path))
    logger.info("Cast %s to FP16 (graph I/O kept FP32).", onnx_path.name)


def autocast_to_fp16(onnx_path: Path, sample_inputs: Mapping[str, Any]) -> None:
    """Convert an exported FP32 ONNX graph to mixed FP16 in place (ModelOpt AutoCast).

    TensorRT engines build strongly typed, so FP16 must live in the graph itself; this
    is the official replacement for the removed ``BuilderFlag.FP16`` weak-typing path.
    I/O tensor types are preserved (``keep_io_types=True``) so the artifact ABI —
    what the Autoware runtime binds against — does not change with the precision.

    ``sample_inputs`` (the stage's trace inputs) drive AutoCast's reference run: its
    magnitude-based node classification then sees real activations, and graphs with
    dynamic spatial dims get valid shapes (AutoCast's random fallback fills dynamic
    dims with 1, which breaks strided convolutions). Same inputs → same partition,
    so the conversion is reproducible.

    Quantized graphs must not pass through here: AutoCast does not support Q/DQ models
    (the caller gates on :func:`onnx_has_qdq`).

    Args:
        onnx_path: Exported FP32 ``.onnx``, overwritten with the mixed-FP16 graph.
        sample_inputs: ONNX input name -> tensor/array with concrete shapes (one batch).
    """
    from modelopt.onnx.autocast import convert_to_mixed_precision

    import numpy as np
    import onnx

    feed = {}
    for name, value in sample_inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        feed[name] = np.asarray(value)
    calibration_path = onnx_path.with_suffix(".autocast_inputs.npz")
    np.savez(calibration_path, **feed)

    logger.info("AutoCast: converting %s to mixed FP16 (I/O types preserved)...", onnx_path.name)
    try:
        model = convert_to_mixed_precision(
            onnx_path=str(onnx_path),
            low_precision_type="fp16",
            keep_io_types=True,
            calibration_data=str(calibration_path),
        )
    finally:
        calibration_path.unlink(missing_ok=True)
    onnx.save(model, str(onnx_path))
    logger.info("AutoCast: wrote mixed-FP16 graph back to %s", onnx_path)
