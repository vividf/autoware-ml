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

"""Build TensorRT engines from ONNX.

Networks are always strongly typed: precision comes from the ONNX graph
(``deploy.onnx.precision`` and its Q/DQ nodes), not from builder flags. TensorRT deprecated
the weak-typing precision flags in 10.12 and removed them in 11.
"""

from __future__ import annotations

import ctypes
import functools
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# tensorrt is imported where an engine is built or loaded: it is optional for every other
# deploy step, and a PyTorch-only environment has none.
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger(__name__)


class ShapeProfile(BaseModel):
    """One input's TensorRT optimization profile (``{min_shape, opt_shape, max_shape}``).

    Validated on construction, so a bad profile is a config error naming the dimension rather
    than a TensorRT build failure. Only the minimum may be 0: TensorRT accepts empty tensors,
    and a point model can get an empty cloud.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]

    @model_validator(mode="after")
    def _check_ranks_and_order(self) -> ShapeProfile:
        ranks = {len(self.min_shape), len(self.opt_shape), len(self.max_shape)}
        if len(ranks) != 1 or 0 in ranks:
            raise ValueError(
                "min/opt/max must describe one tensor, so they need the same non-zero rank; "
                f"got min={self.min_shape}, opt={self.opt_shape}, max={self.max_shape}."
            )
        for index, (low, opt, high) in enumerate(
            zip(self.min_shape, self.opt_shape, self.max_shape)
        ):
            if low < 0 or opt < 1:
                raise ValueError(
                    f"dim {index} has min={low}, opt={opt}; min must be >= 0 and opt >= 1."
                )
            if not low <= opt <= high:
                raise ValueError(
                    f"dim {index} violates min <= opt <= max ({low} <= {opt} <= {high} is false)."
                )
        return self


@functools.cache
def _load_plugin_library(library: str) -> None:
    library_path = Path(library)
    if not library_path.exists():
        raise FileNotFoundError(f"TensorRT plugin library not found: {library_path}")
    ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    logger.info("Loaded TensorRT plugin library: %s", library_path)


def load_tensorrt_plugin_libraries(plugin_libraries: Sequence[str] | None) -> None:
    """Load custom TensorRT plugin libraries, each once; call before the registry initializes."""
    for library in plugin_libraries or ():
        _load_plugin_library(str(library))


@functools.cache
def trt_logger():
    """The one TensorRT logger of the process, shared by every builder, runtime and engine.

    TensorRT keeps a raw pointer to the logger it was given: a logger created per build or
    per engine load is destroyed before the objects that log through it.
    """
    import tensorrt as trt  # noqa: PLC0415

    return trt.Logger(trt.Logger.WARNING)


@functools.cache
def _init_plugin_registry() -> None:
    import tensorrt as trt  # noqa: PLC0415

    trt.init_libnvinfer_plugins(trt_logger(), "")


def _create_builder(workspace_size: int, plugin_libraries: Sequence[str]):
    """Create ``(builder, network, parser, config)`` for one strongly typed engine build."""
    import tensorrt as trt  # noqa: PLC0415

    load_tensorrt_plugin_libraries(plugin_libraries)
    _init_plugin_registry()
    builder = trt.Builder(trt_logger())
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger())
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size))
    logger.info("Workspace size: %.2f GB", workspace_size / (1024**3))
    return builder, network, parser, config


def _parse_onnx_file(parser: Any, onnx_path: Path) -> None:
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()

    if not parser.parse(onnx_data):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        error_msg = "\n".join(f"TensorRT parser error {i}: {err}" for i, err in enumerate(errors))
        if "plugin" in error_msg.lower() or "INVALID_NODE" in error_msg:
            error_msg += (
                "\nHint: if this graph carries custom ops (e.g. autoware::*), make sure "
                "deploy.tensorrt.plugin_libraries lists the plugin .so for this environment."
            )
        raise RuntimeError(f"Failed to parse ONNX file:\n{error_msg}")

    logger.info("Successfully parsed ONNX file")


def _create_optimization_profile(
    builder: Any, network: Any, input_shapes: Mapping[str, ShapeProfile], key: str
):
    """Build the profile of ``network`` from its module's entries.

    Every entry has to name an input of this network: a misspelled key would otherwise build
    an engine without the shape it was meant to set.

    Raises:
        ValueError: When an entry names no input of the network.
    """
    network_inputs = [network.get_input(i).name for i in range(network.num_inputs)]
    unknown = sorted(set(input_shapes) - set(network_inputs))
    if unknown:
        raise ValueError(
            f"{key} names {unknown}, which are not inputs of the graph {network_inputs}."
        )
    profile = builder.create_optimization_profile()
    for input_name, shapes in input_shapes.items():
        profile.set_shape(
            input_name,
            min=list(shapes.min_shape),
            opt=list(shapes.opt_shape),
            max=list(shapes.max_shape),
        )
        logger.info(
            "Optimization profile for '%s': min=%s, opt=%s, max=%s",
            input_name,
            list(shapes.min_shape),
            list(shapes.opt_shape),
            list(shapes.max_shape),
        )
    return profile


def dynamic_inputs_without_profile(
    network: Any, input_shapes: Mapping[str, ShapeProfile] | None
) -> list[str]:
    """Return the network's dynamic inputs that ``input_shapes`` has no entry for.

    TensorRT needs a min / opt / max shape for every input with a dynamic dimension; without
    one it fails inside the build with a message that does not name the config.

    Args:
        network: Parsed TensorRT network.
        input_shapes: The configured profile per input name, if any.

    Returns:
        The input names, in the network's input order.
    """
    configured = set(input_shapes or {})
    inputs = [network.get_input(index) for index in range(network.num_inputs)]
    return [
        tensor.name
        for tensor in inputs
        if -1 in tuple(tensor.shape) and tensor.name not in configured
    ]


def build_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    workspace_size: int = 1 << 30,
    plugin_libraries: Sequence[str] = (),
    input_shapes: Mapping[str, ShapeProfile] | None = None,
    input_shapes_key: str = "deploy.tensorrt.modules.<module>.input_shapes",
) -> None:
    """Build and serialize a strongly typed TensorRT engine from an ONNX file.

    Args:
        onnx_path: The graph to build.
        output_path: Where the serialized engine is written.
        workspace_size: Workspace memory-pool limit in bytes.
        plugin_libraries: Custom plugin libraries to load before parsing.
        input_shapes: Optimization profile per dynamic input; ``None`` for a static graph.
        input_shapes_key: The config key the entries came from, for the error messages.

    Raises:
        ValueError: When an entry names no input of the graph, or a dynamic input of the graph
            has no entry.
        RuntimeError: When ONNX parsing or the engine build fails.
    """
    onnx_path, output_path = Path(onnx_path), Path(output_path)
    logger.info("Building TensorRT engine (strongly typed)...")
    builder, network, parser, config = _create_builder(workspace_size, plugin_libraries)
    _parse_onnx_file(parser, onnx_path)

    if input_shapes:
        config.add_optimization_profile(
            _create_optimization_profile(builder, network, input_shapes, input_shapes_key)
        )

    missing = dynamic_inputs_without_profile(network, input_shapes)
    if missing:
        raise ValueError(
            f"{onnx_path.name} has dynamic inputs {missing} without a {input_shapes_key} "
            "entry; TensorRT needs a min / opt / max shape for each."
        )

    logger.info("Building TensorRT engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info("Successfully built TensorRT engine: %s", output_path)


def build_engine_from_config(
    onnx_path: Path,
    deploy_cfg: DictConfig,
    output_path: Path,
    module_name: str,
) -> None:
    """Build an engine with :func:`build_engine` from the ``deploy.tensorrt`` settings.

    The shapes of a module live under ``deploy.tensorrt.modules.<module_name>.input_shapes``,
    keyed by its ONNX input names, next to its ``deploy.onnx.modules`` entry. A static input
    needs no entry.

    Raises:
        ValueError: When the table is at the old place (``deploy.tensorrt.input_shapes``), or
            an entry is not a valid :class:`ShapeProfile`.
    """
    tensorrt_cfg = deploy_cfg.tensorrt
    key = f"deploy.tensorrt.modules.{module_name}.input_shapes"
    if "input_shapes" in tensorrt_cfg:
        raise ValueError(
            "deploy.tensorrt.input_shapes is not read: the shapes of each exported module live "
            f"under deploy.tensorrt.modules.<module>.input_shapes. Move the entries to {key}."
        )
    module_cfg = (tensorrt_cfg.get("modules") or {}).get(module_name) or {}
    raw_shapes = module_cfg.get("input_shapes", None)
    input_shapes = None
    if raw_shapes:
        input_shapes = {}
        for name, shapes in OmegaConf.to_container(raw_shapes, resolve=True).items():
            try:
                input_shapes[str(name)] = ShapeProfile.model_validate(shapes)
            except ValidationError as error:
                raise ValueError(f"{key}.{name}: {error}") from None
    build_engine(
        onnx_path,
        output_path,
        workspace_size=int(tensorrt_cfg.get("workspace_size", 1 << 30)),
        plugin_libraries=tuple(str(p) for p in (tensorrt_cfg.get("plugin_libraries") or ())),
        input_shapes=input_shapes,
        input_shapes_key=key,
    )
