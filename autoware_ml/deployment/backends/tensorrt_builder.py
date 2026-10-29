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
    import tensorrt as trt

    return trt.Logger(trt.Logger.WARNING)


@functools.cache
def _init_plugin_registry() -> None:
    import tensorrt as trt

    trt.init_libnvinfer_plugins(trt_logger(), "")


def _create_builder(workspace_size: int, plugin_libraries: Sequence[str]):
    """Create ``(builder, network, parser, config)`` for one strongly typed engine build."""
    import tensorrt as trt

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
    builder: Any, network: Any, input_shapes: Mapping[str, ShapeProfile]
):
    """Build a profile over the configured inputs this network has.

    ``deploy.tensorrt.input_shapes`` is shared by every exported module, so entries for another
    module's inputs are skipped rather than handed to TensorRT.
    """
    network_inputs = {network.get_input(i).name for i in range(network.num_inputs)}
    profile = builder.create_optimization_profile()
    applied = 0
    for input_name, shapes in input_shapes.items():
        if input_name not in network_inputs:
            logger.debug(
                "Optimization profile entry '%s' is not an input of this network (%s); skipped.",
                input_name,
                sorted(network_inputs),
            )
            continue
        profile.set_shape(
            input_name,
            min=list(shapes.min_shape),
            opt=list(shapes.opt_shape),
            max=list(shapes.max_shape),
        )
        applied += 1
        logger.info(
            "Optimization profile for '%s': min=%s, opt=%s, max=%s",
            input_name,
            list(shapes.min_shape),
            list(shapes.opt_shape),
            list(shapes.max_shape),
        )
    if applied == 0:
        logger.warning(
            "None of the %d configured optimization-profile entries is an input of this "
            "network (%s); building without a profile.",
            len(input_shapes),
            sorted(network_inputs),
        )
    return profile, applied


def build_engine(
    onnx_path: str | Path,
    output_path: str | Path,
    *,
    workspace_size: int = 1 << 30,
    plugin_libraries: Sequence[str] = (),
    input_shapes: Mapping[str, ShapeProfile] | None = None,
) -> None:
    """Build and serialize a strongly typed TensorRT engine from an ONNX file.

    Args:
        workspace_size: Workspace memory-pool limit in bytes.
        plugin_libraries: Custom plugin libraries to load before parsing.
        input_shapes: Optimization profile per dynamic input; ``None`` for a static graph.

    Raises:
        RuntimeError: When ONNX parsing or the engine build fails.
    """
    onnx_path, output_path = Path(onnx_path), Path(output_path)
    logger.info("Building TensorRT engine (strongly typed)...")
    builder, network, parser, config = _create_builder(workspace_size, plugin_libraries)
    _parse_onnx_file(parser, onnx_path)

    if input_shapes:
        profile, applied = _create_optimization_profile(builder, network, input_shapes)
        if applied:
            config.add_optimization_profile(profile)

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
) -> None:
    """Build an engine with :func:`build_engine` from the ``deploy.tensorrt`` settings."""
    tensorrt_cfg = deploy_cfg.tensorrt
    raw_shapes = tensorrt_cfg.get("input_shapes", None)
    input_shapes = None
    if raw_shapes:
        input_shapes = {}
        for name, shapes in OmegaConf.to_container(raw_shapes, resolve=True).items():
            try:
                input_shapes[str(name)] = ShapeProfile.model_validate(shapes)
            except ValidationError as error:
                raise ValueError(f"deploy.tensorrt.input_shapes.{name}: {error}") from None
    build_engine(
        onnx_path,
        output_path,
        workspace_size=int(tensorrt_cfg.get("workspace_size", 1 << 30)),
        plugin_libraries=tuple(str(p) for p in (tensorrt_cfg.get("plugin_libraries") or ())),
        input_shapes=input_shapes,
    )
