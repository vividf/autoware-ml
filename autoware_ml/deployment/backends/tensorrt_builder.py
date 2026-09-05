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

"""TensorRT engine builder — the build half of the TensorRT backend.

Owns everything between an ONNX file and a serialized ``.engine``: builder/network
creation, the workspace pool, optimization profiles, plugin loading, and
serialization. The runtime half lives next door in :mod:`.tensorrt_runner`.

Every network is built STRONGLY TYPED: the ONNX graph's own tensor types are
binding, and precision therefore lives in the ONNX, not in builder flags —
quantized precisions come from Quantize/DequantizeLinear nodes (explicit
quantization), FP16 from the exported graph's tensor types
(:func:`autoware_ml.deployment.onnx.autocast.autocast_to_fp16`). This matches the
TensorRT direction: the weak-typing precision flags (``BuilderFlag.FP16`` & co)
were deprecated in TensorRT 10.12 and removed in TensorRT 11, where all networks
are strongly typed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from autoware_ml.deployment.config import ShapeProfile

logger = logging.getLogger(__name__)


def load_tensorrt_plugin_libraries(plugin_libraries: Sequence[str] | None) -> None:
    """Load custom TensorRT plugin shared libraries before plugin registry init.

    Args:
        plugin_libraries: Paths to plugin ``.so`` files, or None/empty for none.

    Raises:
        FileNotFoundError: If a configured plugin library does not exist.
    """
    if not plugin_libraries:
        return
    import ctypes

    for library in plugin_libraries:
        library_path = Path(library)
        if not library_path.exists():
            raise FileNotFoundError(f"TensorRT plugin library not found: {library_path}")
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        logger.info("Loaded TensorRT plugin library: %s", library_path)


def _create_builder(workspace_size: int, plugin_libraries: Sequence[str]):
    """Create ``(builder, network, parser, config)`` for one strongly typed engine build."""
    import tensorrt as trt

    # Custom plugins must be loadable before plugin registry initialization.
    load_tensorrt_plugin_libraries(plugin_libraries)

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    builder = trt.Builder(trt_logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, trt_logger)
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


def _create_optimization_profile(builder: Any, input_shapes: Mapping[str, ShapeProfile]):
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


def build_engine(
    onnx_path: Path,
    output_path: Path,
    *,
    workspace_size: int = 1 << 32,
    plugin_libraries: Sequence[str] = (),
    input_shapes: Mapping[str, ShapeProfile] | None = None,
) -> None:
    """Build and serialize one strongly typed TensorRT engine from an ONNX file.

    Precision is read from the ONNX graph (module docstring); there are no
    precision knobs here by design.

    Args:
        onnx_path: Exported ONNX model.
        output_path: Destination ``.engine`` path.
        workspace_size: Workspace memory-pool limit in bytes.
        plugin_libraries: Custom plugin ``.so`` paths to load before parsing.
        input_shapes: Optimization-profile shapes per dynamic input; ``None``/empty
            builds without an explicit profile (static-shape graphs).

    Raises:
        RuntimeError: When ONNX parsing or the engine build fails.
    """
    logger.info("Building TensorRT engine (strongly typed)...")
    builder, network, parser, config = _create_builder(workspace_size, plugin_libraries)
    _parse_onnx_file(parser, onnx_path)

    if input_shapes:
        config.add_optimization_profile(_create_optimization_profile(builder, input_shapes))

    logger.info("Building TensorRT engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    logger.info("Successfully built TensorRT engine: %s", output_path)
