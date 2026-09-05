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

"""Shared TensorRT engine runner (torch-native I/O).

One battle-tested implementation of the TensorRT run loop, reused by every
per-model deployment pipeline so the GPU plumbing cannot drift between
backends. Unlike the classic pycuda variant this runner uses torch CUDA
tensors as device buffers: inputs that already live on the GPU are bound
in place (no host round-trip) and outputs are returned as CUDA tensors,
which is exactly what the downstream torch stages (scatter, decode) want.

Timing brackets only ``execute_async_v3`` with CUDA events on the current
torch stream, so the reported time is the engine's pure GPU compute.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_trt_engine(engine_path: str | Path, *, component_name: str | None = None):
    """Deserialize a TensorRT engine and create its execution context, failing loud.

    Args:
        engine_path: Path to the serialized ``.engine`` file.
        component_name: Optional component label for error messages.

    Returns:
        Tuple of ``(engine, execution_context)``.

    Raises:
        RuntimeError: If deserialization or context creation fails
            (context failure is usually GPU out-of-memory).
    """
    import tensorrt as trt

    engine_path = Path(engine_path)
    label = component_name or engine_path.name
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(
            f"Failed to create TensorRT execution context for {label} (likely GPU out-of-memory)."
        )
    return engine, context


def list_trt_io_names(engine) -> tuple[list[str], list[str]]:
    """Return ``(input_names, output_names)`` in TensorRT tensor-index order."""
    import tensorrt as trt

    inputs: list[str] = []
    outputs: list[str] = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return inputs, outputs


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    """Map a TensorRT dtype to the matching torch dtype, failing loud on unknowns.

    Guessing a size (e.g. defaulting to float32) would mis-size the GPU buffer and
    silently corrupt the data, so unknown dtypes raise via TensorRT's own ``nptype``.
    """
    import tensorrt as trt

    numpy_dtype = np.dtype(trt.nptype(trt_dtype))
    return torch.from_numpy(np.zeros(0, dtype=numpy_dtype)).dtype


class TensorRTModuleRunner:
    """Run one serialized TensorRT engine with torch tensors as device buffers.

    Args:
        engine_path: Path to the serialized ``.engine`` file.
        device: CUDA device the engine executes on.
    """

    def __init__(self, engine_path: str | Path, device: torch.device) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRT requires a CUDA device, got {self.device}.")
        self.engine, self.context = load_trt_engine(engine_path)
        self.input_names, self.output_names = list_trt_io_names(self.engine)
        # Bindings persist across calls: input shapes are re-declared and output buffers
        # re-allocated only when a shape actually changes. Re-binding every call was
        # measured to inflate the reported per-engine time by ~0.5 ms/engine on
        # BEVFusion (the deployment report showed 6.08 ms for a chain whose paired
        # single-window measurement is 5.16 ms).
        self._bound_input_shapes: dict[str, tuple[int, ...]] = {}
        self._output_buffers: dict[str, torch.Tensor] = {}
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        logger.info(
            "Loaded TensorRT engine %s (inputs=%s, outputs=%s)",
            Path(engine_path).name,
            self.input_names,
            self.output_names,
        )

    def _cast_to_binding_dtype(self, tensor_name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``tensor`` on :attr:`device`, contiguous, in the engine binding's dtype.

        Matching the binding dtype is critical for FP16 engines: a graph traced with
        FP32 inputs may bind as ``HALF``, and feeding float32 bytes into a HALF binding
        misaligns the GPU buffer and silently corrupts the activations.
        """
        target_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(tensor_name))
        if tensor.dtype != target_dtype:
            logger.debug(
                "[trt-io] casting tensor %r: %s -> %s", tensor_name, tensor.dtype, target_dtype
            )
            tensor = tensor.to(target_dtype)
        return tensor.to(self.device).contiguous()

    def run(self, inputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        """Run the engine once and return ``(outputs_by_name, pure_gpu_time_ms)``.

        Args:
            inputs: Engine input tensor name -> torch tensor (any device / dtype;
                cast and moved as needed).

        Returns:
            Tuple of (outputs by engine output name as CUDA tensors, pure-GPU time in
            ms measured with CUDA events around ``execute_async_v3`` only).

        NOTE: output tensors are owned by the runner and REUSED on the next ``run``
        with the same shapes — consume (or copy) them before calling ``run`` again.
        Every current caller is strictly sequential per runner (evaluation processes a
        frame to completion; verification compares per batch, and its reference and
        test pipelines hold separate runners).

        Raises:
            RuntimeError: If ``execute_async_v3`` reports a failure status.
        """
        device_inputs = {
            name: self._cast_to_binding_dtype(name, tensor) for name, tensor in inputs.items()
        }
        shapes_changed = False
        for name, tensor in device_inputs.items():
            shape = tuple(tensor.shape)
            if self._bound_input_shapes.get(name) != shape:
                self.context.set_input_shape(name, shape)
                self._bound_input_shapes[name] = shape
                shapes_changed = True
            # Input tensors arrive from the caller, so their addresses change per call.
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        # Output shapes can depend on the input shapes, so re-derive (and re-allocate
        # only what actually changed) when any input shape did.
        if shapes_changed or not self._output_buffers:
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                buffer = self._output_buffers.get(name)
                if buffer is None or tuple(buffer.shape) != shape:
                    buffer = torch.empty(
                        shape,
                        dtype=_trt_dtype_to_torch(self.engine.get_tensor_dtype(name)),
                        device=self.device,
                    )
                    self._output_buffers[name] = buffer
                    self.context.set_tensor_address(name, int(buffer.data_ptr()))

        stream = torch.cuda.current_stream(self.device)
        self._start_event.record(stream)
        succeeded = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not succeeded:
            raise RuntimeError("TensorRT execute_async_v3 returned failure status.")
        self._end_event.record(stream)
        self._end_event.synchronize()
        gpu_time_ms = float(self._start_event.elapsed_time(self._end_event))

        return dict(self._output_buffers), gpu_time_ms
