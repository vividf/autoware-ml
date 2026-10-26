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

"""TensorRT runner for one graph stage."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

from autoware_ml.deployment.backends.tensorrt_builder import load_tensorrt_plugin_libraries

logger = logging.getLogger(__name__)


def load_trt_engine(engine_path: str | Path, *, plugin_libraries: Sequence[str] = ()):
    """Deserialize a TensorRT engine and return ``(engine, execution_context)``.

    Raises:
        RuntimeError: If deserialization or context creation fails.
    """

    engine_path = Path(engine_path)
    label = engine_path.name
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

    # The engine's plugin ops need their libraries in this process, wherever it was built.
    load_tensorrt_plugin_libraries(plugin_libraries)
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
    """Map a TensorRT dtype to torch; an unknown one raises instead of mis-sizing the buffer."""

    numpy_dtype = np.dtype(trt.nptype(trt_dtype))
    return torch.from_numpy(np.zeros(0, dtype=numpy_dtype)).dtype


class TensorRTModuleRunner:
    """Run one serialized TensorRT engine with torch CUDA tensors as I/O buffers.

    Inputs already on the GPU are bound in place and outputs stay on the GPU. The reported
    time covers only ``execute_async_v3``, measured with CUDA events on the current stream.
    """

    def __init__(
        self, engine_path: str | Path, device: torch.device, plugin_libraries: Sequence[str] = ()
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"TensorRT requires a CUDA device, got {self.device}.")
        self.engine, self.context = load_trt_engine(engine_path, plugin_libraries=plugin_libraries)
        self.input_names, self.output_names = list_trt_io_names(self.engine)
        # Re-bind only when a shape changes: re-binding every call added ~0.5 ms per engine to
        # the measured time on BEVFusion.
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
        """Return ``tensor`` on :attr:`device`, contiguous, in the binding's dtype.

        TensorRT reads the raw buffer, so float32 data bound to a ``HALF`` input (an fp16
        export of a graph traced in fp32) would silently corrupt the activations.
        """
        target_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(tensor_name))
        if tensor.dtype != target_dtype:
            logger.debug(
                "[trt-io] casting tensor %r: %s -> %s", tensor_name, tensor.dtype, target_dtype
            )
            tensor = tensor.to(target_dtype)
        return tensor.to(self.device).contiguous()

    def _bind_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        """Declare an input shape, raising when the engine rejects it.

        ``set_input_shape`` only returns False for a wrong rank or a shape outside the profile;
        executing anyway would return the previous call's outputs from the reused buffers. A
        rejected shape is not recorded as bound.
        """
        if self.context.set_input_shape(name, shape):
            self._bound_input_shapes[name] = shape
            return
        self._bound_input_shapes.pop(name, None)
        declared = tuple(self.engine.get_tensor_shape(name))
        allowed = f"engine shape {declared}"
        if -1 in declared:
            low, _, high = self.engine.get_tensor_profile_shape(
                name, self.context.active_optimization_profile
            )
            allowed += f", profile min {tuple(low)} max {tuple(high)}"
        raise RuntimeError(f"TensorRT rejected shape {shape} for input {name!r} ({allowed}).")

    def run(self, inputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        """Run the engine once and return the outputs by name and the GPU time in ms.

        Inputs may be on any device and of any dtype. The outputs are the runner's own buffers,
        which the next call may overwrite: copy them to keep them.

        Raises:
            ValueError: If the input names are not exactly the engine's inputs.
            RuntimeError: If TensorRT rejects an input shape or the execution fails.
        """
        if set(inputs) != set(self.input_names):
            missing = sorted(set(self.input_names) - set(inputs))
            unexpected = sorted(set(inputs) - set(self.input_names))
            raise ValueError(
                f"TensorRT engine inputs are {self.input_names}; "
                f"missing {missing}, unexpected {unexpected}."
            )
        device_inputs = {
            name: self._cast_to_binding_dtype(name, tensor) for name, tensor in inputs.items()
        }
        shapes_changed = False
        for name, tensor in device_inputs.items():
            shape = tuple(tensor.shape)
            if self._bound_input_shapes.get(name) != shape:
                self._bind_input_shape(name, shape)
                shapes_changed = True
            # Input tensors arrive from the caller, so their addresses change per call.
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        # Output shapes can depend on the input shapes.
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
