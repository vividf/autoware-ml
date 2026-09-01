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

"""ONNX Runtime module runner.

One implementation of the ONNX Runtime session plumbing (provider selection,
name discovery, tensor conversion, wall-clock timing) shared by every
per-model deployment pipeline, so the run loop cannot drift between backends.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time

import torch

logger = logging.getLogger(__name__)

# ONNX Runtime element-type string -> torch dtype, for casting feeds to the
# graph's declared input types (feeding fp16 into a float32 graph is an error).
_ORT_TYPE_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int64)": torch.int64,
    "tensor(int32)": torch.int32,
    "tensor(int8)": torch.int8,
    "tensor(uint8)": torch.uint8,
    "tensor(bool)": torch.bool,
}


class OnnxModuleRunner:
    """Run one exported ONNX module through ONNX Runtime.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        device: Torch device the module should execute on. ``cuda`` requires the
            CUDA execution provider.
    """

    def __init__(self, onnx_path: str | Path, device: torch.device) -> None:
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX module not found: {onnx_path}")

        self.device = torch.device(device)
        if self.device.type == "cuda":
            providers = [
                ("CUDAExecutionProvider", {"device_id": self.device.index or 0}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        if (
            self.device.type == "cuda"
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError(
                f"CUDA execution provider unavailable for ONNX module {onnx_path.name} — "
                "ONNX Runtime silently fell back to CPU, which would corrupt latency "
                "comparisons. Install onnxruntime-gpu / check the CUDA setup, or request "
                "device=cpu explicitly."
            )
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]
        self._input_torch_dtypes = {
            node.name: _ORT_TYPE_TO_TORCH_DTYPE.get(node.type) for node in self.session.get_inputs()
        }
        logger.info(
            "Loaded ONNX module %s (inputs=%s, outputs=%s, providers=%s)",
            onnx_path.name,
            self.input_names,
            self.output_names,
            self.session.get_providers(),
        )

    def run(self, inputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float]:
        """Run the module once.

        Args:
            inputs: Input tensor per ONNX input name. Tensors may live on any device.

        Returns:
            Tuple of (outputs by ONNX output name on :attr:`device`, wall-clock time in ms
            for ``session.run`` only — host/device transfers excluded).
        """
        feed = {}
        for name, tensor in inputs.items():
            expected_dtype = self._input_torch_dtypes.get(name)
            if expected_dtype is not None and tensor.dtype != expected_dtype:
                tensor = tensor.to(expected_dtype)
            feed[name] = tensor.detach().cpu().numpy()
        start = time.perf_counter()
        raw_outputs = self.session.run(self.output_names, feed)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        outputs = {
            name: torch.from_numpy(array).to(self.device)
            for name, array in zip(self.output_names, raw_outputs)
        }
        return outputs, elapsed_ms
