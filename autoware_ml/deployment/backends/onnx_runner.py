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

"""ONNX Runtime runner for one graph stage."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import onnxruntime as ort
import torch

from autoware_ml.deployment.onnx.inspect import onnx_has_qdq

logger = logging.getLogger(__name__)

# Feeds are cast to the declared input types; ONNX Runtime rejects a mismatched dtype.
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

    Without the CUDA execution provider, a CUDA ``device`` is an error rather than a silent
    CPU fallback.
    """

    def __init__(self, onnx_path: str | Path, device: torch.device) -> None:
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

        session_options = ort.SessionOptions()
        if onnx_has_qdq(onnx_path):
            # ONNX Runtime's graph optimizer (the whole BASIC level, not one transformer)
            # mis-executes fp16-typed Q/DQ graphs: on the BEVFusion dense stage the heatmap's
            # mean error grows 30x (3.7e-3 vs 1.4e-4) and mAP drops from 0.45 to 0.39, while
            # the unoptimized run matches TensorRT. ORT is the correctness reference, not a
            # latency target, so quantized graphs run as exported.
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            logger.info(
                "ONNX module %s carries Q/DQ nodes: ONNX Runtime graph optimizations disabled "
                "(the optimizer mis-executes fp16-typed Q/DQ).",
                onnx_path.name,
            )
        self.session = ort.InferenceSession(str(onnx_path), session_options, providers=providers)
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
        """Run the module once and return the outputs by name and the time in ms.

        Only ``session.run`` is timed, not the host / device copies. Inputs may be on any
        device; outputs are on :attr:`device`.
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
