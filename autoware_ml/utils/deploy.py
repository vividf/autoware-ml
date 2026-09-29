# Copyright 2025 TIER IV, Inc.
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

"""Small device / path helpers the deploy, quantize and evaluation entrypoints share.

The export contract itself (``ExportSpec``, the ONNX export loop helpers) lives in
:mod:`autoware_ml.deployment.export`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def validate_cuda_available() -> None:
    """Ensure CUDA is available for deployment export."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. TensorRT requires CUDA. "
            "Please run on a machine with CUDA support."
        )


def resolve_output_paths(
    checkpoint_path: Path,
    output_name: str | None,
    output_dir: str | None,
) -> tuple[Path, Path, Path]:
    """Resolve the output directory and export artifact paths."""
    base_name = output_name if output_name else checkpoint_path.stem
    output_directory = Path(output_dir) if output_dir else checkpoint_path.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    onnx_path = output_directory / f"{base_name}.onnx"
    engine_path = output_directory / f"{base_name}.engine"
    return output_directory, onnx_path, engine_path


def move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors nested in common Python containers to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value
