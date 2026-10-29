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

"""Unit tests for the TensorRT runner (GPU)."""

from __future__ import annotations

import gc

import pytest
import torch

from autoware_ml.deployment.backends.onnx_runner import OnnxModuleRunner
from autoware_ml.deployment.tests.test_onnx_runner import _gemm_graph

pytest.importorskip("tensorrt", reason="the TensorRT runner needs the tensorrt package")

from autoware_ml.deployment.backends.tensorrt_builder import build_engine, trt_logger  # noqa: E402
from autoware_ml.deployment.backends.tensorrt_runner import TensorRTModuleRunner  # noqa: E402

REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="TensorRT needs CUDA")


@REQUIRES_CUDA
def test_engine_matches_onnxruntime_and_reuses_its_output_buffers(tmp_path) -> None:
    onnx_path = _gemm_graph(tmp_path, quantized=False)
    engine_path = tmp_path / "plain.engine"
    build_engine(onnx_path, engine_path)

    runner = TensorRTModuleRunner(engine_path, torch.device("cuda"))
    reference = OnnxModuleRunner(onnx_path, torch.device("cpu"))
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    outputs, elapsed_ms = runner.run({"x": x.cuda()})
    expected, _ = reference.run({"x": x})
    assert runner.input_names == ["x"] and runner.output_names == ["y"]
    assert elapsed_ms >= 0.0
    torch.testing.assert_close(outputs["y"].cpu(), expected["y"])

    first_buffer = outputs["y"].data_ptr()
    outputs, _ = runner.run({"x": (x + 1).cuda()})
    assert outputs["y"].data_ptr() == first_buffer
    torch.testing.assert_close(outputs["y"].cpu(), x + 1)


@REQUIRES_CUDA
def test_engines_share_one_logger_and_survive_a_reload(tmp_path) -> None:
    # One logger per process: a logger created per engine is destroyed before the runtime and
    # engine that log through it, which crashed the second load of a plugin engine.
    onnx_path = _gemm_graph(tmp_path, quantized=False)
    engine_path = tmp_path / "plain.engine"
    build_engine(onnx_path, engine_path)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).cuda()

    first = TensorRTModuleRunner(engine_path, torch.device("cuda"))
    assert first.runtime.logger is trt_logger()
    expected, _ = first.run({"x": x})
    expected = expected["y"].clone()
    del first
    gc.collect()

    second = TensorRTModuleRunner(engine_path, torch.device("cuda"))
    outputs, _ = second.run({"x": x})
    torch.testing.assert_close(outputs["y"], expected)


@REQUIRES_CUDA
def test_runner_rejects_wrong_input_names_and_shapes(tmp_path) -> None:
    onnx_path = _gemm_graph(tmp_path, quantized=False)
    engine_path = tmp_path / "plain.engine"
    build_engine(onnx_path, engine_path)
    runner = TensorRTModuleRunner(engine_path, torch.device("cuda"))
    x = torch.ones(2, 4, device="cuda")

    with pytest.raises(ValueError, match=r"missing \['x'\], unexpected \['z'\]"):
        runner.run({"z": x})
    with pytest.raises(ValueError, match=r"unexpected \['z'\]"):
        runner.run({"x": x, "z": x})
    # The graph is static [2, 4].
    with pytest.raises(RuntimeError, match=r"rejected shape \(3, 4\) for input 'x'"):
        runner.run({"x": torch.ones(3, 4, device="cuda")})

    # The rejected shape is not recorded as bound, so a valid call still runs.
    outputs, _ = runner.run({"x": x})
    torch.testing.assert_close(outputs["y"], x)


def test_runner_refuses_a_non_cuda_device(tmp_path) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        TensorRTModuleRunner(tmp_path / "missing.engine", torch.device("cpu"))
