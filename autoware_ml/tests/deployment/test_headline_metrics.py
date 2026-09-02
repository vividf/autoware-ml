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

"""Which metrics a deployment report leads with is the metric suite's declaration, not
the evaluator's: a task the evaluator has never heard of still gets a headline."""

from __future__ import annotations

import logging

from autoware_ml.evaluation.evaluator import EvaluationResult, log_backend_report, log_comparison
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.segmentation3d.suite import Segmentation3DMetricSuite
from autoware_ml.types.backend import Backend


def _result(backend: Backend, headline: tuple[str, ...]) -> EvaluationResult:
    return EvaluationResult(
        backend=backend,
        device="cuda",
        split="test",
        metrics={
            f"test/{backend.value}/seg3d/mIoU": 0.61,
            f"test/{backend.value}/seg3d/iou_car": 0.9,
            f"test/{backend.value}/det3d/mAP": 0.45,
        },
        latency={},
        num_samples=10,
        headline_metrics=headline,
    )


def test_suites_declare_their_headline_metrics() -> None:
    assert Segmentation3DMetricSuite.headline_metrics == ("mIoU", "fwIoU")
    assert Detection3DMetricSuite.headline_metrics == ("mAP", "NDS")


def test_report_leads_with_the_declared_metrics_only(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="autoware_ml.evaluation.evaluator"):
        log_backend_report(_result(Backend.PYTORCH, ("mIoU",)))
    logged = caplog.text
    assert "seg3d/mIoU" in logged
    # Not declared headline: the per-class breakdown and the other task's metric.
    assert "iou_car" not in logged
    assert "mAP" not in logged


def test_comparison_table_rows_come_from_the_declared_metrics(caplog) -> None:
    results = [_result(Backend.PYTORCH, ("mIoU",)), _result(Backend.TENSORRT, ("mIoU",))]
    with caplog.at_level(logging.INFO, logger="autoware_ml.evaluation.evaluator"):
        log_comparison(results)
    assert "test/seg3d/mIoU" in caplog.text
    assert "det3d/mAP" not in caplog.text


def test_a_suite_declaring_nothing_reports_no_headline(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="autoware_ml.evaluation.evaluator"):
        log_backend_report(_result(Backend.PYTORCH, ()))
    assert "mIoU" not in caplog.text


def test_fallback_stages_are_visible_in_report_and_comparison(caplog) -> None:
    """A backend whose stages ran in torch must say so — never a silent pytorch copy."""
    from dataclasses import replace

    starred = replace(_result(Backend.ONNX, ("mIoU",)), fallback_stages=("encoder", "head"))
    clean = _result(Backend.TENSORRT, ("mIoU",))
    with caplog.at_level(logging.INFO, logger="autoware_ml.evaluation.evaluator"):
        log_backend_report(starred)
        log_comparison([starred, clean])
    assert "encoder, head" in caplog.text
    assert "onnx*" in caplog.text
    assert "tensorrt*" not in caplog.text


def test_pipeline_reports_its_fallback_stages(tmp_path) -> None:
    import onnx
    from onnx import TensorProto, helper
    import torch
    from torch import nn

    from autoware_ml.deployment.pipeline import StagedPipeline
    from autoware_ml.deployment.stages import GraphStage, TorchStage

    x = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node("Identity", ["mid"], ["y"])], "g", [x], [y])
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
        str(tmp_path / "plain.onnx"),
    )

    def seed(context):
        return {"x": torch.ones(1, 2)}

    stages = (
        TorchStage("seed", run=seed),
        GraphStage(
            "falls_back",
            module=nn.Identity(),
            inputs=("x",),
            outputs=("mid",),
            torch_fallback_backends=(Backend.ONNX,),
        ),
        GraphStage(
            "plain", module=nn.Identity(), inputs=("mid",), outputs=("y",),
            output_fields=(("y", "y"),),
        ),
    )
    onnx_pipeline = StagedPipeline(
        stages, backend=Backend.ONNX, device=torch.device("cpu"), artifacts_dir=tmp_path
    )
    assert onnx_pipeline.fallback_stage_names == ("falls_back",)
    torch_pipeline = StagedPipeline(stages, backend=Backend.PYTORCH, device=torch.device("cpu"))
    assert torch_pipeline.fallback_stage_names == ()
