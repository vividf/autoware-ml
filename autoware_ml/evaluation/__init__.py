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

"""Ground-truth evaluation of a model on any inference backend.

One evaluate loop scores whatever ran the forward — the PyTorch modules, an ONNX
session, or a TensorRT engine — with the model's own metric suites, and reports
under the same ``{split}/{backend}/{metric}`` keys the Lightning trainer uses
(:mod:`autoware_ml.metrics.report`). Latency is collected per stage alongside.
"""

from autoware_ml.evaluation.evaluator import (
    EvaluationResult,
    evaluate_backend,
    log_backend_report,
    log_comparison,
    log_results_to_mlflow,
)
from autoware_ml.evaluation.latency import LatencyStats

__all__ = [
    "EvaluationResult",
    "LatencyStats",
    "evaluate_backend",
    "log_backend_report",
    "log_comparison",
    "log_results_to_mlflow",
]
