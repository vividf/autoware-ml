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


"""ONNX-side machinery of deployment: export primitive, precision passes, graph surgery."""

from autoware_ml.deployment.onnx.export import (
    build_dynamic_axes,
    build_dynamic_shapes,
    export_to_onnx,
    normalize_dynamic_shapes_for_model,
)
from autoware_ml.deployment.onnx.modify import modify_onnx_graph, should_modify_graph
from autoware_ml.deployment.onnx.autocast import (
    autocast_to_fp16,
    keep_topk_in_fp16,
)
from autoware_ml.deployment.onnx.precision import (
    cast_graph_to_fp16,
    onnx_custom_op_domains,
    onnx_has_qdq,
)

__all__ = [
    "autocast_to_fp16",
    "build_dynamic_axes",
    "build_dynamic_shapes",
    "cast_graph_to_fp16",
    "export_to_onnx",
    "keep_topk_in_fp16",
    "modify_onnx_graph",
    "normalize_dynamic_shapes_for_model",
    "onnx_custom_op_domains",
    "onnx_has_qdq",
    "should_modify_graph",
]
