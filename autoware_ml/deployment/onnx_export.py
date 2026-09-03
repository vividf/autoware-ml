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


"""DEPRECATED module path — the content moved to :mod:`autoware_ml.deployment.onnx`.

Kept as a re-export shim for one deprecation cycle; import from
``autoware_ml.deployment.onnx.{export,precision,modify}`` instead.
"""

from autoware_ml.deployment.onnx.export import (  # noqa: F401
    _log_export_inputs,
    _merge_onnx_external_data,
    build_dynamic_axes,
    build_dynamic_shapes,
    export_to_onnx,
    normalize_dynamic_shapes_for_model,
)
from autoware_ml.deployment.onnx.modify import (  # noqa: F401
    _apply_modifier,
    _instantiate_modifier,
    modify_onnx_graph,
    should_modify_graph,
)
from autoware_ml.deployment.onnx.precision import (  # noqa: F401
    _assign_missing_node_names,
    _quantized_island_names,
    autocast_to_fp16,
    cast_graph_to_fp16,
    onnx_custom_op_domains,
    onnx_has_qdq,
)
