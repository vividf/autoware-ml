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

"""The ONNX ABI of the ``autoware_tensorrt_plugins`` sparse-convolution nodes.

Domain, op names, input slots and attribute values, shared by the export symbolics and the
ONNX passes (source: autoware_universe ``perception/autoware_tensorrt_plugins``).
"""

from __future__ import annotations

AUTOWARE_DOMAIN = "autoware"

GET_INDICE_PAIRS_OP = "GetIndicePairs"
INDICE_CONV_OP = "IndiceConv"
GET_INDICE_PAIRS_IMPLICIT_GEMM_OP = "GetIndicePairsImplicitGemm"
IMPLICIT_GEMM_OP = "ImplicitGemm"


def qualified(op_type: str) -> str:
    """``domain::op`` as ``torch.onnx`` symbolics spell a custom op."""
    return f"{AUTOWARE_DOMAIN}::{op_type}"


#: ``ImplicitGemm`` inputs: features, weight, pair_fwd, pair_mask, mask_argsort and, in
#: slot 5, the optional bias the fusion folds in.
IMPLICIT_GEMM_INPUTS_WITHOUT_BIAS = 5
IMPLICIT_GEMM_BIAS_SLOT = 5

#: ``ImplicitGemm`` attributes the ONNX passes write; the symbolics emit them by name.
ACT_TYPE_ATTR = "act_type"
PRECISION_ATTR = "precision"
INPUT_SCALE_ATTR = "input_scale"
OUTPUT_SCALE_ATTR = "output_scale"
#: ``GetIndicePairsImplicitGemm`` attribute: whether the pair masks are argsorted.
DO_SORT_ATTR = "do_sort"

#: ``ImplicitGemm.act_type`` values; the plugin mirrors cumm's ``tv::gemm::Activation``.
ACT_NONE = 0
ACT_RELU = 1

#: ``ImplicitGemm.precision`` values (``ImplicitGemmParameters::precision``).
PRECISION_FP = 0
PRECISION_INT8 = 1

#: The INT8 form (``precision = 1``) has seven inputs: slot 5 ``channel_scale`` and slot 6
#: ``bias_scaled``, both read by the plugin as ``float*``. A precision cast must leave
#: them fp32 whatever it does to the rest of the graph.
IMPLICIT_GEMM_INT8_INPUTS = 7
IMPLICIT_GEMM_FP32_INPUT_SLOTS: dict[int, tuple[int, ...]] = {IMPLICIT_GEMM_INT8_INPUTS: (5, 6)}
