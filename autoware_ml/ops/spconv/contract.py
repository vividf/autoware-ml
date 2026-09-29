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

"""The exported sparse graph's ABI with ``autoware_tensorrt_plugins``, in one place.

The export symbolics emit these nodes, the ONNX passes rewrite them, the precision cast
types their inputs and the runtime plugin reads them; every one of those spells the same
domain, op names, input slots and attribute enums, so they are defined here and nowhere
else (source: ``autoware_universe/perception/autoware_tensorrt_plugins``).
"""

from __future__ import annotations

#: ONNX domain of every plugin node.
AUTOWARE_DOMAIN = "autoware"

#: Op names inside :data:`AUTOWARE_DOMAIN`.
GET_INDICE_PAIRS_OP = "GetIndicePairs"
INDICE_CONV_OP = "IndiceConv"
GET_INDICE_PAIRS_IMPLICIT_GEMM_OP = "GetIndicePairsImplicitGemm"
IMPLICIT_GEMM_OP = "ImplicitGemm"


def qualified(op_type: str) -> str:
    """``domain::op`` as ``torch.onnx`` symbolics spell a custom op."""
    return f"{AUTOWARE_DOMAIN}::{op_type}"


#: ``ImplicitGemm`` input slots. The first five are the sparse tensor and its rulebook
#: (features, pair_fwd, pair_mask, mask_argsort, weight); slot 5 is the optional bias the
#: bias/activation fusion folds in.
IMPLICIT_GEMM_INPUTS_WITHOUT_BIAS = 5
IMPLICIT_GEMM_BIAS_SLOT = 5

#: ``ImplicitGemm.act_type`` values — the plugin mirrors cumm's ``tv::gemm::Activation``.
ACT_NONE = 0
ACT_RELU = 1

#: The INT8 form (``precision = 1``) has seven inputs: slot 5 ``channel_scale`` and slot 6
#: ``bias_scaled``, both read by the plugin as ``float*``. A precision cast must leave
#: them fp32 whatever it does to the rest of the graph.
IMPLICIT_GEMM_INT8_INPUTS = 7
IMPLICIT_GEMM_FP32_INPUT_SLOTS: dict[int, tuple[int, ...]] = {IMPLICIT_GEMM_INT8_INPUTS: (5, 6)}
