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

"""PTv3-specific quantization declaration.

PTv3's arithmetic is almost entirely GEMMs behind ``nn.Linear``: the attention
projections (``attn.qkv``, ``attn.proj``), the MLP blocks, the patch embedding, and the
head's classifier — 74 of them in the trained segmentation checkpoint. Those are what
INT8 buys here, so both submodules declare the ``linear`` kind.

What is deliberately *not* declared:

- The attention core itself (the ``q @ k`` and ``attn @ v`` batched matmuls). They are
  not modules, so there is nothing for module replacement to swap; quantizing them needs
  functional-level insertion and its own accuracy study.
- The serialization and pooling glue (``spconv``-style scatter/gather, the
  ``cpe`` depthwise convolutions on sparse tensors). Their cost is memory movement, not
  multiply-accumulate, so INT8 would add quantize/dequantize traffic for no gain.
"""

from __future__ import annotations

from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan, QuantRules

#: PTv3's quantization declaration. The GEMM-bearing submodules, nothing else.
PTV3_QUANT_RULES = QuantRules(
    quantize_submodules={
        "encoder": ("linear",),
        "seg3d_head": ("linear",),
    },
)


def build_ptv3_quantization_plan(config: QuantizationConfig) -> QuantizationPlan:
    """Bind PTv3's quantization rules to a parsed config.

    Args:
        config: Parsed ``quantization`` config block.

    Returns:
        The :class:`~autoware_ml.quantization.plan.QuantizationPlan` shared by the
        quantize stage (PTQ / QAT) and the deploy loader.
    """
    return QuantizationPlan(rules=PTV3_QUANT_RULES, config=config)
