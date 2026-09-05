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

"""CenterPoint-specific quantization declaration.

The one model-specific fact the quantization engine deliberately does not own:
*which top-level submodules* carry *which quantizable module kinds*. Everything
else — BN fusion, ``skip_quantize`` resolution, module replacement, architecture
recipes, placement recording — is the generic
:class:`~autoware_ml.quantization.plan.QuantizationPlan`.

Per-submodule module kinds are architecture facts and live here in code, not in
config: ``pts_backbone`` → conv **and** linear (ConvNeXt pointwise, future
backbones), ``pts_neck`` / ``bbox_head`` → conv, ``pts_voxel_encoder`` → linear.
Precision placement stays declarative: config's ``skip_quantize`` is the single
opt-out (a matched module and its whole subtree stay un-quantized), and the recipes are
always-on and class-gated (a plain SECOND backbone matches none of them — zero
matches are normal); ``disable_recipes`` opts a config out of one by name.

Every stage — quantize (PTQ / QAT) and deploy-load — and the QAT callback all reach this
declaration through ``CenterPointDetectionModel.build_quantization_plan`` — the
same plan builds the same tree everywhere, so the calibrated ``state_dict`` and
the deploy ``load_state_dict`` line up by construction (and the plan's placement
record lets the loader verify that).
"""

from __future__ import annotations

from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan, QuantRules

#: CenterPoint's quantization declaration. Submodules absent on a model variant are
#: skipped, so this one rules object serves every CenterPoint composition.
CENTERPOINT_QUANT_RULES = QuantRules(
    quantize_submodules={
        "pts_backbone": ("conv", "linear"),
        "pts_neck": ("conv",),
        "bbox_head": ("conv",),
        "pts_voxel_encoder": ("linear",),
    },
)


def build_centerpoint_quantization_plan(config: QuantizationConfig) -> QuantizationPlan:
    """Bind CenterPoint's quantization rules to a parsed config.

    Args:
        config: Parsed ``quantization`` config block.

    Returns:
        The :class:`~autoware_ml.quantization.plan.QuantizationPlan` shared by
        the quantize stage (PTQ / QAT) and the deploy loader.
    """
    return QuantizationPlan(rules=CENTERPOINT_QUANT_RULES, config=config)
