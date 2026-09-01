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

"""BEVFusion (lidar-only) quantization declaration.

Quantization covers the dense deployment graph only: ``pts_backbone`` /
``pts_neck`` (Conv2d) and the head's Conv2d layers (shared conv, heatmap head).
The sparse side (``pts_voxel_encoder`` / ``pts_middle_encoder``) is deliberately
absent — its INT8 form is the libspconv engine produced by the dedicated sparse
exporter, not Q/DQ replacement. The decoder's attention/Linear/Conv1d layers
carry no quantizable kinds here and stay in floating point.

Every stage — quantize (PTQ / QAT) and deploy-load — reaches this declaration
through ``BEVFusionLidarDetectionModel.build_quantization_plan``, so the same
plan builds the same tree everywhere.
"""

from __future__ import annotations

from autoware_ml.quantization.config import QuantizationConfig
from autoware_ml.quantization.plan import QuantizationPlan, QuantRules

#: BEVFusion lidar quantization declaration (dense graph only; see module docstring).
BEVFUSION_LIDAR_QUANT_RULES = QuantRules(
    quantize_submodules={
        "pts_backbone": ("conv",),
        "pts_neck": ("conv",),
        "bbox_head": ("conv",),
    },
    recipes=("residual_add", "ese", "maxpool"),
)


def build_bevfusion_quantization_plan(config: QuantizationConfig) -> QuantizationPlan:
    """Bind BEVFusion's quantization rules to a parsed config.

    Args:
        config: Parsed ``quantization`` config block.

    Returns:
        The :class:`~autoware_ml.quantization.plan.QuantizationPlan` shared by
        the quantize stage (PTQ / QAT) and the deploy loader.
    """
    return QuantizationPlan(rules=BEVFUSION_LIDAR_QUANT_RULES, config=config)
