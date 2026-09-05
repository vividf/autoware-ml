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

"""CenterPoint-specific pruning declaration and distillation loss.

Two architecture facts the generic engine does not own:

- **What is prunable**: the dense BEV subtree ``pts_backbone -> pts_neck -> bbox_head``
  (81% of the INT8 engine time; pure Conv2d/ConvTranspose2d/BatchNorm2d). The PFN stays
  out (memory-bound, 0.85 ms, its output width is the deployed ``spatial_features`` ABI).
- **How outputs are compared for distillation**: heatmap logits as soft labels, regression
  maps only where the teacher sees an object.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.models.detection3d.main_modules.centerpoint.stages import (
    SPATIAL_FEATURES,
    head_output_fields,
)
from autoware_ml.pruning.spec import PruningSpec

#: ONNX output name -> CenterHead branch attribute (all equal except ``reg`` -> ``regs``).
_BRANCH_ATTR = {"reg": "regs"}
#: Teacher heatmap probability above which a BEV cell counts as foreground for the
#: regression distillation (CenterPoint decodes peaks; background regression is noise).
_FOREGROUND_THRESHOLD = 0.1


class CenterPointDenseSubtree(nn.Module):
    """``spatial_features -> (heatmap, reg, height, dim, rot[, vel])`` over the model's own modules.

    ``SECONDFPN.forward`` (iterates a list input) and ``CenterHead.forward`` (builds a
    pydantic dataclass) cannot be traced by ``torch.fx``; both are inlined here so their
    children are traced — a module traced as a leaf is un-prunable.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.branches = tuple(
            _BRANCH_ATTR.get(onnx_name, onnx_name) for onnx_name, _ in head_output_fields(head)
        )
        for name in self.branches:
            if not isinstance(getattr(head, name, None), nn.Module):
                raise TypeError(f"CenterHead has no prediction branch module {name!r}")

    def forward(self, spatial_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feats = self.backbone(spatial_features)
        bev = torch.cat([block(f) for block, f in zip(self.neck.blocks, feats)], dim=1)
        shared = self.head.shared_conv(bev)
        return tuple(getattr(self.head, name)(shared) for name in self.branches)


def build_centerpoint_pruning_spec(model) -> PruningSpec:
    """CenterPoint's pruning declaration (see the module docstring)."""
    return PruningSpec(
        subtree=CenterPointDenseSubtree(model.pts_backbone, model.pts_neck, model.bbox_head),
        stage_input=SPATIAL_FEATURES,
        submodules=("pts_backbone", "pts_neck", "bbox_head"),
    )


def centerpoint_distillation_loss(
    student: MultiTaskOutputs,
    teacher: MultiTaskOutputs,
    heatmap_weight: float = 1.0,
    regression_weight: float = 1.0,
) -> torch.Tensor:
    """Knowledge-distillation loss between two CenterPoint output sets.

    - heatmaps: binary cross-entropy of the student logits against the teacher's
      probabilities (soft labels over every cell — the teacher's full ranking, not just
      its peaks, is what the pruned network has to reproduce);
    - centers / heights / dims / rots / vels: L1 on cells where the teacher's max class
      probability exceeds :data:`_FOREGROUND_THRESHOLD`, normalized by the number of
      such cells.
    """
    s = student.detection3d_head_outputs.center_head_outputs
    t = teacher.detection3d_head_outputs.center_head_outputs
    heatmap_loss = F.binary_cross_entropy_with_logits(s.heatmaps, torch.sigmoid(t.heatmaps))

    foreground = (
        torch.sigmoid(t.heatmaps).amax(dim=1, keepdim=True) > _FOREGROUND_THRESHOLD
    ).float()
    count = foreground.sum().clamp_min(1.0)
    regression_loss = s.heatmaps.new_zeros(())
    for field in ("centers", "heights", "dims", "rots", "vels"):
        s_map, t_map = getattr(s, field, None), getattr(t, field, None)
        if s_map is None or t_map is None:
            continue
        regression_loss = regression_loss + ((s_map - t_map).abs() * foreground).sum() / (
            count * s_map.shape[1]
        )
    return heatmap_weight * heatmap_loss + regression_weight * regression_loss
