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

"""CenterPoint pruning declaration: the fx-traceable subtree, the KD loss, the aux-loss hook."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.detection3d.head_outputs import (
    CenterHeadOutputs,
    Detection3DHeadOutputs,
)
from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.models.detection3d.main_modules.centerpoint.pruning import (
    CenterPointDenseSubtree,
    centerpoint_distillation_loss,
)
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN

_RANGE = [-10.0, -10.0, -3.0, 10.0, 10.0, 5.0]


def _parts():
    torch.manual_seed(0)
    backbone = SECONDBackbone(
        in_channels=8, out_channels=[16, 32], layer_nums=[1, 1], layer_strides=[1, 2]
    )
    neck = SECONDFPN(in_channels=[16, 32], out_channels=[16, 16], upsample_strides=[1, 2])
    head = CenterHead(
        in_channels=32,
        class_names=["car", "pedestrian", "cone"],
        shared_channels=8,
        point_cloud_range=_RANGE,
        voxel_size=[0.5, 0.5, 8.0],
        out_size_factor=1,
        min_radius=2,
        score_threshold=0.0,
        post_max_size=10,
        nms_min_radius=1.0,
        use_velocity=True,
    )
    return backbone, neck, head


class TestDenseSubtree:
    def test_matches_the_real_forward_and_is_fx_traceable(self):
        backbone, neck, head = _parts()
        subtree = CenterPointDenseSubtree(backbone, neck, head).eval()
        x = torch.randn(1, 8, 16, 16)
        with torch.no_grad():
            outs = subtree(x)
            reference = head(neck(backbone(x)))
        assert len(outs) == 6  # heatmap, reg, height, dim, rot, vel
        assert torch.equal(outs[0], reference.heatmaps)
        assert torch.equal(outs[1], reference.centers)
        assert torch.equal(outs[5], reference.vels)
        graph = torch.fx.symbolic_trace(subtree)
        called = {n.target for n in graph.graph.nodes if n.op == "call_module"}
        # The children are traced (prunable); the neck/head containers are not leaves.
        assert "neck.blocks.0.conv" in called or "neck.blocks.0" in called
        assert "head.shared_conv.conv" in called or "head.shared_conv" in called
        assert "neck" not in called and "head" not in called


def _outputs(heatmaps, scale=1.0):
    n, _, h, w = heatmaps.shape
    g = torch.Generator().manual_seed(1)

    def r(c):
        return torch.randn(n, c, h, w, generator=g) * scale

    return MultiTaskOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(
            center_head_outputs=CenterHeadOutputs(
                heatmaps=heatmaps, centers=r(2), heights=r(1), dims=r(3), rots=r(2), vels=r(2)
            ),
            transfusion_head_outputs=None,
        )
    )


class TestDistillationLoss:
    def test_zero_for_identical_outputs_positive_otherwise(self):
        heat = torch.randn(1, 3, 8, 8)
        teacher = _outputs(heat)
        same = centerpoint_distillation_loss(teacher, teacher)
        # BCE(soft labels, themselves) is the teacher's own entropy, not 0; the regression part is 0.
        entropy = torch.nn.functional.binary_cross_entropy_with_logits(heat, torch.sigmoid(heat))
        assert torch.isclose(same, entropy)
        other = centerpoint_distillation_loss(_outputs(heat + 1.0, scale=3.0), teacher)
        assert other > same

    def test_regression_only_counts_teacher_foreground(self):
        heat = torch.full((1, 3, 8, 8), -10.0)  # teacher sees nothing -> no foreground cell
        teacher = _outputs(heat)
        student = _outputs(heat, scale=5.0)
        loss = centerpoint_distillation_loss(student, teacher, heatmap_weight=0.0)
        assert torch.isclose(loss, torch.zeros(()))

    def test_gradients_reach_the_student(self):
        heat_t = torch.randn(1, 3, 8, 8)
        heat_s = torch.randn(1, 3, 8, 8, requires_grad=True)
        loss = centerpoint_distillation_loss(_outputs(heat_s), _outputs(heat_t))
        loss.backward()
        assert heat_s.grad is not None and heat_s.grad.abs().sum() > 0


class TestAuxiliaryLossRegistry:
    def test_registered_term_is_added_and_logged(self):
        pytest.importorskip("lightning")
        from types import MappingProxyType

        from autoware_ml.models.multi_task_base_model import MultiTaskBaseModel

        model = MultiTaskBaseModel.__new__(MultiTaskBaseModel)
        model._auxiliary_losses = {}
        model.register_auxiliary_loss("kd", lambda i, o: {"loss_kd": torch.tensor(2.0)})
        metrics = model._with_auxiliary_losses(
            None, None, MappingProxyType({"loss": torch.tensor(1.0)})
        )
        assert float(metrics["loss"]) == 3.0 and float(metrics["loss_kd"]) == 2.0
        with pytest.raises(ValueError, match="already registered"):
            model.register_auxiliary_loss("kd", lambda i, o: {})
        model.unregister_auxiliary_loss("kd")
        assert model._auxiliary_losses == {}
