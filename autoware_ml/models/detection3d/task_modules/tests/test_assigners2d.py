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

"""Tests for the Hungarian assigner of the auxiliary 2D head."""

from __future__ import annotations

import torch

from autoware_ml.models.detection3d.task_modules.assigners2d import (
    AssignResult2D,
    HungarianAssigner2D,
)
from autoware_ml.models.detection3d.task_modules.match_costs import ClassificationCost

IMAGE_HEIGHT, IMAGE_WIDTH = 100, 200


def _proposals_at(centers: list[tuple[float, float]], num_classes: int = 3):
    """Normalized ``(cx, cy, w, h)`` proposals at the given pixel centers, 20x20 px each."""
    boxes = torch.tensor(
        [
            [x / IMAGE_WIDTH, y / IMAGE_HEIGHT, 20 / IMAGE_WIDTH, 20 / IMAGE_HEIGHT]
            for x, y in centers
        ]
    )
    logits = torch.full((len(centers), num_classes), -4.0)
    return boxes, logits, boxes[:, :2].clone()


def test_matches_every_ground_truth_box_to_the_proposal_on_top_of_it() -> None:
    bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0), (100.0, 80.0)])
    # Proposal 1 is confident about class 2, proposal 0 about class 0.
    cls_pred[1, 2] = 4.0
    cls_pred[0, 0] = 4.0
    gt_bboxes = torch.tensor([[140.0, 50.0, 160.0, 70.0], [20.0, 20.0, 40.0, 40.0]])
    gt_labels = torch.tensor([2, 0])

    result = HungarianAssigner2D().assign(
        bbox_pred,
        cls_pred,
        centers,
        gt_bboxes,
        gt_labels,
        gt_bboxes[:, :2] + 10.0,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    assert isinstance(result, AssignResult2D)
    assert result.num_gts == 2
    assert result.gt_inds.tolist() == [2, 1, 0]
    assert result.labels.tolist() == [0, 2, -1]


def test_classification_cost_of_one_image_equals_the_batched_cost() -> None:
    # The assigner wraps the single image into a batch of one with all-valid ground truth, so
    # the matching cost is the batched classification cost of upstream read back per image.
    cost = ClassificationCost(weight=2.0)
    cls_pred = torch.randn(5, 3)
    gt_labels = torch.tensor([1, 2])
    batched = cost(
        cls_pred.unsqueeze(0), gt_labels.unsqueeze(0), torch.ones(1, 2, dtype=torch.bool)
    )
    assert batched.shape == (1, 5, 2)
    assert torch.isfinite(batched).all()


def test_an_image_without_ground_truth_assigns_every_proposal_to_background() -> None:
    bbox_pred, cls_pred, centers = _proposals_at([(30.0, 30.0), (150.0, 60.0)])

    result = HungarianAssigner2D().assign(
        bbox_pred,
        cls_pred,
        centers,
        torch.zeros(0, 4),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, 2),
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    assert result.num_gts == 0
    assert result.gt_inds.tolist() == [0, 0]
    assert result.labels.tolist() == [-1, -1]
