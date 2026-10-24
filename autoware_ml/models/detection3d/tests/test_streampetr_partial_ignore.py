"""Tests for the partial annotations of the StreamPETR heads and the auxiliary 2D head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoware_ml.dataclasses.models.detection3d.head_outputs import StreamPETRHeadOutputs
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.detection3d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations

CLASS_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
PARTIAL_ANNOTATIONS = PartialAnnotations(CLASS_NAMES, ["traffic_cone", "barrier"])
POINT_CLOUD_RANGE = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]


def _masks(status: list[bool]) -> torch.Tensor:
    return PARTIAL_ANNOTATIONS.build_annotated_class_masks(torch.tensor(status))


def _build_head() -> StreamPETRHead:
    return StreamPETRHead(
        class_names=CLASS_NAMES,
        in_channels=32,
        hidden_dim=32,
        num_queries=16,
        num_decoder_layers=2,
        num_heads=4,
        feedforward_channels=64,
        memory_len=16,
        topk_proposals=4,
        num_propagated=4,
        with_dn=True,
        with_ego_pos=True,
        depth_num=8,
        LID=True,
        position_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
        scalar=2,
        noise_scale=0.5,
        dn_weight=1.0,
        split=0.5,
        bbox_coder=NMSFreeBBoxCoder3D(
            pc_range=POINT_CLOUD_RANGE,
            post_center_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
            score_threshold=0.01,
            max_num=8,
        ),
        assigner=HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=2.0),
            reg_cost=BBox3DL1Cost(weight=0.25),
            iou_cost=IoU3DCost(weight=0.0),
            point_cloud_range=POINT_CLOUD_RANGE,
        ),
        point_cloud_range=POINT_CLOUD_RANGE,
        code_weights=[2.0, 2.0] + [1.0] * 8,
    )


def test_partial_annotations_mask_only_the_cone_and_barrier_columns() -> None:
    """
    Input: the traffic-cone and barrier status of two frames over the seven classes, True for
    the first frame and False for the second.
    Expected: the first frame supervises every class, the second one every class but
    traffic_cone and barrier.
    Check: the mask shape, every column of the first row, and columns 5 and 6 against the first
    five of the second row.
    """
    masks = _masks([True, False])
    assert masks.shape == (2, 7)
    assert masks[0].all()
    assert not masks[1, [5, 6]].any()
    assert masks[1, :5].all()


def test_sigmoid_focal_loss_classwise_weights_zero_masked_columns() -> None:
    """
    Input: random logits of four queries over seven classes, all background, scored without
    weights, with weights of one, and with weights of zero on columns 5 and 6.
    Expected: weights of one change nothing, and zero-weighted columns drop out of the loss as
    if their classes did not exist.
    Check: the unweighted loss against the one with weights of one, and the masked loss against
    the loss of the first five columns alone.
    """
    loss_fn = SigmoidFocalLoss()
    logits = torch.randn(4, 7)
    targets = torch.zeros(4, 7)
    ones = torch.ones(4, 7)
    baseline = loss_fn(logits, targets, avg_factor=1.0)
    weighted = loss_fn(logits, targets, weights=ones, avg_factor=1.0)
    assert torch.allclose(baseline, weighted)

    masked_weights = ones.clone()
    masked_weights[:, [5, 6]] = 0.0
    masked = loss_fn(logits, targets, weights=masked_weights, avg_factor=1.0)
    manual = loss_fn(logits[:, :5], targets[:, :5], avg_factor=1.0)
    assert torch.allclose(masked, manual, atol=1e-6)


def test_get_targets_weights_every_query_by_the_mask_of_its_frame() -> None:
    """
    Input: the classification targets of a frame with one box, built with the mask of a frame
    annotated without traffic cones and barriers, without a mask, and for a batch of two frames,
    the first annotated with them and the second without.
    Expected: every query of a frame annotated without them loses only the cone and barrier
    columns of its classification weights, no mask keeps the weights uniform, and in a batch
    every frame weights its queries by its own mask.
    Check: the label weights of the masked frame by column, None without a mask, and the weights
    of both frames of the batch.
    """
    head = _build_head()
    num_queries = 16
    cls_logits = torch.randn(1, num_queries, 7)
    box_params = torch.randn(1, num_queries, 10)
    box_params[..., :3] = 0.0
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    targets = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels, _masks([False]))[0]
    assert targets.label_weights is not None
    assert targets.label_weights.shape == (num_queries, 7)

    # Every query of the un-annotated frame loses only the cone/barrier columns.
    assert torch.all(targets.label_weights[:, [5, 6]] == 0.0)
    assert torch.all(targets.label_weights[:, :5] == 1.0)

    # Without a mask the weights stay uniform (None).
    plain = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels)[0]
    assert plain.label_weights is None

    # In a mixed batch every sample carries its own row.
    mixed = head._get_targets(
        cls_logits.repeat(2, 1, 1),
        box_params.repeat(2, 1, 1),
        gt_boxes * 2,
        gt_labels * 2,
        _masks([True, False]),
    )
    assert torch.all(mixed[0].label_weights == 1.0)
    assert torch.all(mixed[1].label_weights[:, [5, 6]] == 0.0)


def test_dn_label_weights_mask_every_denoising_query_of_unannotated_samples() -> None:
    """
    Input: six denoising queries of two samples, sample 1 annotated without traffic cones and
    barriers.
    Expected: every denoising query of sample 1 loses the cone and barrier columns, the queries
    of sample 0 stay fully weighted, and no mask gives no weights.
    Check: the weights of the queries of each sample by column, and None without a mask.
    """
    head = _build_head()
    cls_scores = torch.randn(6, 7)
    known_bids = torch.tensor([0, 0, 1, 1, 1, 0])
    weights = head._dn_label_weights(cls_scores, known_bids, _masks([True, False]))
    assert weights is not None
    # Every row of sample 1 (indices 2, 3, 4) is masked on columns 5/6.
    assert torch.all(weights[[2, 3, 4]][:, [5, 6]] == 0.0)
    assert torch.all(weights[[2, 3, 4]][:, :5] == 1.0)
    # Rows of the annotated sample 0 stay fully weighted.
    assert torch.all(weights[[0, 1, 5]] == 1.0)

    assert head._dn_label_weights(cls_scores, known_bids, None) is None


def test_head_loss_applies_the_annotated_class_masks() -> None:
    """
    Input: the StreamPETR head loss of one frame with one box, without a mask, with the mask of
    a frame annotated with traffic cones and barriers, with the mask of a frame annotated
    without them, and with the masks of two frames.
    Expected: the mask of an annotated frame gives the plain loss, and the other mask removes
    non-negative background terms from the classification loss only; masks of another batch size
    are refused.
    Check: the total losses, the classification and regression losses, and a ValueError naming
    the expected shape.
    """
    torch.manual_seed(0)
    head = _build_head().eval()
    outputs = StreamPETRHeadOutputs(
        all_cls_scores=torch.randn(2, 1, 16, 7),
        all_bbox_preds=torch.rand(2, 1, 16, 10),
        denoising=None,
    )
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]
    real = torch.zeros(1, dtype=torch.bool)

    plain = head.loss(outputs, gt_boxes, gt_labels, padding_mask=real)
    annotated = head.loss(
        outputs, gt_boxes, gt_labels, padding_mask=real, gt_annotated_class_masks=_masks([True])
    )
    ignored = head.loss(
        outputs, gt_boxes, gt_labels, padding_mask=real, gt_annotated_class_masks=_masks([False])
    )
    torch.testing.assert_close(annotated["loss"], plain["loss"])
    assert ignored["loss_cls"] < annotated["loss_cls"]
    assert torch.allclose(ignored["loss_bbox"], annotated["loss_bbox"])

    with pytest.raises(ValueError, match="batch_size, num_classes"):
        head.loss(
            outputs,
            gt_boxes,
            gt_labels,
            padding_mask=real,
            gt_annotated_class_masks=_masks([True, False]),
        )


def test_focal_head_2d_forward_and_loss_with_the_annotated_class_masks() -> None:
    """
    Input: image features of two samples of two cameras through FocalHead2D, one 2D box of
    traffic_cone on the first camera of sample 0, and the masks of sample 0 annotated with
    traffic cones and barriers and sample 1 annotated without them.
    Expected: the head predicts the class, box, center and centerness of every token. The mask
    removes strictly positive background terms of the images of sample 1 from the classification
    loss and touches nothing else, the masks of two annotated samples give the plain loss, and
    masks of another batch size are refused.
    Check: the output shapes, the five losses finite, the classification loss with and without
    the mask, the other losses unchanged, and a ValueError naming the expected shape.
    """
    torch.manual_seed(0)
    head = FocalHead2D(class_names=CLASS_NAMES, in_channels=32, embed_dims=32, stride=16)
    assert head.num_classes == 7
    batch_size, num_cams = 2, 2
    img_features = torch.randn(batch_size, num_cams, 32, 6, 10)
    outputs = head(img_features, image_height=96, image_width=160)
    assert outputs.cls_scores.shape == (4, 60, 7)
    assert outputs.bbox_preds.shape == (4, 60, 4)
    assert outputs.centers_2d.shape == (4, 60, 2)
    assert outputs.centerness.shape == (4, 60, 1)
    assert (outputs.image_height, outputs.image_width) == (96, 160)

    gt_boxes = np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32)
    gt_centers = np.array([[35.0, 35.0]], dtype=np.float32)
    gt_labels = np.array([5], dtype=np.int64)
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_centers = np.zeros((0, 2), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int64)
    annotations = {
        "gt_bboxes_2d": [[gt_boxes, empty_boxes], [empty_boxes, empty_boxes]],
        "gt_labels_2d": [[gt_labels, empty_labels], [empty_labels, empty_labels]],
        "centers_2d": [[gt_centers, empty_centers], [empty_centers, empty_centers]],
        "padding_mask": torch.zeros(batch_size, dtype=torch.bool),
    }

    losses = head.loss(outputs, gt_annotated_class_masks=_masks([True, False]), **annotations)
    for key in (
        "loss_cls2d",
        "loss_bbox2d",
        "loss_iou2d",
        "loss_centers2d",
        "loss_centerness2d",
    ):
        assert torch.isfinite(losses[key]), key

    unmasked = head.loss(outputs, gt_annotated_class_masks=_masks([True, True]), **annotations)
    plain = head.loss(outputs, **annotations)
    torch.testing.assert_close(unmasked["loss_cls2d"], plain["loss_cls2d"])
    assert losses["loss_cls2d"] < unmasked["loss_cls2d"]
    for key in ("loss_bbox2d", "loss_iou2d", "loss_centers2d", "loss_centerness2d"):
        assert torch.allclose(losses[key], unmasked[key]), key

    with pytest.raises(ValueError, match="batch_size, num_classes"):
        head.loss(outputs, gt_annotated_class_masks=_masks([True]), **annotations)


def test_focal_head_2d_class_weights_follow_the_images_of_every_sample() -> None:
    """
    Input: the masks of two samples, the second annotated without traffic cones and barriers,
    for two cameras of three tokens each.
    Expected: every token of every image of a sample takes the mask row of its sample, so only
    the six tokens of the two images of sample 1 lose the cone and barrier columns; no mask
    gives no weights.
    Check: the token weights against the expected matrix, and None without a mask.
    """
    head = FocalHead2D(class_names=CLASS_NAMES, in_channels=32, embed_dims=32)
    tokens_per_image = 3
    num_cams = 2
    weights = head._class_weights(_masks([True, False]), num_cams, tokens_per_image, torch.float32)
    assert weights is not None
    expected = torch.ones(2 * num_cams * tokens_per_image, 7)
    # The two images of sample 1 are rows 6..11.
    expected[6:12, 5:7] = 0.0
    assert torch.equal(weights, expected)

    assert head._class_weights(None, num_cams, tokens_per_image, torch.float32) is None
