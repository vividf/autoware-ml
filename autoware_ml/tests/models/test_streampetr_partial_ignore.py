"""Tests for StreamPETR partial-ignore, the auxiliary 2D head, and conversion."""

from __future__ import annotations

import numpy as np
import torch

from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.common.necks.cp_fpn import CPFPN
from autoware_ml.models.detection3d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.partial_ignore import (
    normalize_status_flags,
    resolve_partial_ignore_labels,
)
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.tools.convert_streampetr_checkpoint import convert_state_dict
from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.utils.schedulers.iter_warmup_epoch_cosine import IterWarmupEpochCosineLR

CLASS_NAMES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]


def _build_head(partial_ignore: bool) -> StreamPETRHead:
    return StreamPETRHead(
        num_classes=7,
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
        use_bottom_center=True,
        bbox_coder=NMSFreeBBoxCoder3D(
            pc_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
            post_center_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
            score_threshold=0.01,
            max_num=8,
        ),
        assigner=HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=2.0),
            reg_cost=BBox3DL1Cost(weight=0.25),
            iou_cost=IoU3DCost(weight=0.0),
        ),
        point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
        code_weights=[2.0, 2.0] + [1.0] * 8,
        class_names=CLASS_NAMES if partial_ignore else None,
        partial_ignore_classes=["traffic_cone", "barrier"] if partial_ignore else None,
    )


def test_resolve_partial_ignore_labels() -> None:
    assert resolve_partial_ignore_labels(CLASS_NAMES, ["traffic_cone", "barrier"]) == [5, 6]
    assert resolve_partial_ignore_labels(CLASS_NAMES, None) is None


def test_normalize_status_flags_handles_tensors_lists_and_missing() -> None:
    assert normalize_status_flags(None, 2) == [True, True]
    assert normalize_status_flags([True, False], 2) == [True, False]
    assert normalize_status_flags(torch.tensor([1.0, 0.0]), 2) == [True, False]
    assert normalize_status_flags([torch.tensor(False)], 2) == [False, True]


def test_sigmoid_focal_loss_classwise_weights_zero_masked_columns() -> None:
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


def test_get_targets_zeroes_ignore_columns_only_on_negative_queries() -> None:
    head = _build_head(partial_ignore=True)
    num_queries = 16
    cls_logits = torch.randn(1, num_queries, 7)
    box_params = torch.randn(1, num_queries, 10)
    box_params[..., :3] = 0.0
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    targets = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels, [False])[0]
    assert targets.label_weights is not None
    assert targets.label_weights.shape == (num_queries, 7)

    pos_mask = targets.labels >= 0
    assert pos_mask.sum() == 1
    # Positive queries keep full supervision on every class column.
    assert torch.all(targets.label_weights[pos_mask] == 1.0)
    # Negative queries lose only the traffic_cone/barrier columns.
    neg_weights = targets.label_weights[~pos_mask]
    assert torch.all(neg_weights[:, [5, 6]] == 0.0)
    assert torch.all(neg_weights[:, :5] == 1.0)

    # Fully annotated frames keep uniform weights (no classwise tensor).
    annotated = head._get_targets(cls_logits, box_params, gt_boxes, gt_labels, [True])[0]
    assert annotated.label_weights is None


def test_dn_label_weights_mask_background_rows_of_unannotated_samples() -> None:
    head = _build_head(partial_ignore=True)
    cls_scores = torch.randn(6, 7)
    known_labels = torch.tensor([0, 7, 7, 1, 7, 5])
    known_bids = torch.tensor([0, 0, 1, 1, 1, 0])
    weights = head._dn_label_weights(cls_scores, known_labels, known_bids, [True, False])
    assert weights is not None
    # Background rows of sample 1 (indices 2 and 4) are masked on columns 5/6.
    assert torch.all(weights[[2, 4]][:, [5, 6]] == 0.0)
    # Background rows of the annotated sample 0 stay fully weighted.
    assert torch.all(weights[1] == 1.0)
    # Foreground rows are never masked.
    assert torch.all(weights[[0, 3, 5]] == 1.0)

    assert head._dn_label_weights(cls_scores, known_labels, known_bids, [True, True]) is None


def test_focal_head_2d_forward_and_loss_with_partial_ignore() -> None:
    torch.manual_seed(0)
    head = FocalHead2D(
        num_classes=7,
        in_channels=32,
        embed_dims=32,
        stride=16,
        class_names=CLASS_NAMES,
        partial_ignore_classes=["traffic_cone", "barrier"],
    )
    batch_size, num_cams = 2, 2
    img_features = torch.randn(batch_size, num_cams, 32, 6, 10)
    outputs = head(img_features, image_height=96, image_width=160)
    assert outputs["enc_cls_scores"].shape == (4, 60, 7)
    assert outputs["enc_bbox_preds"].shape == (4, 60, 4)
    assert outputs["pred_centers2d"].shape == (4, 60, 2)
    assert outputs["centerness"].shape == (4, 60, 1)

    gt_boxes = np.array([[10.0, 10.0, 60.0, 60.0]], dtype=np.float32)
    gt_centers = np.array([[35.0, 35.0]], dtype=np.float32)
    gt_labels = np.array([5], dtype=np.int64)
    empty_boxes = np.zeros((0, 4), dtype=np.float32)
    empty_centers = np.zeros((0, 2), dtype=np.float32)
    empty_labels = np.zeros((0,), dtype=np.int64)

    losses = head.loss(
        outputs,
        gt_bboxes_2d=[[gt_boxes, empty_boxes], [empty_boxes, empty_boxes]],
        gt_labels_2d=[[gt_labels, empty_labels], [empty_labels, empty_labels]],
        centers_2d=[[gt_centers, empty_centers], [empty_centers, empty_centers]],
        traffic_cone_barrier_status=[True, False],
    )
    for key in (
        "loss_cls2d",
        "loss_bbox2d",
        "loss_iou2d",
        "loss_centers2d",
        "loss_centerness2d",
    ):
        assert torch.isfinite(losses[key]), key

    # Masking the cone/barrier columns on the unannotated sample must not
    # increase the classification loss.
    unmasked = head.loss(
        outputs,
        gt_bboxes_2d=[[gt_boxes, empty_boxes], [empty_boxes, empty_boxes]],
        gt_labels_2d=[[gt_labels, empty_labels], [empty_labels, empty_labels]],
        centers_2d=[[gt_centers, empty_centers], [empty_centers, empty_centers]],
        traffic_cone_barrier_status=[True, True],
    )
    assert losses["loss_cls2d"] <= unmasked["loss_cls2d"] + 1e-6


def test_load_annotations_2d_projects_box_in_front_of_camera() -> None:
    transform = LoadAnnotations2DFromBoxes3D()
    lidar2cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[0, 0] = intrinsics[1, 1] = 100.0
    intrinsics[0, 2] = 80.0
    intrinsics[1, 2] = 48.0
    input_dict = {
        "img": np.zeros((1, 3, 96, 160), dtype=np.float32),
        "gt_boxes": np.array([[8.0, 0.0, -1.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0]], dtype=np.float32),
        "gt_labels": np.array([0], dtype=np.int64),
        "lidar2cam": lidar2cam[None],
        "camera_intrinsics": intrinsics[None],
    }
    result = transform(input_dict)
    assert result["gt_bboxes_2d"][0].shape == (1, 4)
    assert result["centers_2d"][0].shape == (1, 2)
    assert result["gt_labels_2d"][0].tolist() == [0]
    x1, y1, x2, y2 = result["gt_bboxes_2d"][0][0]
    assert 0 <= x1 < x2 <= 160
    assert 0 <= y1 < y2 <= 96


def test_cpfpn_outputs_match_input_levels_and_channels() -> None:
    neck = CPFPN(in_channels=[24, 40], out_channels=16)
    high = torch.randn(2, 24, 12, 20)
    low = torch.randn(2, 40, 6, 10)
    outputs = neck((high, low))
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 16, 12, 20)
    assert outputs[1].shape == (2, 16, 6, 10)


def test_iter_warmup_epoch_cosine_schedule_shape() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = IterWarmupEpochCosineLR(
        optimizer,
        total_steps=100,
        max_epochs=10,
        warmup_iters=5,
        warmup_start_factor=1.0 / 3.0,
        eta_min_factor=1e-4,
    )
    assert abs(optimizer.param_groups[0]["lr"] - 1.0 / 3.0) < 1e-6
    lrs = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    # After warmup and within the first epoch the factor is the epoch-0 cosine.
    assert abs(lrs[5] - 1.0) < 1e-6
    # The last epoch approaches the eta_min floor.
    assert lrs[-1] < 0.05


def test_convert_state_dict_maps_into_native_model_names() -> None:
    head = _build_head(partial_ignore=False)
    neck = CPFPN(in_channels=[24, 40], out_channels=16)
    native_keys = {f"bbox_head.{name}": tensor for name, tensor in head.state_dict().items()}
    native_keys.update({f"img_neck.{name}": tensor for name, tensor in neck.state_dict().items()})

    # Build the mm-style source names for the same structures.
    mm_state = {}
    for name, tensor in head.state_dict().items():
        mm_name = name
        for native_prefixed, mm_prefixed in (
            ("featurized_pe.reduce.", "featurized_pe.conv_reduce."),
            ("featurized_pe.expand.", "featurized_pe.conv_expand."),
            ("post_norm.", "transformer.decoder.post_norm."),
        ):
            if mm_name.startswith(native_prefixed):
                mm_name = mm_prefixed + mm_name[len(native_prefixed) :]
        import re

        decoder_match = re.match(r"^decoder\.(\d+)\.(.+)$", mm_name)
        if decoder_match:
            index, rest = decoder_match.groups()
            rest_map = {
                "self_attn.": "attentions.0.attn.",
                "cross_attn.": "attentions.1.attn.",
                "ffn.0.": "ffns.0.layers.0.0.",
                "ffn.3.": "ffns.0.layers.1.",
                "norm1.": "norms.0.",
                "norm2.": "norms.1.",
                "norm3.": "norms.2.",
            }
            for native_part, mm_part in rest_map.items():
                if rest.startswith(native_part):
                    rest = mm_part + rest[len(native_part) :]
                    break
            mm_name = f"transformer.decoder.layers.{index}.{rest}"
        mm_state[f"pts_bbox_head.{mm_name}"] = tensor
    for name, tensor in neck.state_dict().items():
        mm_state[f"img_neck.{name}"] = tensor
    mm_state["pts_bbox_head.coords_d"] = torch.zeros(8)

    converted, skipped = convert_state_dict(mm_state, bgr_to_rgb=False)
    assert "pts_bbox_head.coords_d" in skipped
    for key, tensor in converted.items():
        assert key in native_keys, key
        assert tensor.shape == native_keys[key].shape, key
    # Every native head/neck tensor is covered by the conversion.
    assert set(converted) == set(native_keys)


def test_convert_state_dict_drop_patterns_strip_class_heads() -> None:
    mm_state = {
        "pts_bbox_head.cls_branches.0.6.weight": torch.zeros(10, 32),
        "pts_bbox_head.cls_branches.0.6.bias": torch.zeros(10),
        "pts_bbox_head.cls_branches.0.0.weight": torch.zeros(32, 32),
        "img_roi_head.cls.weight": torch.zeros(10, 32, 1, 1),
        "img_roi_head.shared_cls.0.weight": torch.zeros(32, 32, 3, 3),
    }
    converted, skipped = convert_state_dict(
        mm_state,
        bgr_to_rgb=False,
        drop_patterns=[r"cls_branches\.\d+\.6\.", r"img_roi_head\.cls\."],
    )
    assert sorted(skipped) == [
        "img_roi_head.cls.weight",
        "pts_bbox_head.cls_branches.0.6.bias",
        "pts_bbox_head.cls_branches.0.6.weight",
    ]
    assert set(converted) == {
        "bbox_head.cls_branches.0.0.weight",
        "img_roi_head.shared_cls.0.weight",
    }


def test_multiview_loader_shuffle_order_keeps_sample_consistent(tmp_path) -> None:
    import random

    import cv2

    from autoware_ml.transforms.camera.loading import LoadMultiViewImagesFromFiles

    camera_order = [f"CAM_{i}" for i in range(5)]
    images_meta = {}
    for index, name in enumerate(camera_order):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), np.full((4, 6, 3), index * 10, dtype=np.uint8))
        intrinsics = np.eye(3, dtype=np.float32) * (index + 1)
        images_meta[name] = {
            "img_path": str(path),
            "cam2img": intrinsics,
            "lidar2cam": np.eye(4, dtype=np.float32),
        }

    loader = LoadMultiViewImagesFromFiles(normalize_to_unit=False, shuffle_order=True)
    random.seed(3)
    shuffled_seen = False
    for _ in range(8):
        out = loader({"images": images_meta, "camera_order": camera_order})
        names = out["camera_names"]
        assert sorted(names) == sorted(camera_order)
        if names != camera_order:
            shuffled_seen = True
        for position, name in enumerate(names):
            index = camera_order.index(name)
            # Image content and intrinsics must follow the shuffled order.
            assert float(out["img"][position].mean()) == index * 10
            assert out["camera_intrinsics"][position][0, 0] == index + 1
    assert shuffled_seen

    fixed_loader = LoadMultiViewImagesFromFiles(normalize_to_unit=False)
    out = fixed_loader({"images": images_meta, "camera_order": camera_order})
    assert out["camera_names"] == camera_order
