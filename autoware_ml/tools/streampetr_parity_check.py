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

"""Check StreamPETR forward/loss parity against an AWML reference dump.

Consumes the reference file written by AWML's
``projects/StreamPETR/tools/parity_dump.py`` (one real frame, fp32, eval
mode, no DN/GridMask/dropout), replays the identical tensors through the
native model initialized from a converted checkpoint, and compares image
features, positional embeddings, per-layer head outputs, and every loss
term.

Run inside the autoware-ml docker:

    python -m autoware_ml.tools.streampetr_parity_check \
        --reference /workspace/work_dirs/parity/streampetr_parity_reference.pt \
        --checkpoint /workspace/work_dirs/streampetr_2_7_epoch_20_converted.pth \
        [--config-name tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier]
"""

from __future__ import annotations

import argparse

import hydra
import torch
from hydra import compose, initialize_config_dir

from autoware_ml.utils.checkpoints import load_matching_weights

# AWML FocalHead loss keys -> native FocalHead2D loss keys.
_ROI_LOSS_KEY_MAP = {
    "enc_loss_cls": "loss_cls2d",
    "enc_loss_bbox": "loss_bbox2d",
    "enc_loss_iou": "loss_iou2d",
    "centers2d_losses": "loss_centers2d",
    "centerness_losses": "loss_centerness2d",
}

# A forward comparison passes on either gate: bit-near identity (absolute)
# or numeric-noise scale (relative). Cross-container cuDNN algorithm
# selection makes deep conv stacks differ at ~1e-4 relative even in fp32,
# while geometry-only paths (positional embeddings) stay bit-exact.
FORWARD_ABS_TOLERANCE = 1e-4
FORWARD_REL_TOLERANCE = 1e-3
# Sigmoid/inverse-sigmoid decodes near saturation amplify upstream conv
# noise; end-to-end decoded 2D outputs get a wider gate. Head equivalence
# itself is checked strictly on the reference features (isolated rows).
DECODED_REL_TOLERANCE = 5e-3
LOSS_RELATIVE_TOLERANCE = 1e-3


class _ParityReport:
    """Collect named comparisons and render a pass/fail table."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float, float, bool]] = []

    def check_tensor(
        self,
        name: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        rel_tolerance: float = FORWARD_REL_TOLERANCE,
    ) -> None:
        actual = actual.detach().float().cpu()
        expected = expected.detach().float().cpu()
        if actual.shape != expected.shape:
            self.rows.append(
                (
                    f"{name} shape {tuple(actual.shape)} vs {tuple(expected.shape)}",
                    float("nan"),
                    float("nan"),
                    False,
                )
            )
            return
        abs_diff = (actual - expected).abs()
        max_abs = float(abs_diff.max()) if abs_diff.numel() else 0.0
        denom = expected.abs().max().clamp(min=1e-6)
        rel = float((abs_diff.max() / denom)) if abs_diff.numel() else 0.0
        passed = max_abs <= FORWARD_ABS_TOLERANCE or rel <= rel_tolerance
        self.rows.append((name, max_abs, rel, passed))

    def check_loss(self, name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        actual_value = float(actual)
        expected_value = float(expected)
        abs_diff = abs(actual_value - expected_value)
        rel = abs_diff / max(abs(expected_value), 1e-8)
        self.rows.append((name, abs_diff, rel, rel <= LOSS_RELATIVE_TOLERANCE))

    def render(self) -> bool:
        width = max(len(row[0]) for row in self.rows) + 2
        print(f"{'comparison'.ljust(width)}{'max_abs':>12}{'max_rel':>12}  verdict")
        ok = True
        for name, max_abs, rel, passed in self.rows:
            verdict = "PASS" if passed else "FAIL"
            ok = ok and passed
            print(f"{name.ljust(width)}{max_abs:>12.3e}{rel:>12.3e}  {verdict}")
        return ok


def _transpose_camera_major(per_camera: list) -> list:
    """AWML dumps 2D GT as [camera][sample]; native wants [sample][camera]."""
    num_cameras = len(per_camera)
    num_samples = len(per_camera[0])
    return [
        [per_camera[camera][sample] for camera in range(num_cameras)]
        for sample in range(num_samples)
    ]


def main() -> None:
    """Run the parity comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config-name",
        default="tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2_finetune_cone_barrier",
    )
    parser.add_argument("--config-dir", default="/workspace/autoware_ml/configs")
    args = parser.parse_args()

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    inputs = reference["inputs"]
    gt = reference["gt"]

    with initialize_config_dir(config_dir=args.config_dir, version_base="1.3"):
        cfg = compose(config_name=args.config_name)
    model = hydra.utils.instantiate(cfg.model)
    load_matching_weights(model, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).float().eval()

    # The dump is normalized BGR (AWML pipeline); the converted checkpoint
    # carries a channel-flipped stem, so flip the image to normalized RGB.
    img = inputs["img"].to(device).flip(dims=[2])
    intrinsics = inputs["intrinsics"].to(device)
    lidar2cam = inputs["extrinsics"].to(device)
    lidar2img = inputs["lidar2img"].to(device)
    pad_height, pad_width = (int(x) for x in inputs["pad_shape"][:2])

    report = _ParityReport()
    with torch.no_grad():
        img_feats = model.image_feature_extractor(img)
        report.check_tensor("img_feats", img_feats, reference["intermediates"]["img_feats"])

        pos_embed, cone = model.bbox_head.position_embedding(
            img_feats, intrinsics, lidar2cam, pad_height, pad_width, lidar2img=lidar2img
        )
        report.check_tensor("pos_embed", pos_embed, reference["intermediates"]["pos_embed"])
        report.check_tensor("cone", cone, reference["intermediates"]["cone"])

        gt_boxes = [boxes.to(device) for boxes in gt["gt_boxes"]]
        gt_labels = [labels.to(device) for labels in gt["gt_labels"]]
        model.bbox_head.reset_memory()
        outputs = model.bbox_head(
            img_features=img_feats,
            img=img,
            camera_intrinsics=intrinsics,
            lidar2cam=lidar2cam,
            lidar2img=lidar2img,
            timestamp=inputs["timestamp"].to(device),
            prev_exists=inputs["prev_exists"].to(device),
            ego_pose=inputs["ego_pose"].to(device),
            ego_pose_inv=inputs["ego_pose_inv"].to(device),
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
        )
        report.check_tensor(
            "all_cls_scores", outputs["all_cls_scores"], reference["outputs"]["all_cls_scores"]
        )
        report.check_tensor(
            "all_bbox_preds", outputs["all_bbox_preds"], reference["outputs"]["all_bbox_preds"]
        )

        status = gt["traffic_cone_barrier_status"]
        losses = model.bbox_head.loss(
            outputs, gt_boxes, gt_labels, traffic_cone_barrier_status=status
        )
        for mm_key, value in reference["losses"].items():
            if mm_key in _ROI_LOSS_KEY_MAP:
                continue
            if mm_key not in losses:
                print(f"NOTE: mm loss key '{mm_key}' missing on the native side")
                continue
            report.check_loss(f"loss[{mm_key}]", losses[mm_key], value)

        # Isolated head equivalence: identical (reference) features in, so
        # any diff here is a genuine head/weight mismatch — strict gate.
        reference_feats = reference["intermediates"]["img_feats"].to(device)
        roi_isolated = model.img_roi_head(reference_feats, pad_height, pad_width)
        report.check_tensor(
            "roi_cls (isolated)",
            roi_isolated["enc_cls_scores"],
            reference["outputs"]["roi_enc_cls_scores"],
        )
        report.check_tensor(
            "roi_bbox (isolated)",
            roi_isolated["enc_bbox_preds"],
            reference["outputs"]["roi_enc_bbox_preds"],
        )
        report.check_tensor(
            "roi_centers2d (isolated)",
            roi_isolated["pred_centers2d"],
            reference["outputs"]["roi_pred_centers2d"],
        )
        report.check_tensor(
            "roi_centerness (isolated)",
            roi_isolated["centerness"],
            reference["outputs"]["roi_centerness"],
        )

        # End-to-end (native features in): decoded 2D outputs amplify the
        # backbone's cross-container conv noise near sigmoid saturation.
        roi_outputs = model.img_roi_head(img_feats, pad_height, pad_width)
        report.check_tensor(
            "roi_cls", roi_outputs["enc_cls_scores"], reference["outputs"]["roi_enc_cls_scores"]
        )
        report.check_tensor(
            "roi_bbox",
            roi_outputs["enc_bbox_preds"],
            reference["outputs"]["roi_enc_bbox_preds"],
            rel_tolerance=DECODED_REL_TOLERANCE,
        )
        report.check_tensor(
            "roi_centers2d",
            roi_outputs["pred_centers2d"],
            reference["outputs"]["roi_pred_centers2d"],
            rel_tolerance=DECODED_REL_TOLERANCE,
        )
        report.check_tensor(
            "roi_centerness", roi_outputs["centerness"], reference["outputs"]["roi_centerness"]
        )

        roi_losses = model.img_roi_head.loss(
            roi_outputs,
            gt_bboxes_2d=_transpose_camera_major(gt["gt_bboxes_2d"]),
            gt_labels_2d=_transpose_camera_major(gt["gt_labels_2d"]),
            centers_2d=_transpose_camera_major(gt["centers_2d"]),
            traffic_cone_barrier_status=status,
        )
        for mm_key, native_key in _ROI_LOSS_KEY_MAP.items():
            if mm_key in reference["losses"]:
                report.check_loss(
                    f"loss[{mm_key}->{native_key}]",
                    roi_losses[native_key],
                    reference["losses"][mm_key],
                )

    ok = report.render()
    print("PARITY:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
