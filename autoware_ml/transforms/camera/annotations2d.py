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

"""Derive per-camera 2D annotations from 3D boxes for auxiliary supervision.

The transform projects the (already augmented) 3D ground-truth boxes onto
every camera and emits per-camera 2D boxes, projected gravity centers,
depths, and labels. It must therefore run after all geometric augmentations
(``ResizeCropFlipRotImage``, ``GlobalRotScaleTrans``, ``PadMultiViewImage``)
so the projection matrices and the pixels agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform


def _boxes3d_corners(boxes: np.ndarray) -> np.ndarray:
    """Compute the 8 corners of gravity-center 3D boxes.

    Args:
        boxes: Boxes ``(N, >=7)`` as ``[x, y, z_center, dx, dy, dz, yaw, ...]``,
            where ``z_center`` is the gravity center - the convention this
            repo's 3D boxes carry throughout.

    Returns:
        Corners with shape ``(N, 8, 3)``.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    dims = boxes[:, 3:6]
    signs = np.array(
        [
            [dx, dy, dz]
            for dx in (-0.5, 0.5)
            for dy in (-0.5, 0.5)
            # Centered on z, not growing upward from it: the incoming z is the
            # gravity center. Building corners from z to z+dz put every
            # projected 2D box half an object height too high in the image.
            for dz in (-0.5, 0.5)
        ],
        dtype=np.float32,
    )
    corners = dims[:, None, :] * signs[None, :, :]
    yaw = boxes[:, 6]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotated_x = corners[..., 0] * cos_yaw[:, None] - corners[..., 1] * sin_yaw[:, None]
    rotated_y = corners[..., 0] * sin_yaw[:, None] + corners[..., 1] * cos_yaw[:, None]
    corners = np.stack([rotated_x, rotated_y, corners[..., 2]], axis=-1)
    return corners + boxes[:, None, :3]


def _project_points(
    points: np.ndarray, lidar2cam: np.ndarray, cam2img: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project lidar-frame points into image pixels.

    Returns:
        Pixel coordinates ``(N, 2)`` and an in-front-of-camera mask ``(N,)``.
    """
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points_cam = (lidar2cam @ homogeneous.T).T
    valid = points_cam[:, 2] > 0
    projected = (cam2img[:3, :3] @ points_cam[:, :3].T).T
    projected = projected / np.maximum(projected[:, 2:3], 1e-6)
    return projected[:, :2], valid


class LoadAnnotations2DFromBoxes3D(BaseTransform):
    """Project 3D ground-truth boxes onto every camera as 2D annotations.

    Outputs (per camera, one entry per visible box):
        ``gt_bboxes_2d``: ``(N, 4)`` clipped ``(x1, y1, x2, y2)`` pixel boxes.
        ``centers_2d``: ``(N, 2)`` projected gravity centers in pixels.
        ``gt_labels_2d``: ``(N,)`` class labels.
    """

    _required_keys = ["img", "gt_boxes", "gt_labels", "lidar2cam", "camera_intrinsics"]

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Compute per-camera 2D annotations from the 3D boxes."""
        images = input_dict["img"]
        image_height, image_width = int(images.shape[-2]), int(images.shape[-1])
        gt_boxes = np.asarray(input_dict["gt_boxes"], dtype=np.float32).reshape(-1, 9)
        gt_labels = np.asarray(input_dict["gt_labels"]).reshape(-1)
        corners = _boxes3d_corners(gt_boxes)
        # ``gt_boxes`` already stores the gravity center, so it is the center to
        # project. The previous ``+= dz * 0.5`` treated z as the bottom face and
        # pushed both these centers and the corner-derived 2D boxes half an
        # object height up the image, giving the 2D auxiliary head targets that
        # were systematically too high (verified against an independent
        # projection from the annotation pkl).
        gravity_centers = gt_boxes[:, :3].copy()

        all_bboxes, all_centers, all_labels = [], [], []
        num_cams = len(input_dict["lidar2cam"])
        for camera_index in range(num_cams):
            lidar2cam = np.asarray(input_dict["lidar2cam"][camera_index], dtype=np.float64)
            cam2img = np.asarray(input_dict["camera_intrinsics"][camera_index], dtype=np.float64)
            bboxes, centers, labels = self._project_camera(
                corners,
                gravity_centers,
                gt_boxes,
                gt_labels,
                lidar2cam,
                cam2img,
                image_height,
                image_width,
            )
            all_bboxes.append(bboxes)
            all_centers.append(centers)
            all_labels.append(labels)

        return {
            "gt_bboxes_2d": all_bboxes,
            "centers_2d": all_centers,
            "gt_labels_2d": all_labels,
        }

    def _project_camera(
        self,
        corners: np.ndarray,
        gravity_centers: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        lidar2cam: np.ndarray,
        cam2img: np.ndarray,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_bboxes, valid_centers, valid_labels = [], [], []
        for box_index in range(gt_boxes.shape[0]):
            points = np.concatenate(
                [corners[box_index], gravity_centers[box_index : box_index + 1]], axis=0
            )
            pixels, in_front = _project_points(points, lidar2cam, cam2img)
            # The center must itself be in front of the camera: for a box
            # straddling the camera plane the clamped projection would place
            # the center target at an arbitrary pixel.
            if not in_front[-1]:
                continue
            center_pixel = pixels[-1]
            corner_pixels = pixels[:-1][in_front[:-1]]
            if corner_pixels.shape[0] == 0:
                continue

            x_min = np.clip(corner_pixels[:, 0].min(), 0, image_width)
            x_max = np.clip(corner_pixels[:, 0].max(), 0, image_width)
            y_min = np.clip(corner_pixels[:, 1].min(), 0, image_height)
            y_max = np.clip(corner_pixels[:, 1].max(), 0, image_height)
            if x_min == x_max or y_min == y_max:
                continue

            valid_bboxes.append([x_min, y_min, x_max, y_max])
            valid_centers.append(
                [
                    np.clip(center_pixel[0], 0, image_width),
                    np.clip(center_pixel[1], 0, image_height),
                ]
            )
            valid_labels.append(gt_labels[box_index])

        if valid_bboxes:
            return (
                np.asarray(valid_bboxes, dtype=np.float32),
                np.asarray(valid_centers, dtype=np.float32),
                np.asarray(valid_labels, dtype=np.int64),
            )
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
