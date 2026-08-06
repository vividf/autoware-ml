# Copyright 2025 TIER IV, Inc.
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

"""Unit tests for camera transforms."""

import numpy as np
import numpy.typing as npt
import pytest

from autoware_ml.transforms.camera.annotations2d import LoadAnnotations2DFromBoxes3D
from autoware_ml.transforms.camera.distortion import UndistortImage
from autoware_ml.transforms.camera.masking import GridMask
from autoware_ml.transforms.camera.normalize import NormalizeMultiviewImage
from autoware_ml.transforms.camera.resize import (
    CropAndScale,
    PadMultiViewImage,
    ResizeCropFlipRotImage,
)
from autoware_ml.utils.calibration import CalibrationData


class TestUndistortImage:
    """Tests for UndistortImage transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom alpha."""
        transform = UndistortImage()
        assert transform.alpha == 0.0

        transform = UndistortImage(alpha=0.5)
        assert transform.alpha == 0.5

    def test_missing_img_key(self, sample_calibration_data: CalibrationData) -> None:
        """Test that missing 'img' key raises KeyError."""
        transform = UndistortImage()
        input_dict = {"calibration_data": sample_calibration_data}

        with pytest.raises(KeyError, match="Missing required key 'img'"):
            transform(input_dict)

    def test_missing_calibration_data_key(self, sample_image: npt.NDArray[np.uint8]) -> None:
        """Test that missing 'calibration_data' key raises KeyError."""
        transform = UndistortImage()
        input_dict = {"img": sample_image}

        with pytest.raises(KeyError, match="Missing required key 'calibration_data'"):
            transform(input_dict)

    def test_passthrough_zero_distortion(
        self,
        sample_image: npt.NDArray[np.uint8],
        sample_calibration_data_no_distortion: CalibrationData,
    ) -> None:
        """Test that zero distortion coefficients pass through unchanged."""
        transform = UndistortImage()
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data_no_distortion,
        }

        output_dict = transform(input_dict)

        # Image should be unchanged (same reference since early return)
        assert "img" in output_dict
        assert output_dict["img"].shape == sample_image.shape

    def test_output_shape_preserved(
        self, sample_image: npt.NDArray[np.uint8], sample_calibration_data: CalibrationData
    ) -> None:
        """Test that output image shape matches input shape."""
        transform = UndistortImage()
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        assert output_dict["img"].shape == sample_image.shape
        assert output_dict["img"].dtype == sample_image.dtype

    def test_new_camera_matrix_updated(
        self, sample_image: npt.NDArray[np.uint8], sample_calibration_data: CalibrationData
    ) -> None:
        """Test that new_camera_matrix is updated after undistortion."""
        transform = UndistortImage()

        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)
        output_calibration = output_dict["calibration_data"]

        # new_camera_matrix should be set
        assert output_calibration.new_camera_matrix is not None
        # Distortion coefficients should be zeroed
        assert np.allclose(output_calibration.distortion_coefficients, 0)

    def test_calibration_data_returned(
        self, sample_image: npt.NDArray[np.uint8], sample_calibration_data: CalibrationData
    ) -> None:
        """Test that calibration_data is returned in output."""
        transform = UndistortImage()
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        assert "calibration_data" in output_dict
        assert isinstance(output_dict["calibration_data"], CalibrationData)


class TestCropAndScale:
    """Tests for CropAndScale transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        transform = CropAndScale()
        assert transform.p == 0.5
        assert transform.crop_ratio == 0.8

        transform = CropAndScale(p=0.9, crop_ratio=0.7)
        assert transform.p == 0.9
        assert transform.crop_ratio == 0.7

    def test_missing_keys(
        self, sample_image: npt.NDArray[np.uint8], sample_calibration_data: CalibrationData
    ) -> None:
        """Test that missing required keys raise KeyError."""
        transform = CropAndScale()

        # Missing img
        with pytest.raises(KeyError, match="Missing required key 'img'"):
            transform({"calibration_data": sample_calibration_data})

        # Missing calibration_data
        with pytest.raises(KeyError, match="Missing required key 'calibration_data'"):
            transform({"img": sample_image})

    def test_never_apply(
        self, sample_image: npt.NDArray[np.uint8], sample_calibration_data: CalibrationData
    ) -> None:
        """Test with p=0.0 returns input unchanged."""
        transform = CropAndScale(p=0.0)
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        # Should return same reference when not applied
        assert output_dict["img"].shape == sample_image.shape

    def test_always_apply_shape_preserved(
        self, sample_image: npt.NDArray[np.uint8], sample_calibration_data: CalibrationData
    ) -> None:
        """Test with p=1.0 preserves output shape."""
        transform = CropAndScale(p=1.0, crop_ratio=0.8)
        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        # Output shape should match input shape (resize back)
        assert output_dict["img"].shape == sample_image.shape

    def test_camera_matrix_updated(
        self, sample_image: npt.NDArray[np.uint8], sample_calibration_data: CalibrationData
    ) -> None:
        """Test that camera matrix is updated when transform is applied."""
        transform = CropAndScale(p=1.0, crop_ratio=0.8)
        original_camera_matrix = sample_calibration_data.new_camera_matrix.copy()

        input_dict = {
            "img": sample_image.copy(),
            "calibration_data": sample_calibration_data,
        }

        output_dict = transform(input_dict)

        # Camera matrix should be modified
        assert not np.allclose(
            output_dict["calibration_data"].new_camera_matrix,
            original_camera_matrix,
        )


def test_normalize_multiview_image_handles_chw_stack() -> None:
    images = np.ones((2, 3, 4, 5), dtype=np.float32)

    output = NormalizeMultiviewImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], to_rgb=False)(
        {"img": images}
    )

    assert output["img"].shape == (2, 3, 4, 5)
    assert np.allclose(output["img"], 1.0)


def test_pad_multiview_image_handles_chw_stack() -> None:
    images = np.ones((2, 3, 4, 5), dtype=np.float32)

    output = PadMultiViewImage(size_divisor=4, pad_val=0.0)({"img": images})

    assert output["img"].shape == (2, 3, 4, 8)
    assert output["pad_shape"] == (4, 8)


def test_resize_crop_flip_rot_image_updates_chw_stack_and_aug_matrix() -> None:
    images = np.ones((2, 3, 8, 8), dtype=np.float32)
    intrinsics = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    lidar2cam = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    transform = ResizeCropFlipRotImage(
        data_aug_conf={
            "resize_lim": (1.0, 1.0),
            "final_dim": (6, 6),
            "bot_pct_lim": (0.0, 0.0),
            "rot_lim": (0.0, 0.0),
            "rand_flip": False,
        },
        training=False,
    )

    output = transform({"img": images, "camera_intrinsics": intrinsics, "lidar2cam": lidar2cam})

    assert output["img"].shape == (2, 3, 6, 6)
    assert output["img_aug_matrix"].shape == (2, 4, 4)
    assert output["lidar2img"].shape == (2, 4, 4)


def test_resize_crop_aug_matrix_matches_pixels_for_mismatched_aspect() -> None:
    """The augmentation matrix must describe the actual pixel mapping.

    Regression test: with a source aspect ratio different from final_dim, a
    hidden second resize used to stretch the pixels without entering the
    matrix, desynchronizing the updated intrinsics from the image.
    """
    image = np.zeros((930, 1440, 3), dtype=np.uint8)
    image[450:480, 705:735] = 255  # block centered at pixel (720, 465)
    transform = ResizeCropFlipRotImage(
        data_aug_conf={
            "resize_lim": 0.02,
            "final_dim": (240, 320),
            "bot_pct_lim": (0.0, 0.0),
            "rot_lim": (0.0, 0.0),
            "rand_flip": False,
        },
        training=False,
    )

    matrix, augmented = transform._augment_image(image)

    predicted = matrix @ np.array([720.0, 465.0, 1.0, 1.0])
    rows, cols = np.nonzero(augmented[..., 0] > 128)
    assert augmented.shape[:2] == (240, 320)
    assert abs(cols.mean() - predicted[0]) < 1.0
    assert abs(rows.mean() - predicted[1]) < 1.0


def test_grid_mask_handles_chw_stack() -> None:
    np.random.seed(0)
    images = np.ones((2, 3, 64, 64), dtype=np.float32)

    output = GridMask(p=1.0, ratio=0.5, rotate=0)({"img": images})

    assert output["img"].shape == images.shape
    assert (output["img"] == 0).any()


def test_load_annotations_2d_treats_box_z_as_gravity_center() -> None:
    """The projected 2D box and center must straddle z, not sit above it.

    ``gt_boxes`` stores the gravity center. Treating that z as the bottom face
    (building corners from z upward, or adding dz/2 before projecting the
    center) silently lifts every 2D auxiliary target by half an object height,
    which is invisible in the loss but wrong everywhere.
    """
    # Pinhole camera: fx = fy = 100, principal point at (50, 50).
    cam2img = np.eye(4)
    cam2img[0, 0] = cam2img[1, 1] = 100.0
    cam2img[0, 2] = cam2img[1, 2] = 50.0
    # Lidar (x forward, y left, z up) -> camera (x right, y down, z forward).
    lidar2cam = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # One 2 m cube centered on the optical axis, 10 m ahead.
    gt_boxes = np.array([[10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    output = LoadAnnotations2DFromBoxes3D()(
        {
            "img": np.zeros((1, 3, 100, 100), dtype=np.float32),
            "gt_boxes": gt_boxes,
            "gt_labels": np.array([0]),
            "lidar2cam": [lidar2cam],
            "camera_intrinsics": [cam2img],
        }
    )

    center = output["centers_2d"][0][0]
    x1, y1, x2, y2 = output["gt_bboxes_2d"][0][0]
    # z = 0 is on the optical axis, so the center projects to the principal
    # point. The bottom-face reading would put it at y = 40.
    assert center == pytest.approx([50.0, 50.0], abs=1e-3)
    # The cube spans 9-11 m in depth, so its near face projects largest and
    # sets the extent: 100 * 1 / 9 either side of the principal point.
    extent = 100.0 / 9.0
    assert (y1, y2) == pytest.approx((50.0 - extent, 50.0 + extent), abs=1e-3)
    assert (x1, x2) == pytest.approx((50.0 - extent, 50.0 + extent), abs=1e-3)
    # The center sits at the middle of the box, not on its top edge.
    assert center[1] == pytest.approx((y1 + y2) / 2, abs=1e-3)
