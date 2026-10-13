"""Unit tests for the batched image normalization layer."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.tests.batch_inputs_fixtures import build_batch_inputs
from autoware_ml.preprocessing.camera.normalize import BatchNormalizeImages

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _inputs(images: torch.Tensor) -> ModelBatchInputs:
    batch_size, num_cams = images.shape[:2]
    return build_batch_inputs(
        images=ImageGTBatch(
            images=images,
            depth_maps=None,
            camera_intrinsics=torch.eye(3).expand(batch_size, num_cams, 3, 3),
            image_augmentation_matrices=torch.eye(4).expand(batch_size, num_cams, 4, 4),
            lidar2images=torch.eye(4).expand(batch_size, num_cams, 4, 4),
            lidar2cams=torch.eye(4).expand(batch_size, num_cams, 4, 4),
            calibration_statuses=None,
        )
    )


@pytest.mark.parametrize("is_training", [True, False])
def test_matches_the_runtime_normalization_of_8_bit_pixels(is_training: bool) -> None:
    # The runtime feeds the graph (pixel - mean) / std with 0-255 pixels and statistics.
    pixels = torch.randint(0, 256, (2, 3, 3, 4, 5)).float()
    mean_255 = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
    std_255 = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)

    normalized = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
        _inputs(pixels / 255.0), is_training=is_training
    )

    torch.testing.assert_close(normalized.image_data.images, (pixels - mean_255) / std_255)


def test_keeps_the_calibration() -> None:
    inputs = _inputs(torch.rand(1, 2, 3, 4, 5))

    normalized = BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(inputs, is_training=True)

    assert normalized.image_data.lidar2images is inputs.image_data.lidar2images


def test_rejects_images_of_another_channel_count() -> None:
    with pytest.raises(ValueError, match="channels"):
        BatchNormalizeImages(IMAGENET_MEAN, IMAGENET_STD)(
            _inputs(torch.rand(1, 2, 4, 4, 5)), is_training=True
        )


def test_rejects_inconsistent_statistics() -> None:
    with pytest.raises(ValueError):
        BatchNormalizeImages([0.5, 0.5], IMAGENET_STD)
    with pytest.raises(ValueError):
        BatchNormalizeImages(IMAGENET_MEAN, [0.2, 0.0, 0.2])
