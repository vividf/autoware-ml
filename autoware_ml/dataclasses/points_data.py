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

"""Batched point inputs for models that serialize a point cloud instead of voxelizing it."""

from __future__ import annotations

from typing import NamedTuple

from jaxtyping import Float32, Int32, Int64
import torch


class PointsData(NamedTuple):
    """A grid-quantized point batch, flattened across samples.

    The input contract of serialization-based backbones (PTv3): points from every
    sample are concatenated along dim 0 and ``offset`` marks where each sample ends,
    which is how those models derive per-sample attention windows. ``grid_coord`` is
    the discrete cell each point fell into — the quantization already happened, so a
    model consuming this never re-quantizes.

    Attributes:
        coord: Metric point coordinates, shape ``(num_points, 3)``.
        feat: Point features, shape ``(num_points, num_features)``.
        grid_coord: Discrete grid coordinates, shape ``(num_points, 3)``.
        offset: Inclusive cumulative point count per sample, shape ``(batch_size,)``.
            ``offset[-1] == num_points``.

    Only inputs live here. The mapping back to the pre-quantization points is a target-
    side concern (metrics score original points) and travels with the segmentation GT.
    """

    coord: Float32[torch.Tensor, "num_points 3"]
    feat: Float32[torch.Tensor, "num_points num_features"]
    grid_coord: Int32[torch.Tensor, "num_points 3"]
    offset: Int64[torch.Tensor, " batch_size"]

    def to_device(self, device: torch.device) -> PointsData:
        """Move every tensor to ``device``."""
        return PointsData(
            coord=self.coord.to(device),
            feat=self.feat.to(device),
            grid_coord=self.grid_coord.to(device),
            offset=self.offset.to(device),
        )

    @property
    def batch_size(self) -> int:
        """Number of samples the batch carries."""
        return int(self.offset.shape[0])

    def as_point_dict(self) -> dict[str, torch.Tensor]:
        """The mapping serialization-based encoders consume."""
        return {
            "coord": self.coord,
            "feat": self.feat,
            "grid_coord": self.grid_coord,
            "offset": self.offset,
        }


__all__ = ["PointsData"]
