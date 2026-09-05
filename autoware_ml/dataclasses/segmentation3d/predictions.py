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

"""Decoded 3D semantic segmentation predictions."""

from __future__ import annotations

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict
import torch


class Segmentation3DPredictions(BaseModel):
    """Point-wise semantic predictions for a whole batch.

    Points are flattened across samples the way the inputs are (see
    :class:`~autoware_ml.dataclasses.points_data.PointsData`), so these align with the
    model's *input* points. Scoring the original cloud means scattering them through
    ``Segmentation3DGTBatch.inverse`` first.

    Attributes:
      pred_labels: Winning class per point, shape ``(num_points,)``.
      pred_probs: Class probabilities per point, shape ``(num_points, num_classes)``.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    pred_labels: Int64[torch.Tensor, " num_points"]
    pred_probs: Float32[torch.Tensor, "num_points num_classes"]
