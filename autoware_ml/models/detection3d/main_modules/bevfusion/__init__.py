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

"""BEVFusion (lidar-only): model (:mod:`.model`) and deployment stages (:mod:`.stages`).
Everything BEVFusion-specific lives in this directory; quantization rules land with
the PTQ milestone as :mod:`.quantization`.
"""

from autoware_ml.models.detection3d.main_modules.bevfusion.model import (
    BEVFusionLidarDetectionModel,
)

__all__ = ["BEVFusionLidarDetectionModel"]
