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

"""PTv3: model (:mod:`.model`), deployment stages (:mod:`.stages`), quantization rules
(:mod:`.quantization`) — the same three files every model on this architecture has.
"""

from autoware_ml.models.segmentation3d.main_modules.ptv3.model import PTv3SegmentationModel

__all__ = ["PTv3SegmentationModel"]
