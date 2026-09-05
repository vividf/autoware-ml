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

"""CenterPoint: model (:mod:`.model`), deployment stages (:mod:`.stages`), quantization
rules (:mod:`.quantization`). Everything CenterPoint-specific lives in this directory;
adding a model means adding a sibling directory with the same three files.
"""

from autoware_ml.models.detection3d.main_modules.centerpoint.model import CenterPointDetectionModel

__all__ = ["CenterPointDetectionModel"]
