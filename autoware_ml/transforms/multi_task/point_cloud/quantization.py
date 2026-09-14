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

"""DEPRECATED module path — renamed to :mod:`.grid_sampling`.

"quantization" in this repo means INT8/FP8 fake-quant; this transform subsamples a
point cloud on a voxel grid. Kept as a re-export shim for one deprecation cycle.
"""

from autoware_ml.transforms.multi_task.point_cloud.grid_sampling import GridSample  # noqa: F401
