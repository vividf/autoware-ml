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

"""Sparse-convolution (spconv) primitives — model-agnostic.

TODO(vividf): no callers yet — this package exists for the BEVFusion migration (the
sparse encoder deploys in FP16 and needs one shared SparseConv+BN fold).
"""

from .fusion import fuse_spconv_bn_in_encoder

__all__ = [
    "fuse_spconv_bn_in_encoder",
]
