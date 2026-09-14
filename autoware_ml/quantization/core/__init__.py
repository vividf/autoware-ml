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

"""Model-agnostic quantization engine, on top of nvidia-modelopt.

The generic building blocks that know only about ``nn.Conv2d`` / ``nn.Linear`` and modelopt —
not about any particular model: the per-precision descriptor tables (:mod:`.descriptors`),
the in-place module conversion walker on modelopt's ``QuantModuleRegistry`` (:mod:`.replace`),
BN fusion (:mod:`.fusion`), calibration (:mod:`.calibration`), and quantizer state operations
(:mod:`.quantizer_state`). :mod:`.modelopt` holds the modelopt bug workarounds and is imported
here first so every quantizer built by this package sees them.

Architecture-specific placement lives in :mod:`autoware_ml.quantization.recipes`; the deployment
interface (rules / plan / placement record) lives in :mod:`autoware_ml.quantization.plan`.
"""

from . import modelopt as _modelopt_patches  # noqa: F401  (import applies the patches)
