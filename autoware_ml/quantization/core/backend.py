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

"""DEPRECATED module path — renamed to :mod:`autoware_ml.quantization.core.modelopt`.

"backend" in this repo means a *runtime* backend (pytorch / onnx / tensorrt,
:class:`autoware_ml.types.backend.Backend`); this module is the fake-quant *library*
seam, which is exactly one library. Kept as a re-export shim for one deprecation cycle.
"""

from autoware_ml.quantization.core.modelopt import *  # noqa: F401,F403
from autoware_ml.quantization.core.modelopt import (  # noqa: F401
    _ensure_modelopt_patches,
    available,
    get_preset_desc,
    install_hint,
    make_quant_desc,
    require,
    resolve,
)
