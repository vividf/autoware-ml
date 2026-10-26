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

"""
Model-agnostic quantization on nvidia-modelopt.

The exports load on first access: every model imports :mod:`.rules` through
``models/module_base_model.py``, which runs this ``__init__`` first, and eager re-exports
would import modelopt with every model.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, ...]] = {
    ".config": ("QuantizationConfig", "CalibrationConfig", "PTQConfig", "QATConfig"),
    ".rules": ("QuantRules",),
    ".plan": ("QuantizationPlan", "PlacementRecord", "PlacementDecision"),
    ".checkpoint": (
        "QUANTIZATION_KEY",
        "QuantizationDescription",
        "save_quantized_checkpoint",
        "read_quantization",
        "find_quantization",
    ),
    ".loader": ("load_quantized_model",),
    ".core.replace": (
        "replace_quantizable_modules",
        "expand_skip_quantize",
        "match_skip_quantize_roots",
    ),
    ".core.calibration": ("Calibrator",),
    "autoware_ml.utils.bn_fusion": ("fuse_model_bn",),
    ".core.quantizer_state": (
        "disable_quantizers_in",
        "quantizers_disabled",
        "set_quantizers_enabled",
        "validate_quantizer_amax",
        "print_quantizer_status",
    ),
}
_MODULE_OF = {name: module for module, names in _EXPORTS.items() for name in names}

__all__ = list(_MODULE_OF)


def __getattr__(name: str) -> Any:
    module = _MODULE_OF.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
