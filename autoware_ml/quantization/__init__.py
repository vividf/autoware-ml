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
Quantization framework (model-agnostic).

PTQ / QAT building blocks based on NVIDIA's modelopt toolkit, organized in layers:

- :mod:`~autoware_ml.quantization.plan`    — the single interface between deployment
  stages and quantization: ``QuantRules`` (a model's declaration) + ``QuantizationPlan``
  (rules bound to config; ``prepare`` builds the tree AND records a ``PlacementRecord``
  of every placement decision).
- :mod:`~autoware_ml.quantization.core`    — model-agnostic engine on nvidia-modelopt
  (descriptor tables, in-place module conversion through modelopt's ``QuantModuleRegistry``,
  BN fusion, calibration, quantizer state).
- :mod:`~autoware_ml.quantization.recipes` — architecture-specific Q/DQ placement as
  matcher+action recipes: quantized block classes selected by ``ResidualBlockSpec`` /
  ``ESEBlockSpec`` rows (residual blocks, VoVNet eSE), plus the MaxPool input wrapper.
- :mod:`~autoware_ml.quantization.config`  — typed view of the Hydra ``quantization`` section.
- :mod:`~autoware_ml.quantization.checkpoint` — self-describing quantized checkpoints (config +
  placement record embedded next to the ``state_dict``; no sidecar files).
- :mod:`~autoware_ml.quantization.loader` — rebuild + verify + load from that description.
- :mod:`~autoware_ml.quantization.qat_callback` — Lightning callback that turns a training run
  into frozen-amax QAT fine-tuning.

A model's quantization declaration (e.g. CenterPoint's ``QuantRules``) lives next to the model
and is exposed through the model's ``build_quantization_plan()`` hook — the engine never imports
a model.

The invariant every stage preserves: the quantize stage (PTQ / QAT) and the loader all build the
quantized module tree by calling the *same* ``build_quantization_plan(config).prepare(model)``,
so the calibrated ``state_dict`` and the later ``load_state_dict`` line up by construction —
and the placement record embedded in the checkpoint lets the loader machine-check that instead
of trusting it. Because the config travels inside the checkpoint, ``deploy`` and ``test`` need
no ``quantization`` section at all.

The names exported here are the package's real external API. Deeper internals (descriptor
tables, the single Conv-BN fold, the block registry) stay importable from their defining
modules but are deliberately not re-exported.
"""

from .checkpoint import (
    QUANTIZATION_KEY,
    QuantizationDescription,
    find_quantization,
    read_quantization,
    save_quantized_checkpoint,
)
from .config import CalibrationConfig, PTQConfig, QATConfig, QuantizationConfig
from .core.calibration import Calibrator
from .core.fusion import fuse_model_bn
from .core.quantizer_state import (
    disable_quantizers_in,
    print_quantizer_status,
    quantizers_disabled,
    set_quantizers_enabled,
    validate_quantizer_amax,
)
from .core.replace import (
    expand_skip_quantize,
    match_skip_quantize_roots,
    replace_quantizable_modules,
)
from .loader import load_quantized_model
from .plan import PlacementDecision, PlacementRecord, QuantizationPlan, QuantRules

__all__ = [
    # Typed config
    "QuantizationConfig",
    "CalibrationConfig",
    "PTQConfig",
    "QATConfig",
    # Plan (the single interface between deployment stages and quantization)
    "QuantRules",
    "QuantizationPlan",
    "PlacementRecord",
    "PlacementDecision",
    # Self-describing checkpoints
    "QUANTIZATION_KEY",
    "QuantizationDescription",
    "save_quantized_checkpoint",
    "read_quantization",
    "find_quantization",
    "load_quantized_model",
    # Replace / placement
    "replace_quantizable_modules",
    "expand_skip_quantize",
    "match_skip_quantize_roots",
    # Calibration
    "Calibrator",
    # Fusion
    "fuse_model_bn",
    # Quantizer state
    "disable_quantizers_in",
    "quantizers_disabled",
    "set_quantizers_enabled",
    "validate_quantizer_amax",
    "print_quantizer_status",
]
