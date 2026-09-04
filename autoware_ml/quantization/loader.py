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

"""Loading a self-describing quantized checkpoint into a freshly built model.

The loader rebuilds the *identical* quantized module tree the quantize stage built
— from the config embedded in the checkpoint, via the model's own
``build_quantization_plan`` — verifies it against the embedded placement record, and
only then loads the weights, so the calibrated ``state_dict`` lines up by construction.
It never branches on ``mode``: a QAT checkpoint loads exactly like a PTQ one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from autoware_ml.quantization.checkpoint import QuantizationDescription
from autoware_ml.quantization.core.quantizer_state import (
    disable_quantizers_in,
    validate_quantizer_amax,
)
from autoware_ml.quantization.core.replace import expand_skip_quantize

logger = logging.getLogger(__name__)


def _load_checkpoints_into_model(
    model: torch.nn.Module,
    weight_paths: str | Path | Sequence[str | Path],
    map_location: str | torch.device,
) -> set[str]:
    """Load checkpoint(s) into a model whose quantized module tree is already prepared.

    NOTE: the checkpoint is loaded with a raw ``load_state_dict(strict=False)``, not
    ``apply_matching_weights``: an uncalibrated ``TensorQuantizer`` has no ``_amax``
    buffer yet, so the matching-weights key pre-filter would silently drop every
    calibrated ``_amax`` from the checkpoint. modelopt's (patched)
    ``_load_from_state_dict`` creates the buffer on load instead
    (:mod:`autoware_ml.quantization.core.modelopt`).

    Args:
        model: Model already prepared by the shared quantization plan.
        weight_paths: One checkpoint path or a sequence of them.
        map_location: Device to map the loaded tensors to.

    Returns:
        The set of state_dict keys actually loaded into the model (coverage policy is
        the caller's).
    """
    if isinstance(weight_paths, (str, Path)):
        weight_paths = [weight_paths]
    loaded_keys: set[str] = set()
    for weight_path in weight_paths:
        checkpoint = torch.load(str(weight_path), map_location=map_location, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded_keys.update(set(state_dict.keys()) - set(unexpected))
        logger.info(
            "Loaded checkpoint %s (%d tensors; %d missing here, %d unexpected)",
            weight_path,
            len(state_dict) - len(unexpected),
            len(missing),
            len(unexpected),
        )
        if unexpected:
            logger.warning("Unexpected checkpoint keys (first 10): %s", list(unexpected)[:10])
    return loaded_keys


def load_quantized_model(
    model: torch.nn.Module,
    weight_paths: Sequence[str | Path],
    description: QuantizationDescription,
    device: torch.device,
) -> torch.nn.Module:
    """Rebuild the quantized tree described by ``description`` and load the weights.

    Steps (mirroring the quantize stage, so state_dict keys match exactly):

    1. ``model.build_quantization_plan(description.config).prepare(model)`` — the SAME
       plan the quantize stage built (BN fuse + Q/DQ insert), recording every placement.
    2. Verify the rebuilt placement record against the checkpoint's embedded one — tree
       drift is a hard failure here instead of a silent weight mis-map.
    3. Load the checkpoint(s) with ``load_state_dict(strict=False)`` (see
       :func:`_load_checkpoints_into_model`) and enforce full key coverage.
    4. Move to ``device`` (the calibrated scales are buffers and move with the model) and
       disable the quantizers inside the ``skip_quantize`` subtrees — the same shared
       loop the quantize stage ran.
    5. Validate the remaining (enabled) quantizer amax values (TensorRT requires
       positive finite scales). modelopt's ``TensorQuantizer`` emits Q/DQ natively on
       ONNX export; nothing to configure.

    Args:
        model: Freshly built model with NO weights loaded yet.
        weight_paths: Checkpoint path(s); exactly one carries the quantization payload.
        description: The quantized checkpoint's embedded description.
        device: Target device.

    Returns:
        The model, quantized-tree-rebuilt, weights loaded, in eval mode on ``device``.
    """
    config = description.config
    logger.info(
        "Rebuilding the quantized tree from the checkpoint's description "
        "(mode=%s, fuse_bn=%s, skip_quantize=%s, disable_recipes=%s)",
        config.mode,
        config.fuse_bn,
        list(config.skip_quantize),
        list(config.disable_recipes),
    )
    plan = model.build_quantization_plan(config)
    plan.prepare(model)
    plan.placement_record.verify_matches(
        description.placement_record, source="the checkpoint's embedded placement record"
    )
    skip_layers = expand_skip_quantize(model, config.skip_quantize, log=False)

    loaded_keys = _load_checkpoints_into_model(model, weight_paths, map_location=device)
    uncovered = sorted(set(model.state_dict().keys()) - loaded_keys)
    if uncovered:
        raise RuntimeError(
            f"Model has {len(uncovered)} parameter(s) not covered by any checkpoint: "
            f"{', '.join(uncovered[:20])}{' ...' if len(uncovered) > 20 else ''}. "
            "Supply additional --weights or a checkpoint that includes these keys."
        )

    model.to(device)
    model.eval()
    # Disable BEFORE validating: a skip_quantize quantizer legitimately carries amax=None
    # and must not fail validation (disabled quantizers are skipped).
    disable_quantizers_in(model, skip_layers)
    validate_quantizer_amax(model)

    logger.info("Quantized checkpoint loaded successfully")
    return model
