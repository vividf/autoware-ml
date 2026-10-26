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

"""Load checkpoints into a freshly built model, rebuilding the tree of a quantized one."""

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
from autoware_ml.quantization.plan import QuantizationPlan

logger = logging.getLogger(__name__)


def _load_checkpoints_into_model(
    model: torch.nn.Module,
    weight_paths: str | Path | Sequence[str | Path],
    map_location: str | torch.device,
) -> set[str]:
    """Load checkpoints into an already prepared quantized tree and return the keys loaded.

    Not ``apply_matching_weights``, whose key pre-filter would drop every calibrated
    ``_amax``: a fresh ``TensorQuantizer`` has no ``_amax`` buffer yet. The patched
    ``_load_from_state_dict`` (:mod:`autoware_ml.quantization.core.modelopt`) creates the
    buffers on load.
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
    *,
    enforce_full_coverage: bool = True,
) -> torch.nn.Module:
    """Rebuild the quantized tree of ``description`` in a freshly built model and load the weights.

    The rebuilt placement record must match the embedded one. Returns the model in eval mode
    on ``device``.
    """
    config = description.config
    logger.info(
        "Rebuilding the quantized tree from the checkpoint's description "
        "(mode=%s, skip_quantize=%s)",
        config.mode,
        list(config.skip_quantize),
    )
    plan = QuantizationPlan.for_model(model, config)
    plan.prepare(model)
    plan.placement_record.verify_matches(
        description.placement_record, source="the checkpoint's embedded placement record"
    )
    skip_layers = expand_skip_quantize(model, config.skip_quantize, log=False)

    loaded_keys = _load_checkpoints_into_model(model, weight_paths, map_location=device)
    uncovered = sorted(set(model.state_dict().keys()) - loaded_keys)
    if uncovered and not enforce_full_coverage:
        logger.warning(
            "%d model parameter(s) not covered by the checkpoint(s) (first: %s).",
            len(uncovered),
            ", ".join(uncovered[:5]),
        )
    elif uncovered:
        raise RuntimeError(
            f"Model has {len(uncovered)} parameter(s) not covered by any checkpoint: "
            f"{', '.join(uncovered[:20])}{' ...' if len(uncovered) > 20 else ''}. "
            "Supply additional --weights or a checkpoint that includes these keys."
        )

    model.to(device)
    model.eval()
    # Disable first: skip_quantize quantizers carry amax=None, and validation skips disabled ones.
    disable_quantizers_in(model, skip_layers)
    validate_quantizer_amax(model)

    logger.info("Quantized checkpoint loaded successfully")
    return model


def load_model_weights(
    model: torch.nn.Module,
    weight_paths: str | Path | Sequence[str | Path],
    device: torch.device,
    *,
    set_eval: bool = True,
    enforce_full_coverage: bool = True,
    skip_mismatched_shapes: bool = False,
    allow_quantized: bool = True,
) -> None:
    """Load ``--weights`` into a freshly built model, quantized or not.

    A quantized checkpoint goes through :func:`load_quantized_model`, any other through
    :func:`~autoware_ml.utils.checkpoints.apply_matching_weights`. ``allow_quantized=False``
    refuses a quantized checkpoint, for a stage that needs the FP model, and
    ``skip_mismatched_shapes`` applies to FP checkpoints only: a quantized one must match its
    rebuilt tree exactly.
    """
    from autoware_ml.quantization.checkpoint import find_quantization
    from autoware_ml.utils.checkpoints import apply_matching_weights

    if isinstance(weight_paths, (str, Path)):
        weight_paths = [weight_paths]
    weight_paths = [Path(path) for path in weight_paths]
    quantized = find_quantization(weight_paths)
    if quantized is not None:
        path, description = quantized
        if not allow_quantized:
            raise ValueError(
                f"{path} is a quantized checkpoint (mode={description.config.mode}), which "
                "this stage cannot take: it needs the floating-point model. Train or "
                "quantize from the FP checkpoint; a quantized checkpoint is for deploy / test."
            )
        if skip_mismatched_shapes:
            raise ValueError(
                f"{path} is a quantized checkpoint: its weights must match the quantized tree "
                "rebuilt from its description exactly, so --skip-mismatched-weights cannot apply."
            )
        logger.info(
            "Quantized checkpoint detected (%s, mode=%s): rebuilding the quantized tree from "
            "its embedded description.",
            path,
            description.config.mode,
        )
        load_quantized_model(
            model, weight_paths, description, device, enforce_full_coverage=enforce_full_coverage
        )
        if not set_eval:
            model.train()
        return
    apply_matching_weights(
        model,
        weight_paths,
        map_location=device,
        device=device,
        set_eval=set_eval,
        enforce_full_coverage=enforce_full_coverage,
        skip_mismatched_shapes=skip_mismatched_shapes,
        logger=logger,
    )
