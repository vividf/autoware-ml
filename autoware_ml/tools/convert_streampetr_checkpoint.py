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

"""Convert an mm-format StreamPETR checkpoint (AWML) into autoware-ml layout.

The AWML/mmdetection3d StreamPETR (``Petr3D`` + ``StreamPETRHead`` + ``VoVNet``
+ ``CPFPN`` + flash-attention transformer) and the native autoware-ml
``StreamPETRDetectionModel`` share the same underlying weights but use
different module names. This script renames every parameter, optionally flips
the stem convolution's input channels for the BGR→RGB loader change, and
writes a ``{"state_dict": ...}`` payload consumable by
``autoware-ml train --weights``.

Requirements for a lossless conversion (all hold for the reference recipe):
  * ``img_neck`` must be the native ``CPFPN`` port (``GeneralizedLSSFPN`` is
    not weight-compatible with CPFPN checkpoints).
  * Both attention implementations (FlashMHA and ``nn.MultiheadAttention``)
    pack q/k/v as one ``in_proj_weight``, so no repacking is needed.

Usage:
    python -m autoware_ml.tools.convert_streampetr_checkpoint \
        --input work_dirs/streampetr_2_7/epoch_20.pth \
        --output streampetr_2_7_epoch_20_converted.pth \
        --bgr-to-rgb
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class _StubMeta(type):
    """Resolve arbitrary class attributes (mm pickles methods as getattr)."""

    def __getattr__(cls, name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return _StubObject()


class _StubObject(metaclass=_StubMeta):
    """Placeholder for unpicklable mm metadata objects."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __call__(self, *args: object, **kwargs: object) -> "_StubObject":
        del args, kwargs
        return self

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickle mm checkpoints without the mm packages installed.

    mm-format checkpoints embed mmengine bookkeeping objects (message hub,
    history buffers) next to the tensors. Those classes are irrelevant to the
    conversion, so missing modules resolve to inert stub objects instead of
    failing the load.
    """

    def find_class(self, module: str, name: str) -> type:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            logger.debug("Stubbing unpicklable class %s.%s", module, name)
            return type(name, (_StubObject,), {})


class _TolerantPickleModule:
    """Pickle-module shim handing torch.load the tolerant unpickler."""

    Unpickler = _TolerantUnpickler

    @staticmethod
    def load(file: object, **kwargs: object) -> object:
        return _TolerantUnpickler(file, **kwargs).load()


# mm parameter names that have no counterpart in the native model
# (non-persistent buffers, mm bookkeeping).
_SKIPPED_PATTERNS = (
    re.compile(r"^pts_bbox_head\.coords_d$"),
    re.compile(r"^pts_bbox_head\.code_weights$"),
    re.compile(r"^pts_bbox_head\.match_costs$"),
    re.compile(r"^pts_bbox_head\.pc_range$"),
    re.compile(r"^pts_bbox_head\.position_range$"),
)

_STEM_CONV_KEY = "img_backbone.stem.stem_1/conv.weight"


def _map_decoder_layer(name: str) -> str | None:
    """Map one mm transformer-decoder parameter name to the native layout."""
    match = re.match(
        r"^transformer\.decoder\.layers\.(\d+)\.(.+)$",
        name,
    )
    if match is None:
        return None
    layer_index, remainder = match.group(1), match.group(2)
    replacements = {
        "attentions.0.attn.": f"decoder.{layer_index}.self_attn.",
        "attentions.1.attn.": f"decoder.{layer_index}.cross_attn.",
        "ffns.0.layers.0.0.": f"decoder.{layer_index}.ffn.0.",
        "ffns.0.layers.1.": f"decoder.{layer_index}.ffn.3.",
        "norms.0.": f"decoder.{layer_index}.norm1.",
        "norms.1.": f"decoder.{layer_index}.norm2.",
        "norms.2.": f"decoder.{layer_index}.norm3.",
    }
    for prefix, replacement in replacements.items():
        if remainder.startswith(prefix):
            return replacement + remainder[len(prefix) :]
    return None


def _map_head_key(name: str) -> str | None:
    """Map one ``pts_bbox_head``-relative mm name to the native head layout."""
    decoder_name = _map_decoder_layer(name)
    if decoder_name is not None:
        return decoder_name
    if name.startswith("transformer.decoder.post_norm."):
        return "post_norm." + name[len("transformer.decoder.post_norm.") :]
    # SELayer_Linear: conv_reduce/conv_expand → reduce/expand.
    if name.startswith("featurized_pe.conv_reduce."):
        return "featurized_pe.reduce." + name[len("featurized_pe.conv_reduce.") :]
    if name.startswith("featurized_pe.conv_expand."):
        return "featurized_pe.expand." + name[len("featurized_pe.conv_expand.") :]
    # MLN (spatial_alignment / ego_pose_pe / ego_pose_memory): ln has no
    # parameters; reduce/gamma/beta names match the native ModulatedLayerNorm.
    # Every remaining submodule (position_encoder, memory_embed, query_embedding,
    # time_embedding, reference_points, pseudo_reference_points, cls/reg
    # branches, spatial_alignment.*, ego_pose_*) is name-stable.
    return name


def convert_state_dict(
    mm_state_dict: dict[str, torch.Tensor],
    *,
    bgr_to_rgb: bool,
    drop_patterns: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Convert an mm StreamPETR state dict into the native naming scheme.

    Args:
        mm_state_dict: Raw mm-format ``state_dict``.
        bgr_to_rgb: Flip the stem convolution's input channels so BGR-trained
            weights consume RGB inputs (valid because the normalization
            statistics swap channel-consistently).
        drop_patterns: Optional regexes matched against the *source* (mm) key
            names; matching tensors are dropped. Use this to strip layers
            whose shapes cannot match the target model, e.g. class-count
            dependent heads when initializing a 7-class T4 model from a
            10-class nuScenes checkpoint (mm's ``strict=False`` load silently
            skips those; autoware-ml's loader refuses shape mismatches, so
            they must be dropped at conversion time instead).

    Returns:
        Converted state dict and the list of skipped source keys.
    """
    dropped_regexes = [re.compile(pattern) for pattern in (drop_patterns or [])]
    converted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, tensor in mm_state_dict.items():
        if any(pattern.match(name) for pattern in _SKIPPED_PATTERNS) or any(
            pattern.search(name) for pattern in dropped_regexes
        ):
            skipped.append(name)
            continue
        if name.startswith("img_backbone.") or name.startswith("img_neck."):
            new_name = name
        elif name.startswith("pts_bbox_head."):
            head_name = _map_head_key(name[len("pts_bbox_head.") :])
            if head_name is None:
                skipped.append(name)
                continue
            new_name = "bbox_head." + head_name
        elif name.startswith("img_roi_head."):
            roi_name = name[len("img_roi_head.") :]
            new_name = "img_roi_head." + roi_name
        else:
            skipped.append(name)
            continue
        if new_name in converted:
            raise ValueError(f"Duplicate converted key {new_name!r} (from {name!r}).")
        converted[new_name] = tensor

    if bgr_to_rgb:
        if _STEM_CONV_KEY not in converted:
            raise KeyError(
                f"--bgr-to-rgb requested but {_STEM_CONV_KEY!r} is missing from the checkpoint."
            )
        converted[_STEM_CONV_KEY] = converted[_STEM_CONV_KEY][:, [2, 1, 0]].contiguous()

    return converted, skipped


def load_mm_checkpoint(path: Path) -> dict:
    """Load an mm-format checkpoint without requiring the mm packages."""
    return torch.load(
        str(path),
        map_location="cpu",
        weights_only=False,
        pickle_module=_TolerantPickleModule,
    )


def main() -> None:
    """Convert one checkpoint file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="mm checkpoint (.pth)")
    parser.add_argument("--output", type=Path, required=True, help="converted checkpoint path")
    parser.add_argument(
        "--bgr-to-rgb",
        action="store_true",
        help=(
            "Flip the stem conv input channels. Use when the source was trained "
            "on BGR images (AWML) and the target pipeline loads RGB (autoware-ml "
            "default)."
        ),
    )
    parser.add_argument(
        "--drop-pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help=(
            "Drop source tensors whose mm key matches this regex (repeatable). "
            "Needed when the class count differs between source and target, "
            "e.g. nuScenes (10 classes) -> T4 (7 classes): "
            "--drop-pattern 'cls_branches\\.\\d+\\.6\\.' "
            "--drop-pattern 'img_roi_head\\.cls\\.'"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    payload = load_mm_checkpoint(args.input)
    mm_state_dict = payload.get("state_dict", payload)
    converted, skipped = convert_state_dict(
        mm_state_dict, bgr_to_rgb=args.bgr_to_rgb, drop_patterns=args.drop_pattern
    )

    logger.info("Converted %d tensors; skipped %d.", len(converted), len(skipped))
    for name in skipped:
        logger.info("  skipped: %s", name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": converted}, str(args.output))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
