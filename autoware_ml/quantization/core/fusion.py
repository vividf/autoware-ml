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

"""BatchNorm fusion utilities for quantization.

Fusing BatchNorm into preceding convolutions is important for quantization because:
1. It reduces the number of operations, improving inference speed
2. It eliminates a source of quantization error (BN scaling after quantized conv)
3. It's required for accurate fake quantization during QAT training
"""

import logging
from collections.abc import Iterator

import torch
from torch import nn

logger = logging.getLogger(__name__)


def fuse_bn_weights(
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_eps: float,
    bn_weight: torch.Tensor | None,
    bn_bias: torch.Tensor | None,
    is_transposed: bool = False,
) -> tuple[nn.Parameter, nn.Parameter]:
    """
    Fuse BatchNorm parameters into convolution weights.

    The fused convolution computes:
        y = (W * x + b - mean) * (gamma / sqrt(var + eps)) + beta

    Which can be rewritten as:
        y = (W * gamma / sqrt(var + eps)) * x + (b - mean) * gamma / sqrt(var + eps) + beta

    So the fused weights are:
        W_fused = W * gamma / sqrt(var + eps)
        b_fused = (b - mean) * gamma / sqrt(var + eps) + beta

    Args:
        conv_weight: Convolution weight tensor
            - For Conv2d: [out_channels, in_channels, H, W]
            - For ConvTranspose2d: [in_channels, out_channels, H, W]
        conv_bias: Convolution bias tensor [out_channels] or None
        bn_mean: BatchNorm running mean [out_channels]
        bn_var: BatchNorm running variance [out_channels]
        bn_eps: BatchNorm epsilon
        bn_weight: BatchNorm weight (gamma) [out_channels] or None
        bn_bias: BatchNorm bias (beta) [out_channels] or None
        is_transposed: If True, conv_weight is from ConvTranspose2d with shape
            [in_channels, out_channels, H, W] where scale applies to dim 1

    Returns:
        Tuple of (fused_weight, fused_bias) as nn.Parameters
    """
    # Handle None values
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_mean)
    if bn_weight is None:
        bn_weight = torch.ones_like(bn_mean)
    if bn_bias is None:
        bn_bias = torch.zeros_like(bn_mean)

    # Compute 1 / sqrt(var + eps)
    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)

    # Compute scale factor: gamma / sqrt(var + eps)
    scale = bn_weight * bn_var_rsqrt

    # Reshape for broadcasting with conv weights
    # Conv2d weight shape: [out_channels, in_channels, H, W] -> scale on dim 0
    # ConvTranspose2d weight shape: [in_channels, out_channels, H, W] -> scale on dim 1
    if is_transposed:
        # For ConvTranspose2d: scale applies to dimension 1 (out_channels)
        shape = [1, -1] + [1] * (conv_weight.ndim - 2)
    else:
        # For Conv2d/Linear: scale applies to dimension 0 (out_channels)
        shape = [-1] + [1] * (conv_weight.ndim - 1)

    # Fuse weights: W_fused = W * scale
    fused_weight = conv_weight * scale.reshape(shape)

    # Fuse bias: b_fused = (b - mean) * scale + beta
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias

    return nn.Parameter(fused_weight.contiguous()), nn.Parameter(fused_bias.contiguous())


def fuse_conv_bn(conv: nn.Module, bn: nn.Module):
    """
    Fuse Conv and BatchNorm modules in-place.

    This modifies the conv module's weight and bias parameters to include
    the BatchNorm transformation, so the BN can be replaced with Identity.

    Args:
        conv: Convolution module (Conv1d, Conv2d, ConvTranspose2d, or Linear)
        bn: BatchNorm module (BatchNorm1d or BatchNorm2d)

    Raises:
        AssertionError: If modules are in training mode
    """
    assert not conv.training and not bn.training, "Fusion only works in eval mode"

    # Check if this is a transposed convolution
    is_transposed = isinstance(conv, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))

    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        is_transposed=is_transposed,
    )


#: Attribute a custom container sets (``True``) to state that its ``forward`` applies
#: its registered children one after another, in registration order — what
#: ``nn.Sequential`` does by construction. Only such parents let adjacency stand for
#: dataflow; PTv3's ``PointSequential`` declares it.
SEQUENTIAL_MARKER = "applies_children_in_order"

#: Attribute a custom block sets to name the ``(conv_or_linear, batchnorm)`` child pairs
#: its ``forward`` connects directly (``bn(conv(x))``, a reshape between them included),
#: e.g. ``bn_fusion_pairs = (("conv", "norm"),)``; a named BN that is an ``nn.Identity``
#: (the block was built without one, or was folded already) is skipped. Registration
#: order proves nothing about dataflow in an arbitrary ``nn.Module`` — ``bn(relu(conv(x)))``
#: or a residual block can register the two side by side without connecting them — so
#: the block has to say so.
FUSION_PAIRS_MARKER = "bn_fusion_pairs"

#: Conv/Linear types and the BatchNorm type each folds into.
_CONV_TO_BN: dict[type, type] = {
    nn.Conv1d: nn.BatchNorm1d,
    nn.Conv2d: nn.BatchNorm2d,
    nn.ConvTranspose2d: nn.BatchNorm2d,
    nn.Linear: nn.BatchNorm1d,
}


def _applies_children_in_order(module: nn.Module) -> bool:
    return isinstance(module, nn.Sequential) or bool(getattr(module, SEQUENTIAL_MARKER, False))


def _is_fusible_pair(left: nn.Module, right: nn.Module) -> bool:
    """Whether ``left`` is a Conv/Linear whose output channels ``right`` normalizes."""
    for conv_type, bn_type in _CONV_TO_BN.items():
        if isinstance(left, conv_type) and isinstance(right, bn_type):
            out_channels = left.out_features if isinstance(left, nn.Linear) else left.out_channels
            return out_channels == right.num_features
    return False


def find_conv_bn_pairs(model: nn.Module) -> list[tuple[str, str]]:
    """Find the Conv/Linear + BatchNorm pairs the model's structure proves are connected.

    Two sources, both statements about dataflow rather than guesses from it:

    - Adjacent children of a container that applies its children in order
      (``nn.Sequential``, or a class with ``applies_children_in_order = True``).
    - Pairs a custom block names in its ``bn_fusion_pairs`` attribute.

    Adjacent Conv+BN registrations under any other parent are *not* paired — the
    parent's ``forward`` may put an activation between them or never connect them — and
    are logged at WARNING so the block gets a declaration rather than a silent skip.

    Matches Conv1d+BatchNorm1d, Conv2d+BatchNorm2d, ConvTranspose2d+BatchNorm2d and
    Linear+BatchNorm1d, and only when the channel counts agree.

    Returns:
        List of ``(conv_name, bn_name)`` full module paths, in traversal order.
    """
    pairs: list[tuple[str, str]] = []
    undeclared: list[str] = []
    for parent_name, parent in model.named_modules():
        prefix = f"{parent_name}." if parent_name else ""
        declared = getattr(parent, FUSION_PAIRS_MARKER, None)
        if declared is not None:
            for conv_attr, bn_attr in declared:
                conv, bn = parent._modules.get(conv_attr), parent._modules.get(bn_attr)
                if isinstance(bn, nn.Identity):
                    continue  # the block built without a BN (or it is already folded)
                if conv is None or bn is None or not _is_fusible_pair(conv, bn):
                    raise ValueError(
                        f"{type(parent).__name__}.bn_fusion_pairs names ({conv_attr!r}, "
                        f"{bn_attr!r}) under {parent_name or '<root>'!r}, which is not a "
                        "Conv/Linear + matching BatchNorm child pair."
                    )
                pairs.append((f"{prefix}{conv_attr}", f"{prefix}{bn_attr}"))
            continue
        children = [(name, child) for name, child in parent._modules.items() if child is not None]
        for (left_name, left), (right_name, right) in zip(children, children[1:]):
            if not _is_fusible_pair(left, right):
                continue
            if _applies_children_in_order(parent):
                pairs.append((f"{prefix}{left_name}", f"{prefix}{right_name}"))
            else:
                undeclared.append(f"{prefix}{left_name} + {prefix}{right_name}")
    if undeclared:
        logger.warning(
            "BN fusion skipped %d adjacent Conv/Linear + BatchNorm registration(s) whose "
            "parent neither applies its children in order nor declares bn_fusion_pairs "
            "(registration order does not prove dataflow): %s. If the block computes "
            "bn(conv(x)), declare `bn_fusion_pairs = ((conv_attr, bn_attr),)` on it.",
            len(undeclared),
            ", ".join(undeclared),
        )
    return pairs


def _get_parent_module(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """
    Get parent module and attribute name for a nested module.

    Args:
        model: Root model
        name: Dot-separated path to module (e.g., "backbone.layer1.conv1")

    Returns:
        Tuple of (parent_module, attr_name)
    """
    parent_name, _, attr_name = name.rpartition(".")
    # get_submodule resolves numeric names (container children) as well as attributes,
    # and works for containers that do not implement __getitem__.
    return (model.get_submodule(parent_name) if parent_name else model), attr_name


def _replace_bn_with_identity(model: nn.Module, bn_name: str):
    """Replace a BatchNorm module with ``nn.Identity`` by name."""
    parent, attr = _get_parent_module(model, bn_name)
    # add_module covers plain attributes, Sequential-style numeric names, and custom
    # containers with no __setitem__ (PTv3's PointSequential).
    parent.add_module(attr, nn.Identity())


def fuse_model_bn(model: nn.Module) -> nn.Module:
    """
    Fuse all Conv-BN pairs in the model, in place.

    Args:
        model: PyTorch model (modified in place; also returned for chaining).
    Returns:
        Model with fused BatchNorm layers.

    Example:
        >>> model.eval()
        >>> fuse_model_bn(model)
        >>> # Now all fused BN layers are replaced with Identity
    """
    # Must be in eval mode for fusion
    model.eval()

    # Find all Conv-BN pairs
    pairs = find_conv_bn_pairs(model)
    if len(pairs) == 0:
        logger.info("No Conv-BN pairs found to fuse")
        return model

    # Build modules dict for fast lookup
    modules_dict = dict(model.named_modules())

    # Fuse each pair
    for conv_name, bn_name in pairs:
        conv = modules_dict[conv_name]
        bn = modules_dict[bn_name]
        fuse_conv_bn(conv, bn)
        _replace_bn_with_identity(model, bn_name)

    logger.info("Fused %d Conv-BN pairs", len(pairs))
    return model
