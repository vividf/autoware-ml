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

"""SparseConv + BatchNorm folding for SECOND-style spconv encoders.

.. todo:: TODO(vividf): no callers yet — wire this into the BEVFusion migration as the
   single fold used by the quantize stage, the deploy loader, and the FP16 sparse export
   path, so the module tree (and state_dict keys) cannot drift between them.

The fold itself is **not** quantization — it rewrites the module tree so the exported
sparse ONNX is BN-free and matches the runtime graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fuse_spconv_bn_in_encoder(sparse_encoder: nn.Module) -> int:
    """Fuse each ``SparseConvolution + BatchNorm1d`` pair in ``sparse_encoder`` (eval mode).

    Single source of truth for SECOND-style spconv BN folding: used by the quantize stage, the
    deploy loader, and the FP16 sparse export path, so the module tree — and therefore the
    ``state_dict`` keys — line up on every side. Returns the number of fused Conv-BN pairs.

    Raises:
        RuntimeError: If spconv's quantization utils are not importable — silently skipping
            the fold would change the ``state_dict`` keys and surface later as a misleading
            checkpoint-key mismatch.
    """
    try:
        from spconv.pytorch.quantization.utils import fuse_spconv_bn_eval
    except ImportError as error:
        raise RuntimeError(
            "spconv quantization utils are required to fold SparseConv+BN (the fused module "
            "tree and its state_dict keys depend on it; skipping would surface later as a "
            "misleading state_dict mismatch). Install spconv with quantization support, "
            "e.g. `pip install spconv-cu120` matching your CUDA version."
        ) from error

    from spconv.pytorch.conv import SparseConvolution

    sparse_encoder.eval()
    fused_count = 0

    for module in sparse_encoder.modules():
        children = list(module._modules.items())
        for i in range(len(children) - 1):
            left_name, left_mod = children[i]
            right_name, right_mod = children[i + 1]
            if isinstance(left_mod, SparseConvolution) and isinstance(
                right_mod, torch.nn.BatchNorm1d
            ):
                fused_conv = fuse_spconv_bn_eval(left_mod, right_mod)
                setattr(module, left_name, fused_conv)
                setattr(module, right_name, torch.nn.Identity())
                fused_count += 1

    return fused_count
