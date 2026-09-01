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

"""Quantized forward replacements — per-block ``forward`` rewrites that reposition Q/DQ.

Each ``Quant*Forward`` class reimplements one backbone block's ``forward`` so quantizers land where TensorRT
can fuse them (quantize only the residual/identity branch; single-Q fan-out at a block input).
The block's original ``forward`` is REPLACED (these are not ``register_forward_hook`` hooks).
They are pure ``nn`` reimplementations driven by attributes the :mod:`.attach` functions set on the
module, so they carry no dependency on the quantization engine in
:mod:`autoware_ml.quantization.core`.

Covered architectures:
- VoVNet / V-99-eSE: :class:`QuantOSAModuleForward`, :class:`QuantESEModuleForward`, :class:`QuantBeforePool`
- ResNet / SECOND:   :class:`QuantBasicBlockForward`, :class:`QuantSparseBasicBlockForward`
- ConvNeXt:          :class:`QuantConvNeXtBlockForward`
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class QuantBeforePool(nn.Module):
    """
    Wraps TensorQuantizer + any pool (AvgPool, MaxPool) so QDQ appears before the pool in the graph.
    Used by replacing a pool submodule with this wrapper; ONNX export then sees Quantize -> Dequantize -> Pool.
    """

    def __init__(self, quantizer: nn.Module, pool: nn.Module):
        super().__init__()
        self.quantizer = quantizer
        self.pool = pool

    def forward(self, x):
        return self.pool(self.quantizer(x))


class QuantBasicBlockForward:
    """
    Forward hook for BasicBlock to use residual_quantizer for residual connections.

    This hook replaces the forward method of BasicBlock to quantize only the identity
    branch (residual connection), not the conv path output. This enables TensorRT to
    fuse Conv+Add operations, reducing reformat operations.

    According to TensorRT best practices:
    - Only quantize the residual branch (identity), not the conv path output
    - This allows TensorRT to fuse Conv+Add into a single kernel
    - The conv path output (after norm2) should remain unquantized until after Add
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with quantized residual connection."""
        self = self.obj

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Quantize only the identity branch (residual connection), not the conv path output
        # This enables TensorRT to fuse Conv+Add operations
        if hasattr(self, "residual_quantizer"):
            # Directly call residual_quantizer as a module
            # This is critical for ONNX export to trace the quantizer call
            # The quantizer must be registered as a submodule or accessible as an attribute
            identity = self.residual_quantizer(identity)

        out = out + identity
        out = self.relu(out)
        return out


class QuantSparseBasicBlockForward:
    """
    Forward hook for SparseBasicBlock to use residual_quantizer for residual connections.

    This hook replaces the forward method of SparseBasicBlock to quantize only the identity
    branch (residual connection), not the conv path output. This enables TensorRT to
    fuse Conv+Add operations, reducing reformat operations.

    SparseBasicBlock works with SparseConvTensor which requires replace_feature.
    According to TensorRT best practices:
    - Only quantize the residual branch (identity), not the conv path output
    - This allows TensorRT to fuse Conv+Add into a single kernel
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with quantized residual connection for sparse tensors."""
        self = self.obj

        identity = x
        out = self.conv1(x)

        # Handle ReLU (may be fused in conv1)
        if hasattr(self, "relu") and not getattr(self.conv1, "act_type", None):
            out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Quantize only the identity branch (residual connection), not the conv path output
        # This enables TensorRT to fuse Conv+Add operations
        if hasattr(self, "residual_quantizer"):
            # Directly call residual_quantizer as a module
            # This is critical for ONNX export to trace the quantizer call
            identity = identity.replace_feature(self.residual_quantizer(identity.features))

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out


class QuantConvNeXtBlockForward:
    """
    Forward hook for ConvNeXtBlock to use residual_quantizer on the identity branch.

    Mirrors mmpretrain ConvNeXtBlock.forward, but quantizes the shortcut before add.
    """

    # Norm/GRN types that do NOT accept a ``data_format`` keyword argument.
    # After BN-fusion the norm may become nn.Identity, which also does not
    # accept ``data_format``.
    _PLAIN_NORM_TYPES = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.Identity,
    )

    def __init__(self, obj):
        self.obj = obj

    @staticmethod
    def _safe_call(module, x, data_format):
        """Call *module* with ``data_format`` only when it accepts one."""
        if isinstance(module, QuantConvNeXtBlockForward._PLAIN_NORM_TYPES):
            return module(x)
        return module(x, data_format=data_format)

    def __call__(self, x):
        """Forward pass with quantized residual connection for ConvNeXtBlock."""
        self = self.obj

        def _inner_forward(x):
            identity = x

            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = QuantConvNeXtBlockForward._safe_call(self.norm, x, "channel_last")
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = QuantConvNeXtBlockForward._safe_call(self.grn, x, "channel_last")
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = QuantConvNeXtBlockForward._safe_call(self.norm, x, "channel_first")
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = QuantConvNeXtBlockForward._safe_call(self.grn, x, "channel_first")
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            # Quantize only the identity branch (residual connection).
            if hasattr(self, "residual_quantizer"):
                identity = self.residual_quantizer(identity)

            x = identity + self.drop_path(x)
            return x

        if getattr(self, "with_cp", False) and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)


class QuantOSAModuleForward:
    """
    Forward hook for _OSA_module (VoVNet/V-99-eSE) to use single Q at block input when identity=True.

    When identity=True, the block input is used in three places: first conv, concat branch,
    and Add after eSE. To avoid three FP32 reformats in TRT, use a single Q and fan-out
    to all three. We reuse concat_input_quantizers[0] as that Q (so it receives x and gets
    calibrated). qx = concat_input_quantizers[0](x); use qx for output[0], first layer input,
    and Add. When identity=False, use concat_input_quantizers per branch and no residual Q.
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        """Forward pass with single Q at block input when identity=True (reuse concat_input_quantizers[0])."""
        self = self.obj
        identity_feat = x

        # When identity=True, reuse concat_input_quantizers[0] as the single Q for block input (avoids NaN: it sees x)
        use_single_q = (
            getattr(self, "identity", False)
            and hasattr(self, "concat_input_quantizers")
            and len(self.concat_input_quantizers) == len(self.layers)
        )
        if use_single_q:
            qx = self.concat_input_quantizers[0](x)
            identity_feat = qx
            output = [qx]
            x_in = qx
        else:
            output = [x]
            x_in = x

        if getattr(self, "depthwise", False) and getattr(self, "isReduced", False):
            x_in = self.conv_reduction(x_in)
        for layer in self.layers:
            x_in = layer(x_in)
            output.append(x_in)

        # Q/DQ on branch inputs before Concat. When use_single_q, output[0] is already qx; skip index 0.
        if (
            hasattr(self, "concat_input_quantizers")
            and len(self.concat_input_quantizers) == len(output) - 1
        ):
            start_i = 1 if use_single_q else 0
            for i in range(start_i, len(output) - 1):
                output[i] = self.concat_input_quantizers[i](output[i])

        x = torch.cat(output, dim=1)
        xt = self.concat(x)
        xt = self.ese(xt)

        if self.identity:
            if not use_single_q and hasattr(self, "residual_quantizer"):
                identity_feat = self.residual_quantizer(identity_feat)
            xt = xt + identity_feat

        return xt


class QuantESEModuleForward:
    """
    Forward hook for eSEModule (VoVNet): single Q at the eSE input, fanned out to both paths.

    TRT expects only ONE QuantizeLinear at the eSE input (one FP32→INT8). The same Q output then feeds
    BOTH the global-average-pool (gate) path and the bypass path into the Mul, so ``qx * gate`` has both
    operands in INT8 with a single reformat:

        conv_out FP32 → Reformat → qx (single Q) → { avg_pool→fc→hsigmoid→mul_gate_q ; bypass } → Mul

    ``pool_input_quantizer`` is that single Q, attached by :func:`attach_ese_recipe`; the
    bypass reuses its output (``qx``), so there is no second quantizer on the conv output. This is the
    canonical, reformat-minimizing INT8 eSE placement. (The legacy two-Q ``mul_identity`` variant — a
    second reformat, pool branch left un-quantized — has been removed.) A module without
    ``pool_input_quantizer`` (eSE INT8 recipe not applied) falls back to the plain FP path.
    """

    def __init__(self, obj):
        self.obj = obj

    def __call__(self, x):
        self = self.obj
        pool_input_quantizer = getattr(self, "pool_input_quantizer", None)
        if pool_input_quantizer is None:
            # eSE INT8 recipe not applied here → plain FP path (no quantizers).
            return x * self.hsigmoid(self.fc(self.avg_pool(x)))
        qx = pool_input_quantizer(x)
        gate = self.hsigmoid(self.fc(self.avg_pool(qx)))
        mul_gate_quantizer = getattr(self, "mul_gate_quantizer", None)
        if mul_gate_quantizer is not None:
            gate = mul_gate_quantizer(gate)
        return qx * gate
