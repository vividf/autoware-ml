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

"""Quantized block classes — the ConvertBlock / WrapModule transforms' targets.

A *quantized block* is a modelopt ``QuantModule`` whose ``forward`` re-implements one
architecture block with Q/DQ where TensorRT can fuse it. Residual blocks quantize only the
identity branch before the residual ``Add`` (the conv-path output stays un-quantized until
after the Add, so TensorRT fuses Conv+Add into one INT8 kernel and skips a reformat); the
VoVNet blocks place a single Q at a fan-out point. The block is converted **in place** through
:data:`QuantBlockRegistry` — the instance's class is patched, exactly like modelopt converts
``nn.Conv2d`` — so ``isinstance(block, OriginalCls)`` stays true, the state_dict is unchanged,
and no ``forward`` monkeypatching is involved.

Quantizers are NOT created here: the recipe (:mod:`.attach`) decides whether a block gets a
fresh quantizer (a submodule, so its ``amax`` lands in the state_dict) or shares a conv's
``input_quantizer`` (a plain attribute — a ``TensorQuantizer`` cannot be the child of two
parents), which is why every ``_setup`` is empty. The forwards read those quantizers as
attributes and fall back to the plain FP path when absent.

Blocks covered (each ``forward`` mirrors the original block's forward with the quantizer
calls added — keep them in sync with the upstream class they shadow):

- :class:`QuantSparseBasicBlock` — the repo's spconv residual block (``encoders/sparse.py``).
- :class:`QuantOSAModule` / :class:`QuantESEModule` — VoVNet ``_OSA_module`` / ``eSEModule``
  (CenterPoint VoV99 backbone; StreamPETR's VoVNet-CP uses the same blocks).

The ConvNeXt residual block (CenterPoint ConvNeXt_PC) is parked on branch
``feat/quantization-convnext-recipe`` (``recipes/quant_convnext.py``) until that backbone
migrates.

Supporting a new block = one ``QuantModule`` subclass here plus one spec row
(:class:`~.attach.ResidualBlockSpec` / :class:`~.attach.ESEBlockSpec`) declared by the model
that owns the block class (``QuantRules.residual_blocks`` / ``ese_blocks``), or in
:func:`~.attach.default_block_specs` for blocks the repo itself defines.
"""

from __future__ import annotations

import torch
from modelopt.torch.opt.dynamic import _DMRegistryCls
from modelopt.torch.quantization.nn.modules.quant_module import QuantModule
from torch import nn

#: Registry of residual-block conversions (``OriginalBlock -> QuantOriginalBlock``). Kept
#: separate from modelopt's ``QuantModuleRegistry`` (leaf Conv/Linear) so the two
#: vocabularies — "quantize this GEMM" vs "place Q/DQ inside this block" — never mix.
QuantBlockRegistry = _DMRegistryCls("Quant", QuantModule)


class QuantBeforePool(nn.Module):
    """Wraps a TensorQuantizer + any pool so Q/DQ appears before the pool in the graph.

    Used by replacing a pool submodule with this wrapper (the WrapModule transform); ONNX
    export then sees Quantize -> Dequantize -> Pool.
    """

    def __init__(self, quantizer: nn.Module, pool: nn.Module):
        super().__init__()
        self.quantizer = quantizer
        self.pool = pool

    def forward(self, x):
        return self.pool(self.quantizer(x))


class QuantSparseBasicBlock(QuantModule):
    """``SparseBasicBlock`` (spconv residual block) with the identity branch quantized.

    Mirrors :class:`autoware_ml.models.detection3d.encoders.sparse.SparseBasicBlock.forward`;
    sparse tensors carry features separately, so the quantizer runs on ``.features`` and the
    result goes back through ``replace_feature``.
    """

    def _setup(self) -> None:
        """No quantizer of its own — see the module docstring."""

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        residual_quantizer = getattr(self, "residual_quantizer", None)
        if residual_quantizer is not None:
            identity = residual_quantizer(identity)
        return out.replace_feature(self.relu(out.features + identity))


class QuantOSAModule(QuantModule):
    """VoVNet ``_OSA_module`` with Q/DQ on every Concat input and a single Q at the block input.

    The recipe attaches ``concat_input_quantizers`` (one per Concat input that is a skip
    connection: the block input and every layer output but the last, i.e. ``len(layers)``
    quantizers). With ``identity=True`` the block input feeds three consumers (first layer,
    Concat, the residual Add after eSE); quantizing it once through
    ``concat_input_quantizers[0]`` and fanning ``qx`` out avoids three FP32 reformats in
    TensorRT, so no separate residual quantizer exists. With ``identity=False`` each Concat
    input is quantized on its own and there is no residual Add.
    """

    def _setup(self) -> None:
        """No quantizer of its own — see the module docstring."""

    def forward(self, x):
        quantizers = getattr(self, "concat_input_quantizers", None)
        single_q = bool(self.identity) and quantizers is not None
        if single_q:
            x = quantizers[0](x)
        identity_feat = x
        output = [x]
        x_in = x
        if getattr(self, "depthwise", False) and getattr(self, "isReduced", False):
            x_in = self.conv_reduction(x_in)
        for layer in self.layers:
            x_in = layer(x_in)
            output.append(x_in)
        if quantizers is not None:
            # Concat inputs 0..n-2 are skip connections; the main path (last layer output)
            # stays un-quantized like a ResNet Add. Index 0 is already qx under single_q.
            for i in range(1 if single_q else 0, len(output) - 1):
                output[i] = quantizers[i](output[i])
        xt = self.ese(self.concat(torch.cat(output, dim=1)))
        if self.identity:
            xt = xt + identity_feat
        return xt


class QuantESEModule(QuantModule):
    """VoVNet ``eSEModule`` with ONE Q at its input, fanned out to both ``Mul`` operands.

    TensorRT wants a single QuantizeLinear at the eSE input (one FP32 -> INT8 reformat). The
    same ``qx`` feeds both the gate path (``avg_pool -> fc -> hsigmoid``) and the bypass into
    the ``Mul``; ``mul_gate_quantizer`` quantizes the gate so both ``Mul`` operands are INT8::

        conv_out -> qx = pool_input_quantizer(x) -> { avg_pool -> fc -> hsigmoid -> mul_gate_q ; bypass } -> Mul

    The recipe attaches ``pool_input_quantizer`` and ``mul_gate_quantizer`` as submodules.
    (The legacy two-Q ``mul_identity`` variant — a second reformat with the pool branch left
    un-quantized — was retired in AWML.)
    """

    def _setup(self) -> None:
        """No quantizer of its own — see the module docstring."""

    def forward(self, x):
        pool_input_quantizer = getattr(self, "pool_input_quantizer", None)
        if pool_input_quantizer is None:
            return x * self.hsigmoid(self.fc(self.avg_pool(x)))
        qx = pool_input_quantizer(x)
        gate = self.hsigmoid(self.fc(self.avg_pool(qx)))
        mul_gate_quantizer = getattr(self, "mul_gate_quantizer", None)
        if mul_gate_quantizer is not None:
            gate = mul_gate_quantizer(gate)
        return qx * gate
