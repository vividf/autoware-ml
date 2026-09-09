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

"""Switch exported ``autoware::ImplicitGemm`` nodes into the plugin's INT8 mode.

A sparse convolution is not quantized the way the dense graph is. There are no Q/DQ nodes
around a plugin op — TensorRT would not fuse them into it — so the calibrated scales travel
as *plugin attributes and inputs* instead: the node keeps its op type and gains

    precision = 1
    input_scale = input_amax / 127                      (attribute, per tensor)
    channel_scale = input_scale * weight_amax / 127     (6th input, per output channel)
    bias_scaled = the folded bias                       (7th input, per output channel)

and the plugin quantizes features and weights itself on the way into a cumm INT8
``implicit_gemm``, accumulating in INT32 and writing FP16 out. That contract is
``ImplicitGemmPlugin::enqueueInt8`` in ``autoware_tensorrt_plugins``.

``output_scale`` stays 1.0 on purpose. The plugin folds it into the GEMM's scale and bias
(``channel_scale * output_scale``, ``bias_scaled * output_scale``) *and* divides it back out
when it recovers the weight scale (``w_scale = channel_scale * output_scale / input_scale``),
so it cancels exactly: the epilogue is ``s8s8f16`` and never re-quantizes the result, which
is what makes an activation-chain output scale — AWML's successor-stem walk and its terminal
``_last_int8_conv_output_absmax`` buffer — unnecessary here. ``test_onnx_int8.py`` pins that
cancellation.

Which layers become INT8 is decided upstream, by whether the PyTorch module carries a
calibrated quantizer: a layer left out through ``quantization.skip_quantize`` never gets one,
so it stays an FP16 ``ImplicitGemm`` node here — byte-identical to the FP16 deployment. That
single control point is what keeps calibration honest: an excluded layer is excluded during
PTQ too, so its successor's ``input_scale`` is calibrated against the genuine FP
distribution the engine will feed it.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
from torch import nn

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.ops.spconv.onnx_fusion import AUTOWARE_DOMAIN, IMPLICIT_GEMM_OP

if IS_SPCONV_AVAILABLE:
    # Guarded at the top: spconv is the framework's one optional dependency.
    from spconv.pytorch.conv import SparseConvolution

logger = logging.getLogger(__name__)

#: ``ImplicitGemmParameters::precision`` — floating point (FP16/FP32) vs INT8.
PRECISION_FP = 0
PRECISION_INT8 = 1
#: Signed 8-bit symmetric quantization: amax maps to 127.
_INT8_MAX = 127.0
#: The five sparse tensors every ImplicitGemm node carries; a sixth is the folded bias.
_SPARSE_INPUTS = 5


@dataclass(frozen=True)
class SparseLayerScales:
    """Calibrated INT8 scales of one sparse convolution.

    Attributes:
        input_scale: Activation scale, ``input_amax / 127`` (per tensor).
        weight_scale: Weight scale per output channel, ``weight_amax / 127``.
    """

    input_scale: float
    weight_scale: np.ndarray

    @property
    def channel_scale(self) -> np.ndarray:
        """The plugin's ``channel_scale`` input (``output_scale`` is 1, see module doc)."""
        return (self.input_scale * self.weight_scale).astype(np.float32)


def _amax_array(quantizer: nn.Module, what: str, stem: str) -> np.ndarray:
    amax = getattr(quantizer, "amax", None)
    if amax is None:
        raise ValueError(
            f"Sparse INT8: {stem!r} has an un-calibrated {what} quantizer (amax is None). "
            "The checkpoint was saved before calibration ran, or calibration saw no data "
            "for this layer."
        )
    values = amax.detach().float().cpu().numpy().reshape(-1)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            f"Sparse INT8: {stem!r} has a degenerate {what} amax {values[:4]} — a zero or "
            "non-finite amax makes the layer quantize everything to zero. Re-run PTQ."
        )
    return values


def collect_sparse_layer_scales(encoder: nn.Module) -> dict[str, SparseLayerScales]:
    """Read the calibrated INT8 scales off a quantized sparse encoder.

    Args:
        encoder: The *training-form* sparse encoder (``pts_middle_encoder``), whose
            quantized convolutions carry ``input_quantizer`` / ``weight_quantizer``. The
            export copy does not — it is rebuilt from weights alone, which is what keeps
            the exported sparse graph free of Q/DQ nodes.

    Returns:
        Module name (relative to ``encoder``) -> scales, for every quantized convolution.
        Empty when nothing is quantized, which is the FP16 deployment.

    Raises:
        ValueError: On a quantized layer whose calibration is missing or degenerate.
    """
    if not IS_SPCONV_AVAILABLE:
        return {}
    scales: dict[str, SparseLayerScales] = {}
    for name, module in encoder.named_modules():
        if not isinstance(module, SparseConvolution):
            continue
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if input_quantizer is None or weight_quantizer is None:
            continue  # not quantized: skip_quantize, or an FP16 deployment
        input_amax = _amax_array(input_quantizer, "input", name)
        if input_amax.size != 1:
            raise ValueError(
                f"Sparse INT8: {name!r} has a per-channel input amax of {input_amax.size} "
                "values, but the plugin quantizes features with one scale. The input "
                "quantizer must be per-tensor."
            )
        weight_amax = _amax_array(weight_quantizer, "weight", name)
        if weight_amax.size != module.out_channels:
            raise ValueError(
                f"Sparse INT8: {name!r} has {weight_amax.size} weight amax value(s) but "
                f"{module.out_channels} output channels. The weight quantizer must be "
                "per-output-channel (axis 0 of [C_out, k1, k2, k3, C_in])."
            )
        scales[name] = SparseLayerScales(
            input_scale=float(input_amax[0]) / _INT8_MAX,
            weight_scale=weight_amax / _INT8_MAX,
        )
    return scales


def _stem_of(initializer_name: str, stems: frozenset[str]) -> str | None:
    """Resolve a filter initializer name (``<prefix>.<stem>.weight``) to its module stem."""
    if not initializer_name.endswith(".weight"):
        return None
    body = initializer_name[: -len(".weight")]
    matched = [stem for stem in stems if body == stem or body.endswith(f".{stem}")]
    if not matched:
        return None
    if len(matched) > 1:
        raise ValueError(
            f"Sparse INT8: filter initializer {initializer_name!r} matches several quantized "
            f"layers {sorted(matched)}. Module names must resolve a node unambiguously."
        )
    return matched[0]


def _set_attribute(node: onnx.NodeProto, name: str, value: int | float) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)
    node.attribute.append(helper.make_attribute(name, value))


def quantize_implicit_gemm_nodes(
    model: onnx.ModelProto, scales: dict[str, SparseLayerScales]
) -> tuple[int, int]:
    """Switch every calibrated ``ImplicitGemm`` node in ``model`` into INT8 mode, in place.

    Args:
        model: Exported sparse graph, already bias/activation-fused.
        scales: Output of :func:`collect_sparse_layer_scales`.

    Returns:
        ``(converted, kept_fp)`` node counts.

    Raises:
        ValueError: When a node's scales do not match its filter, or a calibrated layer has
            no node in the graph (a silently un-quantized tower otherwise).
    """
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    stems = frozenset(scales)
    converted: set[str] = set()
    kept_fp = 0

    for node in graph.node:
        if node.op_type != IMPLICIT_GEMM_OP or node.domain != AUTOWARE_DOMAIN:
            continue
        if len(node.input) < _SPARSE_INPUTS:
            raise ValueError(
                f"Sparse INT8: {node.name!r} has {len(node.input)} inputs; an ImplicitGemm "
                "node carries at least the five sparse tensors."
            )
        stem = _stem_of(node.input[1], stems)
        if stem is None:
            kept_fp += 1
            continue

        layer = scales[stem]
        channel_scale = layer.channel_scale
        filters = initializers.get(node.input[1])
        if filters is None:
            raise ValueError(
                f"Sparse INT8: {node.name!r} takes its filters from {node.input[1]!r}, which "
                "is not an initializer — the weight must be constant to be quantized."
            )
        out_channels = filters.dims[0]
        if channel_scale.size != out_channels:
            raise ValueError(
                f"Sparse INT8: {stem!r} has {channel_scale.size} channel scales but its "
                f"filter has {out_channels} output channels."
            )

        # The bias was folded into the node's sixth input by the bias/activation fusion; the
        # plugin adds it after the scaled accumulation, so it becomes bias_scaled unchanged
        # (output_scale is 1). A node without a folded bias contributes zeros.
        if len(node.input) > _SPARSE_INPUTS:
            bias_initializer = initializers.get(node.input[_SPARSE_INPUTS])
            if bias_initializer is None:
                raise ValueError(
                    f"Sparse INT8: {node.name!r} has a fused bias input "
                    f"{node.input[_SPARSE_INPUTS]!r} that is not an initializer."
                )
            bias = numpy_helper.to_array(bias_initializer).astype(np.float32).reshape(-1)
            if bias.size != out_channels:
                raise ValueError(
                    f"Sparse INT8: {node.name!r} fused bias has {bias.size} values for "
                    f"{out_channels} output channels."
                )
        else:
            bias = np.zeros(out_channels, dtype=np.float32)

        # New FP32 initializers: the plugin reads both as float*, and they are created after
        # the precision pass has run, so the FP16 cast leaves them alone by construction.
        channel_scale_name = f"{stem.replace('.', '_')}_channel_scale"
        bias_scaled_name = f"{stem.replace('.', '_')}_bias_scaled"
        graph.initializer.append(
            numpy_helper.from_array(channel_scale.astype(np.float32), name=channel_scale_name)
        )
        graph.initializer.append(numpy_helper.from_array(bias, name=bias_scaled_name))

        del node.input[_SPARSE_INPUTS:]
        node.input.extend([channel_scale_name, bias_scaled_name])
        _set_attribute(node, "precision", PRECISION_INT8)
        _set_attribute(node, "input_scale", float(layer.input_scale))
        _set_attribute(node, "output_scale", 1.0)
        converted.add(stem)
        logger.debug(
            "Sparse INT8: %s -> precision=1 input_scale=%.6f channel_scale[%d]",
            stem,
            layer.input_scale,
            channel_scale.size,
        )

    unmatched = sorted(stems - converted)
    if unmatched:
        raise ValueError(
            f"Sparse INT8: {len(unmatched)} calibrated layer(s) have no ImplicitGemm node in "
            f"the exported graph: {unmatched[:5]}. The tower would deploy in FP16 while the "
            "checkpoint claims INT8."
        )
    return len(converted), kept_fp


def sparse_int8_transform(onnx_path: str | Path, *, encoder: nn.Module) -> Path:
    """Stage transform: write the encoder's calibrated INT8 scales into its exported graph.

    A no-op when the encoder carries no quantizers, so the same stage declaration serves the
    FP16 and INT8 deployments.

    Args:
        onnx_path: The exported sparse graph, rewritten in place.
        encoder: The training-form sparse encoder holding the calibrated quantizers.

    Returns:
        The same path.
    """
    onnx_path = Path(onnx_path)
    scales = collect_sparse_layer_scales(encoder)
    if not scales:
        return onnx_path
    model = onnx.load(str(onnx_path))
    converted, kept_fp = quantize_implicit_gemm_nodes(model, scales)
    onnx.save(model, str(onnx_path))
    logger.info(
        "Sparse INT8 in %s: %d ImplicitGemm node(s) switched to the plugin's INT8 mode, "
        "%d kept FP16.",
        onnx_path.name,
        converted,
        kept_fp,
    )
    return onnx_path
