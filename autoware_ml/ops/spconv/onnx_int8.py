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

"""Write the calibrated INT8 scales of the exported ``autoware::ImplicitGemm`` nodes.

A calibrated node takes the quantized row of the plugin contract that spconv's
``autoware_tensorrt_plugins`` and t4sparse share:

    input 0      INT8, from a QuantizeLinear inserted in front of the node (scale input_scale)
    input 1      INT8 filters, round(w / weight_scale[n]) clipped to +-127
    input 5      weight_scale = weight_amax / 127, FP32 [C_out]
    input 6      the folded bias, FP32 [C_out], or [0] without one
    input_scale  input_amax / 127, the QuantizeLinear's scale
    output_type  "f16"; no output_scale

The plugin accumulates in INT32 and writes ``act(input_scale * weight_scale[n] * acc +
bias[n])`` in FP16: the dequantization is its epilogue, so no DequantizeLinear follows the
node and no output scale is calibrated. TensorRT runs the FP16 -> INT8 conversion and can fuse
it into the producer.

A node becomes INT8 only when its PyTorch module carries a calibrated quantizer. A layer in
``quantization.skip_quantize`` has none and stays an FP16 node; it is skipped during PTQ as
well, so the next layer's ``input_scale`` is calibrated on the FP activations the engine
feeds it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from torch import nn

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.ops.spconv.contract import (
    AUTOWARE_DOMAIN,
    IMPLICIT_GEMM_BASE_INPUTS,
    IMPLICIT_GEMM_BIAS_SLOT,
    IMPLICIT_GEMM_INPUTS,
    IMPLICIT_GEMM_OP,
    IMPLICIT_GEMM_WEIGHT_SCALE_SLOT,
    INPUT_SCALE_ATTR,
    OUTPUT_SCALE_ATTR,
    OUTPUT_TYPE_ATTR,
    OUTPUT_TYPE_FP16,
)
from autoware_ml.ops.spconv.onnx_fusion import replace_attribute, weight_scale_name

if IS_SPCONV_AVAILABLE:
    from spconv.pytorch.conv import SparseConvolution

logger = logging.getLogger(__name__)

_INT8_MAX = 127.0


@dataclass(frozen=True)
class SparseLayerScales:
    """Calibrated INT8 scales of one sparse convolution.

    Attributes:
        input_scale: Activation scale, ``input_amax / 127`` (per tensor).
        weight_scale: Weight scale per output channel, ``weight_amax / 127``.
    """

    input_scale: float
    weight_scale: np.ndarray


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


def collect_sparse_layer_scales(module: nn.Module) -> dict[str, SparseLayerScales]:
    """Read the calibrated INT8 scales of the quantized sparse convolutions in ``module``.

    Args:
        module: The training-form module (BEVFusion's ``pts_middle_encoder``, PTv3's
            encoder or decoder head). Its export copy is rebuilt from the weights alone and
            has no quantizers, which keeps the exported graph free of Q/DQ nodes.

    Returns:
        Scales keyed by module name relative to ``module``; empty when nothing is quantized.

    Raises:
        ValueError: On a quantized layer whose calibration is missing or degenerate.
    """
    if not IS_SPCONV_AVAILABLE:
        return {}
    scales: dict[str, SparseLayerScales] = {}
    for name, submodule in module.named_modules():
        if not isinstance(submodule, SparseConvolution):
            continue
        input_quantizer = getattr(submodule, "input_quantizer", None)
        weight_quantizer = getattr(submodule, "weight_quantizer", None)
        if input_quantizer is None or weight_quantizer is None:
            continue
        input_amax = _amax_array(input_quantizer, "input", name)
        if input_amax.size != 1:
            raise ValueError(
                f"Sparse INT8: {name!r} has a per-channel input amax of {input_amax.size} "
                "values, but the plugin quantizes features with one scale. The input "
                "quantizer must be per-tensor."
            )
        weight_amax = _amax_array(weight_quantizer, "weight", name)
        if weight_amax.size != submodule.out_channels:
            raise ValueError(
                f"Sparse INT8: {name!r} has {weight_amax.size} weight amax value(s) but "
                f"{submodule.out_channels} output channels. The weight quantizer must be "
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


def _set_initializer(graph: onnx.GraphProto, name: str, values: np.ndarray) -> None:
    """Write ``values`` as initializer ``name``, replacing one of that name."""
    tensor = numpy_helper.from_array(values, name)
    existing = next((init for init in graph.initializer if init.name == name), None)
    if existing is None:
        graph.initializer.append(tensor)
    else:
        existing.CopyFrom(tensor)


def _drop_attribute(node: onnx.NodeProto, name: str) -> None:
    kept = [attribute for attribute in node.attribute if attribute.name != name]
    del node.attribute[:]
    node.attribute.extend(kept)


def _int8_filter_codes(filters: onnx.TensorProto, weight_scale: np.ndarray) -> np.ndarray:
    """``round(w / weight_scale[n])`` clipped to +-127; axis 0 of the filter is ``C_out``."""
    weights = numpy_helper.to_array(filters).astype(np.float32)
    per_channel = weight_scale.astype(np.float32).reshape((-1,) + (1,) * (weights.ndim - 1))
    return np.clip(np.rint(weights / per_channel), -_INT8_MAX, _INT8_MAX).astype(np.int8)


def _folded_bias(
    node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto], out_channels: int
) -> str | None:
    """Name of the node's slot-6 bias, or ``None`` when it has no slot 6."""
    if len(node.input) == IMPLICIT_GEMM_BASE_INPUTS:
        return None
    weight_scale = initializers.get(node.input[IMPLICIT_GEMM_WEIGHT_SCALE_SLOT])
    if weight_scale is None or list(weight_scale.dims) != [0]:
        raise ValueError(
            f"Sparse INT8: {node.name!r} already carries a weight scale in input "
            f"{IMPLICIT_GEMM_WEIGHT_SCALE_SLOT}; an exported node has an empty one."
        )
    bias = initializers.get(node.input[IMPLICIT_GEMM_BIAS_SLOT])
    if bias is None or bias.data_type != TensorProto.FLOAT:
        raise ValueError(
            f"Sparse INT8: {node.name!r} input {IMPLICIT_GEMM_BIAS_SLOT} is not an FP32 bias "
            "initializer."
        )
    if list(bias.dims) not in ([out_channels], [0]):
        raise ValueError(
            f"Sparse INT8: {node.name!r} bias has shape {list(bias.dims)} for {out_channels} "
            "output channels."
        )
    return bias.name


def quantize_implicit_gemm_nodes(
    model: onnx.ModelProto, scales: dict[str, SparseLayerScales]
) -> tuple[int, int]:
    """Rewrite every calibrated ``ImplicitGemm`` node into the contract's INT8 row, in place.

    Run it after the bias/activation fusion, if the stage has one: a folded bias stays in
    input 6. Each converted node gets its own QuantizeLinear, placed right before it.

    Returns:
        ``(converted, kept_fp)`` node counts.

    Raises:
        ValueError: When a node is not in a contract form, its scales do not match its
            filter, two nodes share a quantized filter, or a calibrated layer has no node in
            the graph.
    """
    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    stems = frozenset(scales)
    converted: set[str] = set()
    kept_fp = 0
    nodes: list[onnx.NodeProto] = []

    for node in graph.node:
        if node.op_type != IMPLICIT_GEMM_OP or node.domain != AUTOWARE_DOMAIN:
            nodes.append(node)
            continue
        if len(node.input) not in (IMPLICIT_GEMM_BASE_INPUTS, IMPLICIT_GEMM_INPUTS):
            raise ValueError(
                f"Sparse INT8: {node.name!r} has {len(node.input)} inputs; the plugin contract "
                f"has {IMPLICIT_GEMM_BASE_INPUTS} or {IMPLICIT_GEMM_INPUTS}."
            )
        stem = _stem_of(node.input[1], stems)
        if stem is None:
            kept_fp += 1
            nodes.append(node)
            continue
        if stem in converted:
            raise ValueError(
                f"Sparse INT8: {node.name!r} shares the filter of {stem!r} with another node; "
                "a quantized filter must belong to one node."
            )

        layer = scales[stem]
        filters = initializers.get(node.input[1])
        if filters is None:
            raise ValueError(
                f"Sparse INT8: {node.name!r} takes its filters from {node.input[1]!r}, which "
                "is not an initializer — the weight must be constant to be quantized."
            )
        out_channels = filters.dims[0]
        if layer.weight_scale.size != out_channels:
            raise ValueError(
                f"Sparse INT8: {stem!r} has {layer.weight_scale.size} weight scales but its "
                f"filter has {out_channels} output channels."
            )
        bias = _folded_bias(node, initializers, out_channels)

        filters.CopyFrom(
            numpy_helper.from_array(_int8_filter_codes(filters, layer.weight_scale), filters.name)
        )
        weight_scale = weight_scale_name(node)
        _set_initializer(graph, weight_scale, layer.weight_scale.astype(np.float32))
        if bias is None:
            bias = f"{node.input[1].removesuffix('.weight')}.no_bias"
            _set_initializer(graph, bias, np.zeros(0, np.float32))
        del node.input[IMPLICIT_GEMM_BASE_INPUTS:]
        node.input.extend([weight_scale, bias])

        # The fp16 cast stores the Quantize scale in fp16, and the plugin has to dequantize
        # with the value TensorRT quantized with, so both carry the fp16-rounded scale.
        input_scale = float(np.float16(layer.input_scale))
        prefix = node.input[1].removesuffix(".weight")
        scale, zero_point, quantized = (
            f"{prefix}.input_scale",
            f"{prefix}.input_zero_point",
            f"{prefix}.input_int8",
        )
        _set_initializer(graph, scale, np.array(input_scale, np.float32))
        _set_initializer(graph, zero_point, np.array(0, np.int8))
        nodes.append(
            helper.make_node(
                "QuantizeLinear",
                [node.input[0], scale, zero_point],
                [quantized],
                name=f"{node.name}/QuantizeLinear",
            )
        )
        node.input[0] = quantized

        replace_attribute(node, INPUT_SCALE_ATTR, input_scale)
        replace_attribute(node, OUTPUT_TYPE_ATTR, OUTPUT_TYPE_FP16)
        # The exporter writes output_scale = 1.0; the INT8 row must not carry one.
        _drop_attribute(node, OUTPUT_SCALE_ATTR)
        nodes.append(node)
        converted.add(stem)
        logger.debug(
            "Sparse INT8: %s -> input_scale=%.6g weight_scale[%d]",
            stem,
            input_scale,
            out_channels,
        )

    unmatched = sorted(stems - converted)
    if unmatched:
        raise ValueError(
            f"Sparse INT8: {len(unmatched)} calibrated layer(s) have no ImplicitGemm node in "
            f"the exported graph: {unmatched[:5]}. The tower would deploy in FP16 while the "
            "checkpoint claims INT8."
        )
    del graph.node[:]
    graph.node.extend(nodes)
    return len(converted), kept_fp


def sparse_int8_transform(onnx_path: str | Path, *, module: nn.Module) -> Path:
    """Stage transform: write the module's calibrated INT8 scales into its exported graph.

    ``module`` is the training-form module, not its export copy, and ``onnx_path`` is
    rewritten in place. A no-op when the module carries no quantizers, so the same stage
    declaration serves the FP16 and INT8 deployments.
    """
    onnx_path = Path(onnx_path)
    scales = collect_sparse_layer_scales(module)
    if not scales:
        return onnx_path
    model = onnx.load(str(onnx_path))
    converted, kept_fp = quantize_implicit_gemm_nodes(model, scales)
    onnx.save(model, str(onnx_path))
    logger.info(
        "Sparse INT8 in %s: %d ImplicitGemm node(s) quantized, %d kept FP16.",
        onnx_path.name,
        converted,
        kept_fp,
    )
    return onnx_path
