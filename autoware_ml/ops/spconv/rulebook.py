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

"""Precomputed rulebooks for the down-sampling sparse convolutions of a deployed graph.

A sparse convolution runs in two steps: ``GetIndicePairsImplicitGemm`` builds the *rulebook*
(which input voxel feeds which output voxel, per kernel offset), then ``ImplicitGemm`` does
the arithmetic. For a submanifold layer the output voxels are the input voxels, so the
rulebook's shape is known before it runs. For a down-sampling layer the output voxel set is
new, and its size is only known once the kernel has run — a data-dependent shape. TensorRT
handles that by copying the size back to the host and waiting for it before it can schedule
what follows: the ``[trainStation]`` regions in an engine's layer list, measured at 0.83 ms of
pure stall per frame on BEVFusion's four down-sampling layers (23% of its sparse engine).

The rulebook chain is a pure function of the voxel coordinates: nothing in it depends on
features, weights or precision. So it is computed here, outside the graph, in the Torch stage
that already produces the graph's ``coors`` input, and handed to the graph as inputs. The
down-sampling layers find it under their indice key and reuse it
(:meth:`~autoware_ml.ops.spconv.sparse_conv.SparseConvolution._resolve_implicit_gemm_plan`),
so the exported graph carries no ``GetIndicePairsImplicitGemm`` node for them and no
data-dependent shape. The submanifold layers keep generating their own rulebooks in-graph:
they have no dynamic shape to remove, and moving them would only relocate the work.

The graph-input contract (``rulebook/<indice_key>/<slot>`` names, the ``rulebook_stages``
metadata) is what ``autoware_bevfusion``'s runtime binds against; keep the two in step.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import logging
from pathlib import Path

import onnx
from spconv.core import ConvAlgo
from spconv.pytorch.conv import SparseConvolution as SparseConvolutionBase
from spconv.pytorch.core import ImplicitGemmIndiceData
from spconv.tools import CUDAKernelTimer
import torch

from autoware_ml.ops.spconv.sparse_conv import resolve_output_spatial_shape
from autoware_ml.ops.spconv.sparse_functional import get_indice_pairs_implicit_gemm
from autoware_ml.utils.onnx_meta import meta_value_to_str

logger = logging.getLogger(__name__)

#: The four tensors of one rulebook, in the order ``GetIndicePairsImplicitGemm`` emits them.
RULEBOOK_SLOTS = ("out_indices", "pair_fwd", "pair_mask", "mask_argsort")
#: Graph-input namespace; Netron groups the inputs of one stage under it.
INPUT_NAMESPACE = "rulebook"
#: ONNX ``metadata_props`` key carrying the stage geometry the runtime recomputes from.
STAGES_METADATA_KEY = "rulebook_stages"
#: ONNX ``metadata_props`` key carrying the ``coors`` -> convolution-coordinate column order.
COORS_PERMUTATION_METADATA_KEY = "rulebook_coors_permutation"


@dataclass(frozen=True)
class DownsampleStage:
    """One down-sampling sparse convolution whose rulebook is precomputed.

    Attributes:
        indice_key: The layer's spconv indice key — the name the layer looks its rulebook
            up under, and the stage's identity in the graph-input names.
        algo: Convolution algorithm the layer runs; a rulebook is only reusable by a layer
            with the same one.
        kernel_size: Kernel size per spatial axis.
        stride: Stride per spatial axis.
        padding: Padding per spatial axis.
        dilation: Dilation per spatial axis.
        output_padding: Transposed-convolution output padding (zeros for a forward layer).
        spatial_shape: Sparse spatial shape the layer's *input* lives in.
        out_spatial_shape: Sparse spatial shape the layer produces.
    """

    indice_key: str
    algo: ConvAlgo
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    output_padding: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    out_spatial_shape: tuple[int, ...]

    @property
    def kernel_volume(self) -> int:
        """Number of kernel offsets — the leading dimension of ``pair_fwd``."""
        volume = 1
        for size in self.kernel_size:
            volume *= size
        return volume

    @property
    def onnx_base(self) -> str:
        """Graph-input name prefix of this stage's four tensors."""
        return f"{INPUT_NAMESPACE}/{self.indice_key}"

    def input_name(self, slot: str) -> str:
        """Graph-input name of one rulebook tensor (``slot`` from :data:`RULEBOOK_SLOTS`)."""
        if slot not in RULEBOOK_SLOTS:
            raise ValueError(f"Unknown rulebook slot {slot!r}; expected one of {RULEBOOK_SLOTS}.")
        return f"{self.onnx_base}/{slot}"

    @property
    def input_names(self) -> tuple[str, ...]:
        """This stage's four graph-input names, in :data:`RULEBOOK_SLOTS` order."""
        return tuple(self.input_name(slot) for slot in RULEBOOK_SLOTS)

    def metadata(self) -> dict[str, object]:
        """The stage as the runtime reads it from ``rulebook_stages`` (spconv's key names)."""
        return {
            "onnx_base": self.onnx_base,
            "ksize": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "spatial_shape": list(self.spatial_shape),
        }


def downsample_stages(
    layers: Iterable[SparseConvolutionBase], sparse_shape: Iterable[int]
) -> tuple[DownsampleStage, ...]:
    """Describe the down-sampling layers of a sparse encoder, with their input shapes.

    Args:
        layers: The encoder's sparse convolutions **in forward order** — the spatial shape
            cascades through them, so the order is part of the answer.
        sparse_shape: Spatial shape of the encoder's input sparse tensor.

    Returns:
        One stage per non-submanifold layer, in forward order.

    Raises:
        ValueError: On a layer without an indice key (nothing to look its rulebook up by), a
            transposed layer (not a down-sampling), or two down-sampling layers sharing a key.
    """
    stages: list[DownsampleStage] = []
    spatial_shape = [int(size) for size in sparse_shape]
    for layer in layers:
        out_spatial_shape = resolve_output_spatial_shape(
            spatial_shape,
            subm=layer.subm,
            transposed=layer.transposed,
            kernel_size=list(layer.kernel_size),
            stride=list(layer.stride),
            padding=list(layer.padding),
            dilation=list(layer.dilation),
            output_padding=list(layer.output_padding),
        )
        if not layer.subm:
            if layer.transposed:
                raise ValueError("Rulebook precompute supports forward down-sampling layers only.")
            if layer.indice_key is None:
                raise ValueError(
                    "A down-sampling sparse convolution needs an indice_key for its rulebook "
                    "to be precomputed: the layer looks the rulebook up by that key."
                )
            if any(stage.indice_key == layer.indice_key for stage in stages):
                raise ValueError(
                    f"Two down-sampling layers share indice_key {layer.indice_key!r}; each "
                    "needs its own rulebook."
                )
            stages.append(
                DownsampleStage(
                    indice_key=layer.indice_key,
                    algo=layer.algo,
                    kernel_size=tuple(int(k) for k in layer.kernel_size),
                    stride=tuple(int(s) for s in layer.stride),
                    padding=tuple(int(p) for p in layer.padding),
                    dilation=tuple(int(d) for d in layer.dilation),
                    output_padding=tuple(int(p) for p in layer.output_padding),
                    spatial_shape=tuple(spatial_shape),
                    out_spatial_shape=tuple(int(size) for size in out_spatial_shape),
                )
            )
        spatial_shape = [int(size) for size in out_spatial_shape]
    return tuple(stages)


def rulebook_input_names(stages: Iterable[DownsampleStage]) -> tuple[str, ...]:
    """All graph-input names of ``stages``, stage-major, slot order within a stage."""
    return tuple(name for stage in stages for name in stage.input_names)


def rulebook_dynamic_axes(stages: Iterable[DownsampleStage]) -> dict[str, dict[int, str]]:
    """ONNX dynamic axes of the rulebook inputs: the active-voxel count of every stage."""
    axes: dict[str, dict[int, str]] = {}
    for stage in stages:
        count = f"{stage.indice_key}_num"
        axes[stage.input_name("out_indices")] = {0: count}
        axes[stage.input_name("pair_fwd")] = {1: count}
        axes[stage.input_name("pair_mask")] = {0: count}
        axes[stage.input_name("mask_argsort")] = {0: count}
    return axes


def precompute_rulebooks(
    coords: torch.Tensor,
    batch_size: int,
    stages: Iterable[DownsampleStage],
    *,
    do_sort: bool,
) -> dict[str, torch.Tensor]:
    """Generate the rulebook of every stage, cascading each stage's output voxels into the next.

    Runs the same ``GetIndicePairsImplicitGemm`` the layer would have run in the graph, with
    the same arguments, so the tensors are the ones the graph's ``ImplicitGemm`` nodes were
    exported against.

    Args:
        coords: ``[N, 1 + ndim]`` int32 ``[batch, *spatial]`` coordinates **in the column
            order the encoder's convolutions use** (for the BEVFusion encoder,
            :meth:`~autoware_ml.models.detection3d.encoders.sparse.SparseEncoder.conv_coords`).
        batch_size: Number of samples in ``coords``.
        stages: From :func:`downsample_stages`, in forward order.
        do_sort: The encoder's ``export_do_sort`` — must match the graph the rulebooks feed.

    Returns:
        Graph-input name -> int32 tensor, for every stage and slot.
    """
    timer = CUDAKernelTimer(False)
    tensors: dict[str, torch.Tensor] = {}
    current = coords.int().contiguous()
    with torch.no_grad():
        for stage in stages:
            # Same call the exportable layer makes (SparseConvolution._resolve_implicit_gemm_plan):
            # is_train is `not subm`, the allocator is the fresh tensor's (None).
            out_indices, pair_fwd, pair_mask, mask_argsort, _ = get_indice_pairs_implicit_gemm(
                current,
                batch_size,
                list(stage.spatial_shape),
                stage.algo,
                list(stage.kernel_size),
                list(stage.stride),
                list(stage.padding),
                list(stage.dilation),
                list(stage.output_padding),
                False,
                False,
                True,
                None,
                timer,
                do_sort,
            )
            for slot, tensor in zip(
                RULEBOOK_SLOTS, (out_indices, pair_fwd, pair_mask, mask_argsort)
            ):
                tensors[stage.input_name(slot)] = tensor.int().contiguous()
            current = tensors[stage.input_name("out_indices")]
    return tensors


def rulebook_indice_data(
    stage: DownsampleStage, tensors: Mapping[str, torch.Tensor]
) -> ImplicitGemmIndiceData:
    """Wrap one stage's tensors as the cached indice data its layer reuses.

    Args:
        stage: The stage the tensors belong to.
        tensors: Graph-input name -> tensor (the whole rulebook mapping is accepted).

    Returns:
        The object to store under ``stage.indice_key`` in the input tensor's ``indice_dict``.
        ``indices`` (the layer's *input* coordinates) is not carried: the graph inputs hold
        the layer's outputs, and the layer validates geometry instead.
    """
    out_indices = tensors[stage.input_name("out_indices")]
    return ImplicitGemmIndiceData(
        out_indices,
        None,
        tensors[stage.input_name("pair_fwd")],
        None,
        tensors[stage.input_name("pair_mask")],
        None,
        tensors[stage.input_name("mask_argsort")],
        mask_argsort_bwd_splits=None,
        masks=[None],
        spatial_shape=list(stage.spatial_shape),
        out_spatial_shape=list(stage.out_spatial_shape),
        is_subm=False,
        algo=stage.algo,
        ksize=list(stage.kernel_size),
        stride=list(stage.stride),
        padding=list(stage.padding),
        dilation=list(stage.dilation),
        out_voxel_num=int(out_indices.shape[0]),
    )


def embed_rulebook_metadata(
    onnx_path: str | Path,
    *,
    stages: Iterable[DownsampleStage],
    coors_permutation: Iterable[int],
) -> Path:
    """Stage transform: record the rulebook geometry in the exported graph's metadata.

    The deployed runtime has to regenerate exactly these rulebooks from the frame's voxel
    coordinates, so the graph carries what it needs: every stage's kernel geometry and input
    spatial shape, and the column order that turns the graph's ``coors`` input into the
    coordinates the convolutions were exported with.

    Args:
        onnx_path: The exported sparse graph, rewritten in place.
        stages: From :func:`downsample_stages`.
        coors_permutation: For each convolution spatial column, the ``coors`` column it comes
            from — ``(1, 2, 0)`` maps ``coors = [z, y, x]`` onto ``[y, x, z]``.

    Returns:
        The same path.
    """
    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    props = {prop.key: prop.value for prop in model.metadata_props}
    props[STAGES_METADATA_KEY] = json.dumps(
        [stage.metadata() for stage in stages], separators=(",", ":")
    )
    props[COORS_PERMUTATION_METADATA_KEY] = meta_value_to_str(
        [int(column) for column in coors_permutation]
    )
    onnx.helper.set_model_props(model, props)
    onnx.save(model, str(onnx_path))
    logger.info(
        "Rulebook metadata in %s: %d down-sampling stage(s) recorded for the runtime.",
        onnx_path.name,
        len(list(stages)),
    )
    return onnx_path
