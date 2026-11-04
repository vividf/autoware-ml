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

A down-sampling layer's rulebook (which input voxel feeds which output voxel, per kernel
offset) has a data-dependent size, and TensorRT stalls on the host until it is known (the
``[trainStation]`` regions of an engine): 0.83 ms per frame on BEVFusion's four down-sampling
layers, 23% of its sparse engine. The rulebooks depend only on the voxel coordinates, so a
Torch stage computes them and the graph takes them as inputs. Submanifold layers keep their
in-graph rulebooks: their output voxels are their input voxels, so the shape is static.

``autoware_bevfusion``'s runtime binds the ``rulebook/<indice_key>/<slot>`` inputs and reads
the ``rulebook_stages`` metadata; keep the two in step.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from spconv.core import ConvAlgo
from spconv.pytorch.conv import SparseConvolution as SparseConvolutionBase
from spconv.pytorch.core import ImplicitGemmIndiceData
from spconv.tools import CUDAKernelTimer

from autoware_ml.ops.spconv.sparse_conv import resolve_output_spatial_shape
from autoware_ml.ops.spconv.sparse_functional import get_indice_pairs_implicit_gemm
from autoware_ml.utils.onnx_meta import meta_value_to_str

logger = logging.getLogger(__name__)

#: The four tensors of one rulebook, in the order ``GetIndicePairsImplicitGemm`` emits them.
RULEBOOK_SLOTS = ("out_indices", "pair_fwd", "pair_mask", "mask_argsort")
INPUT_NAMESPACE = "rulebook"
# metadata_props keys the runtime reads: the stage geometry it regenerates the rulebooks
# from, the coors -> convolution column order, and whether the pair masks are argsorted
# (its rulebooks must be generated the same way).
STAGES_METADATA_KEY = "rulebook_stages"
COORS_PERMUTATION_METADATA_KEY = "rulebook_coors_permutation"
DO_SORT_METADATA_KEY = "rulebook_do_sort"


@dataclass(frozen=True)
class DownsampleStage:
    """One down-sampling sparse convolution whose rulebook is precomputed.

    Attributes:
        indice_key: The key the layer looks its rulebook up under; also names the stage's
            graph inputs.
        algo: The layer's convolution algorithm, which the rulebook layout depends on.
        spatial_shape: Spatial shape of the layer's input.
        out_spatial_shape: Spatial shape of the layer's output.
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
        """Number of kernel offsets, the leading dimension of ``pair_fwd``."""
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
        """The stage's ``rulebook_stages`` entry, with spconv's key names."""
        return {
            "onnx_base": self.onnx_base,
            "algo": int(self.algo.value),
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

    ``layers`` are all of the encoder's sparse convolutions in forward order, since the spatial
    shape cascades through them; ``sparse_shape`` is the encoder's input shape.

    Raises:
        ValueError: On a down-sampling layer without an indice key or sharing one, or on a
            transposed layer.
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

    Makes the same ``GetIndicePairsImplicitGemm`` call as the layer in the graph. The output
    voxels come in the order of spconv's thrust-based generation; a runtime using the direct
    table lists them in another order, which is fine: the graph only needs the four tensors of
    a rulebook to agree, and the dense BEV scatter does not depend on the voxel order.

    Args:
        coords: ``[N, 1 + ndim]`` ``[batch, *spatial]`` coordinates in the convolutions' column
            order (``SparseEncoder.conv_coords``).
        batch_size: Number of samples in ``coords``.
        stages: From :func:`downsample_stages`.
        do_sort: The encoder's ``export_do_sort``; must match the graph the rulebooks feed.

    Returns:
        Graph-input name -> int32 tensor, for every stage and slot.
    """
    timer = CUDAKernelTimer(False)
    tensors: dict[str, torch.Tensor] = {}
    current = coords.int().contiguous()
    with torch.no_grad():
        for stage in stages:
            # Same arguments as SparseConvolution._resolve_implicit_gemm_plan: subm and
            # transposed False, is_train = not subm, no allocator on a fresh tensor.
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
    """Wrap one stage's tensors as the indice data its layer finds under ``stage.indice_key``.

    ``tensors`` may hold every stage's rulebook. The layer's input ``indices`` are not
    carried; the layer validates the geometry instead.
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
    do_sort: bool,
) -> Path:
    """Stage transform: record what the runtime needs to regenerate the rulebooks.

    Stores every stage's kernel geometry and input spatial shape, ``do_sort``, and
    ``coors_permutation``: for each convolution spatial column, the column of the graph's
    ``coors`` it comes from (``(1, 2, 0)`` maps ``[z, y, x]`` onto ``[y, x, z]``). Rewrites
    the file in place and returns its path.
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
    props[DO_SORT_METADATA_KEY] = meta_value_to_str(bool(do_sort))
    onnx.helper.set_model_props(model, props)
    onnx.save(model, str(onnx_path))
    logger.info(
        "Rulebook metadata in %s: %d down-sampling stage(s) recorded for the runtime.",
        onnx_path.name,
        len(list(stages)),
    )
    return onnx_path
