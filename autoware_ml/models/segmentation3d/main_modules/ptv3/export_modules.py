"""PTv3 export toolbox: the deployment name contract and the export-only modules.

Owned by the stage-graph deployment of PTv3 (:mod:`.stages`): the serialized-pooling
metadata layout, the generated input/output name rules (``point_feat_{i}``,
``serialized_pooling_{i}_{field}``, ``pooling_cluster_{i}``), their dynamic axes, and
the export-only encoder/head modules. The legacy ``ExportSpec`` path
(:mod:`autoware_ml.models.segmentation3d.ptv3_base`) builds on the same toolbox through
re-imports until it is deleted (Q5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointTransformerV3Encoder,
    SerializedPoolingMeta,
    _pooling_depth,
    build_serialized_pooling_meta,
    collect_encoder_stage_points,
)
from autoware_ml.ops.indexing.operators import argsort
from autoware_ml.utils.point_cloud.structures import Point, invert_permutation


_BLOCK_STAGE_META_FIELDS = ("serialized_order", "serialized_inverse", "grid_coord")

SERIALIZED_POOLING_FIELDS = tuple(field.name for field in fields(SerializedPoolingMeta))
# The encoder-only encoder graph never consumes `cluster` (it only drives the
# heads' unpooling), so the split encoder export excludes it.
ENCODER_EXPORT_POOLING_FIELDS = tuple(
    name for name in SERIALIZED_POOLING_FIELDS if name != "cluster"
)
SERIALIZED_POOLING_INPUT_SIZED_FIELDS = frozenset({"indices", "cluster"})
SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS = frozenset({"indptr"})
SERIALIZED_POOLING_ORDER_FIELDS = frozenset({"serialized_order", "serialized_inverse"})


def split_block_parameters(
    module: nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split trainable parameters into non-block and attention-block groups.

    Args:
        module: Module hierarchy whose parameters are grouped structurally.

    Returns:
        ``(default_params, block_params)`` where block parameters belong to
        :class:`Block` submodules and default parameters are all the rest.
    """
    block_parameter_ids = {
        id(parameter)
        for child in module.modules()
        if isinstance(child, Block)
        for parameter in child.parameters()
        if parameter.requires_grad
    }
    default_params: list[torch.nn.Parameter] = []
    block_params: list[torch.nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in block_parameter_ids:
            block_params.append(parameter)
        else:
            default_params.append(parameter)
    return default_params, block_params



def build_serialized_pooling_metadata(
    grid_coord: torch.Tensor,
    serialized_code: torch.Tensor,
    serialized_order: torch.Tensor,
    strides: Sequence[int],
) -> list[SerializedPoolingMeta]:
    """Build serialized-pooling metadata for every encoder pooling stage."""
    metadata = []
    for stride in strides:
        meta, serialized_code = build_serialized_pooling_meta(
            grid_coord, serialized_code, serialized_order, stride
        )
        metadata.append(meta)
        grid_coord = meta.grid_coord
        serialized_order = meta.serialized_order
    return metadata


def flatten_serialized_pooling_inputs(
    metadata: Sequence[SerializedPoolingMeta],
    field_names: Sequence[str] = SERIALIZED_POOLING_FIELDS,
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """Flatten per-stage metadata into ONNX args and input names.

    Args:
        metadata: Per-pooling-stage metadata objects.
        field_names: Metadata fields exported, in order. The encoder-only
            encoder graph excludes ``cluster`` (it only drives head-side
            unpooling), the combined graphs export every field.
    """
    inputs: list[torch.Tensor] = []
    names: list[str] = []
    for stage_index, meta in enumerate(metadata):
        for field in field_names:
            inputs.append(getattr(meta, field))
            names.append(f"serialized_pooling_{stage_index}_{field}")
    return tuple(inputs), names



def _serialized_pooling_dynamic_axis(input_name: str) -> dict[int, str]:
    _, _, stage_index, field = input_name.split("_", 3)
    stage_prefix = f"serialized_pooling_{stage_index}"
    if field in SERIALIZED_POOLING_INPUT_SIZED_FIELDS:
        return {0: f"{stage_prefix}_in_voxels"}
    if field in SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS:
        return {0: f"{stage_prefix}_out_voxels_plus_one"}
    if field in SERIALIZED_POOLING_ORDER_FIELDS:
        return {1: f"{stage_prefix}_out_voxels"}
    return {0: f"{stage_prefix}_out_voxels"}


def stage_voxel_axis_name(stage_index: int) -> str:
    """Return the dynamic-axis name for the voxel count of one encoder stage."""
    if stage_index == 0:
        return "num_voxels"
    return f"serialized_pooling_{stage_index - 1}_out_voxels"


def stage_feature_names(stage_count: int) -> list[str]:
    """Return the per-stage encoder feature tensor names, finest to deepest."""
    return [f"point_feat_{stage_index}" for stage_index in range(stage_count)]


def pooling_cluster_names(stage_count: int) -> list[str]:
    """Return the per-pooling cluster tensor names consumed by the decoder."""
    return [f"pooling_cluster_{stage_index}" for stage_index in range(stage_count - 1)]


def build_stage_feature_dynamic_axes(stage_count: int) -> dict[str, dict[int, str]]:
    """Build dynamic axes for per-stage encoder feature tensors."""
    return {
        name: {0: stage_voxel_axis_name(stage_index)}
        for stage_index, name in enumerate(stage_feature_names(stage_count))
    }


def build_pooling_cluster_dynamic_axes(stage_count: int) -> dict[str, dict[int, str]]:
    """Build dynamic axes for per-pooling cluster tensors."""
    return {
        name: {0: f"serialized_pooling_{stage_index}_in_voxels"}
        for stage_index, name in enumerate(pooling_cluster_names(stage_count))
    }


def build_point_feature_dynamic_axes(tensor_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Build dynamic axes for tensors indexed by the decoded point/voxel count."""
    return {tensor_name: {0: "num_voxels"} for tensor_name in tensor_names}


def build_ptv3_input_dynamic_axes(input_names: Sequence[str]) -> dict[str, dict[int, str]]:
    """Build dynamic axes for generated PTv3 encoder export inputs."""
    dynamic_axes: dict[str, dict[int, str]] = {}
    for input_name in input_names:
        if input_name in {"grid_coord", "feat"}:
            dynamic_axes[input_name] = {0: "num_voxels"}
        elif input_name == "serialized_code":
            dynamic_axes[input_name] = {1: "num_voxels"}
        elif input_name.startswith("serialized_pooling_"):
            dynamic_axes[input_name] = _serialized_pooling_dynamic_axis(input_name)
    return dynamic_axes


def build_ptv3_encoder_dynamic_axes(
    input_names: Sequence[str], stage_count: int
) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split PTv3 encoder export graph."""
    dynamic_axes = build_ptv3_input_dynamic_axes(input_names)
    dynamic_axes.update(build_stage_feature_dynamic_axes(stage_count))
    return dynamic_axes


def make_serialized_pooling_from_flat_inputs(
    serialized_pooling_inputs: tuple[torch.Tensor, ...],
    field_names: Sequence[str] = SERIALIZED_POOLING_FIELDS,
) -> list[SerializedPoolingMeta]:
    """Reconstruct per-stage metadata objects from flattened ONNX graph inputs.

    Fields absent from ``field_names`` are filled with empty placeholders;
    only fields the target graph never consumes may be omitted.
    """
    num_fields = len(field_names)
    if len(serialized_pooling_inputs) % num_fields != 0:
        raise ValueError("serialized-pooling inputs are not divisible by metadata field count.")
    metadata: list[SerializedPoolingMeta] = []
    for index in range(0, len(serialized_pooling_inputs), num_fields):
        values = dict(zip(field_names, serialized_pooling_inputs[index : index + num_fields]))
        placeholder = values[field_names[0]].new_zeros(0)
        for field in SERIALIZED_POOLING_FIELDS:
            values.setdefault(field, placeholder)
        metadata.append(SerializedPoolingMeta(**values))
    return metadata



class PTv3EncoderExportBase(nn.Module):
    """Share the encoder half of every PTv3 export graph.

    Subclasses add their own task head and declare their outputs; the encoder
    inputs, the baked geometry buffers, and the metadata unpacking are identical
    across segmentation, detection, and the joint model.
    """

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
        pooling_field_names: Sequence[str] = SERIALIZED_POOLING_FIELDS,
    ) -> None:
        """Initialize the shared encoder export half.

        Args:
            encoder: Export-prepared PTv3 encoder copy.
            sparse_shape: Static sparse shape baked at export time.
            serialized_depth: Serialization depth baked at export time.
            pooling_field_names: Metadata fields the graph declares per pooling
                stage. The split encoder graph excludes ``cluster``.
        """
        super().__init__()
        self.encoder = encoder
        self.pooling_field_names = tuple(pooling_field_names)
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def run_encoder(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> Point:
        """Run the encoder over the declared inputs.

        Args:
            grid_coord: Discretized grid coordinates.
            feat: Point features whose first three channels are xyz.
            serialized_code: Base serialization codes.
            serialized_pooling_inputs: Flattened per-stage pooling metadata.

        Returns:
            Deepest encoder point with the full pooling chain attached.
        """
        return _run_ptv3_encoder_export(
            self.encoder,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_code,
            self._sparse_shape,
            *serialized_pooling_inputs,
            pooling_field_names=self.pooling_field_names,
        )


def _run_ptv3_encoder_export(
    encoder: PointTransformerV3Encoder,
    grid_coord: torch.Tensor,
    feat: torch.Tensor,
    serialized_depth: torch.Tensor,
    serialized_code: torch.Tensor,
    sparse_shape: torch.Tensor,
    *serialized_pooling_inputs: torch.Tensor,
    pooling_field_names: "Sequence[str]" = SERIALIZED_POOLING_FIELDS,
) -> Point:
    """Run the shared tensor-only PTv3 encoder export path.

    Args:
        encoder: Export-prepared encoder.
        grid_coord: Discretized grid coordinates.
        feat: Point features whose first three channels are xyz.
        serialized_depth: Baked serialization depth.
        serialized_code: Base serialization codes.
        sparse_shape: Baked sparse shape.
        serialized_pooling_inputs: Flattened per-stage pooling metadata.
        pooling_field_names: Metadata fields the flattened tensors carry.

    Returns:
        Deepest encoder point with the full pooling chain attached.
    """
    point_count = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
    serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
    serialized_inverse = invert_permutation(serialized_order)
    return encoder.export_forward(
        {
            "coord": feat[:, :3],
            "feat": feat,
            "grid_coord": grid_coord,
            "offset": point_count,
            "serialized_depth": serialized_depth,
            "serialized_code": serialized_code,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "serialized_pooling": make_serialized_pooling_from_flat_inputs(
                serialized_pooling_inputs, pooling_field_names
            ),
            "sparse_shape": sparse_shape,
        }
    )




class _PTv3EncoderExportModule(PTv3EncoderExportBase):
    """Export-only PTv3 encoder producing per-stage point features."""

    def forward(
        self,
        grid_coord: torch.Tensor,
        feat: torch.Tensor,
        serialized_code: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the encoder and return per-stage features, finest to deepest."""
        point = self.run_encoder(grid_coord, feat, serialized_code, *serialized_pooling_inputs)
        return tuple(stage.feat for stage in collect_encoder_stage_points(point))


def link_stage_points(
    stage_feats: Sequence[torch.Tensor],
    clusters: Sequence[torch.Tensor],
    block_stage_metadata: Mapping[int, tuple[torch.Tensor, ...]] | None = None,
) -> Point:
    """Rebuild the encoder pooling chain from per-stage tensors.

    Args:
        stage_feats: Per-stage features ordered finest to deepest.
        clusters: Per-pooling cluster tensors mapping each finer-stage voxel
            to its pooled voxel.
        block_stage_metadata: For every stage whose decoder has attention
            blocks, ``(serialized_order, serialized_inverse, grid_coord,
            sparse_shape)`` used to rebuild the serialization and sparse-conv
            views the blocks read. Single-sample export is assumed for the
            derived batch offsets.

    Returns:
        Deepest point whose ``pooling_parent``/``pooling_inverse`` chain links
        every finer stage.
    """
    if len(clusters) != len(stage_feats) - 1:
        raise ValueError(
            f"Expected {len(stage_feats) - 1} cluster tensors for {len(stage_feats)} stages, "
            f"got {len(clusters)}."
        )
    points = [Point(feat=feat) for feat in stage_feats]
    for stage_index in range(1, len(points)):
        points[stage_index]["pooling_parent"] = points[stage_index - 1]
        points[stage_index]["pooling_inverse"] = clusters[stage_index - 1]
    for stage_index, metadata in (block_stage_metadata or {}).items():
        serialized_order, serialized_inverse, grid_coord, sparse_shape = metadata
        point = points[stage_index]
        point["serialized_order"] = serialized_order
        point["serialized_inverse"] = serialized_inverse
        point["grid_coord"] = grid_coord
        point["offset"] = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
        point["batch"] = torch.zeros_like(grid_coord[:, 0]).long()
        point["sparse_shape"] = sparse_shape
        point.sparsify()
    return points[-1]


def _block_stage_indices(dec_depths: Sequence[int]) -> list[int]:
    """Return the decoder stages that contain blocks of any kind."""
    return [stage for stage, depth in enumerate(dec_depths) if depth > 0]


def seg_head_export_input_names(stage_count: int, dec_depths: Sequence[int]) -> list[str]:
    """Return the split seg-head export input names for a decoder configuration.

    The rule is the deployment contract: per-stage features and pooling
    clusters always; for every stage with decoder blocks, that stage's
    serialization metadata (the same tensors, under the same names, that the
    encoder graph consumes - stage 0 uses the base ``serialized_code`` and
    ``grid_coord`` inputs instead).

    Args:
        stage_count: Number of encoder stages.
        dec_depths: Decoder block counts per stage (``stage_count - 1`` entries).
    """
    if len(dec_depths) != stage_count - 1:
        raise ValueError(
            f"dec_depths must have {stage_count - 1} entries for {stage_count} stages, "
            f"got {len(dec_depths)}."
        )
    names = [*stage_feature_names(stage_count), *pooling_cluster_names(stage_count)]
    for stage in _block_stage_indices(dec_depths):
        if stage == 0:
            names += ["serialized_code", "grid_coord"]
        else:
            prefix = f"serialized_pooling_{stage - 1}_"
            names += [prefix + field for field in _BLOCK_STAGE_META_FIELDS]
    return names


def build_seg_head_export_args(
    stage_feats: Sequence[torch.Tensor],
    pooling_metadata: Sequence[SerializedPoolingMeta],
    serialized_code: torch.Tensor,
    grid_coord: torch.Tensor,
    dec_depths: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Assemble the split seg-head export args matching the input-name rule."""
    args = [*stage_feats, *(meta.cluster for meta in pooling_metadata)]
    for stage in _block_stage_indices(dec_depths):
        if stage == 0:
            args += [serialized_code, grid_coord]
        else:
            meta = pooling_metadata[stage - 1]
            args += [getattr(meta, field) for field in _BLOCK_STAGE_META_FIELDS]
    return tuple(args)


def build_seg_head_input_dynamic_axes(
    stage_count: int, dec_depths: Sequence[int]
) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split seg-head export inputs."""
    dynamic_axes = build_stage_feature_dynamic_axes(stage_count)
    dynamic_axes.update(build_pooling_cluster_dynamic_axes(stage_count))
    for stage in _block_stage_indices(dec_depths):
        if stage == 0:
            dynamic_axes["serialized_code"] = {1: "num_voxels"}
            dynamic_axes["grid_coord"] = {0: "num_voxels"}
        else:
            prefix = f"serialized_pooling_{stage - 1}_"
            for field in _BLOCK_STAGE_META_FIELDS:
                dynamic_axes[prefix + field] = _serialized_pooling_dynamic_axis(prefix + field)
    return dynamic_axes



class _PTv3SegHeadExportModule(nn.Module):
    """Export-only segmentation head decoding per-stage encoder features."""

    def __init__(
        self,
        seg3d_head: nn.Module,
        stage_count: int,
        sparse_shape: torch.Tensor,
        strides: Sequence[int],
    ) -> None:
        """Initialize the segmentation head export module.

        Args:
            seg3d_head: Export-prepared decoder head copy.
            stage_count: Number of encoder stages feeding the decoder.
            sparse_shape: Static base sparse shape baked at export time;
                block stages use it right-shifted by their cumulative pooling
                depth.
            strides: Encoder pooling strides (one per pooling stage).
        """
        super().__init__()
        self.seg3d_head = seg3d_head
        self.stage_count = int(stage_count)
        self.dec_depths = list(seg3d_head.dec_depths)
        cumulative_depth = 0
        stage_depths = [0]
        for stride in strides:
            cumulative_depth += _pooling_depth(int(stride))
            stage_depths.append(cumulative_depth)
        for stage in _block_stage_indices(self.dec_depths):
            self.register_buffer(
                f"_sparse_shape_{stage}",
                sparse_shape.to(dtype=torch.long) >> stage_depths[stage],
                persistent=False,
            )

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode per-stage features and return labels and class probabilities.

        Args:
            tensors: ``stage_count`` per-stage feature tensors, then
                ``stage_count - 1`` pooling cluster tensors, then the
                per-block-stage serialization tensors in the order defined by
                :func:`seg_head_export_input_names`.
        """
        stage_feats = tensors[: self.stage_count]
        clusters = tensors[self.stage_count : 2 * self.stage_count - 1]
        extras = list(tensors[2 * self.stage_count - 1 :])

        block_stage_metadata: dict[int, tuple[torch.Tensor, ...]] = {}
        for stage in _block_stage_indices(self.dec_depths):
            if stage == 0:
                serialized_code = extras.pop(0)
                grid_coord = extras.pop(0)
                serialized_order = torch.stack([argsort(code) for code in serialized_code], dim=0)
                serialized_inverse = invert_permutation(serialized_order)
            else:
                serialized_order = extras.pop(0)
                serialized_inverse = extras.pop(0)
                grid_coord = extras.pop(0)
            block_stage_metadata[stage] = (
                serialized_order,
                serialized_inverse,
                grid_coord,
                getattr(self, f"_sparse_shape_{stage}"),
            )

        logits = self.seg3d_head(link_stage_points(stage_feats, clusters, block_stage_metadata))
        probs = torch.softmax(logits, dim=1)
        return probs.argmax(dim=1), probs
