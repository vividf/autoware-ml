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
from dataclasses import fields
import json
import logging
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointTransformerV3Encoder,
    SerializedPoolingMeta,
    _pooling_depth,
    build_patch_order,
    build_serialized_pooling_meta,
    collect_encoder_stage_points,
    collect_stage_patch_sizes,
)
from autoware_ml.utils.point_cloud.structures import Point

logger = logging.getLogger(__name__)

#: The input level's serialization tensors, the graph inputs that replace the in-graph
#: Argsort of `serialized_code`: the serialize stage computes the order (and its inverse)
#: anyway to derive the pooling metadata, so the graph reads them instead of sorting again.
#: `patch_order` is the order padded to whole attention windows (build_patch_order); its
#: first `count` entries are the order itself, which is why the bare order is not an input —
#: the tracer would prune an input nothing reads, and the deployed runtime binds by name.
INPUT_LEVEL_SERIALIZATION_INPUTS = ("serialized_inverse", "patch_order")
#: ONNX `metadata_props` key under which the exporter records the per-level attention window,
#: the one constant the deployed runtime needs to rebuild `patch_order`.
PATCH_SIZES_METADATA_KEY = "patch_sizes"

# The per-stage serialization tensors a decoder block stage reads: the same names, and the
# same tensors, the encoder graph consumes for that stage.
_BLOCK_STAGE_META_FIELDS = ("serialized_inverse", "grid_coord", "patch_order")

SERIALIZED_POOLING_FIELDS = tuple(field.name for field in fields(SerializedPoolingMeta))
# Fields no exported graph reads: `cluster` only drives the heads' unpooling (the joint
# single-graph exports do consume it), and `serialized_order` is carried by `patch_order`
# (its first `count` entries) — the blocks gather through the padded order and un-permute
# through the inverse. A declared input the tracer prunes would break the deployed
# runtime, which binds every declared name.
GRAPH_UNREAD_POOLING_FIELDS = frozenset({"serialized_order"})
# The encoder-only encoder graph never consumes `cluster` (it only drives the
# heads' unpooling), so the split encoder export excludes it.
ENCODER_EXPORT_POOLING_FIELDS = tuple(
    name
    for name in SERIALIZED_POOLING_FIELDS
    if name != "cluster" and name not in GRAPH_UNREAD_POOLING_FIELDS
)
# Every graph-facing field: what the joint single-graph exports declare.
GRAPH_POOLING_FIELDS = tuple(
    name for name in SERIALIZED_POOLING_FIELDS if name not in GRAPH_UNREAD_POOLING_FIELDS
)
SERIALIZED_POOLING_INPUT_SIZED_FIELDS = frozenset({"indices", "cluster"})
SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS = frozenset({"indptr"})
SERIALIZED_POOLING_ORDER_FIELDS = frozenset({"serialized_order", "serialized_inverse"})
SERIALIZED_POOLING_PADDED_FIELDS = frozenset({"patch_order"})


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
    patch_sizes: Sequence[int | None],
) -> list[SerializedPoolingMeta]:
    """Build serialized-pooling metadata for every encoder pooling stage.

    Args:
        grid_coord: Input-level voxel coordinates.
        serialized_code: Input-level codes, ``[num_orders, count]``.
        serialized_order: Input-level orders, ``[num_orders, count]``.
        strides: One pooling stride per stage.
        patch_sizes: Attention window per *level* (``len(strides) + 1`` entries, the input
            level first); pooling stage ``i`` produces level ``i + 1``.
    """
    if len(patch_sizes) != len(strides) + 1:
        raise ValueError(
            f"patch_sizes must have one entry per level ({len(strides) + 1}), got {len(patch_sizes)}."
        )
    metadata = []
    for stride, patch_size in zip(strides, patch_sizes[1:]):
        meta, serialized_code = build_serialized_pooling_meta(
            grid_coord, serialized_code, serialized_order, stride, patch_size
        )
        metadata.append(meta)
        grid_coord = meta.grid_coord
        serialized_order = meta.serialized_order
    return metadata


def export_patch_sizes(model: Any) -> list[int | None]:
    """The attention window of every hierarchy level, as the export graphs need it.

    One ``patch_order`` per level is the graph contract, so a level's encoder blocks and its
    decoder blocks (when the model has a segmentation head with blocks there) must agree on
    the window; both PTv3 configurations do, and a disagreement is a config error reported
    here rather than a silently mis-padded gather.
    """
    patch_sizes = collect_stage_patch_sizes(model.encoder.enc)
    head = getattr(model, "seg3d_head", None)
    if head is not None:
        for stage, dec_patch_size in enumerate(collect_stage_patch_sizes(head.dec)):
            if dec_patch_size is None:
                continue
            if patch_sizes[stage] is None:
                patch_sizes[stage] = dec_patch_size
            elif patch_sizes[stage] != dec_patch_size:
                raise ValueError(
                    f"Level {stage}: encoder patch_size {patch_sizes[stage]} != decoder "
                    f"patch_size {dec_patch_size}; the deployed graphs share one patch_order per "
                    "level, so enc_patch_size and dec_patch_size must match where both have "
                    "attention blocks."
                )
    return patch_sizes


def build_input_level_serialization(
    point: Point, patch_size: int | None
) -> dict[str, torch.Tensor]:
    """The input level's serialization graph inputs from a serialized ``Point``."""
    return {
        "serialized_inverse": point["serialized_inverse"],
        "patch_order": build_patch_order(point["serialized_order"], patch_size),
    }


def embed_patch_sizes_metadata(onnx_path: str | Path, *, patch_sizes: Sequence[int | None]) -> Path:
    """Stage transform: record the per-level attention window in the exported graph.

    The deployed runtime rebuilds every ``patch_order`` input from the frame's serialization
    and this constant, so the graph carries it.
    """
    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    props = {prop.key: prop.value for prop in model.metadata_props}
    props[PATCH_SIZES_METADATA_KEY] = json.dumps(list(patch_sizes), separators=(",", ":"))
    onnx.helper.set_model_props(model, props)
    onnx.save(model, str(onnx_path))
    logger.info("Patch sizes %s recorded in %s for the runtime.", list(patch_sizes), onnx_path.name)
    return onnx_path


def flatten_serialized_pooling_inputs(
    metadata: Sequence[SerializedPoolingMeta],
    field_names: Sequence[str] = GRAPH_POOLING_FIELDS,
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
    if field in SERIALIZED_POOLING_PADDED_FIELDS:
        return {1: f"{stage_prefix}_padded_voxels"}
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
        elif input_name == "serialized_inverse":
            dynamic_axes[input_name] = {1: "num_voxels"}
        elif input_name == "patch_order":
            dynamic_axes[input_name] = {1: "padded_voxels"}
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
    field_names: Sequence[str] = GRAPH_POOLING_FIELDS,
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
        pooling_field_names: Sequence[str] = GRAPH_POOLING_FIELDS,
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
        serialized_inverse: torch.Tensor,
        patch_order: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> Point:
        """Run the encoder over the declared inputs.

        Args:
            grid_coord: Discretized grid coordinates.
            feat: Point features whose first three channels are xyz.
            serialized_inverse: Input-level inverse serialization orders,
                ``[num_orders, count]``.
            patch_order: The serialization orders padded to whole attention windows.
            serialized_pooling_inputs: Flattened per-stage pooling metadata.

        Returns:
            Deepest encoder point with the full pooling chain attached.
        """
        return _run_ptv3_encoder_export(
            self.encoder,
            grid_coord,
            feat,
            self._serialized_depth,
            serialized_inverse,
            patch_order,
            self._sparse_shape,
            *serialized_pooling_inputs,
            pooling_field_names=self.pooling_field_names,
        )


def _run_ptv3_encoder_export(
    encoder: PointTransformerV3Encoder,
    grid_coord: torch.Tensor,
    feat: torch.Tensor,
    serialized_depth: torch.Tensor,
    serialized_inverse: torch.Tensor,
    patch_order: torch.Tensor,
    sparse_shape: torch.Tensor,
    *serialized_pooling_inputs: torch.Tensor,
    pooling_field_names: Sequence[str] = GRAPH_POOLING_FIELDS,
) -> Point:
    """Run the shared tensor-only PTv3 encoder export path.

    The serialization arrives fully precomputed: the serialize stage sorts the input level
    to derive the pooling metadata anyway, so the graph carries no Argsort of its own.

    Args:
        encoder: Export-prepared encoder.
        grid_coord: Discretized grid coordinates.
        feat: Point features whose first three channels are xyz.
        serialized_depth: Baked serialization depth.
        serialized_inverse: Input-level inverse serialization orders.
        patch_order: The orders padded to whole attention windows (their first ``count``
            entries are the orders themselves).
        sparse_shape: Baked sparse shape.
        serialized_pooling_inputs: Flattened per-stage pooling metadata.
        pooling_field_names: Metadata fields the flattened tensors carry.

    Returns:
        Deepest encoder point with the full pooling chain attached.
    """
    point_count = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
    return encoder.export_forward(
        {
            "coord": feat[:, :3],
            "feat": feat,
            "grid_coord": grid_coord,
            "offset": point_count,
            "serialized_depth": serialized_depth,
            "serialized_inverse": serialized_inverse,
            "patch_order": patch_order,
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
        serialized_inverse: torch.Tensor,
        patch_order: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the encoder and return per-stage features, finest to deepest."""
        point = self.run_encoder(
            grid_coord, feat, serialized_inverse, patch_order, *serialized_pooling_inputs
        )
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
            blocks, ``(serialized_inverse, grid_coord, patch_order,
            sparse_shape)`` used to rebuild the serialization and sparse-conv
            views the blocks read. Single-sample export is assumed
            for the derived batch offsets.

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
        serialized_inverse, grid_coord, patch_order, sparse_shape = metadata
        point = points[stage_index]
        point["serialized_inverse"] = serialized_inverse
        point["patch_order"] = patch_order
        point["grid_coord"] = grid_coord
        point["offset"] = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
        point["batch"] = torch.zeros_like(grid_coord[:, 0]).long()
        point["sparse_shape"] = sparse_shape
        point.sparsify()
    return points[-1]


def _block_stage_indices(dec_depths: Sequence[int]) -> list[int]:
    """Return the decoder stages that contain blocks of any kind."""
    return [stage for stage, depth in enumerate(dec_depths) if depth > 0]


# --- Detection head -----------------------------------------------------------
# The det head's export pieces live here, next to the seg head's, because the stage
# graph for both tasks is built in this package. Keeping them in the legacy
# `models/detection3d/ptv3.py` would make the stage graph import that module, and
# that module imports back into this package — an import cycle whose only previous
# answer was a function-level import.


def det_head_export_input_names(stage_count: int) -> list[str]:
    """Return the split det-head export input names for a given stage count."""
    skip_stage = stage_count - 2
    deep_stage = stage_count - 1
    return [
        f"point_feat_{skip_stage}",
        f"point_feat_{deep_stage}",
        f"pooling_cluster_{skip_stage}",
        f"point_grid_coord_{skip_stage}",
    ]


class _PTv3DetHeadExportModule(nn.Module):
    """Export-only detection head consuming the two coarsest encoder stages."""

    def __init__(
        self,
        bev_neck: nn.Module,
        bbox_head: nn.Module,
        output_names: Sequence[str],
    ) -> None:
        """Initialize the export-only detection head module.

        Args:
            bev_neck: Export-ready detection BEV neck (``PTv3DetBEVNeck``).
            bbox_head: Export-ready detection head module.
            output_names: Ordered output tensor names emitted by ``bbox_head``.
        """
        super().__init__()
        self.bev_neck = bev_neck
        self.bbox_head = bbox_head
        self.output_names = list(output_names)

    def forward(
        self,
        skip_feat: torch.Tensor,
        deepest_feat: torch.Tensor,
        cluster: torch.Tensor,
        skip_grid_coord: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Rebuild the coarse pooling link, project to BEV, and run the head."""
        offset = shape_as_tensor(skip_feat)[:1].to(skip_feat.device)
        parent = Point(feat=skip_feat, grid_coord=skip_grid_coord, offset=offset)
        point = Point(feat=deepest_feat, pooling_parent=parent, pooling_inverse=cluster)
        bev = self.bev_neck(point)
        outputs = self.bbox_head(bev)
        return tuple(outputs[name] for name in self.output_names)


def seg_head_export_input_names(stage_count: int, dec_depths: Sequence[int]) -> list[str]:
    """Return the split seg-head export input names for a decoder configuration.

    The rule is the deployment contract: per-stage features and pooling
    clusters always; for every stage with decoder blocks, that stage's
    serialization metadata (the same tensors, under the same names, that the
    encoder graph consumes - stage 0 uses the base input-level tensors
    ``serialized_inverse`` / ``patch_order`` and ``grid_coord`` instead).

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
            names += [*INPUT_LEVEL_SERIALIZATION_INPUTS, "grid_coord"]
        else:
            prefix = f"serialized_pooling_{stage - 1}_"
            names += [prefix + field for field in _BLOCK_STAGE_META_FIELDS]
    return names


def build_seg_head_export_args(
    stage_feats: Sequence[torch.Tensor],
    pooling_metadata: Sequence[SerializedPoolingMeta],
    input_level: Mapping[str, torch.Tensor],
    grid_coord: torch.Tensor,
    dec_depths: Sequence[int],
) -> tuple[torch.Tensor, ...]:
    """Assemble the split seg-head export args matching the input-name rule.

    Args:
        stage_feats: Per-stage encoder features.
        pooling_metadata: Per-pooling-stage metadata.
        input_level: The input level's serialization tensors keyed by their input names
            (:func:`build_input_level_serialization`).
        grid_coord: Input-level voxel coordinates.
        dec_depths: Decoder block counts per stage.
    """
    args = [*stage_feats, *(meta.cluster for meta in pooling_metadata)]
    for stage in _block_stage_indices(dec_depths):
        if stage == 0:
            args += [*(input_level[name] for name in INPUT_LEVEL_SERIALIZATION_INPUTS), grid_coord]
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
            dynamic_axes.update(
                build_ptv3_input_dynamic_axes([*INPUT_LEVEL_SERIALIZATION_INPUTS, "grid_coord"])
            )
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
                serialized_inverse = extras.pop(0)
                patch_order = extras.pop(0)
                grid_coord = extras.pop(0)
            else:
                # Unpack through the same field tuple the input names and the export args
                # are built from. `serialized_order` and `serialized_inverse` have identical
                # shapes, so a hand-written order that disagreed with the name list would
                # not raise — it would quietly decode with the permutation reversed, and
                # only mIoU would notice.
                by_field = dict(
                    zip(
                        _BLOCK_STAGE_META_FIELDS,
                        [extras.pop(0) for _ in _BLOCK_STAGE_META_FIELDS],
                    )
                )
                serialized_inverse = by_field["serialized_inverse"]
                grid_coord = by_field["grid_coord"]
                patch_order = by_field["patch_order"]
            block_stage_metadata[stage] = (
                serialized_inverse,
                grid_coord,
                patch_order,
                getattr(self, f"_sparse_shape_{stage}"),
            )

        if extras:
            raise ValueError(
                f"{len(extras)} unconsumed export input(s) for the seg head — the call does "
                "not match seg_head_export_input_names() for this decoder configuration."
            )

        logits = self.seg3d_head(link_stage_points(stage_feats, clusters, block_stage_metadata))
        probs = torch.softmax(logits, dim=1)
        return probs.argmax(dim=1), probs
