"""Abstract base class and shared export modules for PTv3-based task models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.base import BaseModel
from autoware_ml.deployment.stages import GraphStage, RunState, Stage, TorchStage
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointTransformerV3Encoder,
    SerializedPooling,
    SerializedPoolingMeta,
    _pooling_depth,
    build_serialized_pooling_meta,
    collect_encoder_stage_points,
)
from autoware_ml.models.segmentation3d.encoders.voxel import (
    TIME_LAG_COLUMN,
    VoxelFeatureEncoder,
)
from autoware_ml.types.backend import Backend
from autoware_ml.utils.point_cloud.batching import batch_to_offset
from autoware_ml.utils.point_cloud.structures import (
    Point,
    bit_length_tensor,
    pooled_sparse_shape,
    serialize_point_cloud_batch,
)

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
# Voxel-level export inputs shared by every PTv3 graph, ahead of the pooling metadata.
ENCODER_EXPORT_INPUT_NAMES = (
    "voxels",
    "num_points_per_voxel",
    "grid_coord",
    "serialized_order",
    "serialized_inverse",
)


def validate_serialization_geometry(
    encoder: nn.Module, grid_size: float, point_cloud_range: Sequence[float]
) -> None:
    """Raise if the configured geometry cannot cover the encoder's pooling hierarchy."""
    pooling_depth = sum(
        m.pooling_depth for m in encoder.modules() if isinstance(m, SerializedPooling)
    )
    extent = max(point_cloud_range[i + 3] - point_cloud_range[i] for i in range(3))
    if int(bit_length_tensor(extent / grid_size).item()) < pooling_depth:
        raise ValueError(
            f"point_cloud_range {tuple(point_cloud_range)} with grid_size {grid_size} cannot "
            f"cover the encoder's cumulative pooling depth {pooling_depth}."
        )


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


class PTv3BaseModel(BaseModel):
    """Abstract base class for all PTv3 task models.

    Provides shared encoder management, export geometry computation, and
    export helpers. Detection and segmentation subclasses inherit from this
    class (potentially with additional base classes via MRO).
    """

    EXPORT_ORDER = ("z", "z-trans")

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        grid_size: float | None,
        point_cloud_range: Sequence[float] | None,
        freeze_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 base model.

        Args:
            encoder: PTv3 encoder module.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            grid_size: Voxel grid size used to derive sparse shape and
                serialization depth for export.
            point_cloud_range: Six-element sequence ``[x_min, y_min, z_min,
                x_max, y_max, z_max]`` used to derive sparse shape for export.
            freeze_encoder: When ``True``, the encoder is permanently kept
                in eval mode with its parameters frozen.
            **kwargs: Keyword arguments forwarded to :class:`BaseModel` (and
                further up the MRO chain).
        """
        super().__init__(**kwargs)
        if encoder.in_channels != voxel_encoder.out_channels:
            raise ValueError(
                f"The encoder reads {encoder.in_channels} point features but the voxel encoder "
                f"describes a voxel with {voxel_encoder.out_channels}."
            )
        self.encoder = encoder
        self.voxel_encoder = voxel_encoder
        self.grid_size = grid_size
        self.point_cloud_range = (
            tuple(float(v) for v in point_cloud_range) if point_cloud_range is not None else None
        )
        if self.grid_size is not None and self.point_cloud_range is not None:
            validate_serialization_geometry(encoder, self.grid_size, self.point_cloud_range)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True) -> PTv3BaseModel:
        """Keep the frozen encoder in eval mode during training.

        Args:
            mode: When ``True``, set the model to training mode; otherwise to
                evaluation mode.

        Returns:
            This model instance.
        """
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Record encoder-freeze provenance in saved checkpoints.

        Args:
            checkpoint: Mutable checkpoint dictionary to annotate.
        """
        checkpoint["autoware_ml_checkpoint_recipe"] = {
            "type": "ptv3",
            "freeze_encoder": self.freeze_encoder,
        }

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        """Pick the voxels of the batch and the column of the point time lag.

        Args:
            batch_inputs: Model inputs holding the point cloud and its voxels.

        Returns:
            The padded voxels, their point counts, their ``[batch, x, y, z]`` coordinates, the
            occupied voxels the voxelizer discarded and the time lag column of the points.

        Raises:
            ValueError: If the batch carries no point cloud or no voxels.
        """
        point_batch = batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        voxels_data = batch_inputs.voxels_data
        if point_batch is None or voxels_data is None:
            raise ValueError("PTv3 needs the points of the batch and the voxels of a voxelizer.")
        return {
            "voxels": voxels_data.voxels,
            "num_points": voxels_data.num_points,
            "voxel_coords": voxels_data.batch_xyz_coords(),
            "num_dropped_voxels": voxels_data.num_dropped_voxels,
            "time_lag_column": point_batch.timestamp_difference_dim,
        }

    def build_encoder_inputs(
        self, voxels: torch.Tensor, num_points: torch.Tensor, voxel_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Turn preprocessed voxels into the PTv3 encoder input dictionary.

        Args:
            voxels: Padded voxel points ``(num_voxels, max_points, channels)``.
            num_points: Valid point count per voxel ``(num_voxels,)``.
            voxel_coords: Voxel coordinates ``(num_voxels, 4)`` laid out as
                ``(batch, x, y, z)`` with voxels grouped by batch index.

        Returns:
            Dictionary with ``coord``, ``feat``, ``grid_coord`` and ``offset``.
        """
        feat = self.voxel_encoder(voxels, num_points)
        return {
            "coord": feat[:, :3],
            "feat": feat,
            "grid_coord": voxel_coords[:, 1:],
            "offset": batch_to_offset(voxel_coords[:, 0].long()),
        }

    def encode(
        self,
        voxels: torch.Tensor,
        num_points: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_dropped_voxels: torch.Tensor,
        time_lag_column: int,
    ) -> Point:
        """Run the voxel encoder and the PTv3 encoder on preprocessed voxels.

        Args:
            voxels: Padded voxel points ``(num_voxels, max_points, channels)``.
            num_points: Valid point count per voxel ``(num_voxels,)``.
            voxel_coords: Voxel coordinates ``(num_voxels, 4)`` with a leading batch column.
            num_dropped_voxels: Occupied voxels the voxelizer discarded.
            time_lag_column: Point column holding the time lag.

        Returns:
            Deepest encoder point with the full pooling chain attached.

        Raises:
            ValueError: If the points carry the time lag in another column than
                the voxel encoder reads, or hold future points the voxel encoder does not read.
            RuntimeError: If the voxelizer dropped occupied voxels.
        """
        self.validate_voxel_inputs(voxels, num_dropped_voxels, time_lag_column)
        return self.encoder(self.build_encoder_inputs(voxels, num_points, voxel_coords))

    def validate_voxel_inputs(
        self, voxels: torch.Tensor, num_dropped_voxels: torch.Tensor, time_lag_column: int
    ) -> None:
        """Refuse voxels the voxel encoder cannot read (see :meth:`encode` for the cases).

        Args:
            voxels: Padded voxel points ``(num_voxels, max_points, channels)``.
            num_dropped_voxels: Occupied voxels the voxelizer discarded.
            time_lag_column: Point column holding the time lag.
        """
        if time_lag_column != TIME_LAG_COLUMN:
            raise ValueError(
                f"PTv3 needs the time lag of the points in column {TIME_LAG_COLUMN}, the batch "
                f"has it in column {time_lag_column}. Enable use_timestamp_difference in the "
                "sweep loader."
            )
        if not self.voxel_encoder.reads_future and bool((voxels[..., TIME_LAG_COLUMN] < 0).any()):
            raise ValueError(
                f"{type(self.voxel_encoder).__name__} reads no future points, but the batch has "
                "some. Load no future sweeps or use PastFutureVoxelFeatureEncoder."
            )
        if int(num_dropped_voxels) > 0:
            raise RuntimeError(
                f"The voxelizer dropped {int(num_dropped_voxels)} occupied voxels. Increase "
                "'max_voxels' and 'eval_max_voxels' of the data preprocessing so every voxel "
                "is kept."
            )

    def _prepare_encoder_export(self) -> PointTransformerV3Encoder:
        """Return an export-ready copy of the encoder.

        Returns:
            Copy of the encoder prepared for ONNX export with the configured
            export order.
        """
        return self.encoder.prepare_for_export(self.EXPORT_ORDER)


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
        if input_name in {"voxels", "num_points_per_voxel", "grid_coord"}:
            dynamic_axes[input_name] = {0: "num_voxels"}
        elif input_name in {"serialized_order", "serialized_inverse"}:
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


def _run_ptv3_encoder_export(
    encoder: PointTransformerV3Encoder,
    voxel_encoder: VoxelFeatureEncoder,
    voxels: torch.Tensor,
    num_points: torch.Tensor,
    grid_coord: torch.Tensor,
    serialized_depth: torch.Tensor,
    serialized_order: torch.Tensor,
    serialized_inverse: torch.Tensor,
    sparse_shape: torch.Tensor,
    *serialized_pooling_inputs: torch.Tensor,
    pooling_field_names: "Sequence[str]" = SERIALIZED_POOLING_FIELDS,
) -> Point:
    """Run the shared tensor-only PTv3 encoder export path.

    The finest level's serialization order arrives as a graph input, exactly like
    every pooled level's does. Deriving it in-graph instead would add an argsort
    over all voxels that the preprocessing pipeline has already performed.

    Args:
        encoder: Export-prepared encoder.
        voxel_encoder: Voxel feature encoder feeding the embedding stem.
        voxels: Padded voxel points.
        num_points: Valid point count per voxel.
        grid_coord: Discretized grid coordinates.
        serialized_depth: Baked serialization depth.
        serialized_order: Level-0 serialization order, one row per curve.
        serialized_inverse: Inverse of ``serialized_order``.
        sparse_shape: Baked sparse shape.
        serialized_pooling_inputs: Flattened per-stage pooling metadata.
        pooling_field_names: Metadata fields the flattened tensors carry.

    Returns:
        Deepest encoder point with the full pooling chain attached.
    """
    feat = voxel_encoder(voxels, num_points)
    point_count = shape_as_tensor(grid_coord)[:1].to(grid_coord.device)
    return encoder.export_forward(
        {
            "coord": feat[:, :3],
            "feat": feat,
            "grid_coord": grid_coord,
            "offset": point_count,
            "serialized_depth": serialized_depth,
            "serialized_order": serialized_order,
            "serialized_inverse": serialized_inverse,
            "serialized_pooling": make_serialized_pooling_from_flat_inputs(
                serialized_pooling_inputs, pooling_field_names
            ),
            "sparse_shape": sparse_shape,
        }
    )


class _PTv3EncoderExportModule(nn.Module):
    """The ``ptv3_encoder`` graph: voxels and serialization in, per-stage point features out."""

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        voxel_encoder: VoxelFeatureEncoder,
        sparse_shape: torch.Tensor,
        serialized_depth: torch.Tensor,
    ) -> None:
        """Initialize the encoder export module.

        Args:
            encoder: Export-prepared PTv3 encoder copy.
            voxel_encoder: Voxel feature encoder feeding the embedding stem.
            sparse_shape: Static sparse shape baked at export time.
            serialized_depth: Serialization depth baked at export time.
        """
        super().__init__()
        self.encoder = encoder
        self.voxel_encoder = voxel_encoder
        self.register_buffer("_sparse_shape", sparse_shape.to(dtype=torch.long), persistent=False)
        self.register_buffer("_serialized_depth", serialized_depth, persistent=False)

    def forward(
        self,
        voxels: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        grid_coord: torch.Tensor,
        serialized_order: torch.Tensor,
        serialized_inverse: torch.Tensor,
        *serialized_pooling_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the encoder and return per-stage features, finest to deepest.

        Args:
            voxels: Padded voxel points.
            num_points_per_voxel: Valid point count per voxel.
            grid_coord: Discretized grid coordinates.
            serialized_order: Level-0 serialization order, one row per curve.
            serialized_inverse: Inverse of ``serialized_order``.
            serialized_pooling_inputs: Flattened per-stage pooling metadata
                (:data:`ENCODER_EXPORT_POOLING_FIELDS` per stage).
        """
        point = _run_ptv3_encoder_export(
            self.encoder,
            self.voxel_encoder,
            voxels,
            num_points_per_voxel,
            grid_coord,
            self._serialized_depth,
            serialized_order,
            serialized_inverse,
            self._sparse_shape,
            *serialized_pooling_inputs,
            pooling_field_names=ENCODER_EXPORT_POOLING_FIELDS,
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
    encoder graph consumes - stage 0 uses the unprefixed base inputs).

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
        prefix = "" if stage == 0 else f"serialized_pooling_{stage - 1}_"
        names += [prefix + field for field in _BLOCK_STAGE_META_FIELDS]
    return names


def build_seg_head_input_dynamic_axes(
    stage_count: int, dec_depths: Sequence[int]
) -> dict[str, dict[int, str]]:
    """Build dynamic axes for the split seg-head export inputs."""
    dynamic_axes = build_stage_feature_dynamic_axes(stage_count)
    dynamic_axes.update(build_pooling_cluster_dynamic_axes(stage_count))
    for stage in _block_stage_indices(dec_depths):
        if stage == 0:
            dynamic_axes["serialized_order"] = {1: "num_voxels"}
            dynamic_axes["serialized_inverse"] = {1: "num_voxels"}
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
                block stages use it pooled by their cumulative pooling depth
                (rounded up, see :func:`pooled_sparse_shape`).
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
                pooled_sparse_shape(sparse_shape, stage_depths[stage]),
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
            # Every stage, including the finest, receives its serialization order directly.
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


# ---------------------------------------------------------------------------------------
# Stage graph: the declaration the deployment framework derives the split export from.
# ---------------------------------------------------------------------------------------

SERIALIZE_STAGE = "serialize_points"
ENCODER_STAGE = "ptv3_encoder"
SEG_HEAD_STAGE = "ptv3_seg3d_head"
#: The seg head graph's outputs: the argmax labels and the softmax scores per voxel.
SEG_HEAD_OUTPUT_NAMES = ("pred_labels", "pred_probs")
DET_HEAD_STAGE = "ptv3_det3d_head"


def export_geometry(model: PTv3BaseModel) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sparse_shape, serialization_depth)`` from the model config alone (CPU tensors).

    ``serialization_depth`` is the bit length of the largest axis extent in grid cells.

    CPU is fine here: the serialize stage moves the depth to the batch device itself, and
    the export modules read ``sparse_shape`` as a Python list (:meth:`Point.sparsify`).
    """
    point_cloud_range = torch.tensor(model.point_cloud_range, dtype=torch.float32)
    axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / model.grid_size
    serialization_depth = bit_length_tensor(torch.max(axis_extents))
    sparse_shape = torch.round(axis_extents).to(dtype=torch.long)
    return sparse_shape, serialization_depth


def encoder_stage_input_names(num_poolings: int) -> list[str]:
    """The encoder graph's input names: the voxels, the level-0 serialization and the
    per-stage pooling metadata (without ``cluster``, which only the heads read)."""
    return [
        *ENCODER_EXPORT_INPUT_NAMES,
        *(
            f"serialized_pooling_{stage}_{field}"
            for stage in range(num_poolings)
            for field in ENCODER_EXPORT_POOLING_FIELDS
        ),
    ]


def serialize_stage_output_names(num_poolings: int) -> tuple[str, ...]:
    """The serialize glue's declared outputs: the union of what the encoder and the head
    graphs read — the encoder inputs, every pooling field (``cluster`` included), the
    per-stage clusters under the head graphs' names, and the det head's skip-level grid."""
    names = list(encoder_stage_input_names(num_poolings))
    names += [f"serialized_pooling_{stage}_cluster" for stage in range(num_poolings)]
    names += pooling_cluster_names(num_poolings + 1)
    names.append(f"point_grid_coord_{num_poolings - 1}")
    return tuple(names)


def build_serialize_stage(model: PTv3BaseModel) -> TorchStage:
    """Glue: serialize the voxel batch and precompute every pooling tensor.

    Reads the voxels the model's ``forward_inputs`` picks from the typed batch and produces
    the union of what the encoder and both head graphs read, under the names the deployed
    graphs have always used as inputs; unused names in the run state are free.
    """
    num_poolings = len(model.encoder.stride)
    _, serialization_depth = export_geometry(model)

    def serialize_points(state: RunState) -> dict[str, torch.Tensor]:
        if state.batch.batch_size() != 1:
            # The export modules are single-frame by contract (one offset, batch index 0
            # everywhere); a multi-frame batch would be merged into one frame silently.
            raise ValueError(
                f"The PTv3 deployment stage graph takes one frame per batch, got "
                f"{state.batch.batch_size()} (deploy with batch_size=1)."
            )
        inputs = model.forward_inputs(state.batch)
        voxels, num_points = inputs["voxels"], inputs["num_points"]
        model.validate_voxel_inputs(voxels, inputs["num_dropped_voxels"], inputs["time_lag_column"])
        encoder_inputs = model.build_encoder_inputs(voxels, num_points, inputs["voxel_coords"])
        depth = serialization_depth.to(state.device)
        point, _ = serialize_point_cloud_batch(encoder_inputs, model.EXPORT_ORDER, depth)
        metadata = build_serialized_pooling_metadata(
            point["grid_coord"],
            point["serialized_code"],
            point["serialized_order"],
            model.encoder.stride,
        )
        produced: dict[str, torch.Tensor] = {
            "voxels": voxels,
            "num_points_per_voxel": num_points,
            "grid_coord": encoder_inputs["grid_coord"],
            "serialized_order": point["serialized_order"],
            "serialized_inverse": point["serialized_inverse"],
        }
        for stage_index, meta in enumerate(metadata):
            for field in SERIALIZED_POOLING_FIELDS:
                produced[f"serialized_pooling_{stage_index}_{field}"] = getattr(meta, field)
            produced[f"pooling_cluster_{stage_index}"] = meta.cluster
        # The detection head reads the skip stage's grid coordinates.
        skip_stage = num_poolings - 1
        produced[f"point_grid_coord_{skip_stage}"] = (
            encoder_inputs["grid_coord"] if skip_stage == 0 else metadata[skip_stage - 1].grid_coord
        )
        return produced

    return TorchStage(
        SERIALIZE_STAGE,
        run=serialize_points,
        inputs=(),  # reads only the batch
        outputs=serialize_stage_output_names(num_poolings),
    )


def build_encoder_stage(model: PTv3BaseModel) -> GraphStage:
    """The shared ``ptv3_encoder`` graph: voxels and serialization in, per-stage point
    features out.

    The derived ``ExportSpec`` supports ONNX and TensorRT; the graph's autoware:: plugin
    ops need the autoware TensorRT plugin library when the engine is built.
    """
    num_poolings = len(model.encoder.stride)
    stage_count = num_poolings + 1
    sparse_shape, serialization_depth = export_geometry(model)
    module = _PTv3EncoderExportModule(
        encoder=model._prepare_encoder_export(),
        voxel_encoder=model.voxel_encoder,
        sparse_shape=sparse_shape,
        serialized_depth=serialization_depth,
    ).eval()
    input_names = encoder_stage_input_names(num_poolings)
    return GraphStage(
        ENCODER_STAGE,
        module=module,
        inputs=tuple(input_names),
        outputs=tuple(stage_feature_names(stage_count)),
        # Every tensor is indexed by a voxel count: the axes belong to the graph.
        onnx_dynamic_axes=build_ptv3_encoder_dynamic_axes(input_names, stage_count),
        # The graph carries autoware:: plugin ops (the cpe sparse convolutions). TensorRT
        # executes them through the autoware TensorRT plugin library; ONNX Runtime has no
        # implementation, so only that backend falls back to torch.
        torch_fallback_backends=(Backend.ONNX,),
    )


def build_seg_head_stage(model: PTv3BaseModel, seg3d_head: nn.Module) -> GraphStage:
    """The ``ptv3_seg3d_head`` graph over the encoder's per-stage features, emitting
    :data:`SEG_HEAD_OUTPUT_NAMES`."""
    stage_count = len(model.encoder.stride) + 1
    sparse_shape, _ = export_geometry(model)
    head = seg3d_head.prepare_for_export(model.EXPORT_ORDER)
    module = _PTv3SegHeadExportModule(
        head, stage_count, sparse_shape, tuple(model.encoder.stride)
    ).eval()
    output_names = SEG_HEAD_OUTPUT_NAMES
    dynamic_axes = build_seg_head_input_dynamic_axes(stage_count, head.dec_depths)
    dynamic_axes.update(build_point_feature_dynamic_axes(output_names))
    return GraphStage(
        SEG_HEAD_STAGE,
        module=module,
        inputs=tuple(seg_head_export_input_names(stage_count, head.dec_depths)),
        # The graph emits the argmax and the softmax scores; build_deployed_eval_output
        # scores them as they are (no second softmax).
        outputs=output_names,
        onnx_dynamic_axes=dynamic_axes,
        torch_fallback_backends=(Backend.ONNX,),
    )


def build_ptv3_stages(model: PTv3BaseModel, *head_stages: GraphStage) -> tuple[Stage, ...]:
    """``serialize_points (torch) -> ptv3_encoder (graph) -> <head> (graph) ...``.

    Every head graph reads the encoder's per-stage features (and the serialize glue's
    tensors) and nothing from another head, so the heads are independent graphs and all
    of their outputs are model outputs, in the order the heads are given.
    """
    return (build_serialize_stage(model), build_encoder_stage(model), *head_stages)
