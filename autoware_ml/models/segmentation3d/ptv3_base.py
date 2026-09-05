"""Abstract base class and shared export modules for PTv3-based task models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn
from torch.onnx.operators import shape_as_tensor

from autoware_ml.models.base import BaseModel
from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    PointTransformerV3Encoder,
    SerializedPooling,
    SerializedPoolingMeta,
    _pooling_depth,
    build_serialized_pooling_meta,
    collect_encoder_stage_points,
)
from autoware_ml.ops.indexing.operators import argsort
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import (
    Point,
    bit_length_tensor,
    invert_permutation,
    serialize_point_cloud_batch,
)


# The export toolbox moved next to the stage-graph deployment (main_modules/ptv3);
# the legacy ExportSpec path below builds on it through these re-imports until Q5.
from autoware_ml.models.segmentation3d.main_modules.ptv3.export_modules import (  # noqa: F401
    ENCODER_EXPORT_POOLING_FIELDS,
    SERIALIZED_POOLING_FIELDS,
    SERIALIZED_POOLING_INPUT_SIZED_FIELDS,
    SERIALIZED_POOLING_ORDER_FIELDS,
    SERIALIZED_POOLING_OUTPUT_PLUS_ONE_FIELDS,
    PTv3EncoderExportBase,
    _BLOCK_STAGE_META_FIELDS,
    _PTv3EncoderExportModule,
    _PTv3SegHeadExportModule,
    _block_stage_indices,
    _run_ptv3_encoder_export,
    _serialized_pooling_dynamic_axis,
    build_point_feature_dynamic_axes,
    build_pooling_cluster_dynamic_axes,
    build_ptv3_encoder_dynamic_axes,
    build_ptv3_input_dynamic_axes,
    build_seg_head_export_args,
    build_seg_head_input_dynamic_axes,
    build_serialized_pooling_metadata,
    build_stage_feature_dynamic_axes,
    flatten_serialized_pooling_inputs,
    link_stage_points,
    make_serialized_pooling_from_flat_inputs,
    pooling_cluster_names,
    seg_head_export_input_names,
    split_block_parameters,
    stage_feature_names,
    stage_voxel_axis_name,
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



class PTv3BaseModel(BaseModel):
    """Abstract base class for all PTv3 task models.

    Provides shared encoder management, export geometry computation, and
    export helpers. Detection and segmentation subclasses inherit from this
    class (potentially with additional base classes via MRO).
    """

    EXPORT_ORDER = ("z", "z-trans")
    EXPORT_SUPPORTED_STAGES = frozenset({"onnx"})

    def __init__(
        self,
        encoder: PointTransformerV3Encoder,
        grid_size: float | None,
        point_cloud_range: Sequence[float] | None,
        freeze_encoder: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the PTv3 base model.

        Args:
            encoder: PTv3 encoder module.
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
        self.encoder = encoder
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

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int | None:
        """Infer the effective sample batch size for logging.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Sample batch size when it can be inferred, otherwise ``None``.
        """
        if "gt_boxes" in batch_inputs_dict:
            return len(batch_inputs_dict["gt_boxes"])
        if "offset" in batch_inputs_dict:
            return int(batch_inputs_dict["offset"].numel())
        return super().get_log_batch_size(batch_inputs_dict)

    def _compute_export_geometry(
        self, batch_inputs_dict: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute sparse shape and serialization depth for export.

        Args:
            batch_inputs_dict: Preprocessed batch containing at least
                ``coord`` (used for device inference).

        Returns:
            ``(sparse_shape, serialization_depth)`` as long tensors on the
            same device as ``batch_inputs_dict["coord"]``.
        """
        device = batch_inputs_dict["coord"].device
        point_cloud_range = torch.tensor(self.point_cloud_range, dtype=torch.float32, device=device)
        axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / self.grid_size
        serialization_depth = bit_length_tensor(torch.max(axis_extents))
        sparse_shape = torch.round(axis_extents).to(dtype=torch.long)
        return sparse_shape, serialization_depth

    def _prepare_encoder_export(self) -> PointTransformerV3Encoder:
        """Return an export-ready copy of the encoder.

        Returns:
            Copy of the encoder prepared for ONNX export with the configured
            export order.
        """
        return self.encoder.prepare_for_export(self.EXPORT_ORDER)



class PTv3ExportContext:
    """Shared front half of every split PTv3 export.

    Built once per export: the serialized batch, per-stage pooling metadata,
    the export-ready encoder module, and its per-stage features. Artifact
    spec builders pair this context with their own input-name rule.
    """

    sparse_shape: torch.Tensor
    serialization_depth: torch.Tensor
    grid_coord: torch.Tensor
    feat: torch.Tensor
    serialized_code: torch.Tensor
    strides: tuple[int, ...]
    pooling_metadata: tuple[SerializedPoolingMeta, ...]
    serialized_pooling_inputs: tuple[torch.Tensor, ...]
    serialized_pooling_input_names: tuple[str, ...]
    encoder_module: nn.Module
    stage_feats: tuple[torch.Tensor, ...]

    @property
    def stage_count(self) -> int:
        return len(self.stage_feats)

    @property
    def encoder_input_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.grid_coord,
            self.feat,
            self.serialized_code,
            *self.serialized_pooling_inputs,
        )

    @property
    def encoder_input_names(self) -> list[str]:
        return ["grid_coord", "feat", "serialized_code", *self.serialized_pooling_input_names]


def build_ptv3_export_context(
    model: "PTv3BaseModel", batch: Mapping[str, torch.Tensor]
) -> PTv3ExportContext:
    """Serialize the batch, precompute pooling metadata, and run the encoder once."""
    sparse_shape, serialization_depth = model._compute_export_geometry(batch)
    point, input_args = serialize_point_cloud_batch(batch, model.EXPORT_ORDER, serialization_depth)
    pooling_metadata = build_serialized_pooling_metadata(
        point["grid_coord"],
        point["serialized_code"],
        point["serialized_order"],
        model.encoder.stride,
    )
    serialized_pooling_inputs, serialized_pooling_input_names = flatten_serialized_pooling_inputs(
        pooling_metadata, ENCODER_EXPORT_POOLING_FIELDS
    )
    encoder_module = _PTv3EncoderExportModule(
        encoder=model._prepare_encoder_export(),
        sparse_shape=sparse_shape,
        serialized_depth=serialization_depth,
        pooling_field_names=ENCODER_EXPORT_POOLING_FIELDS,
    ).eval()
    with torch.no_grad():
        stage_feats = encoder_module(
            input_args[0], input_args[1], input_args[3], *serialized_pooling_inputs
        )
    return PTv3ExportContext(
        sparse_shape=sparse_shape,
        serialization_depth=serialization_depth,
        grid_coord=input_args[0],
        feat=input_args[1],
        serialized_code=input_args[3],
        strides=tuple(model.encoder.stride),
        pooling_metadata=tuple(pooling_metadata),
        serialized_pooling_inputs=tuple(serialized_pooling_inputs),
        serialized_pooling_input_names=tuple(serialized_pooling_input_names),
        encoder_module=encoder_module,
        stage_feats=tuple(stage_feats),
    )


# TODO(vividf): legacy ExportSpec builders — migrate the PTv3 family to MultiTaskBaseModel.build_stages() (stage-graph export).
@dataclass(frozen=True)
class MonolithicExportInputs:
    """Encoder-side inputs shared by every single-graph PTv3 export."""

    sparse_shape: torch.Tensor
    serialization_depth: torch.Tensor
    args: tuple[torch.Tensor, ...]
    input_names: list[str]


def build_monolithic_export_inputs(
    model: "PTv3BaseModel", batch: Mapping[str, torch.Tensor]
) -> MonolithicExportInputs:
    """Serialize a batch and derive the encoder inputs for a single-graph export.

    Single-graph exports keep the whole model in one engine, so unlike the split
    encoder graph they do consume ``cluster`` for head-side unpooling.

    Args:
        model: Task model being exported.
        batch: Preprocessed batch with ``coord``, ``feat``, ``grid_coord``, and
            ``offset``.

    Returns:
        Baked geometry and the sample inputs matching the declared input names.
    """
    sparse_shape, serialization_depth = model._compute_export_geometry(batch)
    point, input_args = serialize_point_cloud_batch(batch, model.EXPORT_ORDER, serialization_depth)
    serialized_pooling_inputs, serialized_pooling_input_names = flatten_serialized_pooling_inputs(
        build_serialized_pooling_metadata(
            point["grid_coord"],
            point["serialized_code"],
            point["serialized_order"],
            model.encoder.stride,
        )
    )
    return MonolithicExportInputs(
        sparse_shape=sparse_shape,
        serialization_depth=serialization_depth,
        args=(input_args[0], input_args[1], input_args[3], *serialized_pooling_inputs),
        input_names=["grid_coord", "feat", "serialized_code", *serialized_pooling_input_names],
    )


def build_encoder_export_spec(context: PTv3ExportContext) -> "ExportSpec":
    """Build the shared per-stage-feature encoder export spec."""
    input_names = context.encoder_input_names
    return ExportSpec(
        module=context.encoder_module,
        args=context.encoder_input_args,
        input_param_names=input_names,
        output_names=stage_feature_names(context.stage_count),
        dynamic_axes=build_ptv3_encoder_dynamic_axes(input_names, context.stage_count),
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )


def build_seg_head_export_spec(
    context: PTv3ExportContext, seg3d_head: nn.Module, output_names: Sequence[str]
) -> "ExportSpec":
    """Build the segmentation-head export spec for any decoder configuration.

    Args:
        context: Shared export context.
        seg3d_head: Export-prepared decoder head copy.
        output_names: Ordered head output names.
    """
    module = _PTv3SegHeadExportModule(
        seg3d_head, context.stage_count, context.sparse_shape, context.strides
    ).eval()
    input_names = seg_head_export_input_names(context.stage_count, seg3d_head.dec_depths)
    dynamic_axes = build_seg_head_input_dynamic_axes(context.stage_count, seg3d_head.dec_depths)
    dynamic_axes.update(build_point_feature_dynamic_axes(output_names))
    return ExportSpec(
        module=module,
        args=build_seg_head_export_args(
            context.stage_feats,
            context.pooling_metadata,
            context.serialized_code,
            context.grid_coord,
            seg3d_head.dec_depths,
        ),
        input_param_names=input_names,
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        supported_stages=PTv3BaseModel.EXPORT_SUPPORTED_STAGES,
    )

