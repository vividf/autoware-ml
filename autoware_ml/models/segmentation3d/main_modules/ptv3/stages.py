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

"""PTv3 deployment stage graphs — declared ahead of the interface migration.

Both PTv3 task forms share one front half and differ only in the head graph:

    serialize_points (torch)  ->  encoder (graph)  ->  seg3d_head (graph)   [segmentation]
    serialize_points (torch)  ->  encoder (graph)  ->  det3d_head (graph)   [detection]

The export modules and generated name rules (``point_feat_{i}``,
``serialized_pooling_{i}_{field}``, ``pooling_cluster_{i}``) live next door in
:mod:`.export_modules`; the legacy ExportSpec path re-imports the same toolbox
until it is deleted (Q5).
Stage names keep the legacy artifact names (``encoder.onnx`` / ``seg3d_head.onnx``
/ ``det3d_head.onnx``).

Contract with the interface migration:

- **Submodules / attributes**: ``encoder`` (with ``.stride``), ``point_cloud_range``,
  ``grid_size``, ``EXPORT_ORDER``, ``_prepare_encoder_export()``; segmentation adds
  ``seg3d_head`` (with ``.dec_depths`` and ``prepare_for_export(order)``) and
  ``get_export_output_names()``; detection adds ``bev_neck``, ``bbox_head``
  (with ``prepare_for_export()``) and ``export_output_names``.
- **Batch inputs**: ``MultiTaskBatchInputs`` grows a ``points_data`` mapping with
  ``coord`` / ``feat`` / ``grid_coord`` / ``offset`` (the preprocessed point batch
  the legacy export consumed) — the glue stage reads it.
- **Typed outputs**: the ``output_fields`` drafts below map ONNX outputs onto
  same-named fields; final field names land with the outputs-dataclass extension.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch

from autoware_ml.deployment.stages import GraphStage, Stage, StageContext, TorchStage
from autoware_ml.types.backend import Backend
from autoware_ml.models.segmentation3d.main_modules.ptv3.export_modules import (
    ENCODER_EXPORT_POOLING_FIELDS,
    _PTv3EncoderExportModule,
    _PTv3SegHeadExportModule,
    build_point_feature_dynamic_axes,
    build_ptv3_encoder_dynamic_axes,
    build_seg_head_input_dynamic_axes,
    build_serialized_pooling_metadata,
    seg_head_export_input_names,
    stage_feature_names,
)
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch

# Stage / artifact names (legacy artifact ABI: <name>.onnx).
SERIALIZE_STAGE = "serialize_points"
ENCODER_STAGE = "encoder"
SEG_HEAD_STAGE = "seg3d_head"
DET_HEAD_STAGE = "det3d_head"


def _export_geometry(model: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse shape and serialization depth — pure model-config facts (CPU tensors).

    Mirrors ``PTv3BaseModel._compute_export_geometry`` without needing a batch
    (that method only borrowed the batch's device); the export module registers
    both as buffers, so they follow the module to its execution device.
    """
    from autoware_ml.utils.point_cloud.structures import bit_length_tensor

    point_cloud_range = torch.tensor(model.point_cloud_range, dtype=torch.float32)
    axis_extents = (point_cloud_range[3:] - point_cloud_range[:3]) / model.grid_size
    serialization_depth = bit_length_tensor(torch.max(axis_extents))
    sparse_shape = torch.round(axis_extents).to(dtype=torch.long)
    return sparse_shape, serialization_depth


def _pooling_count(model: Any) -> int:
    """Number of pooling stages (one metadata block per encoder stride entry)."""
    return len(model.encoder.stride)


def encoder_input_names(num_poolings: int) -> list[str]:
    """The encoder graph's ONNX input names (the legacy export name rule)."""
    return [
        "grid_coord",
        "feat",
        "serialized_code",
        *(
            f"serialized_pooling_{stage}_{field}"
            for stage in range(num_poolings)
            for field in ENCODER_EXPORT_POOLING_FIELDS
        ),
    ]


def serialize_output_names(num_poolings: int) -> set[str]:
    """Every context tensor the serialize glue produces (for declaration checks)."""
    names = set(encoder_input_names(num_poolings))
    names.update(f"pooling_cluster_{stage}" for stage in range(num_poolings))
    skip_stage = num_poolings - 1  # stage_count - 2 with stage_count = num_poolings + 1
    names.add(f"point_grid_coord_{skip_stage}")
    return names


def _serialize_stage(model: Any) -> TorchStage:
    """Glue: serialize the point batch and precompute every pooling tensor.

    Produces the union of what the encoder and both head graphs read; unused
    names in the context bag are free.
    """
    num_poolings = _pooling_count(model)
    _, serialization_depth = _export_geometry(model)

    def serialize_points(context: StageContext) -> Mapping[str, torch.Tensor]:
        points = context.batch_inputs.points_data
        if points is None:
            raise ValueError("MultiTaskBatchInputs must carry points_data for PTv3.")
        depth = serialization_depth.to(context.device)
        point, (grid_coord, feat, _depth, serialized_code) = serialize_point_cloud_batch(
            points.as_point_dict(), model.EXPORT_ORDER, depth
        )
        metadata = build_serialized_pooling_metadata(
            point["grid_coord"],
            point["serialized_code"],
            point["serialized_order"],
            model.encoder.stride,
        )
        produced: dict[str, torch.Tensor] = {
            "grid_coord": grid_coord,
            "feat": feat,
            "serialized_code": serialized_code,
        }
        for stage_index, meta in enumerate(metadata):
            for field in ENCODER_EXPORT_POOLING_FIELDS:
                produced[f"serialized_pooling_{stage_index}_{field}"] = getattr(meta, field)
            produced[f"pooling_cluster_{stage_index}"] = meta.cluster
        skip_stage = num_poolings - 1
        produced[f"point_grid_coord_{skip_stage}"] = (
            grid_coord if skip_stage == 0 else metadata[skip_stage - 1].grid_coord
        )
        return produced

    return TorchStage(SERIALIZE_STAGE, run=serialize_points)


def _encoder_stage(model: Any) -> GraphStage:
    """The shared encoder graph: serialized batch in, per-stage point features out."""
    num_poolings = _pooling_count(model)
    sparse_shape, serialization_depth = _export_geometry(model)
    module = _PTv3EncoderExportModule(
        encoder=model._prepare_encoder_export(),
        sparse_shape=sparse_shape,
        serialized_depth=serialization_depth,
        # The stage declares one input per field in ENCODER_EXPORT_POOLING_FIELDS, so the
        # module must unpack them with the same field list (it excludes `cluster`, which
        # only the head graphs consume).
        pooling_field_names=ENCODER_EXPORT_POOLING_FIELDS,
    ).eval()
    input_names = encoder_input_names(num_poolings)
    return GraphStage(
        ENCODER_STAGE,
        module=module,
        inputs=tuple(input_names),
        outputs=tuple(stage_feature_names(num_poolings + 1)),
        # Every tensor here is indexed by a point count, so the axes belong to the graph
        # rather than to a configuration.
        onnx_dynamic_axes=build_ptv3_encoder_dynamic_axes(input_names, num_poolings + 1),
        # The graph carries autoware:: plugin ops (sparse convolution, argsort,
        # segment_csr). TensorRT executes them from deploy.tensorrt.plugin_libraries;
        # ONNX Runtime has no implementation, so only that backend falls back to torch.
        torch_fallback_backends=(Backend.ONNX,),
    )


def build_ptv3_seg_stages(model: Any) -> tuple[Stage, ...]:
    """Declare the PTv3 segmentation stage graph over ``model``'s submodules."""
    num_poolings = _pooling_count(model)
    stage_count = num_poolings + 1
    sparse_shape, _ = _export_geometry(model)
    head = model.seg3d_head.prepare_for_export(model.EXPORT_ORDER)
    head_module = _PTv3SegHeadExportModule(
        head, stage_count, sparse_shape, tuple(model.encoder.stride)
    ).eval()
    output_names = tuple(model.get_export_output_names())
    head_dynamic_axes = build_seg_head_input_dynamic_axes(stage_count, head.dec_depths)
    head_dynamic_axes.update(build_point_feature_dynamic_axes(output_names))
    return (
        _serialize_stage(model),
        _encoder_stage(model),
        GraphStage(
            SEG_HEAD_STAGE,
            module=head_module,
            inputs=tuple(seg_head_export_input_names(stage_count, head.dec_depths)),
            outputs=output_names,
            onnx_dynamic_axes=head_dynamic_axes,
            # Same plugin ops as the encoder graph; see _encoder_stage.
            torch_fallback_backends=(Backend.ONNX,),
            # Field names are a draft until the outputs dataclass gains segmentation slots.
            output_fields=tuple((name, name) for name in output_names),
        ),
    )


def build_ptv3_det_stages(model: Any) -> tuple[Stage, ...]:
    """Declare the PTv3 detection stage graph over ``model``'s submodules."""
    from autoware_ml.models.detection3d.ptv3 import (
        _PTv3DetHeadExportModule,
        det_head_export_input_names,
    )

    num_poolings = _pooling_count(model)
    stage_count = num_poolings + 1
    output_names = tuple(model.get_export_output_names())
    head_module = _PTv3DetHeadExportModule(
        bev_neck=deepcopy(model.bev_neck).eval(),
        bbox_head=model.bbox_head.prepare_for_export(),
        output_names=output_names,
    ).eval()
    return (
        _serialize_stage(model),
        _encoder_stage(model),
        GraphStage(
            DET_HEAD_STAGE,
            module=head_module,
            inputs=tuple(det_head_export_input_names(stage_count)),
            outputs=output_names,
            # Field names are a draft until the outputs dataclass gains PTv3-det slots.
            output_fields=tuple((name, name) for name in output_names),
        ),
    )
