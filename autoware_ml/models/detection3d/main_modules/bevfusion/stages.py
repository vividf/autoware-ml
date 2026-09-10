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

"""BEVFusion (lidar-only) deployment stage graph.

Split form (the INT8 deployment line, mirroring AWML's
``bevfusion_split_int8_deployment`` artifacts):

    fetch_voxels (torch) -> precompute_rulebooks (torch) -> bevfusion_sparse (graph)
                                                          -> bevfusion_dense (graph)

Runtime ABI carried over from the AWML split artifacts:

- ``bevfusion_sparse``: ``voxels`` / ``coors`` / ``num_points_per_voxel`` in,
  ``lidar_bev`` out. Exported through :meth:`SparseEncoder.prepare_for_export`,
  which swaps the native spconv layers for the wrappers in
  :mod:`autoware_ml.ops.spconv`; their symbolics emit the runtime's libspconv
  ABI (``autoware::GetIndicePairsImplicitGemm`` / ``autoware::ImplicitGemm`` with
  the rulebook tensors as graph inputs). TensorRT executes those nodes through
  ``libautoware_tensorrt_plugins.so``, which the image builds and every deploy
  config lists in ``deploy.tensorrt.plugin_libraries``; ONNX Runtime has no
  implementation for them, so only that backend falls back to PyTorch here.
- ``bevfusion_dense``: ``lidar_bev`` in; ``bbox_pred`` / ``score`` / ``label_pred``
  out — the AWML dense graph DECODES in-graph (unlike CenterPoint's raw-map ABI),
  so the wrapper ends at the head's export decode.

Contract with the interface migration:

- **Submodules**: ``pts_voxel_encoder``, ``pts_middle_encoder`` (spconv),
  ``pts_backbone``, ``pts_neck``, ``bbox_head``.
- **Batch inputs**: ``MultiTaskBatchInputs.voxels_data`` provides voxel features,
  coordinates and per-voxel point counts (the ``_first_sample_voxel_inputs``
  tensors of the legacy export).
- **Backend evaluation decode**: because the graph decodes in-graph, a backend returns
  detections rather than head outputs, so the model implements ``assemble_predictions``
  (not ``assemble_outputs``) and reaches it through :func:`decode_packed_detections`.

``precompute_rulebooks`` generates the rulebooks of the encoder's four down-sampling
convolutions from the frame's voxel coordinates and hands them to the sparse graph as
inputs (``rulebook/<indice_key>/<slot>``), so the graph carries no
``GetIndicePairsImplicitGemm`` node with a data-dependent output shape — the
``[trainStation]`` host synchronizations TensorRT inserted for them are gone
(:mod:`autoware_ml.ops.spconv.rulebook`). The submanifold layers still generate theirs
in-graph; they have no dynamic shape to remove.

The sparse stage deploys in INT8 when the checkpoint carries calibrated sparse quantizers:
:func:`~autoware_ml.ops.spconv.onnx_int8.sparse_int8_transform` writes their scales into the
exported graph as plugin attributes and inputs (``precision=1`` + ``channel_scale`` /
``bias_scaled``), because a plugin op cannot absorb Q/DQ nodes. It is a no-op for an
un-quantized encoder, so one stage declaration serves both deployments.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from autoware_ml.deployment.stages import GraphStage, Stage, StageContext, TorchStage
from autoware_ml.models.detection3d.feature_extractors import LidarBEVFeatureExtractor
from autoware_ml.deployment.onnx.autocast import keep_topk_in_fp16
from autoware_ml.ops.spconv.onnx_fusion import fuse_sparse_graph
from autoware_ml.ops.spconv.onnx_int8 import sparse_int8_transform
from autoware_ml.ops.spconv.rulebook import (
    embed_rulebook_metadata,
    precompute_rulebooks,
    rulebook_dynamic_axes,
    rulebook_indice_data,
    rulebook_input_names,
)
from autoware_ml.types.backend import Backend

# Stage / artifact names (AWML split-deployment ABI: <name>.onnx / .engine).
FETCH_VOXELS_STAGE = "fetch_voxels"
PRECOMPUTE_RULEBOOKS_STAGE = "precompute_rulebooks"
SPARSE_STAGE = "bevfusion_sparse"
DENSE_STAGE = "bevfusion_dense"

# Context tensor names — the ONNX input/output names (AWML runtime ABI).
VOXELS = "voxels"
COORS = "coors"
NUM_POINTS_PER_VOXEL = "num_points_per_voxel"
LIDAR_BEV = "lidar_bev"
#: Dynamic-axis name shared by every per-voxel graph input.
VOXELS_NUM = "voxels_num"
BBOX_PRED = "bbox_pred"
SCORE = "score"
LABEL_PRED = "label_pred"

# ONNX output name -> the key ``decode_packed_detections`` reads it under. This graph
# emits detections, not head outputs, so the names simply carry through.
OUTPUT_FIELDS: tuple[tuple[str, str], ...] = (
    (BBOX_PRED, "bbox_pred"),
    (SCORE, "score"),
    (LABEL_PRED, "label_pred"),
)


class BEVFusionSparseExportWrapper(nn.Module):
    """Voxel inputs -> dense lidar BEV features (VFE + spconv middle encoder).

    Single-sample graph in the runtime layout: ``coors`` is ``(z, y, x)`` without a
    batch column (the runtime voxelizes with spconv's Point2Voxel), so a zero batch
    column is prepended — the same adaptation the legacy ``_forward_export`` does.
    """

    def __init__(self, voxel_encoder: nn.Module, middle_encoder: nn.Module) -> None:
        super().__init__()
        # Export-ready deep copy: native spconv layers swapped for the wrappers in
        # autoware_ml.ops.spconv, whose symbolics emit the runtime's
        # autoware::GetIndicePairsImplicitGemm / autoware::ImplicitGemm nodes.
        self.extractor = LidarBEVFeatureExtractor(
            pts_voxel_encoder=voxel_encoder,
            pts_middle_encoder=(
                middle_encoder.prepare_for_export()
                if hasattr(middle_encoder, "prepare_for_export")
                else middle_encoder
            ),
            pts_backbone=None,
            pts_neck=None,
        )
        # The down-sampling layers whose rulebooks arrive as graph inputs, in the order the
        # inputs are declared (rulebook_input_names). Read off the export copy so the two
        # agree by construction.
        self.rulebook_stages = self.extractor.pts_middle_encoder.downsample_stages()

    def forward(
        self,
        voxels: torch.Tensor,
        coors: torch.Tensor,
        num_points_per_voxel: torch.Tensor,
        *rulebooks: torch.Tensor,
    ) -> torch.Tensor:
        names = rulebook_input_names(self.rulebook_stages)
        if len(rulebooks) != len(names):
            raise ValueError(
                f"BEVFusion sparse graph expects {len(names)} rulebook input(s) "
                f"({len(self.rulebook_stages)} down-sampling stage(s) x {len(names) // max(len(self.rulebook_stages), 1)}), "
                f"got {len(rulebooks)}."
            )
        tensors = dict(zip(names, rulebooks))
        precomputed = {
            stage.indice_key: rulebook_indice_data(stage, tensors) for stage in self.rulebook_stages
        }
        return self.extractor(
            voxels,
            num_points_per_voxel,
            _with_batch_column(coors),
            batch_size=1,
            precomputed_rulebooks=precomputed,
        )


class BEVFusionDenseExportWrapper(nn.Module):
    """Lidar BEV features -> packed runtime detections (backbone + neck + head).

    The output packing mirrors the legacy ``_export_detection_outputs`` (the
    runtime ABI): raw regression channels concatenated into ``bbox_pred`` plus the
    fused ``score`` and ``label_pred`` — the runtime decodes and NMS-filters
    itself, so no metric-space decoding happens in the graph.
    """

    def __init__(self, backbone: nn.Module, neck: nn.Module, bbox_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        # Export-ready deep copy: decoder attention swapped for the exportable
        # equivalent (torch.onnx cannot trace nn.MultiheadAttention faithfully).
        self.bbox_head = (
            bbox_head.prepare_for_export()
            if hasattr(bbox_head, "prepare_for_export")
            else bbox_head
        )

    def forward(self, lidar_bev: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.bbox_head(self.neck(self.backbone(lidar_bev)))
        num_proposals = self.bbox_head.num_proposals
        query_labels = outputs["query_labels"]
        heatmap = outputs["heatmap"][..., -num_proposals:].sigmoid()
        one_hot = (
            F.one_hot(query_labels, num_classes=self.bbox_head.num_classes)
            .permute(0, 2, 1)
            .to(heatmap.dtype)
        )
        score = (heatmap * outputs["query_heatmap_score"] * one_hot)[0].max(dim=0).values
        if outputs.get("vel") is None:
            raise ValueError("BEVFusion export requires a velocity branch in the detection head.")
        bbox_pred = torch.cat(
            [
                outputs[key][0, :, -num_proposals:]
                for key in ("center", "height", "dim", "rot", "vel")
            ],
            dim=0,
        )
        return bbox_pred, score, query_labels[0]


def _with_batch_column(coors: torch.Tensor) -> torch.Tensor:
    """``[z, y, x]`` graph coordinates -> the ``[batch, z, y, x]`` the encoder takes (batch 0).

    The graph is single-sample and its ``coors`` input carries no batch column (the runtime
    voxelizes with spconv's Point2Voxel); the encoder and the rulebook precompute have to
    prepend the same column, so it lives in one place.
    """
    batch_column = torch.zeros((coors.shape[0], 1), dtype=coors.dtype, device=coors.device)
    return torch.cat((batch_column, coors), dim=1)


def build_bevfusion_lidar_stages(model: Any) -> tuple[Stage, ...]:
    """Declare the lidar-only BEVFusion split stage graph over ``model``'s submodules."""
    encoder = model.pts_middle_encoder
    rulebook_stages = encoder.downsample_stages()

    def fetch_voxels(context: StageContext) -> Mapping[str, torch.Tensor]:
        voxels_data = context.batch_inputs.voxels_data
        if voxels_data is None:
            raise ValueError("MultiTaskBatchInputs must contain voxels_data for BEVFusion.")
        # Single-sample export graph: keep the first sample's voxels only. Deployment
        # runs batch_size=1 (predict_dataloader), and taking the first sample of a larger
        # batch would drop the rest from every metric without a word — so say it here
        # rather than relying on a setting in another file.
        batch_size = int(voxels_data.batch_indices.max().item()) + 1
        if batch_size != 1:
            raise ValueError(
                f"BEVFusion's export stage graph is single-sample, but the batch holds "
                f"{batch_size} samples. Set predict_dataloader.batch_size=1 for deploy."
            )
        first_sample = voxels_data.batch_indices == 0
        return {
            VOXELS: voxels_data.voxels[first_sample],
            COORS: voxels_data.coords[first_sample].int().contiguous(),
            NUM_POINTS_PER_VOXEL: voxels_data.num_points[first_sample].int(),
        }

    def precompute_rulebooks_stage(context: StageContext) -> Mapping[str, torch.Tensor]:
        # The same coordinates the graph's convolutions will see, so the rulebooks are the
        # ones its ImplicitGemm nodes were exported against; sorting follows the encoder's
        # export setting for the same reason.
        coords = encoder.conv_coords(_with_batch_column(context[COORS]))
        return precompute_rulebooks(coords, 1, rulebook_stages, do_sort=encoder.export_do_sort)

    return (
        TorchStage(FETCH_VOXELS_STAGE, run=fetch_voxels),
        TorchStage(PRECOMPUTE_RULEBOOKS_STAGE, run=precompute_rulebooks_stage),
        GraphStage(
            SPARSE_STAGE,
            module=BEVFusionSparseExportWrapper(model.pts_voxel_encoder, model.pts_middle_encoder),
            inputs=(VOXELS, COORS, NUM_POINTS_PER_VOXEL, *rulebook_input_names(rulebook_stages)),
            outputs=(LIDAR_BEV,),
            # Every input is indexed by a voxel count that varies per frame, so the axes
            # belong to the graph rather than to a configuration.
            onnx_dynamic_axes={
                VOXELS: {0: VOXELS_NUM},
                COORS: {0: VOXELS_NUM},
                NUM_POINTS_PER_VOXEL: {0: VOXELS_NUM},
                **rulebook_dynamic_axes(rulebook_stages),
            },
            # The exported graph carries autoware::GetIndicePairsImplicitGemm /
            # autoware::ImplicitGemm custom ops. TensorRT executes them through
            # libautoware_tensorrt_plugins.so (deploy.tensorrt.plugin_libraries); ONNX
            # Runtime has no implementation at all, so only that backend falls back.
            torch_fallback_backends=(Backend.ONNX,),
            # TensorRT cannot fuse a standard operator into a plugin node, so the traced
            # bias adds and block ReLUs are folded into the plugin's own bias input and
            # act_type instead.
            # Order matters: the INT8 pass reads the bias out of the node's sixth input, so
            # it has to run after the fusion that puts it there. Both run after the
            # precision pass, which is what leaves the new FP32 scale initializers FP32
            # while the filters and features are already FP16 — the plugin's INT8 contract.
            onnx_transforms=(
                fuse_sparse_graph,
                partial(sparse_int8_transform, encoder=encoder),
                # The runtime regenerates the rulebooks from this geometry; it is a fact of
                # the exported graph, so it travels inside the file.
                partial(
                    embed_rulebook_metadata,
                    stages=rulebook_stages,
                    coors_permutation=encoder.coors_permutation,
                ),
            ),
        ),
        GraphStage(
            DENSE_STAGE,
            module=BEVFusionDenseExportWrapper(model.pts_backbone, model.pts_neck, model.bbox_head),
            inputs=(LIDAR_BEV,),
            outputs=(BBOX_PRED, SCORE, LABEL_PRED),
            output_fields=OUTPUT_FIELDS,
            # The proposal TopK ranks FP16 scores directly instead of an FP32 copy of the
            # whole flattened heatmap (measured 0.81 -> 0.45 ms). Near-ties may reorder,
            # which this model already declares (verification_caveat); the gate is mAP.
            onnx_transforms=(keep_topk_in_fp16,),
        ),
    )


def decode_packed_detections(
    bbox_head: Any, outputs: Mapping[str, torch.Tensor]
) -> list[dict[str, torch.Tensor]]:
    """Turn the deployed graph's packed tensors into the head's detection dicts.

    Only the unpacking is deployment-specific. The graph already fused the per-proposal
    score and picked the winning label, so the class scores are re-scattered into the
    per-class layout the head's post-processing expects, and that post-processing —
    metric-space decoding, score and range filtering, NMS — is the head's own
    :meth:`TransFusionHead.decode_detections`, not a copy of it.

    Args:
        bbox_head: The model's detection head, providing the post-processing.
        outputs: Field name -> tensor for the final stage (``bbox_pred`` / ``score`` /
            ``label_pred``), single sample.

    Returns:
        One ``{bboxes_3d, scores_3d, labels_3d}`` dict, matching the head's own return.
    """
    bbox_pred = outputs[BBOX_PRED]
    scores = outputs[SCORE]
    labels = outputs[LABEL_PRED].long()
    num_proposals = scores.shape[0]
    score_matrix = bbox_pred.new_zeros((1, bbox_head.num_classes, num_proposals))
    score_matrix[0, labels, torch.arange(num_proposals, device=bbox_pred.device)] = scores
    return bbox_head.decode_detections(
        score_matrix,
        bbox_pred[6:8].unsqueeze(0),
        bbox_pred[3:6].unsqueeze(0),
        bbox_pred[0:2].unsqueeze(0),
        bbox_pred[2:3].unsqueeze(0),
        bbox_pred[8:10].unsqueeze(0),
    )
