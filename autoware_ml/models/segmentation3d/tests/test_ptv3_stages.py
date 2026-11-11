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

"""Unit tests for the PTv3 deployment graphs."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import onnx
import torch
from omegaconf import OmegaConf

from autoware_ml.deployment.stages import validate_graphs
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_seg3d_model,
    build_trans_model,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET3D_HEAD_STAGE,
    ENCODER_EXPORT_INPUT_NAMES,
    ENCODER_STAGE,
    SEG3D_HEAD_OUTPUT_NAMES,
    SEG3D_HEAD_STAGE,
    build_seg3d_head_stage,
    encoder_stage_input_names,
    serialize_points,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.types.backend import Backend
from autoware_ml.utils.deploy import export_to_onnx

REQUIRES_SPARSE_CUDA = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(), "PTv3 export needs spconv and CUDA"
)


@REQUIRES_SPARSE_CUDA
class TestSeg3dDeployment(unittest.TestCase):
    """The segmentation model's deployment graphs and the export specs derived from them."""

    def test_seg3d_declaration_names_the_runtime_modules_and_is_name_covered(self) -> None:
        model = build_seg3d_model().cuda().eval()
        graphs = validate_graphs(model.deploy_graphs())
        self.assertEqual([graph.name for graph in graphs], [ENCODER_STAGE, SEG3D_HEAD_STAGE])
        encoder, head = graphs
        self.assertEqual(encoder.inputs[:5], ENCODER_EXPORT_INPUT_NAMES)
        self.assertEqual(encoder.torch_fallback_backends, (Backend.ONNX,))
        self.assertFalse(any(name.endswith("_cluster") for name in encoder.inputs))
        self.assertEqual(head.outputs, ("pred_labels", "pred_probs"))
        # The serialized points hold every graph input the encoder does not produce.
        tensors = serialize_points(
            model, build_inputs(device=torch.device("cuda")), torch.device("cuda")
        )
        self.assertLessEqual(set(encoder.inputs), set(tensors))
        self.assertLessEqual(set(head.inputs), set(tensors) | set(encoder.outputs))

    def test_seg3d_export_specs_are_derived_and_keep_the_split_contract(self) -> None:
        model = build_seg3d_model().cuda().eval()
        batch = build_inputs(device=torch.device("cuda"))
        specs = model.build_export_specs(batch)
        self.assertEqual(list(specs), ["ptv3_encoder", "ptv3_seg3d_head"])
        encoder_spec = specs["ptv3_encoder"]
        self.assertEqual(
            encoder_spec.input_names, encoder_stage_input_names(len(model.encoder.stride))
        )
        self.assertIn("serialized_pooling_0_patch_order", encoder_spec.input_names)
        self.assertFalse(any("_cluster" in name for name in encoder_spec.input_names))
        self.assertEqual(encoder_spec.dynamic_axes["voxels"], {0: "num_voxels"})
        with torch.no_grad():
            stage_feats = encoder_spec.module(*encoder_spec.args)
            pred_labels, pred_probs = specs["ptv3_seg3d_head"].module(
                *specs["ptv3_seg3d_head"].args
            )
        num_voxels = batch.voxels_data.voxels.shape[0]
        self.assertEqual(len(stage_feats), len(model.encoder.stride) + 1)
        self.assertEqual(pred_labels.shape, (num_voxels,))
        self.assertEqual(pred_probs.shape[0], num_voxels)

    def test_derived_specs_export_onnx_with_the_declared_io_contract(self) -> None:
        # forward() is no numeric reference: it reshuffles the serialization orders and adapts
        # the attention windows, which the export copy pins.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tmp_path = Path(directory.name)
        model = build_seg3d_model().cuda().eval()
        batch = build_inputs(device=torch.device("cuda"))
        graphs = {graph.name: graph for graph in model.deploy_graphs()}
        onnx_cfg = OmegaConf.create(
            {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
        )
        for name, spec in model.build_export_specs(batch).items():
            path = tmp_path / f"{name}.onnx"
            export_to_onnx(spec, onnx_cfg, path)
            graph = onnx.load(path).graph
            self.assertEqual([tensor.name for tensor in graph.input], list(graphs[name].inputs))
            self.assertEqual([tensor.name for tensor in graph.output], list(graphs[name].outputs))
            for tensor in graph.input:
                axes = graphs[name].onnx_dynamic_axes.get(tensor.name, {})
                dims = tensor.type.tensor_type.shape.dim
                self.assertEqual({i: dims[i].dim_param for i in axes}, dict(axes), tensor.name)

    def test_seg3d_head_stage_emits_the_head_graph_names(self) -> None:
        model = build_seg3d_model().cuda().eval()
        stage = build_seg3d_head_stage(model, model.seg3d_head)
        self.assertEqual(stage.outputs, SEG3D_HEAD_OUTPUT_NAMES)
        self.assertEqual(SEG3D_HEAD_OUTPUT_NAMES, ("pred_labels", "pred_probs"))
        self.assertGreaterEqual(set(stage.onnx_dynamic_axes), set(SEG3D_HEAD_OUTPUT_NAMES))


@REQUIRES_SPARSE_CUDA
class TestDet3dDeployment(unittest.TestCase):
    """The detection model's deployment graphs and their specs."""

    def test_det3d_declaration_and_specs(self) -> None:
        model = build_trans_model().cuda().eval()
        graphs = validate_graphs(model.deploy_graphs())
        self.assertEqual([graph.name for graph in graphs], [ENCODER_STAGE, DET3D_HEAD_STAGE])
        batch = build_inputs(device=torch.device("cuda"))
        specs = model.build_export_specs(batch)
        self.assertEqual(list(specs), ["ptv3_encoder", "ptv3_det3d_head"])
        self.assertEqual(
            specs["ptv3_det3d_head"].output_names, list(model.bbox_head.export_output_names)
        )
