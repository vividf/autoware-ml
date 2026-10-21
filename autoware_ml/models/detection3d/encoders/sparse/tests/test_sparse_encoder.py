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

"""Unit tests for the sparse 3D voxel middle encoder.

The export conversion only rebuilds modules, so it runs wherever spconv is importable. The
forward pass runs through spconv CUDA kernels and is tested on the GPU only.
"""

from __future__ import annotations

from copy import deepcopy
import unittest

from spconv.pytorch import SparseConv3d as NativeSparseConv3d
from spconv.pytorch import SparseSequential
from spconv.pytorch import SubMConv3d as NativeSubMConv3d
import torch
from torch import nn

from autoware_ml.models.detection3d.encoders.sparse.sparse_encoder import (
    SparseEncoder,
    fold_sparse_encoder_bn,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.ops.spconv.sparse_conv import ExportableSparseConv3d, ExportableSubMConv3d


@unittest.skipUnless(IS_SPCONV_AVAILABLE, "SparseEncoder is built on spconv.")
class TestSparseEncoderPrepareForExport(unittest.TestCase):
    """Unit tests for the export conversion of the sparse encoder."""

    def setUp(self) -> None:
        """Set up a small sparse encoder on a 16x16x5 grid that collapses to a 2x2 BEV."""
        torch.manual_seed(0)
        self.encoder = SparseEncoder(
            in_channels=32,
            sparse_shape=[16, 16, 5],
            output_channels=16,
            dense_output_shapes=[2, 2, 1],
        )

    def test_bev_output_shape_is_the_dense_yx_grid(self) -> None:
        """Test that the branch-facing BEV shape is the ``(Y, X)`` part of the dense output."""
        self.assertEqual(self.encoder.bev_output_shape, (2, 2))

    def test_prepare_for_export_replaces_sparse_convolutions(self) -> None:
        """
        Test that the export copy swaps every native spconv layer for the exportable wrapper,
        shares no parameters with the training encoder, and is in eval mode.
        """
        self.encoder.train()

        export_encoder = self.encoder.prepare_for_export()

        self.assertIsNot(export_encoder, self.encoder)
        self.assertFalse(export_encoder.training)
        self.assertTrue(self.encoder.training)
        self.assertIsInstance(self.encoder.conv_input[0], NativeSubMConv3d)
        self.assertIsInstance(export_encoder.conv_input[0], ExportableSubMConv3d)
        self.assertFalse(
            any(
                isinstance(module, (NativeSubMConv3d, NativeSparseConv3d))
                for module in export_encoder.modules()
            )
        )
        self.assertTrue(
            any(isinstance(module, ExportableSparseConv3d) for module in export_encoder.modules())
        )
        self.assertTrue(
            any(isinstance(module, ExportableSubMConv3d) for module in export_encoder.modules())
        )

        # Weights are copied, not shared.
        torch.testing.assert_close(
            export_encoder.conv_input[0].weight, self.encoder.conv_input[0].weight
        )
        export_encoder.conv_input[0].weight.data.add_(1.0)
        self.assertFalse(
            torch.equal(export_encoder.conv_input[0].weight, self.encoder.conv_input[0].weight)
        )

    def test_prepare_for_export_passes_the_export_sort_flag_to_the_wrappers(self) -> None:
        """Test that every exportable wrapper carries the encoder's ``export_do_sort``."""
        encoder = SparseEncoder(
            in_channels=32,
            sparse_shape=[16, 16, 5],
            output_channels=16,
            dense_output_shapes=[2, 2, 1],
            export_do_sort=False,
        )

        export_encoder = encoder.prepare_for_export()

        wrappers = [
            module
            for module in export_encoder.modules()
            if isinstance(module, (ExportableSparseConv3d, ExportableSubMConv3d))
        ]
        self.assertTrue(wrappers)
        self.assertTrue(all(module.export_do_sort is False for module in wrappers))
        self.assertTrue(
            all(
                module.export_do_sort is True
                for module in self.encoder.prepare_for_export().modules()
                if isinstance(module, (ExportableSparseConv3d, ExportableSubMConv3d))
            )
        )

    def test_fold_sparse_encoder_bn_folds_every_pair_into_the_export_copy(self) -> None:
        """
        Test that the fold leaves the export copy BatchNorm-free with the folded weight and
        the bias the fold introduces, while the training encoder keeps its BN layers.
        """
        self.encoder.eval()
        export_encoder = deepcopy(self.encoder)
        batch_norm = self.encoder.conv_input[1]

        fused = fold_sparse_encoder_bn(export_encoder)
        export_encoder = export_encoder.prepare_for_export()

        self.assertGreater(fused, 0)
        self.assertIsInstance(export_encoder.conv_input[0], ExportableSubMConv3d)
        self.assertFalse(
            any(isinstance(module, nn.BatchNorm1d) for module in export_encoder.modules())
        )
        self.assertTrue(
            any(isinstance(module, nn.BatchNorm1d) for module in self.encoder.modules())
        )
        fold_scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
        # spconv kernels are (out_channels, *kernel, in_channels), so the per-output-channel
        # fold scale applies along dim 0.
        expected_weight = self.encoder.conv_input[0].weight * fold_scale.view(-1, 1, 1, 1, 1)
        torch.testing.assert_close(export_encoder.conv_input[0].weight, expected_weight)
        # The bias the fold introduces is beta - mu * fold_scale; a wrapper built without it
        # would leave the fold half-applied and silently shift every output.
        expected_bias = batch_norm.bias - batch_norm.running_mean * fold_scale
        self.assertIsNotNone(export_encoder.conv_input[0].bias)
        torch.testing.assert_close(
            export_encoder.conv_input[0].bias, expected_bias, atol=1e-6, rtol=0
        )

    def test_fold_sparse_encoder_bn_refuses_to_leave_a_batchnorm(self) -> None:
        """
        Test that "the deployed sparse graph has no BatchNormalization node" is enforced: a
        norm that is not adjacent to its convolution is not paired, and the fold says so
        instead of leaving the node for a plugin that cannot absorb it.
        """
        encoder = SparseEncoder(
            in_channels=4, sparse_shape=[16, 16, 9], dense_output_shapes=[2, 2, 2]
        )
        conv, norm = encoder.conv_input[0], encoder.conv_input[1]
        encoder.conv_input = SparseSequential(conv, nn.Identity(), norm, encoder.conv_input[2])

        with self.assertRaisesRegex(RuntimeError, "survived the export fold"):
            fold_sparse_encoder_bn(encoder.eval())

    def test_prepare_for_export_keeps_the_grid_configuration(self) -> None:
        """Test that the export copy keeps the sparse and dense grid shapes of the original."""
        export_encoder = self.encoder.prepare_for_export()

        self.assertEqual(export_encoder.sparse_shape, self.encoder.sparse_shape)
        self.assertEqual(export_encoder.dense_output_shapes, self.encoder.dense_output_shapes)
        self.assertEqual(export_encoder.bev_output_shape, self.encoder.bev_output_shape)


@unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(),
    "SparseEncoder runs through spconv CUDA kernels.",
)
class TestSparseEncoderForward(unittest.TestCase):
    """Integration tests for the sparse encoder forward pass on the GPU."""

    def setUp(self) -> None:
        """Set up the full-size TransFusion grid: 1440x1440x41 voxels down to a 180x180 BEV."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0")
        self.batch_size = 2
        self.num_voxels = 3000
        self.in_channels = 32
        self.output_channels = 128
        self.sparse_shape = (1440, 1440, 41)  # (Y, X, Z)
        self.dense_output_shapes = (180, 180, 2)  # (Y, X, Z)
        self.encoder = SparseEncoder(
            in_channels=self.in_channels,
            sparse_shape=list(self.sparse_shape),
            output_channels=self.output_channels,
            dense_output_shapes=list(self.dense_output_shapes),
        ).to(self.device)

    def _build_coords(self) -> torch.Tensor:
        """Build random ``[batch, z, y, x]`` voxel coordinates inside the sparse grid."""
        height, width, depth = self.sparse_shape
        coords = torch.zeros(self.num_voxels, 4, dtype=torch.int32, device=self.device)
        coords[:, 0] = torch.randint(0, self.batch_size, (self.num_voxels,), device=self.device)
        coords[:, 1] = torch.randint(0, depth, (self.num_voxels,), device=self.device)  # z
        coords[:, 2] = torch.randint(0, height, (self.num_voxels,), device=self.device)  # y
        coords[:, 3] = torch.randint(0, width, (self.num_voxels,), device=self.device)  # x
        return coords

    def test_forward_produces_dense_bev_with_gradients(self) -> None:
        """
        Test that voxel features become a finite dense ``(B, C * Z, Y, X)`` BEV map on the
        ``bev_output_shape`` grid, and that gradients flow back to the voxel features.
        """
        features = torch.randn(
            self.num_voxels, self.in_channels, device=self.device, requires_grad=True
        )

        bev = self.encoder(features, self._build_coords(), self.batch_size)

        height, width, depth = self.dense_output_shapes
        self.assertEqual(bev.shape, (self.batch_size, self.output_channels * depth, height, width))
        self.assertEqual((height, width), self.encoder.bev_output_shape)
        self.assertTrue(torch.isfinite(bev).all())
        bev.sum().backward()
        self.assertIsNotNone(features.grad)
        assert features.grad is not None
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_forward_keeps_samples_apart(self) -> None:
        """Test that a batch is scattered per sample, so a sample without voxels stays empty."""
        self.encoder.eval()
        features = torch.randn(self.num_voxels, self.in_channels, device=self.device)
        coords = self._build_coords()
        coords[:, 0] = 0  # every voxel belongs to the first sample

        with torch.no_grad():
            bev = self.encoder(features, coords, self.batch_size)

        self.assertGreater(bev[0].abs().sum().item(), 0.0)
        self.assertEqual(bev[1].abs().sum().item(), 0.0)

    def test_export_copy_matches_training_forward(self) -> None:
        """Test that the exportable encoder computes the same dense BEV as the native one."""
        self.encoder.eval()
        export_encoder = self.encoder.prepare_for_export()
        features = torch.randn(self.num_voxels, self.in_channels, device=self.device)
        coords = self._build_coords()

        with torch.no_grad():
            expected = self.encoder(features, coords, self.batch_size)
            exported = export_encoder(features, coords, self.batch_size)

        torch.testing.assert_close(exported, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
