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

"""Precomputed down-sampling rulebooks: geometry, graph-input contract, and equivalence."""

from __future__ import annotations

import json

import onnx
from onnx import TensorProto, helper
import pytest
import torch

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

pytestmark = pytest.mark.skipif(not IS_SPCONV_AVAILABLE, reason="spconv is not installed")

if IS_SPCONV_AVAILABLE:
    from autoware_ml.models.detection3d.encoders.sparse import SparseEncoder
    from autoware_ml.ops.spconv.rulebook import (
        COORS_PERMUTATION_METADATA_KEY,
        RULEBOOK_SLOTS,
        STAGES_METADATA_KEY,
        embed_rulebook_metadata,
        precompute_rulebooks,
        rulebook_dynamic_axes,
        rulebook_indice_data,
        rulebook_input_names,
    )


def _encoder() -> "SparseEncoder":
    """Three stride-2 stages plus conv_out — the deployed encoder's shape, at toy size.

    z starts at 32 so that three stride-2 stages (32 -> 16 -> 8 -> 4) leave conv_out's
    (1, 1, 3) / stride-2 kernel a valid output extent; spconv rejects a zero-sized one.
    """
    # Per stage: basic blocks at the stage width, then a stride-2 block into the next width
    # (the deployed encoder's plan, 16/32/64/128, at toy widths). Stage 3 pads z by 0 like the
    # deployed one, so z goes 32 -> 16 -> 8 -> 3, and conv_out's (1, 1, 3) kernel leaves 1.
    return SparseEncoder(
        in_channels=4,
        sparse_shape=(64, 64, 32),
        base_channels=4,
        encoder_channels=((4, 4, 8), (8, 8, 8), (8, 8, 8), (8, 8)),
        encoder_paddings=((1, 1, 1), (1, 1, 1), (1, 1, (1, 1, 0)), (1, 1)),
        output_channels=8,
        dense_output_shapes=(8, 8, 1),
    ).eval()


def _voxels(encoder: "SparseEncoder", count: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    y, x, z = (torch.tensor(s) for s in encoder.sparse_shape)
    coords = torch.stack(
        [
            torch.zeros(count, dtype=torch.int32),
            torch.randint(0, int(z), (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, int(y), (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, int(x), (count,), generator=generator, dtype=torch.int32),
        ],
        dim=1,
    )
    coords = torch.unique(coords, dim=0)  # [batch, z, y, x], the framework's order
    features = torch.randn(coords.shape[0], 4, generator=generator)
    return features.to(device), coords.to(device)


def test_downsample_stages_cascade_the_spatial_shape_in_forward_order():
    stages = _encoder().downsample_stages()

    assert [stage.indice_key for stage in stages] == [
        "spconv1",
        "spconv2",
        "spconv3",
        "spconv_down2",
    ]
    assert [stage.spatial_shape for stage in stages] == [
        (64, 64, 32),
        (32, 32, 16),
        (16, 16, 8),
        (8, 8, 3),  # stage 3 pads z by 0: (8 - 3) // 2 + 1
    ]
    # conv_out strides only along z, with a (1, 1, 3) kernel: (3 - 3) // 2 + 1 = 1.
    assert stages[-1].kernel_size == (1, 1, 3) and stages[-1].kernel_volume == 3
    assert stages[-1].out_spatial_shape == (8, 8, 1)


def test_graph_input_names_and_axes_follow_the_runtime_contract():
    stages = _encoder().downsample_stages()

    names = rulebook_input_names(stages)
    assert len(names) == 4 * len(stages)
    assert names[:4] == (
        "rulebook/spconv1/out_indices",
        "rulebook/spconv1/pair_fwd",
        "rulebook/spconv1/pair_mask",
        "rulebook/spconv1/mask_argsort",
    )
    axes = rulebook_dynamic_axes(stages)
    # pair_fwd is [kernel_volume, count]; the other three are count-first.
    assert axes["rulebook/spconv1/pair_fwd"] == {1: "spconv1_num"}
    assert axes["rulebook/spconv1/out_indices"] == {0: "spconv1_num"}


def test_metadata_carries_the_geometry_and_the_coordinate_order(tmp_path):
    encoder = _encoder()
    stages = encoder.downsample_stages()
    path = tmp_path / "sparse.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    onnx.helper.set_model_props(model, {"unrelated": "kept"})
    onnx.save(model, str(path))

    embed_rulebook_metadata(path, stages=stages, coors_permutation=encoder.coors_permutation)

    props = {prop.key: prop.value for prop in onnx.load(str(path)).metadata_props}
    assert props["unrelated"] == "kept"
    recorded = json.loads(props[STAGES_METADATA_KEY])
    assert [entry["onnx_base"] for entry in recorded] == [stage.onnx_base for stage in stages]
    assert recorded[0] == {
        "onnx_base": "rulebook/spconv1",
        "ksize": [3, 3, 3],
        "stride": [2, 2, 2],
        "padding": [1, 1, 1],
        "dilation": [1, 1, 1],
        "spatial_shape": [64, 64, 32],
    }
    # coors is [z, y, x]; the convolutions run on [y, x, z].
    assert json.loads(props[COORS_PERMUTATION_METADATA_KEY]) == [1, 2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="spconv kernels need CUDA")
def test_precomputed_rulebooks_reproduce_the_in_graph_result():
    """Seeding the export encoder with precomputed rulebooks changes nothing but the graph.

    This is the whole correctness argument for removing the down-sampling
    GetIndicePairsImplicitGemm nodes: the layers consume the tensors the precompute made
    instead of generating their own, and the BEV map must be identical.
    """
    encoder = _encoder().cuda()
    exported = encoder.prepare_for_export()
    features, coords = _voxels(encoder, 3000, "cuda")
    stages = exported.downsample_stages()

    rulebooks = precompute_rulebooks(
        exported.conv_coords(coords), 1, stages, do_sort=exported.export_do_sort
    )
    seeded = {stage.indice_key: rulebook_indice_data(stage, rulebooks) for stage in stages}
    with torch.no_grad():
        reference = exported(features, coords, 1)
        with_rulebooks = exported(features, coords, 1, precomputed_rulebooks=seeded)

    assert set(rulebooks) == set(rulebook_input_names(stages))
    for stage in stages:
        count = rulebooks[stage.input_name("out_indices")].shape[0]
        assert rulebooks[stage.input_name("pair_fwd")].shape == (stage.kernel_volume, count)
        assert rulebooks[stage.input_name("pair_mask")].shape == (count, 1)
        assert rulebooks[stage.input_name("mask_argsort")].shape == (count,)
    torch.testing.assert_close(with_rulebooks, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="spconv kernels need CUDA")
def test_a_rulebook_for_the_wrong_geometry_is_rejected():
    encoder = _encoder().cuda()
    exported = encoder.prepare_for_export()
    features, coords = _voxels(encoder, 500, "cuda")
    stages = exported.downsample_stages()
    rulebooks = precompute_rulebooks(
        exported.conv_coords(coords), 1, stages, do_sort=exported.export_do_sort
    )
    # Hand stage 1's rulebook to stage 2's layer: different spatial shape.
    swapped = {stages[1].indice_key: rulebook_indice_data(stages[0], rulebooks)}

    with pytest.raises(ValueError, match="spatial_shape"):
        with torch.no_grad():
            exported(features, coords, 1, precomputed_rulebooks=swapped)


def test_every_slot_has_a_name():
    stage = _encoder().downsample_stages()[0]
    assert stage.input_names == tuple(stage.input_name(slot) for slot in RULEBOOK_SLOTS)
    with pytest.raises(ValueError, match="Unknown rulebook slot"):
        stage.input_name("pair_bwd")
