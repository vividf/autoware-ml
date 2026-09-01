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

"""The exported sparse graph's contract with the runtime plugin.

Every attribute here is read by ``libautoware_tensorrt_plugins.so`` at engine-build
time, and the plugin's tolerance for a missing one is temporary: it logs a "legacy
ONNX ... will be removed later" warning and assumes a default. A silent regression on
either side therefore surfaces as a vehicle-side failure, so the emitted attribute
names are pinned here rather than left to be noticed in a TensorRT log.

Tracks spconv 2.3.6 (``spconv-cu120`` in pyproject) and the plugin sources in
autoware_universe ``perception/autoware_tensorrt_plugins``.
"""

from __future__ import annotations

import inspect

from autoware_ml.ops.spconv import sparse_functional


def _emitted_attributes(symbolic) -> set[str]:
    """Attribute names a symbolic passes to ``g.op`` (``name_i=`` / ``_f=`` / ``_s=``)."""
    source = inspect.getsource(symbolic)
    body = source[source.index("g.op(") :]
    return {
        line.split("=")[0].strip().rsplit("_", 1)[0]
        for line in body.splitlines()
        if "=" in line and line.strip().split("=")[0].strip().endswith(("_i", "_f", "_s"))
    }


def test_get_indice_pairs_implicit_gemm_emits_the_plugin_field_set() -> None:
    """The 12-field form; without ``do_sort`` the plugin takes its deprecated path."""
    expected = {
        "batch_size",
        "spatial_shape",
        "algo",
        "ksize",
        "stride",
        "padding",
        "dilation",
        "out_padding",
        "subm",
        "transpose",
        "is_train",
        "do_sort",
    }
    assert _emitted_attributes(sparse_functional.GetIndicePairsImplicitGemm.symbolic) == expected


def test_implicit_gemm_emits_the_plugin_field_set() -> None:
    """The 7-field form; ``act_type`` is what the post-export ReLU fusion sets."""
    expected = {
        "is_train",
        "is_subm",
        "fp32_accum",
        "act_alpha",
        "act_beta",
        "output_scale",
        "output_add_scale",
        "act_type",
    }
    assert _emitted_attributes(sparse_functional.ImplicitGemm.symbolic) == expected


def test_the_sparse_stage_wires_the_bias_activation_fusion() -> None:
    """The fusion is only useful if the stage declaration actually runs it."""
    from torch import nn

    from autoware_ml.models.detection3d.main_modules.bevfusion.stages import (
        SPARSE_STAGE,
        build_bevfusion_lidar_stages,
    )
    from autoware_ml.ops.spconv.onnx_fusion import fuse_sparse_graph

    model = type(
        "Stub",
        (),
        {
            "pts_voxel_encoder": nn.Identity(),
            "pts_middle_encoder": nn.Identity(),
            "pts_backbone": nn.Identity(),
            "pts_neck": nn.Identity(),
            "bbox_head": nn.Identity(),
        },
    )()
    sparse = next(
        stage for stage in build_bevfusion_lidar_stages(model) if stage.name == SPARSE_STAGE
    )
    assert fuse_sparse_graph in sparse.onnx_transforms
