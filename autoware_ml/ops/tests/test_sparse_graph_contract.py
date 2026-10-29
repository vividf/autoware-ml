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

"""Unit tests for the attributes the sparse export symbolics emit for the runtime plugin.

The plugin still accepts a graph without one of them (it warns about legacy ONNX and
assumes a default), so a dropped attribute would otherwise only show in a TensorRT log.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("spconv", reason="the export symbolics import spconv")

from autoware_ml.ops.spconv import contract, sparse_functional  # noqa: E402


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
        contract.DO_SORT_ATTR,
    }
    assert _emitted_attributes(sparse_functional.GetIndicePairsImplicitGemm.symbolic) == expected


def test_implicit_gemm_emits_the_plugin_field_set() -> None:
    expected = {
        "is_train",
        "is_subm",
        # Not a plugin field: the plugin's parser drops it.
        "fp32_accum",
        "act_alpha",
        "act_beta",
        contract.OUTPUT_SCALE_ATTR,
        "output_add_scale",
        contract.ACT_TYPE_ATTR,
    }
    assert _emitted_attributes(sparse_functional.ImplicitGemm.symbolic) == expected
