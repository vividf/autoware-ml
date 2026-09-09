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

"""Legacy ExportSpec fallback in the deploy entrypoint.

TODO(vividf): delete this file together with the fallback once every BaseModel
migrates to MultiTaskBaseModel.build_stages().
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

pytest.importorskip("mlflow")
from autoware_ml.scripts.deploy import is_legacy_deploy_config, run_legacy_export  # noqa: E402
from autoware_ml.utils.deploy import ExportSpec, merge_module_onnx_cfg  # noqa: E402


class _Module:
    pass


def _legacy_cfg(tensorrt_enabled: bool = False):
    return OmegaConf.create(
        {
            "onnx": {
                "enabled": True,
                "opset_version": 17,
                "modules": {"encoder": {"output_names": ["feat"]}},
            },
            "tensorrt": {"enabled": tensorrt_enabled},
        }
    )


def test_legacy_schema_is_detected():
    assert is_legacy_deploy_config(_legacy_cfg())
    assert not is_legacy_deploy_config(
        OmegaConf.create({"onnx": {"dynamo": False}, "stages": {"s": {}}})
    )
    assert not is_legacy_deploy_config(OmegaConf.create({}))


def test_merge_module_onnx_cfg_overrides_shared_and_drops_modules():
    merged = merge_module_onnx_cfg(_legacy_cfg().onnx, "encoder")
    assert merged.opset_version == 17
    assert list(merged.output_names) == ["feat"]
    assert "modules" not in merged
    with pytest.raises(KeyError, match="not found in deploy.onnx.modules"):
        merge_module_onnx_cfg(_legacy_cfg().onnx, "missing")


def test_run_legacy_export_drives_export_spec(tmp_path: Path):
    spec = ExportSpec(
        module=_Module(),
        args=(1,),
        input_param_names=["x"],
        output_names=["feat"],
    )
    with (
        patch("autoware_ml.scripts.deploy.resolve_export_specs", return_value={"encoder": spec}),
        patch("autoware_ml.scripts.deploy.export_to_onnx") as export_mock,
    ):
        run_legacy_export(_legacy_cfg(), tmp_path, datamodule=None, model=_Module(), device=None)
    export_mock.assert_called_once()
    assert export_mock.call_args.args[6] == tmp_path / "encoder.onnx"


def test_run_legacy_export_rejects_unsupported_stage(tmp_path: Path):
    spec = ExportSpec(
        module=_Module(),
        args=(1,),
        input_param_names=["x"],
        supported_stages=frozenset({"tensorrt"}),
    )
    with (
        patch("autoware_ml.scripts.deploy.resolve_export_specs", return_value={"encoder": spec}),
        pytest.raises(RuntimeError, match="does not support ONNX export"),
    ):
        run_legacy_export(_legacy_cfg(), tmp_path, datamodule=None, model=_Module(), device=None)
