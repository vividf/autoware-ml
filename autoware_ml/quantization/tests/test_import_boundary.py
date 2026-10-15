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

"""Every model imports ``quantization.rules``; that must not load modelopt or the engine."""

from __future__ import annotations

import subprocess
import sys

import pytest

import autoware_ml.quantization as quantization


def _loaded_after(statement: str) -> list[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{statement}\nimport sys\n"
            "print('modelopt' in sys.modules)\n"
            "print('autoware_ml.quantization.core' in sys.modules)\n",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.splitlines()[-2:]


def test_importing_the_model_base_loads_neither_modelopt_nor_the_engine() -> None:
    assert _loaded_after("import autoware_ml.models.base") == ["False", "False"]


def test_package_exports_load_on_access() -> None:
    assert _loaded_after("from autoware_ml.quantization import QuantRules") == ["False", "False"]
    assert _loaded_after("from autoware_ml.quantization import QuantizationPlan") == [
        "True",
        "True",
    ]


def test_every_export_resolves() -> None:
    for name in quantization.__all__:
        assert getattr(quantization, name) is not None
    with pytest.raises(AttributeError, match="no attribute 'not_an_export'"):
        quantization.not_an_export  # noqa: B018
