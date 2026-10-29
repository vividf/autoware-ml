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
    assert _loaded_after("import autoware_ml.models.module_base_model") == ["False", "False"]


def test_the_rules_and_config_modules_load_without_the_engine() -> None:
    assert _loaded_after("from autoware_ml.quantization.rules import QuantRules") == [
        "False",
        "False",
    ]
    assert _loaded_after("from autoware_ml.quantization.config import QuantizationConfig") == [
        "False",
        "False",
    ]


def test_the_plan_loads_the_engine() -> None:
    assert _loaded_after("from autoware_ml.quantization.plan import prepare_quantized_tree") == [
        "True",
        "True",
    ]
