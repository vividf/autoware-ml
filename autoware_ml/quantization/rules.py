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

"""A model's quantization declaration.

This module is what a model imports to state its rules. It depends on nothing but the
typed config, so declaring rules costs a model nothing at import time — the engine
(nvidia-modelopt, :mod:`autoware_ml.quantization.core`) is first imported by
:mod:`autoware_ml.quantization.plan`, when a plan is actually built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from autoware_ml.quantization.config import VALID_MODULE_KINDS, Precision


@dataclass(frozen=True)
class QuantRules:
    """A model's quantization declaration: which submodules carry which quantizable kinds.

    Per-submodule module kinds are architecture facts and belong in the model's
    rules, not in config; config (``skip_quantize``) only subtracts from what the
    rules declare.

    Attributes:
        quantize_submodules: Top-level model attribute name -> module kinds to
            replace inside it. Two spellings:

            - ``("conv", "linear")`` — every kind at the config's
              ``default_precision`` (the common case);
            - ``{"conv": "int8", "linear": "fp8"}`` — per-kind precision, for a model
              whose layer families tolerate different precisions. A kind mapped to
              ``None`` follows ``default_precision``.

            A submodule absent on the model is skipped silently, so one rules object
            can serve model variants.
    """

    quantize_submodules: Mapping[str, tuple[str, ...] | Mapping[str, str | None]]

    def __post_init__(self) -> None:
        for submodule_name, kinds in self.quantize_submodules.items():
            unknown = set(kinds) - set(VALID_MODULE_KINDS)
            if unknown:
                raise ValueError(
                    f"QuantRules submodule {submodule_name!r} declares unknown module kind(s) "
                    f"{sorted(unknown)}; valid kinds: {list(VALID_MODULE_KINDS)}."
                )
            if isinstance(kinds, Mapping):
                for precision_name in kinds.values():
                    if precision_name is not None:
                        Precision(precision_name)  # raises ValueError on an unknown precision

    def resolved_kinds(
        self, submodule_name: str, default_precision: Precision
    ) -> Mapping[str, Precision]:
        """The submodule's kinds with every precision resolved.

        Args:
            submodule_name: Key of :attr:`quantize_submodules`.
            default_precision: Config precision used for kinds without their own.
        """
        kinds = self.quantize_submodules[submodule_name]
        if isinstance(kinds, Mapping):
            return {
                kind: (Precision(name) if name is not None else default_precision)
                for kind, name in kinds.items()
            }
        return {kind: default_precision for kind in kinds}
