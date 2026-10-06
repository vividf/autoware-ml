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

"""Typo guard for typed config parsing.

Every typed config section (``deploy``, ``quantization``) rejects unknown keys through
this one helper, so a misspelled option fails loudly instead of silently falling back
to a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reject_unknown_keys(
    raw: Mapping[str, Any], known: frozenset[str], where: str, hint: str = ""
) -> None:
    """Raise when ``raw`` contains keys outside ``known``.

    Args:
        raw: The mapping being parsed.
        known: The section's full key set.
        where: Dotted config path for the error message (e.g. ``"deploy.onnx"``).
        hint: Optional sentence appended to the error (why the typo would hurt).
    """
    unknown = set(raw) - known
    if unknown:
        message = f"Unknown {where} key(s): {sorted(unknown)}. Valid keys: {sorted(known)}."
        if hint:
            message += f" {hint}"
        raise ValueError(message)
