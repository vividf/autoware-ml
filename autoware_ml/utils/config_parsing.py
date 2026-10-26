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

"""Typo guard for typed config sections: a misspelled key fails instead of using a default."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def reject_unknown_keys(
    raw: Mapping[str, Any], known: frozenset[str], where: str, hint: str = ""
) -> None:
    """Raise when ``raw`` has keys outside ``known``.

    The error names the section by ``where`` (``"deploy.evaluation"``) and ends with ``hint``.
    """
    unknown = set(raw) - known
    if unknown:
        message = f"Unknown {where} key(s): {sorted(unknown)}. Valid keys: {sorted(known)}."
        if hint:
            message += f" {hint}"
        raise ValueError(message)
