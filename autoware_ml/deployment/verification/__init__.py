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

"""Cross-backend numerical verification of exported deployment artifacts."""

from autoware_ml.deployment.verification.backend_verifier import (
    BackendVerifier,
    VerificationScenario,
)
from autoware_ml.deployment.verification.output_comparator import (
    OutputComparator,
    OutputDiffSummary,
    TensorDiffDetail,
)

__all__ = [
    "BackendVerifier",
    "VerificationScenario",
    "OutputComparator",
    "OutputDiffSummary",
    "TensorDiffDetail",
]
