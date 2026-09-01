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

"""Model-architecture-specific quantization recipes.

The generic engine in :mod:`autoware_ml.quantization.core` inserts Q/DQ into any Conv2d/Linear submodule.
This package holds the parts that must know a specific backbone's block structure: the forward
hooks that reposition Q/DQ for TensorRT-friendly fusion (:mod:`.quant_forwards`) and the functions
that walk a model to attach quantizers + install those hooks (:mod:`.attach`).

No re-exports on purpose: every consumer imports from the concrete submodule
(``recipes.attach`` / ``recipes.quant_forwards``), which is also the only place these names are
maintained.
"""
