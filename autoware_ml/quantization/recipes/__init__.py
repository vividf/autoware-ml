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

The generic engine in :mod:`autoware_ml.quantization.core` converts any Conv2d/Linear leaf.
This package holds the parts that must know a block's structure: the quantized block
classes that reposition Q/DQ for TensorRT-friendly fusion (:mod:`.quant_blocks`) and the
recipes that walk a model to convert blocks / wrap pools (:mod:`.attach`).

No re-exports on purpose: every consumer imports from the concrete submodule
(``recipes.attach`` / ``recipes.quant_blocks``), which is also the only place these names are
maintained.
"""
