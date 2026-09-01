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

"""Quantization-aware nn.Module subclasses (Conv2d / ConvTranspose2d / Linear)."""

from .quant_conv import QuantConv2d, QuantConvTranspose2d
from .quant_linear import QuantLinear

__all__ = [
    "QuantConv2d",
    "QuantConvTranspose2d",
    "QuantLinear",
]
