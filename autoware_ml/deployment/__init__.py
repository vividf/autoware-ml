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

"""Model-agnostic deployment: stage graph -> export -> backend pipelines -> verification.

- :mod:`.stages`       — the declaration a model returns from ``build_stages()``.
- :mod:`.config`       — typed ``deploy`` config (``DeployConfig``).
- :mod:`.export`       — derive ONNX / TensorRT artifacts from the stage graph.
- :mod:`.onnx`         — the ``torch.onnx.export`` primitive, precision passes, graph surgery.
- :mod:`.pipeline`     — run the stage graph on pytorch / onnx / tensorrt.
- :mod:`.backends`     — TensorRT engine builder + ONNX Runtime / TensorRT runners.
- :mod:`.verification` — cross-backend numerical parity.

Ground-truth evaluation of a backend lives in :mod:`autoware_ml.evaluation`, which
treats PyTorch as one more backend. No model name appears anywhere in this package.
"""
