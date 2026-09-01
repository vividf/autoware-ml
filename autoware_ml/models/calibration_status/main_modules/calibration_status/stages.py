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

"""Calibration-status classifier deployment stage graph — declared ahead of the
interface migration.

The deployed classifier is one exported graph fed by one glue stage:

    fetch_fused_image (torch)  ->  calibration_classifier (graph)

Written against the legacy :class:`CalibrationStatusClassifier`'s submodules so the
deployment side can land before the ``MultiTaskBaseModel`` migration. Contract with
the interface migration (kept in one place, here):

- **Submodule names**: the model exposes ``backbone``, ``neck``, ``head``; ``head``
  is callable (logits) and has ``predict(logits)`` (probabilities).
- **Batch inputs**: ``MultiTaskBatchInputs`` grows a ``fused_img`` tensor field
  (the 5-channel fused camera/depth image) — the glue stage reads it.
- **Typed outputs**: ``assemble_outputs`` maps the ``output`` tensor to a
  ``calibration_probabilities`` field (name is a draft until the outputs dataclass
  gains a classification slot).

Runtime ABI (carried over from AWML ``CalibrationStatusClassification``): tensor
names ``input`` / ``output``, probabilities (not logits) as the output, opset 16,
dynamic dims ``input {0: batch_size, 2: height, 3: width}`` / ``output
{0: batch_size}``. AWML names the artifact ``end2end.onnx``; here the artifact is
``calibration_classifier.onnx`` (stage name) — the runtime node takes the ONNX path
as a parameter, so only the tensor names are load-bearing.

Deploy config draft for the future experiment config::

    deploy:
      onnx: { dynamo: false, opset_version: 16, precision: fp16 }
      stages:
        calibration_classifier:
          onnx:
            dynamic_axes:
              input:  { 0: batch_size, 2: height, 3: width }
              output: { 0: batch_size }
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from autoware_ml.deployment.stages import GraphStage, Stage, StageContext, TorchStage

# Stage / artifact name.
CLASSIFIER_STAGE = "calibration_classifier"

# Context tensor names — the ONNX input/output names (AWML runtime ABI).
FUSED_IMAGE = "input"
PROBABILITIES = "output"

# ONNX output name -> typed-outputs field consumed by ``assemble_outputs``.
OUTPUT_FIELDS: tuple[tuple[str, str], ...] = ((PROBABILITIES, "calibration_probabilities"),)


class CalibrationClassifierExportWrapper(nn.Module):
    """Export backbone -> neck -> head as one graph returning probabilities."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, fused_img: torch.Tensor) -> torch.Tensor:
        """Classify one batch of fused images into calibration-status probabilities."""
        logits = self.head(self.neck(self.backbone(fused_img)))
        return self.head.predict(logits)


def build_calibration_status_stages(model: Any) -> tuple[Stage, ...]:
    """Declare the classifier's stage graph over ``model``'s submodules."""

    def fetch_fused_image(context: StageContext) -> Mapping[str, torch.Tensor]:
        fused_img = context.batch_inputs.fused_img
        if fused_img is None:
            raise ValueError(
                "MultiTaskBatchInputs must carry fused_img for the calibration classifier."
            )
        return {FUSED_IMAGE: fused_img.to(context.device)}

    return (
        TorchStage("fetch_fused_image", run=fetch_fused_image),
        GraphStage(
            CLASSIFIER_STAGE,
            module=CalibrationClassifierExportWrapper(model.backbone, model.neck, model.head),
            inputs=(FUSED_IMAGE,),
            outputs=(PROBABILITIES,),
            output_fields=OUTPUT_FIELDS,
        ),
    )
