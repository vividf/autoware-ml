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

"""The check every deployed model's tests run: the deployment reproduces ``forward()``."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple
import unittest

import torch

from autoware_ml.deployment.pipeline import DeployedPipeline
from autoware_ml.deployment.stages import DeployedModel, DeploymentRun, GraphStage
from autoware_ml.types.backend import Backend


class DeploymentForwardCheck(NamedTuple):
    """What :func:`assert_deployment_reproduces_forward` ran, for checks beyond the outputs."""

    graphs: tuple[GraphStage, ...]
    forward_outputs: Any
    run: DeploymentRun


def assert_deployment_reproduces_forward(
    model: Any,
    batch: Any,
    device: torch.device,
    forward_tensors: Callable[[Any], Mapping[str, torch.Tensor]],
    *,
    rtol: float,
    atol: float,
) -> DeploymentForwardCheck:
    """Run ``forward()`` and the deployed function on one batch and compare every model output.

    ``forward()`` and ``deployed_forward()`` compose the same modules twice, once for training
    and once for deployment, and nothing but this check keeps the two compositions equal.
    Both runs share the weights, so only kernel-selection noise remains; a wrong dataflow is
    off by O(1).

    Args:
        model: Model in eval mode declaring ``deploy_graphs()`` and ``deployed_forward()``.
        batch: Model inputs both runs start from.
        device: Device the graphs run on.
        forward_tensors: Maps the outputs ``forward()`` returns to the model outputs the
            deployed function returns, by name.
        rtol: Relative tolerance of :func:`torch.testing.assert_close`.
        atol: Absolute tolerance of :func:`torch.testing.assert_close`.

    Returns:
        The graphs, what ``forward()`` returned and the deployment run.

    Raises:
        AssertionError: When the model declares no graphs, ``forward_tensors`` does not name
            exactly the model outputs, or a model output differs from ``forward()``'s.
    """
    case = unittest.TestCase()
    graphs = model.deploy_graphs()
    case.assertIsNotNone(graphs, f"{type(model).__name__} declares no deployment graphs.")
    deployment = DeployedModel(tuple(graphs), model.deployed_forward)
    with torch.no_grad():
        forward_outputs = model(batch)
    reference = forward_tensors(forward_outputs)
    run = DeployedPipeline(deployment, Backend.PYTORCH, device).trace(batch)
    case.assertEqual(
        set(reference),
        set(run.outputs),
        "forward_tensors must name exactly the model outputs of the deployed function.",
    )
    for name, tensor in run.outputs.items():
        torch.testing.assert_close(
            tensor,
            reference[name],
            rtol=rtol,
            atol=atol,
            msg=lambda detail, name=name: f"Model output {name!r} differs from forward(): {detail}",
        )
    return DeploymentForwardCheck(deployment.graphs, forward_outputs, run)
