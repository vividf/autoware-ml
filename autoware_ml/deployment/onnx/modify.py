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


"""User-configured ONNX graph surgery (``deploy.onnx.modify_graph``).

Deliberately separate from the export primitive and the precision passes: a modifier
is an arbitrary user hook instantiated from config, applied to the exported file.
Fusions intrinsic to a graph's deployed form belong to the stage declaration
(``GraphStage.onnx_transforms``) instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _instantiate_modifier(modify_graph_cfg: Any) -> Any:
    import hydra

    modifier = hydra.utils.instantiate(modify_graph_cfg)
    if callable(modifier):
        return modifier
    if hasattr(modifier, "modify"):
        return modifier
    raise ValueError(f"Modifier {modifier} must be callable or have a 'modify' method.")


def _apply_modifier(modifier: Any, onnx_path: Path) -> Path:
    modified_path = modifier(onnx_path) if callable(modifier) else modifier.modify(onnx_path)
    if modified_path is None:
        raise ValueError("Modifier returned None. Must return Path or str.")
    return Path(modified_path)


def should_modify_graph(modify_graph_cfg: Any) -> bool:
    """Return whether graph modification is enabled (a non-None modifier config)."""
    if modify_graph_cfg is None:
        return False
    from omegaconf import DictConfig, OmegaConf

    if isinstance(modify_graph_cfg, DictConfig):
        return OmegaConf.to_container(modify_graph_cfg, resolve=False) is not None
    return True


def modify_onnx_graph(onnx_path: Path, modify_graph_cfg: Any) -> Path:
    """Apply the configured (Hydra-instantiable) modifier to an exported ONNX file."""
    logger.info("Modifying ONNX graph...")
    modifier = _instantiate_modifier(modify_graph_cfg)
    modified_path = _apply_modifier(modifier, onnx_path)
    logger.info("Successfully modified ONNX graph: %s", modified_path)
    return modified_path

