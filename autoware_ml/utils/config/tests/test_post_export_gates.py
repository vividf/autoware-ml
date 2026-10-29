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


"""An export option a config enables has its post-export gate enabled with it."""

from __future__ import annotations

import pytest

from autoware_ml.utils.config.tests.config_roots import compose_task

BEVFUSION_CONFIGS = [
    "tasks/detection3d/bevfusion/t4dataset/bevfusion_lidar/"
    "lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m",
]


@pytest.mark.parametrize("config_name", BEVFUSION_CONFIGS)
def test_fused_export_attention_is_gated_by_the_deploy_evaluation(config_name: str) -> None:
    # The fused attention is only numerically safe where TensorRT fuses it; nothing at export
    # time checks that, so a config that enables it must also score the deployed graphs.
    cfg = compose_task(config_name)
    if cfg.model.bbox_head.get("fuse_export_attention", False):
        assert cfg.deploy.evaluation.enabled, f"{config_name}: fused attention without its gate"
        assert cfg.deploy.evaluation.num_samples != 0
