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

"""Every StreamPETR stage config composes, streams, and builds its taxonomy, pipeline and model."""

from __future__ import annotations

import hydra
import pytest

from autoware_ml.configs.tests.test_task_configs import compose_task

# The pretrain stage is not matched by the file name filter of the shared task config test.
STAGES = [
    "tasks/detection3d/streampetr/vov_320x800_nuscenes",
    "tasks/detection3d/streampetr/vov_320x800_nuscenes_pretrain",
    "tasks/detection3d/streampetr/vov_480x640_t4dataset_base",
    "tasks/detection3d/streampetr/vov_480x640_t4dataset_j6gen2",
]
SEVEN_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]


@pytest.mark.parametrize("config_name", STAGES)
def test_every_stage_streams_the_seven_classes(config_name: str) -> None:
    cfg = compose_task(config_name)

    taxonomy = hydra.utils.instantiate(cfg.database.taxonomy)
    hydra.utils.instantiate(cfg.data_preprocessing)
    hydra.utils.instantiate(cfg.datamodule.train_dataset.transforms)
    hydra.utils.instantiate(cfg.datamodule.test_dataset.transforms)
    model = hydra.utils.instantiate(cfg.model)

    # Every stage trains the same classes in the same order, so a stage initializes from
    # the previous one without dropping the classification layer.
    assert list(taxonomy.detection3d.class_names) == SEVEN_CLASSES
    assert model.bbox_head.num_classes == len(SEVEN_CLASSES)
    assert cfg.datamodule.streaming
    assert not cfg.trainer.use_distributed_sampler
