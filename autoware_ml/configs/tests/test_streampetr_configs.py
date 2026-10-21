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

"""Invariants of the StreamPETR stages the generic experiment config test does not check."""

from __future__ import annotations

import hydra
import pytest
from omegaconf import OmegaConf

from autoware_ml.configs.tests.test_task_configs import EXPERIMENT_CONFIGS, compose_experiment

STAGES = [config for config in EXPERIMENT_CONFIGS if "/streampetr/" in config]
SEVEN_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]


def test_the_three_stages_are_bundled() -> None:
    assert [stage.rsplit("/", 1)[1] for stage in STAGES] == [
        "vov99_cpfpn_320x800_b8_30e_nuscenes_54m_pretrain",
        "vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base",
        "vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base",
    ]


@pytest.mark.parametrize("config_name", STAGES)
def test_every_stage_streams_the_seven_classes_in_the_same_order(config_name: str) -> None:
    cfg = compose_experiment(config_name)

    # Every stage trains the same classes in the same order, so a stage initializes from the
    # previous one without dropping or permuting the classification layers.
    for database in (cfg.database, cfg.get("gen2_database")):
        if database is not None:
            taxonomy = hydra.utils.instantiate(database.taxonomy)
            assert list(taxonomy.detection3d.class_names) == SEVEN_CLASSES
    assert list(OmegaConf.to_container(cfg.model.bbox_head.class_names)) == SEVEN_CLASSES
    assert list(OmegaConf.to_container(cfg.model.img_roi_head.class_names)) == SEVEN_CLASSES
    # The head carries a temporal memory across batches, so every dataloader streams scenes.
    assert cfg.datamodule.streaming
    assert not cfg.datamodule.train_dataloader.shuffle
    assert cfg.datamodule.validation_dataloader.batch_size == 1
    assert cfg.datamodule.test_dataloader.batch_size == 1
    # The schedule is bound to the trainer rather than auto-filled.
    assert cfg.model.scheduler.max_epochs == cfg.trainer.max_epochs
    # Only the initial weights are left for the command line.
    missing = [key for key in ("weights",) if OmegaConf.is_missing(cfg, key)]
    assert missing == ["weights"]


@pytest.mark.parametrize("config_name", STAGES)
def test_every_pipeline_filters_the_physical_boxes_before_anything_else(config_name: str) -> None:
    cfg = compose_experiment(config_name)
    for split in ("train", "test"):
        steps = [step["_target_"] for step in cfg.datasets[split].transforms.pipeline]
        assert steps[0].endswith("boxes3d.filters.BBoxesPhysicalFilter"), (config_name, split)


@pytest.mark.parametrize("config_name", [stage for stage in STAGES if "/t4dataset/" in stage])
def test_t4_stages_ignore_the_cones_and_barriers_of_the_scenes_without_them(
    config_name: str,
) -> None:
    cfg = compose_experiment(config_name)
    partial_annotations = hydra.utils.instantiate(cfg.model.partial_annotations)
    assert list(partial_annotations.partial_ignore_class_ids) == [5, 6]


def test_the_nuscenes_stage_supervises_every_class() -> None:
    cfg = compose_experiment([stage for stage in STAGES if "/nuscenes/" in stage][0])
    assert cfg.model.partial_annotations is None


def test_the_gen_base_stage_joins_the_gen1_and_gen2_corpora() -> None:
    cfg = compose_experiment([stage for stage in STAGES if "gen1_gen2" in stage][0])
    assert "gen1_base" in cfg.database.version
    assert "gen2_base" in cfg.gen2_database.version
    assert set(cfg.datamodule.sources) == {"main", "gen2"}
    # Both sources read the same split datasets, only from their own database.
    for split in ("train", "test"):
        gen2 = cfg.datasets[f"gen2_{split}"]
        assert gen2.transforms == cfg.datasets[split].transforms
        assert gen2.camera_names == cfg.datasets[split].camera_names
        assert gen2.database_root_path == cfg.gen2_database.root_path
