"""Tests for the bundled T4 detection taxonomy."""

from __future__ import annotations

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.databases.taxonomy import DatabaseTaxonomy

T4_DETECTION3D_CONFIG = (
    "tasks/detection3d/ptv3/t4dataset/voxel012_tiny_b8_30e_t4dataset_122m_j6gen2_base"
)
NUSCENES_DETECTION3D_CONFIG = "tasks/detection3d/ptv3/nuscenes/voxel005_base_b16_30e_nuscenes_51m"


def _taxonomy(config_name: str, taxonomy: str | None = None) -> DatabaseTaxonomy:
    register_config_resolvers()
    GlobalHydra.instance().clear()
    overrides = (
        [] if taxonomy is None else [f"database/t4dataset/taxonomy@database.taxonomy={taxonomy}"]
    )
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return hydra.utils.instantiate(cfg.database.taxonomy)


def _t4_taxonomy() -> DatabaseTaxonomy:
    return _taxonomy(T4_DETECTION3D_CONFIG)


def test_every_raw_name_of_the_vocabulary_resolves() -> None:
    # Every fine name is either trained or dropped on purpose, none falls to the ignore index.
    taxonomy = _t4_taxonomy().detection3d

    for fine_name in taxonomy.vocabulary.fine_names:
        assert fine_name in taxonomy.class_mapping


@pytest.mark.parametrize("raw_name", ["static_object.bicycle_rack", "static_object.bicycle rack"])
def test_bicycle_rack_source_labels_resolve(raw_name: str) -> None:
    # Both bicycle rack spellings map to one fine name. Detection drops it, segmentation keeps
    # it as manmade.
    taxonomy = _t4_taxonomy()

    assert taxonomy.detection3d.fine_name(raw_name) == "bicycle_rack"
    assert taxonomy.detection3d.class_name("bicycle_rack") is None
    assert taxonomy.segmentation3d.class_name("bicycle_rack") == "manmade"


def test_seven_class_is_the_dynamic_taxonomy_in_the_nuscenes_class_order() -> None:
    # The camera detectors pretrain on nuScenes and the deployed node decodes the nuScenes
    # order, so seven_class must keep that order; its content is the dynamic taxonomy's.
    seven = _taxonomy(T4_DETECTION3D_CONFIG, "seven_class").detection3d
    dynamic = _taxonomy(T4_DETECTION3D_CONFIG, "default_dynamic").detection3d
    nuscenes = _taxonomy(NUSCENES_DETECTION3D_CONFIG).detection3d

    assert list(seven.class_names) == list(nuscenes.class_names)
    assert sorted(seven.class_names) == sorted(dynamic.class_names)
    assert seven.class_mapping == dynamic.class_mapping
    assert seven.class_groups == dynamic.class_groups
    assert seven.eval_range == dynamic.eval_range
    assert seven.collision_kinds == dynamic.collision_kinds
    assert seven.living_speeds == dynamic.living_speeds
