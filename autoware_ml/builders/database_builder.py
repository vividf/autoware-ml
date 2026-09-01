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

import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from autoware_ml.databases.database_interface import DatabaseInterface
from autoware_ml.datamodule.multi_task.multi_task_data_module import MultiTaskDataModule

logger = logging.getLogger(__name__)


def build_database(cfg: DictConfig) -> DatabaseInterface:
    """
    Build a Database from the Hydra configuration.

    Args:
        cfg: Hydra configuration
    Returns:
        A DatabaseInterface object.
    """
    logger.info("Building database...")
    database: DatabaseInterface = instantiate(cfg.database)
    logger.info("Database built successfully.")
    return database


def build_datamodule(cfg: DictConfig, database: DatabaseInterface) -> MultiTaskDataModule:
    """
    Build a DataModule from the Hydra configuration.

    Args:
        cfg: Hydra configuration
        database: A DatabaseInterface object
    Returns:
        A MultiTaskDataModule object.
    """
    logger.info("Building datamodule...")
    datamodule = instantiate(cfg.datamodule, database=database)
    logger.info("Datamodule built successfully.")
    return datamodule
