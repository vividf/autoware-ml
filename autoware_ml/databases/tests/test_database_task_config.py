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

"""Unit tests for the database task configuration."""

from __future__ import annotations

import pickle
from types import MappingProxyType
import unittest

from autoware_ml.databases.database_task_config import DatabaseTaskConfig
from autoware_ml.types.tasks import TaskType


class TestDatabaseTaskConfig(unittest.TestCase):
    """Unit tests for DatabaseTaskConfig."""

    def _build_task_config(
        self, label_remapper: dict[str, str] | None = None
    ) -> DatabaseTaskConfig:
        """Build a detection3d task configuration."""
        return DatabaseTaskConfig(
            task_type=TaskType.DETECTION3D,
            label_names=["car", "pedestrian"],
            ignore_label_index=-1,
            label_remapper={"bus": "car"} if label_remapper is None else label_remapper,
        )

    def test_label_remapper_is_wrapped(self) -> None:
        """
        Test that a plain dict label remapper is wrapped, since that is what Hydra composes.
        """
        task_config = self._build_task_config()

        self.assertIsInstance(task_config.label_remapper, MappingProxyType)
        self.assertEqual(dict(task_config.label_remapper), {"bus": "car"})

    def test_pickle_round_trip_keeps_the_label_remapper(self) -> None:
        """
        Test that the task configuration survives pickling, which is what sends it to the
        worker processes that generate the database records. A MappingProxyType cannot be
        pickled on its own, so the label remapper is unwrapped and re-wrapped around it.
        """
        task_config = self._build_task_config()

        restored = pickle.loads(pickle.dumps(task_config))

        self.assertIsInstance(restored.label_remapper, MappingProxyType)
        self.assertEqual(dict(restored.label_remapper), {"bus": "car"})
        self.assertEqual(restored, task_config)
        self.assertEqual(restored.hash_repr, task_config.hash_repr)

    def test_pickle_round_trip_without_a_label_remapper(self) -> None:
        """Test that a task configuration without a label remapper pickles unchanged."""
        task_config = DatabaseTaskConfig(
            task_type=TaskType.SEGMENTATION3D,
            label_names=["car"],
            ignore_label_index=-1,
            label_remapper=None,
        )

        restored = pickle.loads(pickle.dumps(task_config))

        self.assertIsNone(restored.label_remapper)
        self.assertEqual(restored, task_config)


if __name__ == "__main__":
    unittest.main()
