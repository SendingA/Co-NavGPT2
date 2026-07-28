"""Regression tests for the native Habitat navigation-step metric."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from arguments import get_args, load_config
from habitat.core.registry import registry
import habitat.tasks.registration  # noqa: F401  (registers built-in measures)


class NavigationStepsMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", ["main.py"]):
            cls.config = load_config(get_args())

    def test_active_objectnav_config_enables_native_num_steps(self):
        measure_config = self.config.habitat.task.measurements.num_steps

        self.assertEqual(measure_config.type, "NumStepsMeasure")

    def test_num_steps_resets_and_counts_one_per_task_step(self):
        measure_class = registry.get_measure("NumStepsMeasure")
        self.assertIsNotNone(measure_class)

        measure = measure_class(
            sim=None,
            config=self.config.habitat.task.measurements.num_steps,
        )
        measure.reset_metric(episode=None, task=None, observations={})
        self.assertEqual(measure.get_metric(), 0)

        measure.update_metric(episode=None, task=None, observations={})
        measure.update_metric(episode=None, task=None, observations={})
        self.assertEqual(measure.get_metric(), 2)


if __name__ == "__main__":
    unittest.main()
