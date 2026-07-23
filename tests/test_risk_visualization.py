"""Headless tests for risk-map benchmark artifacts."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from utils.risk.visualization import compose_risk_dashboard


class RiskVisualizationTests(unittest.TestCase):
    def test_dashboard_contains_all_six_component_panels(self) -> None:
        shape = (12, 10)
        layers = SimpleNamespace(
            flame=np.full(shape, 0.1, dtype=np.float32),
            temperature=np.full(shape, 0.2, dtype=np.float32),
            smoke=np.full(shape, 0.3, dtype=np.float32),
            physical_risk=np.full(shape, 0.4, dtype=np.float32),
            confidence=np.full(shape, 0.8, dtype=np.float32),
            unknown=np.zeros(shape, dtype=bool),
            hard_unsafe=np.zeros(shape, dtype=bool),
        )
        image = compose_risk_dashboard(
            layers,
            planning_risk=np.full(shape, 0.6, dtype=np.float32),
            obstacle_map=np.zeros(shape, dtype=np.uint8),
            agent_cells=[(6, 5)],
            frontier_points=[(3, 2)],
            panel_size=(40, 32),
        )
        self.assertEqual(image.shape, (34 + 64, 120, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(int(image.max()), 0)

    def test_unknown_cells_are_visually_distinct(self) -> None:
        shape = (4, 4)
        layers = SimpleNamespace(
            flame=np.zeros(shape),
            temperature=np.zeros(shape),
            smoke=np.zeros(shape),
            physical_risk=np.zeros(shape),
            confidence=np.zeros(shape),
            unknown=np.ones(shape, dtype=bool),
            hard_unsafe=np.zeros(shape, dtype=bool),
        )
        image = compose_risk_dashboard(layers, panel_size=(40, 40))
        # Below each 27-pixel panel label, unknown is the fixed neutral grey.
        np.testing.assert_array_equal(image[34 + 35, 20], [72, 72, 72])


if __name__ == "__main__":
    unittest.main()
