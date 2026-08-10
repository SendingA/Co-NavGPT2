"""Regressions for occupancy-map frontier candidate construction."""
from __future__ import annotations

import unittest

import numpy as np

from utils.explored_map_utils import Global_Map_Proc


class GlobalMapFrontierDetectionTests(unittest.TestCase):
    def test_six_candidate_cap_retains_largest_components(self) -> None:
        processor = Global_Map_Proc.__new__(Global_Map_Proc)
        processor.explored_map = np.zeros((120, 120), dtype=np.float32)
        processor.explored_map[10:110, 10:110] = 1.0
        processor.obstacle_map = np.zeros((120, 120), dtype=np.float32)
        for cell in (
            (10, 20),
            (10, 35),
            (10, 55),
            (10, 85),
            (30, 109),
            (70, 109),
            (109, 75),
            (109, 30),
            (50, 10),
        ):
            processor.obstacle_map[cell] = 1.0

        areas, labels, points = processor.Frontier_Det(threshold_point=1)

        self.assertEqual(len(areas), 6)
        self.assertEqual(len(points), 6)
        self.assertEqual(areas, sorted(areas, reverse=True))
        self.assertEqual(
            set(np.unique(labels).astype(int).tolist()),
            set(range(7)),
        )


if __name__ == "__main__":
    unittest.main()
