"""Regressions for occupancy-map frontier candidate construction."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from utils.explored_map_utils import Global_Map_Proc


class GlobalMapFrontierDetectionTests(unittest.TestCase):
    def test_map_extraction_keeps_highest_color_across_frames(self) -> None:
        args = SimpleNamespace(
            map_size_cm=100,
            map_resolution=10,
            map_height_cm=200,
        )
        processor = Global_Map_Proc(args)
        first = SimpleNamespace(
            points=np.asarray(
                [
                    [0.01, 0.2, 0.01],
                    [0.02, 0.8, 0.02],
                    [0.21, 0.4, 0.01],
                ],
                dtype=np.float64,
            ),
            colors=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
        )
        processor.Map_Extraction(first, camera_position_z=0.5)

        center = args.map_size_cm // args.map_resolution // 2
        np.testing.assert_array_equal(
            processor.top_view_map[center, center],
            np.asarray([0, 255, 0], dtype=np.uint8),
        )
        self.assertEqual(processor.z_buffer[center, center], 0.8)

        lower_later = SimpleNamespace(
            points=np.asarray([[0.01, 0.6, 0.01]], dtype=np.float64),
            colors=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
        )
        processor.Map_Extraction(lower_later, camera_position_z=0.5)
        np.testing.assert_array_equal(
            processor.top_view_map[center, center],
            np.asarray([0, 255, 0], dtype=np.uint8),
        )
        self.assertEqual(processor.z_buffer[center, center], 0.8)

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
