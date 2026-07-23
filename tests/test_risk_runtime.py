"""Tests for runtime coordinate alignment and GT/sensed separation."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from utils.risk.runtime import navigation_grid_frame


class RiskRuntimeFrameTests(unittest.TestCase):
    def test_initial_camera_maps_to_center_cell(self) -> None:
        args = SimpleNamespace(map_resolution=5)
        reference = SimpleNamespace(
            init_sim_rotation=np.eye(3),
            init_sim_position=np.array([3.0, 0.88, -4.0]),
            local_w=480,
            local_h=480,
            args=args,
        )
        frame = navigation_grid_frame(reference)
        cell = frame.world_to_grid(reference.init_sim_position)
        np.testing.assert_array_equal(cell, [240, 240])
        recovered = frame.grid_to_world(cell, map_y_m=0.0)
        self.assertLess(
            float(np.linalg.norm(recovered[[0, 2]] - reference.init_sim_position[[0, 2]])),
            frame.resolution_m,
        )

    def test_yaw_rotated_world_point_matches_agent_local_axes(self) -> None:
        # Camera yaw of +90 degrees.  The explicit affine should still round
        # trip every point, independent of the navigation-axis permutation.
        rotation = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        reference = SimpleNamespace(
            init_sim_rotation=rotation,
            init_sim_position=np.array([1.0, 0.88, 2.0]),
            local_w=200,
            local_h=200,
            args=SimpleNamespace(map_resolution=10),
        )
        frame = navigation_grid_frame(reference)
        point = np.array([2.0, 0.88, 3.0])
        local = frame.world_to_map_points(point)
        recovered = frame.map_to_world_points(local)
        np.testing.assert_allclose(recovered, point, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
