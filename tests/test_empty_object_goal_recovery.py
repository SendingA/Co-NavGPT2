"""Regressions for detected-object clouds that project to no map goal."""
from __future__ import annotations

import importlib
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import open3d as o3d

from agents.vlm_agents import VLM_Agent as SingleProcessAgent
from agents.vlm_multi_agents import VLM_Agent as VectorAgent
from utils import fmm_planner as fmm_module
from utils.fmm_planner import FMMPlanner


AGENT_CLASSES = (SingleProcessAgent, VectorAgent)


def _point_cloud(points) -> o3d.geometry.PointCloud:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(values)
    cloud.colors = o3d.utility.Vector3dVector(np.zeros_like(values))
    return cloud


def _goal_agent(agent_class, object_points):
    agent = agent_class.__new__(agent_class)
    agent.agent_id = 0
    agent.l_step = 17
    agent.local_w = agent.local_h = 20
    agent.origins_grid = [10, 10]
    agent.origins_real = [0.0, 0.0]
    agent.map_real_halfsize = 0.5
    agent.args = SimpleNamespace(map_resolution=5)
    agent.camera_position = np.asarray([0.0, 1.2, 0.0])
    agent.object_pcd = _point_cloud(object_points)
    agent.goal_map = np.zeros((20, 20), dtype=np.float32)
    agent.found_goal = False
    agent.nearest_point = None
    return agent


class DetectedObjectProjectionTests(unittest.TestCase):
    def test_out_of_map_object_falls_back_to_frontier(self) -> None:
        # +0.50m is the inclusive world bound but maps to grid index 20,
        # immediately outside a 20x20 grid. It must be rejected safely.
        points = [[0.5, 0.0, 0.5]] * 12
        for agent_class in AGENT_CLASSES:
            with self.subTest(agent=agent_class.__module__):
                agent = _goal_agent(agent_class, points)
                module = importlib.import_module(agent_class.__module__)
                with mock.patch.object(
                    module, "process_pcd", side_effect=lambda cloud: cloud
                ):
                    is_object_goal, target = agent._select_navigation_target(
                        [3, 4]
                    )

                self.assertFalse(is_object_goal)
                self.assertFalse(agent.found_goal)
                self.assertEqual(len(agent.object_pcd.points), 0)
                self.assertEqual(int(np.sum(agent.goal_map)), 1)
                self.assertEqual(agent.goal_map[3, 4], 1)
                np.testing.assert_allclose(target, [-0.35, 1.2, -0.3])

    def test_valid_object_projection_preserves_object_navigation(self) -> None:
        points = [
            [0.10 + index * 1e-4, 0.0, 0.10]
            for index in range(12)
        ]
        for agent_class in AGENT_CLASSES:
            with self.subTest(agent=agent_class.__module__):
                agent = _goal_agent(agent_class, points)
                module = importlib.import_module(agent_class.__module__)
                with mock.patch.object(
                    module, "process_pcd", side_effect=lambda cloud: cloud
                ):
                    is_object_goal, target = agent._select_navigation_target(
                        [3, 4]
                    )

                self.assertTrue(is_object_goal)
                self.assertTrue(agent.found_goal)
                self.assertGreater(int(np.sum(agent.goal_map)), 0)
                self.assertEqual(agent.goal_map[12, 12], 1)
                self.assertTrue(np.all(np.isfinite(target)))
                self.assertEqual(len(agent.object_pcd.points), 12)

    def test_non_finite_object_points_are_not_projected(self) -> None:
        points = [[np.nan, 0.0, 0.0], [0.0, 0.0, np.inf]]
        for agent_class in AGENT_CLASSES:
            with self.subTest(agent=agent_class.__module__):
                agent = _goal_agent(agent_class, points)
                rows, cols = agent.object_map_building(agent.object_pcd)
                self.assertEqual(len(rows), 0)
                self.assertEqual(len(cols), 0)


class EmptyLocalGoalRecoveryTests(unittest.TestCase):
    @staticmethod
    def _planner_agent(agent_class):
        agent = agent_class.__new__(agent_class)
        agent.agent_id = 0
        agent.l_step = 9
        agent.local_w = agent.local_h = 31
        agent.visited_vis = np.zeros((31, 31), dtype=np.uint8)
        agent.collision_map = np.zeros((31, 31), dtype=np.uint8)
        agent.goal_name = "plant"
        agent.goal_map = np.zeros((31, 31), dtype=np.float32)
        agent.replan_count = 0
        agent.risk_navigation_enabled = False
        agent.risk_map = None
        agent.hard_unsafe_mask = None
        agent.risk_alpha = 1.0
        agent._astar_path_cache = None
        agent._risk_escape_active = False
        agent._risk_escape_reason = None
        agent._local_goal_recovery_active = False
        agent.object_pcd = _point_cloud([[1.0, 0.0, 1.0]])
        agent.found_goal = True
        agent.is_running = True
        agent.args = SimpleNamespace(
            local_planner="fmm",
            rl_local_checkpoint=None,
            rl_local_device="cpu",
            rl_local_deterministic=1,
            rl_local_crop_size=15,
            rl_local_rollout_steps=3,
            turn_angle=30,
        )
        return agent

    def test_agent_converts_empty_goal_to_nonterminal_recovery(self) -> None:
        for agent_class in AGENT_CLASSES:
            with self.subTest(agent=agent_class.__module__):
                agent = self._planner_agent(agent_class)
                stg, stop, path = agent._get_stg(
                    np.zeros((31, 31), dtype=np.uint8),
                    [15, 15],
                    np.zeros((31, 31), dtype=np.float32),
                )

                self.assertTrue(agent._local_goal_recovery_active)
                self.assertFalse(agent.found_goal)
                self.assertEqual(len(agent.object_pcd.points), 0)
                self.assertTrue(stop)
                self.assertGreaterEqual(len(path), 2)
                agent.stg = stg
                agent.stop = stop
                self.assertEqual(agent.ffm_act(), 2)

    def test_fmm_rejects_empty_goal_before_skfmm(self) -> None:
        planner = FMMPlanner(np.ones((11, 11), dtype=np.uint8))
        with mock.patch.object(fmm_module.skfmm, "distance") as distance:
            with self.assertRaisesRegex(
                ValueError, "at least one goal cell"
            ):
                planner.set_multi_goal(np.zeros((11, 11), dtype=np.uint8))
        distance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
