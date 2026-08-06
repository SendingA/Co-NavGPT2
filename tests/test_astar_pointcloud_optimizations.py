"""Regressions for empty-cloud handling and single-search A* planning."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from utils.local_planners import AStarPathCache, AStarPlanner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EmptyPointCloudTests(unittest.TestCase):
    def test_all_no_return_depth_skips_open3d_kdtree(self) -> None:
        script = """
import numpy as np
import open3d as o3d
from types import SimpleNamespace
from utils.explored_map_utils import build_full_scene_pcd
from utils.mapping import process_pcd

camera = SimpleNamespace(fx=1.0, fy=1.0, cx=1.5, cy=1.5)
depth = np.zeros((4, 4), dtype=np.float32)
rgb = np.zeros((4, 4, 3), dtype=np.uint8)
scene = build_full_scene_pcd(depth, rgb, camera)
assert len(scene.points) == 0
denoised = process_pcd(o3d.geometry.PointCloud())
assert len(denoised.points) == 0
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        combined_output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, combined_output)
        self.assertNotIn("KDTreeFlann::SetRawData", combined_output)


class AStarOptimizationTests(unittest.TestCase):
    @staticmethod
    def _planner_inputs(size: int = 41):
        traversible = np.ones((size, size), dtype=np.uint8)
        traversible[size // 2, 4:-4] = 0
        traversible[size // 2, -5] = 1
        goal = np.zeros_like(traversible)
        goal[-5, -5] = 1
        return traversible, goal

    def test_one_search_and_one_goal_distance_per_plan(self) -> None:
        traversible, goal = self._planner_inputs()
        planner = AStarPlanner(traversible, step_size=5)
        planner.set_multi_goal(goal)

        result = planner.plan((4, 4))

        self.assertFalse(result.replan)
        self.assertTrue(result.searched)
        self.assertEqual(planner.search_count, 1)
        self.assertEqual(planner.goal_distance_compute_count, 1)
        planner.goal_distance()
        self.assertEqual(planner.goal_distance_compute_count, 1)

    def test_visualization_waypoints_come_from_the_same_path(self) -> None:
        traversible, goal = self._planner_inputs()
        planner = AStarPlanner(traversible, step_size=5)
        planner.set_multi_goal(goal)

        result = planner.plan((4, 4))
        waypoints = planner.sample_path(result.path, max_waypoints=10)

        self.assertGreater(len(waypoints), 1)
        self.assertLessEqual(len(waypoints), 11)
        self.assertEqual(
            (float(waypoints[1][0]), float(waypoints[1][1])),
            result.stg,
        )
        self.assertEqual(planner.search_count, 1)

    def test_failed_path_still_reuses_goal_distance_only(self) -> None:
        traversible = np.zeros((21, 21), dtype=np.uint8)
        traversible[2, 2] = 1
        traversible[-3, -3] = 1
        goal = np.zeros_like(traversible)
        goal[-3, -3] = 1

        first = AStarPlanner(traversible, step_size=5)
        first.set_multi_goal(goal)
        failed = first.plan((2, 2))
        self.assertTrue(failed.replan)
        self.assertFalse(failed.path)
        cache = AStarPathCache.capture(first, failed.path)

        second = AStarPlanner(traversible, step_size=5)
        second.set_multi_goal(goal)
        self.assertTrue(cache.restore_goal_distance(second))
        self.assertIsNone(cache.reusable_suffix(second, (2, 2)))
        second.plan((2, 2))
        self.assertEqual(second.search_count, 1)
        self.assertEqual(second.goal_distance_compute_count, 0)

    def test_unchanged_inputs_reuse_path_suffix_and_goal_distance(self) -> None:
        traversible, goal = self._planner_inputs()
        first = AStarPlanner(traversible, step_size=5)
        first.set_multi_goal(goal)
        initial = first.plan((4, 4))
        cache = AStarPathCache.capture(first, initial.path)
        next_start = tuple(map(int, initial.stg))

        second = AStarPlanner(traversible, step_size=5)
        second.set_multi_goal(goal)
        self.assertTrue(cache.restore_goal_distance(second))
        suffix = cache.reusable_suffix(second, next_start)
        reused = second.plan(next_start, reusable_path=suffix)

        self.assertFalse(reused.searched)
        self.assertEqual(second.search_count, 0)
        self.assertEqual(second.goal_distance_compute_count, 0)
        self.assertEqual(reused.path, suffix)

        changed = traversible.copy()
        changed[0, 0] = 0
        third = AStarPlanner(changed, step_size=5)
        third.set_multi_goal(goal)
        self.assertTrue(cache.restore_goal_distance(third))
        self.assertIsNone(cache.reusable_suffix(third, next_start))
        replanned = third.plan(next_start)
        self.assertTrue(replanned.searched)
        self.assertEqual(third.search_count, 1)
        self.assertEqual(third.goal_distance_compute_count, 0)

        risk_map = np.zeros_like(traversible, dtype=np.float32)
        risk_map[next_start] = 1.0
        risk_changed = AStarPlanner(
            traversible,
            step_size=5,
            risk_map=risk_map,
            risk_alpha=4.0,
        )
        risk_changed.set_multi_goal(goal)
        self.assertTrue(cache.restore_goal_distance(risk_changed))
        self.assertIsNone(
            cache.reusable_suffix(risk_changed, next_start)
        )

        changed_goal = np.zeros_like(goal)
        changed_goal[-6, -6] = 1
        goal_changed = AStarPlanner(traversible, step_size=5)
        goal_changed.set_multi_goal(changed_goal)
        self.assertFalse(cache.restore_goal_distance(goal_changed))
        self.assertIsNone(cache.reusable_suffix(goal_changed, next_start))

    @staticmethod
    def _agent_fixture(agent_class):
        agent = agent_class.__new__(agent_class)
        agent.local_w = agent.local_h = 61
        agent.visited_vis = np.zeros((61, 61), dtype=np.uint8)
        agent.collision_map = np.zeros((61, 61), dtype=np.uint8)
        agent.goal_name = "chair"
        agent.goal_map = np.zeros((61, 61), dtype=np.float32)
        agent.goal_map[30, 50] = 1
        agent.replan_count = 0
        agent.risk_navigation_enabled = False
        agent.risk_map = None
        agent.hard_unsafe_mask = None
        agent.risk_alpha = 4.0
        agent._astar_path_cache = None
        agent.args = SimpleNamespace(
            local_planner="astar",
            rl_local_checkpoint=None,
            rl_local_device="cpu",
            rl_local_deterministic=1,
            rl_local_crop_size=15,
            rl_local_rollout_steps=3,
        )
        return agent

    def test_each_agent_searches_once_then_reuses_unchanged_path(self) -> None:
        from agents.vlm_agents import VLM_Agent as SingleProcessAgent
        from agents.vlm_multi_agents import VLM_Agent as VectorAgent

        original_plan_path = AStarPlanner._plan_path
        calls = []

        def counted_plan_path(planner, *args, **kwargs):
            calls.append(planner)
            return original_plan_path(planner, *args, **kwargs)

        with mock.patch.object(
            AStarPlanner,
            "_plan_path",
            new=counted_plan_path,
        ):
            for agent_class in (SingleProcessAgent, VectorAgent):
                with self.subTest(agent=agent_class.__module__):
                    agent = self._agent_fixture(agent_class)
                    call_start = len(calls)
                    stg, stop, path = agent._get_stg(
                        np.zeros((61, 61), dtype=np.uint8),
                        [30, 10],
                        agent.goal_map.copy(),
                    )
                    self.assertFalse(stop)
                    self.assertGreater(len(path), 1)
                    self.assertEqual(len(calls) - call_start, 1)

                    agent._get_stg(
                        np.zeros((61, 61), dtype=np.uint8),
                        [int(round(stg[0])), int(round(stg[1]))],
                        agent.goal_map.copy(),
                    )
                    self.assertEqual(len(calls) - call_start, 1)


if __name__ == "__main__":
    unittest.main()
