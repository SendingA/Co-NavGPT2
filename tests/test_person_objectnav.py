"""Unit tests for static-person ObjectNav dataset/runtime helpers."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from constants import category_to_id
from scripts.build_person_objectnav_dataset import (
    ViewPointSamplingConfig,
    _coverage_probe_positions,
    _sample_view_points,
)
from utils.fire_sensors.humans_thermal import (
    HumanThermalTarget,
    project_humans_to_thermal,
)
from utils.person_objectnav import (
    PERSON_CATEGORY_ID,
    add_person_category_mappings,
    objectnav_goal_debug_info,
    person_goal_positions,
    person_goals_key,
    refresh_simulator_observations,
    validate_person_dataset_dict,
    yaw_quaternion_facing,
)


def _valid_dataset() -> dict:
    scene_id = "hm3d_v0.2/val/00000-Test/Test.basis.glb"
    return {
        "category_to_task_category_id": {"chair": 0, "person": 6},
        "category_to_scene_annotation_category_id": {
            "chair": 0,
            "person": 6,
        },
        "goals_by_category": {
            person_goals_key(scene_id): [
                {
                    "position": [1.0, 0.0, 2.0],
                    "view_points": [
                        {
                            "agent_state": {
                                "position": [1.0, 0.0, 2.5],
                                "rotation": [0.0, 0.0, 0.0, 1.0],
                            }
                        }
                    ],
                }
            ]
        },
        "episodes": [
            {
                "scene_id": scene_id,
                "object_category": "person",
            }
        ],
    }


class PersonObjectNavTests(unittest.TestCase):
    def test_person_id_is_aligned_with_detector_order(self) -> None:
        detector_classes = [
            "chair", "bed", "potted plant", "toilet",
            "tv_screen", "couch", "person", "fire",
        ]
        self.assertEqual(category_to_id[PERSON_CATEGORY_ID], "person")
        self.assertEqual(detector_classes[PERSON_CATEGORY_ID], "person")

    def test_category_mapping_rejects_id_collision(self) -> None:
        dataset = {
            "category_to_task_category_id": {"other": 6},
            "category_to_scene_annotation_category_id": {},
        }
        with self.assertRaises(ValueError):
            add_person_category_mappings(dataset)

    def test_valid_generated_dataset_shape(self) -> None:
        validate_person_dataset_dict(_valid_dataset())

    def test_person_goal_positions_only_for_person_episode(self) -> None:
        goal = SimpleNamespace(position=[1.0, 0.0, 2.0])
        episode = SimpleNamespace(object_category="person", goals=[goal])
        positions = person_goal_positions(episode)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].tolist(), [1.0, 0.0, 2.0])

        chair_episode = SimpleNamespace(object_category="chair", goals=[goal])
        self.assertEqual(person_goal_positions(chair_episode), [])

    def test_view_rotation_faces_target(self) -> None:
        quat = yaw_quaternion_facing([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
        self.assertTrue(math.isclose(quat[1], 0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(quat[3], 1.0, abs_tol=1e-8))

    def test_refresh_replaces_rendered_frames_and_keeps_task_sensors(self) -> None:
        class FakeSim:
            def __init__(self):
                self.last_action = "not-called"

            def step(self, action):
                self.last_action = action
                return [
                    {
                        "rgb": f"fresh-rgb-{agent_id}-processed",
                        "depth": f"fresh-depth-{agent_id}-processed",
                    }
                    for agent_id in range(2)
                ]

        reset_observations = [
            {
                "rgb": "stale-rgb-0",
                "depth": "stale-depth-0",
                "objectgoal": [6],
                "gps": [1.0, 2.0],
            },
            {
                "rgb": "stale-rgb-1",
                "depth": "stale-depth-1",
                "objectgoal": [6],
                "compass": [0.5],
            },
        ]
        sim = FakeSim()
        refreshed = refresh_simulator_observations(sim, reset_observations, 2)

        self.assertIsNone(sim.last_action)
        self.assertEqual(refreshed[0]["rgb"], "fresh-rgb-0-processed")
        self.assertEqual(refreshed[1]["depth"], "fresh-depth-1-processed")
        self.assertEqual(refreshed[0]["objectgoal"], [6])
        self.assertEqual(refreshed[0]["gps"], [1.0, 2.0])
        self.assertEqual(refreshed[1]["compass"], [0.5])

    def test_goal_debug_uses_both_agents_and_all_goals(self) -> None:
        class FakeSim:
            positions = (
                np.array([0.0, 0.0, 0.0]),
                np.array([2.0, 0.0, 0.0]),
            )

            def get_agent_state(self, agent_id):
                return SimpleNamespace(position=self.positions[agent_id])

        def goal(position, view_positions):
            return SimpleNamespace(
                position=position,
                view_points=[
                    SimpleNamespace(
                        agent_state=SimpleNamespace(position=view_position)
                    )
                    for view_position in view_positions
                ],
            )

        episode = SimpleNamespace(
            goals=[
                goal([10.0, 0.0, 0.0], [[0.1, 0.0, 0.0]]),
                goal([3.0, 0.0, 0.0], [[9.0, 0.0, 0.0]]),
            ]
        )
        info = objectnav_goal_debug_info(FakeSim(), episode, 2)

        self.assertEqual(len(info["agent_positions"]), 2)
        self.assertEqual(info["nearest_agent_id"], 1)
        self.assertEqual(info["goal_index"], 1)
        np.testing.assert_allclose(info["goal_position"], [3.0, 0.0, 0.0])
        self.assertAlmostEqual(info["nearest_goal_l2"], 1.0)

    def test_goal_debug_reports_nearest_agent_for_single_goal(self) -> None:
        class FakeSim:
            def get_agent_state(self, agent_id):
                return SimpleNamespace(
                    position=np.array([float(agent_id), 0.0, 0.0])
                )

        episode = SimpleNamespace(
            goals=[SimpleNamespace(position=[1.25, 0.0, 0.0], view_points=[])]
        )
        info = objectnav_goal_debug_info(FakeSim(), episode, 2)

        self.assertEqual(info["nearest_agent_id"], 1)
        self.assertAlmostEqual(info["nearest_goal_l2"], 0.25)

    def test_dataset_rejects_person_goal_without_close_view_point(self) -> None:
        dataset = _valid_dataset()
        goal = next(iter(dataset["goals_by_category"].values()))[0]
        goal["view_points"][0]["agent_state"]["position"] = [1.0, 0.0, 3.0]

        with self.assertRaisesRegex(ValueError, "no close view point"):
            validate_person_dataset_dict(dataset)

    def test_dense_view_points_cover_close_region_and_historical_stop(self) -> None:
        class FlatPathfinder:
            @staticmethod
            def get_island(_position):
                return 0

            @staticmethod
            def snap_point(position, island_index=None):
                del island_index
                return np.asarray(position, dtype=np.float64)

        sim = SimpleNamespace(pathfinder=FlatPathfinder())
        config = ViewPointSamplingConfig()
        goal = np.asarray(
            [-2.2226400375, 0.0465418845, -1.7866499424],
            dtype=np.float64,
        )
        with mock.patch(
            "scripts.build_person_objectnav_dataset._line_of_sight",
            return_value=True,
        ):
            view_points = _sample_view_points(sim, goal, config)
            probes = _coverage_probe_positions(sim, goal, config)

        positions = np.asarray(
            [view["agent_state"]["position"] for view in view_points]
        )
        self.assertGreater(len(positions), 500)
        radii = np.linalg.norm((positions - goal)[:, [0, 2]], axis=1)
        self.assertLess(float(np.min(radii)), 0.30)

        max_probe_error = max(
            float(np.min(np.linalg.norm(
                (positions - probe)[:, [0, 2]], axis=1
            )))
            for probe in probes
        )
        self.assertLess(max_probe_error, config.coverage_distance)

        historical_stop = np.asarray([-2.27, 0.05, -2.03])
        nearest_historical_view = float(np.min(np.linalg.norm(
            positions - historical_stop, axis=1
        )))
        self.assertLess(nearest_historical_view, config.coverage_distance)

    def test_person_and_chair_share_detected_object_navigation(self) -> None:
        try:
            from agents.vlm_agents import VLM_Agent as SingleProcessAgent
            from agents.vlm_multi_agents import VLM_Agent as VectorAgent
        except ImportError as exc:  # pragma: no cover - dependency-only skip
            self.skipTest(f"VLM agent dependencies unavailable: {exc}")

        for agent_class in (SingleProcessAgent, VectorAgent):
            results = {}
            for category in ("chair", "person"):
                with self.subTest(
                    agent_module=agent_class.__module__, category=category
                ):
                    agent = agent_class.__new__(agent_class)
                    agent.goal_name = category
                    agent.object_pcd = SimpleNamespace(points=[object()])
                    agent.found_goal = False
                    agent.goal_map = np.zeros((31, 31), dtype=np.float32)
                    agent.local_w = agent.local_h = 31
                    agent.last_goal = None
                    agent.curr_frontier_count = 0
                    agent.camera_position = np.asarray([0.0, 0.0, 0.0])
                    agent.origins_grid = np.asarray([15, 15])
                    agent.init_sim_rotation = np.eye(3)
                    agent.init_agent_position = np.zeros(3)
                    agent.args = SimpleNamespace(map_resolution=5)
                    agent.object_map_building = lambda _pcd: (
                        np.asarray([17]), np.asarray([14])
                    )
                    agent.find_nearest_point_cloud = lambda _pcd, _camera: \
                        np.asarray([0.3, 0.0, -0.2])
                    agent.search_navigable_path = lambda goal: [
                        np.zeros(3), np.asarray(goal)
                    ]
                    agent.greedy_follower_act = lambda _path: 1

                    module_name = agent_class.__module__
                    with mock.patch(
                        f"{module_name}.process_pcd",
                        return_value=SimpleNamespace(points=[object()]),
                    ):
                        action = agent.act([5, 6])
                    results[category] = (
                        agent.goal_map.copy(),
                        agent.nearest_point.copy(),
                        agent.habitat_goal_pose.copy(),
                        action,
                    )

            np.testing.assert_array_equal(results["chair"][0], results["person"][0])
            np.testing.assert_allclose(results["chair"][1], results["person"][1])
            np.testing.assert_allclose(results["chair"][2], results["person"][2])
            self.assertEqual(results["chair"][3], results["person"][3])

    def test_person_and_chair_share_fmm_and_greedy_stop(self) -> None:
        try:
            from agents.vlm_agents import VLM_Agent as SingleProcessAgent
            from agents.vlm_multi_agents import VLM_Agent as VectorAgent
        except ImportError as exc:  # pragma: no cover - dependency-only skip
            self.skipTest(f"VLM agent dependencies unavailable: {exc}")

        class StopFollower:
            @staticmethod
            def get_next_action(*_args):
                return 0

        for agent_class in (SingleProcessAgent, VectorAgent):
            fmm_results = {}
            greedy_results = {}
            for category in ("chair", "person"):
                agent = agent_class.__new__(agent_class)
                agent.local_w = agent.local_h = 51
                agent.visited_vis = np.zeros((51, 51), dtype=np.uint8)
                agent.collision_map = np.zeros((51, 51), dtype=np.uint8)
                agent.goal_name = category
                agent.goal_map = np.zeros((51, 51), dtype=np.float32)
                agent.goal_map[25, 25] = 1
                agent.replan_count = 0
                fmm_results[category] = agent._get_stg(
                    np.zeros((51, 51), dtype=np.uint8),
                    [40, 25],
                    agent.goal_map.copy(),
                )

                agent.is_running = True
                agent.follower = StopFollower()
                agent.habitat_goal_pose = np.zeros(3)
                agent.current_grid_pose = [25, 25]
                agent.relative_angle = 0.0
                agent.origins_grid = [25, 25]
                agent.args = SimpleNamespace(map_resolution=5, turn_angle=30)
                agent.found_goal = True
                agent.greedy_stop_count = 0
                agent.l_step = 0
                agent.explored_map = np.ones((51, 51), dtype=np.uint8)
                agent.map_size = 51
                agent.eve_angle = 0
                greedy_results[category] = agent.greedy_follower_act(
                    np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
                )

            self.assertEqual(fmm_results["chair"][0], fmm_results["person"][0])
            self.assertEqual(fmm_results["chair"][1], fmm_results["person"][1])
            np.testing.assert_allclose(
                fmm_results["chair"][2], fmm_results["person"][2]
            )
            self.assertEqual(greedy_results["chair"], 0)
            self.assertEqual(greedy_results["person"], 0)

    def test_depth_wall_is_not_fallback_thermal_person(self) -> None:
        sensor_state = SimpleNamespace(
            position=np.zeros(3, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
        )
        agent_state = SimpleNamespace(sensor_states={"depth": sensor_state})
        camera_k = np.array(
            [[60.0, 0.0, 50.0], [0.0, 60.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        target = HumanThermalTarget(
            position=np.array([0.0, 0.0, -2.0], dtype=np.float64)
        )

        # A flat surface at the expected target depth used to be selected as
        # one huge connected component, then injected as person:0.95.
        wall_depth = np.full((100, 100), 2.0, dtype=np.float32)
        mask = project_humans_to_thermal(
            [target], agent_state, camera_k, (100, 100), wall_depth,
            max_depth_m=5.0,
        )
        self.assertEqual(float(mask.max()), 0.0)

        # With no depth sensor at all the legacy visualization fallback is
        # still available for non-benchmark callers.
        no_depth_mask = project_humans_to_thermal(
            [target], agent_state, camera_k, (100, 100), None,
            max_depth_m=5.0,
        )
        self.assertGreater(float(no_depth_mask.max()), 0.0)

if __name__ == "__main__":
    unittest.main()
