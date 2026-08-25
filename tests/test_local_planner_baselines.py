"""Regression tests for FMM, A* and learned local-planner baselines."""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from arguments import get_args
from utils.fmm_planner import FMMPlanner
from utils.local_planners import (
    AStarPlanner,
    RLGridPlanner,
    RLGridPolicy,
    create_local_planner,
    resolve_fmm_backend,
    save_rl_checkpoint,
    validate_local_planner_config,
)
from utils.local_planners.rl import build_rl_observation


class LocalPlannerConfigTests(unittest.TestCase):
    def test_fmm_is_the_default_and_preserves_existing_backend(self) -> None:
        with mock.patch.object(sys, "argv", ["test_local_planners"]):
            args = get_args()
        self.assertEqual(args.local_planner, "fmm")
        self.assertFalse(hasattr(args, "local_planner_risk_aware"))
        planner = create_local_planner(
            args.local_planner, np.ones((5, 5), dtype=np.uint8)
        )
        self.assertIsInstance(planner, FMMPlanner)

    def test_fire_none_and_oracle_resolve_to_the_same_grid_fmm(self) -> None:
        common = {
            "fmm_backend": "auto",
            "fire_world": 1,
            "risk_enabled": 1,
        }
        none = SimpleNamespace(risk_source="none", **common)
        oracle = SimpleNamespace(risk_source="oracle", **common)

        self.assertEqual(resolve_fmm_backend(none), "grid")
        self.assertEqual(resolve_fmm_backend(oracle), "grid")

    def test_normal_auto_fmm_keeps_navmesh_backend(self) -> None:
        args = SimpleNamespace(
            fmm_backend="auto",
            fire_world=0,
            risk_enabled=0,
        )
        self.assertEqual(resolve_fmm_backend(args), "navmesh")

    def test_local_awareness_follows_effective_risk_source(self) -> None:
        common = {
            "local_planner": "fmm",
            "rl_local_crop_size": 31,
            "rl_local_rollout_steps": 5,
        }
        self.assertFalse(validate_local_planner_config(SimpleNamespace(
            risk_enabled=0, risk_source="sensed", **common
        )))
        self.assertTrue(validate_local_planner_config(SimpleNamespace(
            risk_enabled=1, risk_source="sensed", **common
        )))
        self.assertTrue(validate_local_planner_config(SimpleNamespace(
            risk_enabled=1, risk_source="oracle", **common
        )))
        self.assertFalse(validate_local_planner_config(SimpleNamespace(
            risk_enabled=1, risk_source="none", **common
        )))

    def test_removed_local_awareness_cli_is_rejected(self) -> None:
        argv = [
            "test_local_planners",
            "--local_planner_risk_aware",
            "off",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    get_args()


class AStarPlannerTests(unittest.TestCase):
    def test_astar_reaches_multi_goal_and_returns_stg(self) -> None:
        traversible = np.ones((13, 13), dtype=np.uint8)
        traversible[6, 1:11] = 0
        traversible[6, 10] = 1
        goal = np.zeros_like(traversible)
        goal[10, 10] = 1
        goal[11, 10] = 1
        planner = AStarPlanner(traversible, step_size=3)
        planner.set_multi_goal(goal)
        stg_x, stg_y, replan, stop = planner.get_short_term_goal((2, 2))
        self.assertFalse(replan)
        self.assertFalse(stop)
        self.assertGreater(len(planner.last_path), 1)
        self.assertTrue(bool(goal[planner.last_path[-1]]))
        self.assertGreater(
            np.linalg.norm(np.subtract((stg_x, stg_y), (2, 2))), 0
        )

    def test_astar_does_not_cut_diagonal_obstacle_corner(self) -> None:
        traversible = np.ones((4, 4), dtype=np.uint8)
        traversible[0, 1] = 0
        traversible[1, 0] = 0
        goal = np.zeros_like(traversible)
        goal[1, 1] = 1
        planner = AStarPlanner(traversible)
        planner.set_multi_goal(goal)
        _, _, replan, stop = planner.get_short_term_goal((0, 0))
        self.assertTrue(replan)
        self.assertFalse(stop)
        self.assertEqual(planner.last_path, [])

    def test_risk_aware_astar_detours_while_blind_astar_is_direct(self) -> None:
        traversible = np.ones((9, 11), dtype=np.uint8)
        start = (4, 1)
        goal = np.zeros_like(traversible)
        goal[4, 9] = 1
        risk = np.zeros_like(traversible, dtype=np.float32)
        risk[4, 2:9] = 1.0

        blind = AStarPlanner(traversible, step_size=2)
        blind.set_multi_goal(goal)
        blind.get_short_term_goal(start)

        aware = AStarPlanner(
            traversible,
            step_size=2,
            risk_map=risk,
            risk_alpha=10.0,
        )
        aware.set_multi_goal(goal)
        aware.get_short_term_goal(start)

        blind_exposure = sum(float(risk[cell]) for cell in blind.last_path)
        aware_exposure = sum(float(risk[cell]) for cell in aware.last_path)
        self.assertGreater(blind_exposure, aware_exposure)
        self.assertTrue(any(row != start[0] for row, _ in aware.last_path[1:-1]))

    def test_hard_mask_and_emergency_escape_are_supported(self) -> None:
        traversible = np.ones((7, 7), dtype=np.uint8)
        hard = np.zeros_like(traversible, dtype=bool)
        hard[3, 1:6] = True
        planner = AStarPlanner(
            traversible, hard_unsafe_mask=hard, step_size=2
        )
        escape = planner.prepare_emergency_escape((3, 3))
        self.assertIsNotNone(escape)
        self.assertEqual(int(np.sum(escape)), 1)
        self.assertFalse(planner.hard_unsafe_mask[3, 3])


def _constant_checkpoint(
    directory: str,
    *,
    risk_aware: bool,
    crop_size: int,
    preferred_action: int,
) -> Path:
    policy = RLGridPolicy(
        risk_aware=risk_aware,
        crop_size=crop_size,
        hidden_size=32,
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.policy_head.bias[preferred_action] = 5.0
    path = Path(directory) / (
        "aware.pth" if risk_aware else "blind.pth"
    )
    return save_rl_checkpoint(path, policy)


class RLGridPlannerTests(unittest.TestCase):
    def test_checkpointed_policy_produces_expected_waypoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _constant_checkpoint(
                directory,
                risk_aware=False,
                crop_size=15,
                preferred_action=3,  # move right
            )
            traversible = np.ones((15, 15), dtype=np.uint8)
            goal = np.zeros_like(traversible)
            goal[7, 12] = 1
            planner = RLGridPlanner(
                traversible,
                checkpoint_path=checkpoint,
                crop_size=15,
                rollout_steps=3,
                risk_aware=False,
            )
            planner.set_multi_goal(goal)
            stg_x, stg_y, replan, stop = planner.get_short_term_goal((7, 3))
            self.assertEqual((stg_x, stg_y), (7.0, 6.0))
            self.assertFalse(replan)
            self.assertFalse(stop)

    def test_aware_policy_masks_hard_unsafe_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _constant_checkpoint(
                directory,
                risk_aware=True,
                crop_size=15,
                preferred_action=3,
            )
            traversible = np.ones((15, 15), dtype=np.uint8)
            hard = np.zeros_like(traversible, dtype=bool)
            hard[7, 4] = True
            goal = np.zeros_like(traversible)
            goal[7, 12] = 1
            planner = RLGridPlanner(
                traversible,
                checkpoint_path=checkpoint,
                crop_size=15,
                rollout_steps=1,
                risk_aware=True,
                risk_map=np.zeros_like(traversible, dtype=np.float32),
                hard_unsafe_mask=hard,
            )
            planner.set_multi_goal(goal)
            planner.get_short_term_goal((7, 3))
            self.assertNotEqual(planner.last_path[-1], (7, 4))
            maps, _ = planner.policy_observation(
                (7, 3), np.zeros_like(traversible, dtype=bool)
            )
            self.assertEqual(maps.shape[0], 5)

    def test_checkpoint_awareness_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = _constant_checkpoint(
                directory,
                risk_aware=False,
                crop_size=15,
                preferred_action=3,
            )
            with self.assertRaisesRegex(ValueError, "does not match requested"):
                RLGridPlanner(
                    np.ones((15, 15), dtype=np.uint8),
                    checkpoint_path=checkpoint,
                    crop_size=15,
                    risk_aware=True,
                )

    def test_observation_schema_has_three_or_five_channels(self) -> None:
        shape = (9, 9)
        traversible = np.ones(shape, dtype=np.uint8)
        goal = np.zeros(shape, dtype=bool)
        goal[7, 7] = True
        visited = np.zeros(shape, dtype=bool)
        blind, _ = build_rl_observation(
            traversible,
            goal,
            visited,
            (4, 4),
            crop_size=9,
            risk_aware=False,
        )
        aware, _ = build_rl_observation(
            traversible,
            goal,
            visited,
            (4, 4),
            crop_size=9,
            risk_aware=True,
            risk_map=np.zeros(shape, dtype=np.float32),
            hard_unsafe_mask=np.zeros(shape, dtype=bool),
        )
        self.assertEqual(blind.shape, (3, 9, 9))
        self.assertEqual(aware.shape, (5, 9, 9))


class AgentPlannerIntegrationTests(unittest.TestCase):
    @staticmethod
    def _agent_fixture(agent_class, planner_name, checkpoint=None):
        agent = agent_class.__new__(agent_class)
        agent.local_w = agent.local_h = 41
        agent.visited_vis = np.zeros((41, 41), dtype=np.uint8)
        agent.collision_map = np.zeros((41, 41), dtype=np.uint8)
        agent.goal_name = "chair"
        agent.goal_map = np.zeros((41, 41), dtype=np.float32)
        agent.goal_map[20, 30] = 1
        agent.replan_count = 0
        agent.risk_navigation_enabled = False
        agent.risk_map = None
        agent.hard_unsafe_mask = None
        agent.risk_alpha = 4.0
        agent.args = SimpleNamespace(
            local_planner=planner_name,
            fmm_backend="grid",
            fire_world=1,
            risk_enabled=1,
            risk_source="none",
            rl_local_checkpoint=None if checkpoint is None else str(checkpoint),
            rl_local_device="cpu",
            rl_local_deterministic=1,
            rl_local_crop_size=15,
            rl_local_rollout_steps=3,
        )
        return agent

    def test_fire_none_and_oracle_use_identical_grid_geometry(self) -> None:
        from agents.vlm_agents import VLM_Agent as SingleProcessAgent
        from agents.vlm_multi_agents import VLM_Agent as VectorAgent

        class PlannerSpy:
            calls = []

            def __init__(self, traversible, **kwargs):
                self.traversible = np.asarray(traversible).copy()
                self.hard_unsafe_mask = np.zeros_like(
                    self.traversible, dtype=bool
                )
                self.states = []
                type(self).calls.append(self)

            def set_multi_goal(self, goal):
                self.goal = np.asarray(goal).copy()

            def get_short_term_goal(self, state):
                self.states.append(tuple(state))
                return float(state[0]), float(state[1]), False, False

        for agent_class in (SingleProcessAgent, VectorAgent):
            module = agent_class.__module__
            PlannerSpy.calls = []
            blind = self._agent_fixture(agent_class, "fmm")
            aware = self._agent_fixture(agent_class, "fmm")
            aware.risk_navigation_enabled = True
            aware.risk_map = np.zeros((41, 41), dtype=np.float32)
            aware.args.risk_source = "oracle"

            with mock.patch(f"{module}.FMMPlanner", PlannerSpy):
                blind._get_stg(
                    np.zeros((41, 41), dtype=np.uint8),
                    [20, 10],
                    blind.goal_map.copy(),
                )
                aware._get_stg(
                    np.zeros((41, 41), dtype=np.uint8),
                    [20, 10],
                    aware.goal_map.copy(),
                )

            self.assertEqual(len(PlannerSpy.calls), 2)
            none_planner, oracle_planner = PlannerSpy.calls
            np.testing.assert_array_equal(
                none_planner.traversible, oracle_planner.traversible
            )
            np.testing.assert_array_equal(
                none_planner.goal, oracle_planner.goal
            )
            self.assertEqual(
                none_planner.states[0], oracle_planner.states[0]
            )
            self.assertEqual(none_planner.states[0], (21, 11))
            self.assertTrue(np.all(none_planner.traversible[0, :] == 0))
            self.assertTrue(np.all(none_planner.traversible[-1, :] == 0))

    def test_agent_variants_run_astar_in_blind_and_aware_modes(self) -> None:
        from agents.vlm_agents import VLM_Agent as SingleProcessAgent
        from agents.vlm_multi_agents import VLM_Agent as VectorAgent

        for agent_class in (SingleProcessAgent, VectorAgent):
            for aware in (False, True):
                with self.subTest(agent=agent_class.__module__, aware=aware):
                    agent = self._agent_fixture(agent_class, "astar")
                    agent.risk_navigation_enabled = aware
                    if aware:
                        agent.risk_map = np.zeros((41, 41), dtype=np.float32)
                        agent.risk_map[20, 22:29] = 1.0
                        agent.hard_unsafe_mask = np.zeros(
                            (41, 41), dtype=bool
                        )
                    stg, stop, path = agent._get_stg(
                        np.zeros((41, 41), dtype=np.uint8),
                        [20, 10],
                        agent.goal_map.copy(),
                    )
                    self.assertEqual(len(stg), 2)
                    self.assertFalse(stop)
                    self.assertGreater(len(path), 1)

    def test_fmm_and_astar_ignore_hard_masks_in_soft_risk_mode(self) -> None:
        from agents.vlm_agents import VLM_Agent as SingleProcessAgent
        from agents.vlm_multi_agents import VLM_Agent as VectorAgent

        for agent_class in (SingleProcessAgent, VectorAgent):
            for planner_name in ("fmm", "astar"):
                with self.subTest(
                    agent=agent_class.__module__, planner=planner_name
                ):
                    agent = self._agent_fixture(agent_class, planner_name)
                    agent.risk_navigation_enabled = True
                    agent.risk_map = np.zeros((41, 41), dtype=np.float32)
                    agent.risk_map[20, 22:29] = 1.0
                    # If this mask reached the planner, the goal would be
                    # converted into a safety waypoint or escape action.
                    agent.hard_unsafe_mask = np.ones(
                        (41, 41), dtype=bool
                    )

                    stg, stop, path = agent._get_stg(
                        np.zeros((41, 41), dtype=np.uint8),
                        [20, 10],
                        agent.goal_map.copy(),
                    )

                    self.assertEqual(len(stg), 2)
                    self.assertFalse(stop)
                    self.assertGreater(len(path), 1)
                    self.assertFalse(agent._risk_escape_active)
                    self.assertIsNone(agent._risk_escape_reason)

    def test_agent_variants_run_rl_in_blind_and_aware_modes(self) -> None:
        from agents.vlm_agents import VLM_Agent as SingleProcessAgent
        from agents.vlm_multi_agents import VLM_Agent as VectorAgent

        with tempfile.TemporaryDirectory() as directory:
            checkpoints = {
                aware: _constant_checkpoint(
                    directory,
                    risk_aware=aware,
                    crop_size=15,
                    preferred_action=3,
                )
                for aware in (False, True)
            }
            for agent_class in (SingleProcessAgent, VectorAgent):
                for aware in (False, True):
                    with self.subTest(
                        agent=agent_class.__module__, aware=aware
                    ):
                        agent = self._agent_fixture(
                            agent_class, "rl", checkpoints[aware]
                        )
                        agent.risk_navigation_enabled = aware
                        if aware:
                            agent.risk_map = np.zeros(
                                (41, 41), dtype=np.float32
                            )
                            agent.hard_unsafe_mask = np.zeros(
                                (41, 41), dtype=bool
                            )
                        stg, stop, path = agent._get_stg(
                            np.zeros((41, 41), dtype=np.uint8),
                            [20, 10],
                            agent.goal_map.copy(),
                        )
                        self.assertEqual(len(stg), 2)
                        self.assertFalse(stop)
                        self.assertGreater(len(path), 1)


if __name__ == "__main__":
    unittest.main()
