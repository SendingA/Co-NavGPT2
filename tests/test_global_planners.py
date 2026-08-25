"""Headless regressions for the global planner interface."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from utils.global_planners import (
    AgentFrontierMap,
    GlobalPlannerContext,
    RiskPlanningContext,
    create_global_planner,
)
from utils.global_planners.co_ut import CostUtilityGlobalPlanner
from utils.global_planners.errors import GPTResponseError
from utils.global_planners.gpt import GPTGlobalPlanner
from utils.global_planners.risk_aware import RiskAwareGlobalPlanner
from utils.global_planners.risk_aware import risk_utility_weights
from utils.global_planners.risk_module import (
    grid_line_cells,
    risk_aware_route_cells,
)


def _context(
    *,
    points=((2, 2), (8, 8)),
    scores=(1.0, 5.0),
    poses=((1, 1, 0.0), (9, 9, 0.0)),
    cells=((1, 1), (9, 9)),
    local_step=25,
    risk=None,
    risk_by_agent=None,
    episode_index=0,
):
    shape = (12, 12)
    labels = np.zeros(shape, dtype=np.int32)
    for frontier_id, point in enumerate(points):
        labels[int(point[0]), int(point[1])] = frontier_id + 1
    return GlobalPlannerContext(
        target_score=scores,
        target_edge_map=labels,
        target_points=[list(point) for point in points],
        poses=[list(pose) for pose in poses],
        agent_cells=[list(cell) for cell in cells],
        obstacle_map=np.zeros(shape, dtype=np.float32),
        explored_map=np.ones(shape, dtype=np.float32),
        top_view_map=np.zeros((*shape, 3), dtype=np.uint8),
        goal_name="chair",
        local_step=local_step,
        navigation_step=37,
        num_agents=len(poses),
        risk=risk,
        risk_by_agent=risk_by_agent,
        episode_index=episode_index,
    )


def _agent_frontier_map(
    points,
    scores,
    *,
    explored=None,
    obstacle=None,
):
    shape = (12, 12)
    labels = np.zeros(shape, dtype=np.int32)
    for frontier_id, point in enumerate(points):
        labels[int(point[0]), int(point[1])] = frontier_id + 1
    return AgentFrontierMap(
        target_score=scores,
        target_edge_map=labels,
        target_points=[list(point) for point in points],
        obstacle_map=(
            np.zeros(shape, dtype=np.float32)
            if obstacle is None
            else np.asarray(obstacle, dtype=np.float32)
        ),
        explored_map=(
            np.ones(shape, dtype=np.float32)
            if explored is None
            else np.asarray(explored, dtype=np.float32)
        ),
        top_view_map=np.zeros((*shape, 3), dtype=np.uint8),
    )


class ClassicalGlobalPlannerTests(unittest.TestCase):
    def test_co_ut_default_lambda_is_point_five(self) -> None:
        self.assertEqual(CostUtilityGlobalPlanner().cost_utility_lambda, 0.5)
        result = create_global_planner("co_ut").plan(
            _context(
                scores=(10.0, 15.0),
                poses=((1, 1, 0.0), (9, 9, 0.0)),
                cells=((1, 1), (9, 9)),
            )
        )
        self.assertEqual(result.frontier_assignments, {0: 1, 1: 1})

    def test_classical_modes_use_local_frontiers_and_exclude_collision(
        self,
    ) -> None:
        agent_maps = [
            _agent_frontier_map(
                ((2, 2), (2, 9)),
                (20.0, 1.0),
            ),
            _agent_frontier_map(
                ((2, 3), (9, 9)),
                (20.0, 5.0),
            ),
        ]
        # The two local segmentations describe the same physical frontier but
        # have different centroids. Component overlap, rather than point
        # equality alone, must keep the second robot from selecting it again.
        agent_maps[0].target_edge_map[2, 3] = 1
        context = _context(
            poses=((1, 1, 0.0), (9, 9, 0.0)),
            cells=((1, 1), (9, 9)),
        )

        for mode in ("nearest", "co_ut", "fill"):
            with self.subTest(mode=mode):
                result = create_global_planner(mode).plan_individual_maps(
                    context,
                    agent_maps,
                )
                self.assertEqual(result.goal_points, [[2, 2], [9, 9]])
                self.assertEqual(
                    result.frontier_assignments,
                    {0: 0, 1: 3},
                )

    def test_random_samples_from_each_robot_local_map(self) -> None:
        first_explored = np.zeros((12, 12), dtype=np.float32)
        first_explored[1:5, 1:5] = 1.0
        second_explored = np.zeros((12, 12), dtype=np.float32)
        second_explored[7:11, 7:11] = 1.0
        agent_maps = [
            _agent_frontier_map((), (), explored=first_explored),
            _agent_frontier_map((), (), explored=second_explored),
        ]
        context = _context(
            points=(),
            scores=(),
            poses=((1, 1, 0.0), (9, 9, 0.0)),
            cells=((1, 1), (9, 9)),
            episode_index=9,
        )

        result = create_global_planner(
            "random",
            random_seed=7,
            random_goal_min_distance_m=0.0,
        ).plan_individual_maps(context, agent_maps)

        self.assertGreater(first_explored[tuple(result.goal_points[0])], 0.0)
        self.assertGreater(second_explored[tuple(result.goal_points[1])], 0.0)
        self.assertNotEqual(result.goal_points[0], result.goal_points[1])
        self.assertEqual(result.frontier_assignments, {0: None, 1: None})

    def test_nearest_preserves_independent_shared_assignment(self) -> None:
        result = create_global_planner("nearest").plan(_context())

        self.assertEqual(result.goal_points, [[2, 2], [8, 8]])
        self.assertEqual(result.frontier_assignments, {0: 0, 1: 1})

    def test_co_ut_maximizes_frontier_size_minus_lambda_distance(self) -> None:
        context = _context(
            scores=(10.0, 15.0),
            poses=((1, 1, 0.0), (5, 5, 0.0)),
            cells=((1, 1), (5, 5)),
        )
        result = create_global_planner(
            "co_ut",
            cost_utility_lambda=1.0,
        ).plan(context)

        self.assertEqual(result.frontier_assignments, {0: 0, 1: 1})
        self.assertEqual(result.goal_points, [[2, 2], [8, 8]])

    def test_co_ut_lambda_zero_selects_largest_frontier(self) -> None:
        result = create_global_planner(
            "co_ut",
            cost_utility_lambda=0.0,
        ).plan(_context(scores=(1.0, 5.0)))

        self.assertEqual(result.frontier_assignments, {0: 1, 1: 1})

    def test_co_ut_rejects_invalid_lambda(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            create_global_planner("co_ut", cost_utility_lambda=-0.1)

    def test_co_ut_recovers_frontier_size_from_label_map(self) -> None:
        context = _context(scores=None)
        context.target_edge_map[2, 3:6] = 1
        result = create_global_planner(
            "co_ut",
            cost_utility_lambda=0.0,
        ).plan(context)

        self.assertEqual(result.frontier_assignments, {0: 0, 1: 0})

    def test_fill_uses_highest_frontier_score_for_every_robot(self) -> None:
        result = create_global_planner("fill").plan(_context())

        self.assertEqual(result.frontier_assignments, {0: 1, 1: 1})
        self.assertEqual(result.goal_points, [[8, 8], [8, 8]])

    def test_empty_frontier_random_fallback_matches_legacy_rng(self) -> None:
        context = _context(points=(), scores=())
        np.random.seed(13)
        expected = []
        for _ in range(context.num_agents):
            action = np.random.rand(1, 2).squeeze() * 11
            expected.append([int(action[0]), int(action[1])])

        np.random.seed(13)
        result = create_global_planner("nearest").plan(context)

        self.assertEqual(result.goal_points, expected)
        self.assertEqual(result.frontier_assignments, {0: None, 1: None})

    def test_random_samples_reachable_free_goals_reproducibly(self) -> None:
        context = _context(
            points=(),
            scores=(),
            poses=((1, 1, 0.0), (4, 4, 0.0)),
            cells=((1, 1), (4, 4)),
            episode_index=7,
        )
        context.explored_map[:] = 0.0
        context.explored_map[1:6, 1:6] = 1.0
        context.obstacle_map[3, 3] = 1.0
        first = create_global_planner(
            "random",
            random_seed=19,
            random_goal_min_distance_m=1.0,
            map_resolution_cm=100.0,
        ).plan(context)
        second = create_global_planner(
            "random",
            random_seed=19,
            random_goal_min_distance_m=1.0,
            map_resolution_cm=100.0,
        ).plan(context)

        self.assertEqual(first.goal_points, second.goal_points)
        self.assertEqual(first.frontier_assignments, {0: None, 1: None})
        self.assertEqual(len({tuple(goal) for goal in first.goal_points}), 2)
        for row, col in first.goal_points:
            self.assertGreater(context.explored_map[row, col], 0.0)
            self.assertLessEqual(context.obstacle_map[row, col], 0.5)

    def test_random_seed_changes_sampled_goal(self) -> None:
        context = _context(points=(), scores=(), episode_index=2)
        first = create_global_planner(
            "random",
            random_seed=3,
            random_goal_min_distance_m=0.0,
        ).plan(context)
        second = create_global_planner(
            "random",
            random_seed=4,
            random_goal_min_distance_m=0.0,
        ).plan(context)

        self.assertNotEqual(first.goal_points, second.goal_points)

    def test_factory_rejects_unknown_planner(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown global planner"):
            create_global_planner("greedy")


class _FakeChatBackend:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def get_all_candidate_maps(self, edge_map, top_view_map, poses):
        self.calls.append(("candidate_maps", edge_map, top_view_map, poses))
        return ["frontier-0", "frontier-1"]

    def message_prepare(
        self,
        prompt,
        candidate_maps,
        goal_name,
        *,
        num_agents,
    ):
        self.calls.append(
            (
                "normal_message",
                prompt,
                candidate_maps,
                goal_name,
                num_agents,
            )
        )
        return ["normal"]

    def risk_message_prepare(
        self,
        prompt,
        candidate_maps,
        goal_name,
        *,
        risk_context,
        num_agents,
    ):
        self.calls.append(
            (
                "risk_message",
                prompt,
                candidate_maps,
                goal_name,
                risk_context,
                num_agents,
            )
        )
        return ["risk"]

    def chat_with_gpt4v(self, message, **kwargs):
        self.calls.append(("chat", message, kwargs))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class GPTGlobalPlannerTests(unittest.TestCase):
    def test_gpt_fallback_default_lambda_is_point_five(self) -> None:
        planner = GPTGlobalPlanner(
            chat_backend=_FakeChatBackend({}),
            prompts=SimpleNamespace(
                system_prompt="normal prompt",
                risk_prompt="risk prompt",
            ),
        )
        self.assertEqual(planner._fallback.cost_utility_lambda, 0.5)

    def test_gpt_explicitly_keeps_shared_frontier_map(self) -> None:
        planner = create_global_planner(
            "gpt",
            chat_backend=_FakeChatBackend(
                {"robot_0": "frontier_0", "robot_1": "frontier_1"}
            ),
            prompts=SimpleNamespace(
                system_prompt="normal prompt",
                risk_prompt="risk prompt",
            ),
        )

        self.assertTrue(planner.uses_shared_frontier_map)
        with self.assertRaisesRegex(RuntimeError, "shared frontier map"):
            planner.plan_individual_maps(
                _context(),
                [
                    _agent_frontier_map(((2, 2),), (1.0,)),
                    _agent_frontier_map(((8, 8),), (1.0,)),
                ],
            )

    def test_normal_gpt_uses_configurable_robot_count(self) -> None:
        backend = _FakeChatBackend(
            {"robot_0": "frontier_1", "robot_1": "frontier_0"}
        )
        planner = RiskAwareGlobalPlanner(
            GPTGlobalPlanner(
                chat_backend=backend,
                prompts=SimpleNamespace(
                    system_prompt="normal prompt",
                    risk_prompt="risk prompt",
                ),
            )
        )
        result = planner.plan(_context())

        self.assertEqual(result.goal_points, [[8, 8], [2, 2]])
        normal_call = next(call for call in backend.calls if call[0] == "normal_message")
        self.assertEqual(normal_call[-1], 2)
        self.assertEqual(normal_call[-2], "chair")
        chat_call = next(call for call in backend.calls if call[0] == "chat")
        self.assertEqual(
            chat_call[-1],
            {"num_agents": 2, "num_frontiers": 2},
        )

    def test_gpt_first_step_uses_random_fallback_without_api_call(self) -> None:
        backend = _FakeChatBackend(
            {"robot_0": "frontier_0", "robot_1": "frontier_1"}
        )
        planner = RiskAwareGlobalPlanner(
            GPTGlobalPlanner(
                chat_backend=backend,
                prompts=SimpleNamespace(
                    system_prompt="normal prompt",
                    risk_prompt="risk prompt",
                ),
            )
        )
        np.random.seed(5)
        result = planner.plan(_context(local_step=0))

        self.assertEqual(len(result.goal_points), 2)
        self.assertEqual(backend.calls, [])

    def test_failed_gpt_request_falls_back_to_normal_co_ut(self) -> None:
        backend = _FakeChatBackend(
            GPTResponseError(
                "empty response",
                reason="empty_content",
                attempts=5,
            )
        )
        planner = RiskAwareGlobalPlanner(
            GPTGlobalPlanner(
                chat_backend=backend,
                prompts=SimpleNamespace(
                    system_prompt="normal prompt",
                    risk_prompt="risk prompt",
                ),
            )
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            result = planner.plan(_context())

        self.assertEqual(result.frontier_assignments, {0: 0, 1: 1})
        self.assertEqual(result.goal_points, [[2, 2], [8, 8]])
        payload = json.loads(
            stream.getvalue().split("[gpt-fallback] ", 1)[1]
        )
        self.assertEqual(payload["mode"], "normal")
        self.assertEqual(payload["fallback_planner"], "co_ut")
        self.assertEqual(payload["reason"], "empty_content")
        self.assertEqual(payload["attempts"], 5)


class RiskAwareGlobalPlannerTests(unittest.TestCase):
    @staticmethod
    def _risk(hard_at=None):
        shape = (12, 12)
        planning_risk = np.zeros(shape, dtype=np.float32)
        hard = np.zeros(shape, dtype=bool)
        if hard_at is not None:
            planning_risk[hard_at] = 1.0
            hard[hard_at] = True
        return RiskPlanningContext(
            planning_risk=planning_risk,
            confidence=np.ones(shape, dtype=np.float32),
            hard_unsafe=hard,
            danger_threshold=0.55,
            hard_frontier_threshold=0.85,
            frontier_weight=2.0,
            map_resolution_cm=5.0,
        )

    def test_hard_frontier_is_reported_but_not_rejected(self) -> None:
        result = create_global_planner("nearest").plan(
            _context(risk=self._risk(hard_at=(2, 2)))
        )

        self.assertEqual(result.frontier_assignments, {0: 0, 1: 1})
        self.assertEqual(result.goal_points, [[2, 2], [8, 8]])
        self.assertTrue(result.frontier_reports[0].hard_blocked)
        self.assertEqual(result.frontier_computed_step, 37)

    def test_frontier_route_goes_around_fire_instead_of_scoring_a_line(self) -> None:
        shape = (15, 15)
        risk = np.zeros(shape, dtype=np.float32)
        hard = np.zeros(shape, dtype=bool)
        risk[7, 3:12] = 1.0
        hard[7, 3:12] = True

        route = risk_aware_route_cells(
            (7, 1),
            (7, 13),
            np.zeros(shape, dtype=np.float32),
            np.ones(shape, dtype=np.float32),
            risk,
            hard,
            risk_alpha=4.0,
        )

        self.assertIsNotNone(route)
        route_array = np.asarray(route, dtype=int)
        self.assertTrue(np.any(route_array[:, 0] != 7))
        self.assertTrue(
            np.any(
                risk[
                    np.asarray(grid_line_cells((7, 1), (7, 13), shape))[:, 0],
                    np.asarray(grid_line_cells((7, 1), (7, 13), shape))[:, 1],
                ]
                > 0.0
            )
        )
        self.assertFalse(np.any(hard[route_array[:, 0], route_array[:, 1]]))
        self.assertEqual(
            float(risk[route_array[:, 0], route_array[:, 1]].max()),
            0.0,
        )

    def test_co_ut_keeps_valuable_frontier_when_safe_detour_exists(self) -> None:
        shape = (12, 12)
        planning_risk = np.zeros(shape, dtype=np.float32)
        hard = np.zeros(shape, dtype=bool)
        planning_risk[6, 2:9] = 1.0
        hard[6, 2:9] = True
        risk = RiskPlanningContext(
            planning_risk=planning_risk,
            confidence=np.ones(shape, dtype=np.float32),
            hard_unsafe=hard,
            danger_threshold=0.55,
            hard_frontier_threshold=0.85,
            frontier_weight=2.0,
            map_resolution_cm=5.0,
            route_risk_alpha=4.0,
        )

        result = create_global_planner("co_ut").plan(
            _context(
                points=((6, 10), (2, 2)),
                scores=(50.0, 1.0),
                poses=((6, 1, 0.0),),
                cells=((6, 1),),
                risk=risk,
            )
        )

        self.assertEqual(result.frontier_assignments, {0: 0})
        self.assertFalse(result.frontier_reports[0].route_is_proxy)
        self.assertEqual(result.frontier_reports[0].route_max_risk, 0.0)

    def test_risk_reports_remain_in_each_agent_frontier_namespace(
        self,
    ) -> None:
        risk = self._risk(hard_at=(2, 2))
        agent_maps = [
            _agent_frontier_map(
                ((2, 2), (2, 8)),
                (20.0, 1.0),
            ),
            _agent_frontier_map(
                ((2, 2), (9, 9)),
                (20.0, 5.0),
            ),
        ]
        context = _context(
            poses=((1, 1, 0.0), (9, 9, 0.0)),
            cells=((1, 1), (9, 9)),
            risk=risk,
        )

        result = create_global_planner("fill").plan_individual_maps(
            context,
            agent_maps,
        )

        self.assertEqual(result.goal_points, [[2, 2], [9, 9]])
        self.assertEqual(result.frontier_assignments, {0: 0, 1: 3})
        self.assertEqual(
            [report.frontier_id for report in result.frontier_reports],
            [0, 1, 3],
        )
        self.assertEqual(
            result.frontier_report_agent_ids,
            [0, 0, 1],
        )

    def test_individual_maps_use_each_agents_matching_risk_floor(self) -> None:
        first_floor = self._risk(hard_at=(2, 2))
        second_floor = self._risk(hard_at=(9, 9))
        agent_maps = [
            _agent_frontier_map(
                ((2, 2), (2, 8)),
                (20.0, 1.0),
            ),
            _agent_frontier_map(
                ((2, 2), (9, 9)),
                (20.0, 5.0),
            ),
        ]
        context = _context(
            poses=((1, 1, 0.0), (9, 8, 0.0)),
            cells=((1, 1), (9, 8)),
            risk=first_floor,
            risk_by_agent=[first_floor, second_floor],
        )

        result = create_global_planner("fill").plan_individual_maps(
            context,
            agent_maps,
        )

        self.assertEqual(result.goal_points, [[2, 2], [9, 9]])
        self.assertEqual(result.frontier_assignments, {0: 0, 1: 3})
        self.assertEqual(result.frontier_report_agent_ids, [0, 0, 1])
        self.assertTrue(result.frontier_reports[0].hard_blocked)
        self.assertTrue(result.frontier_reports[2].hard_blocked)

    def test_zero_risk_preserves_every_classical_normal_policy(self) -> None:
        context_kwargs = {
            "scores": (10.0, 15.0),
            "poses": ((1, 1, 0.0), (5, 5, 0.0)),
            "cells": ((1, 1), (5, 5)),
            "episode_index": 8,
        }
        for mode in ("nearest", "co_ut", "fill", "random"):
            with self.subTest(mode=mode):
                kwargs = {
                    "random_seed": 17,
                    "random_goal_min_distance_m": 0.0,
                }
                normal = create_global_planner(mode, **kwargs).plan(
                    _context(**context_kwargs)
                )
                aware = create_global_planner(mode, **kwargs).plan(
                    _context(risk=self._risk(), **context_kwargs)
                )
                self.assertEqual(
                    aware.frontier_assignments,
                    normal.frontier_assignments,
                )
                self.assertEqual(aware.goal_points, normal.goal_points)

    def test_risk_does_not_replace_a_clearly_more_valuable_frontier(
        self,
    ) -> None:
        risk = self._risk()
        risk.planning_risk[2, 2] = 0.60
        context_kwargs = {
            "scores": (20.0, 1.0),
            "poses": ((1, 1, 0.0), (1, 2, 0.0)),
            "cells": ((1, 1), (1, 2)),
        }
        for mode in ("nearest", "co_ut", "fill"):
            with self.subTest(mode=mode):
                normal = create_global_planner(mode).plan(
                    _context(**context_kwargs)
                )
                aware = create_global_planner(mode).plan(
                    _context(risk=risk, **context_kwargs)
                )
                self.assertEqual(normal.frontier_assignments, {0: 0, 1: 0})
                self.assertEqual(
                    aware.frontier_assignments,
                    normal.frontier_assignments,
                )

    def test_comparable_fill_frontiers_use_lower_risk_as_tie_break(self) -> None:
        risk = self._risk()
        risk.planning_risk[2, 2] = 0.8
        result = create_global_planner("fill").plan(
            _context(
                points=((2, 2), (2, 3)),
                scores=(10.0, 9.5),
                poses=((1, 2, 0.0),),
                cells=((1, 2),),
                risk=risk,
            )
        )

        self.assertEqual(result.frontier_assignments, {0: 1})

    def test_uncertainty_does_not_change_frontier_choice(self) -> None:
        risk = self._risk()
        risk.confidence[2, 2] = 0.0
        result = create_global_planner("fill").plan(
            _context(
                points=((2, 2), (2, 3)),
                scores=(10.0, 9.5),
                poses=((1, 2, 0.0),),
                cells=((1, 2),),
                risk=risk,
            )
        )

        self.assertEqual(result.frontier_assignments, {0: 0})

    def test_risk_co_ut_weights_preserve_size_distance_tradeoff(self) -> None:
        weights = risk_utility_weights(
            "co_ut",
            frontier_weight=2.0,
            cost_utility_lambda=0.4,
        )

        self.assertEqual(weights.information_gain, 1.0)
        self.assertEqual(weights.distance, 0.4)
        self.assertEqual(weights.redundancy, 0.0)
        self.assertEqual(
            risk_utility_weights("co_ut", frontier_weight=2.0).distance,
            0.5,
        )

    def test_random_risk_mode_keeps_the_normal_sampling_domain(self) -> None:
        risk = self._risk()
        risk.planning_risk[:] = 1.0
        risk.planning_risk[1:5, 1:5] = 0.1
        risk.planning_risk[8:11, 8:11] = 0.1
        aware_context = _context(risk=risk, episode_index=4)
        normal_context = _context(episode_index=4)
        planner = create_global_planner(
            "random",
            random_seed=23,
            random_goal_min_distance_m=0.0,
        )
        result = planner.plan(aware_context)
        normal = planner.plan(normal_context)

        self.assertEqual(result.frontier_assignments, {0: None, 1: None})
        self.assertEqual(len(result.frontier_reports), 2)
        self.assertEqual(result.frontier_computed_step, 37)
        self.assertEqual(result.goal_points, normal.goal_points)

    def test_gpt_risk_choice_is_not_hard_rejected(self) -> None:
        backend = _FakeChatBackend(
            {"robot_0": "frontier_0", "robot_1": "frontier_0"}
        )
        planner = RiskAwareGlobalPlanner(
            GPTGlobalPlanner(
                chat_backend=backend,
                prompts=SimpleNamespace(
                    system_prompt="normal prompt",
                    risk_prompt="risk prompt",
                ),
            )
        )
        result = planner.plan(
            _context(risk=self._risk(hard_at=(2, 2)))
        )

        self.assertEqual(result.frontier_assignments, {0: 0, 1: 0})
        self.assertEqual(result.goal_points, [[2, 2], [2, 2]])
        risk_call = next(call for call in backend.calls if call[0] == "risk_message")
        self.assertEqual(risk_call[-1], 2)
        self.assertEqual(
            len(risk_call[-2]["hazard_report"]),
            len(result.frontier_reports),
        )

    def test_failed_risk_gpt_request_uses_risk_aware_co_ut(self) -> None:
        backend = _FakeChatBackend(
            GPTResponseError(
                "filtered",
                reason="content_filter",
                attempts=1,
            )
        )
        planner = RiskAwareGlobalPlanner(
            GPTGlobalPlanner(
                chat_backend=backend,
                prompts=SimpleNamespace(
                    system_prompt="normal prompt",
                    risk_prompt="risk prompt",
                ),
            )
        )

        stream = io.StringIO()
        with redirect_stdout(stream):
            result = planner.plan(_context(risk=self._risk()))

        expected = create_global_planner("co_ut").plan(
            _context(risk=self._risk())
        )
        self.assertEqual(
            result.frontier_assignments,
            expected.frontier_assignments,
        )
        self.assertEqual(result.goal_points, expected.goal_points)
        self.assertEqual(result.frontier_assignments, {0: 0, 1: 1})
        payload = json.loads(
            stream.getvalue().split("[gpt-fallback] ", 1)[1]
        )
        self.assertEqual(payload["mode"], "risk")
        self.assertEqual(payload["fallback_planner"], "co_ut")
        self.assertEqual(payload["reason"], "content_filter")


class MainGlobalPlannerBoundaryTests(unittest.TestCase):
    def test_entrypoints_delegate_assignment_without_mode_branches(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for filename in ("main.py", "main_vec.py"):
            with self.subTest(filename=filename):
                source = (root / filename).read_text(encoding="utf-8")
                self.assertIn(
                    "global_planner = create_global_planner(",
                    source,
                )
                self.assertIn("global_planner.plan(planner_context)", source)
                self.assertIn(
                    "global_planner.plan_individual_maps(",
                    source,
                )
                self.assertIn("individual_map_processes", source)
                self.assertNotIn('elif args.nav_mode == "nearest"', source)
                self.assertNotIn('elif args.nav_mode == "co_ut"', source)
                self.assertNotIn('elif args.nav_mode == "fill"', source)
                self.assertNotIn('elif args.nav_mode == "random"', source)


if __name__ == "__main__":
    unittest.main()
