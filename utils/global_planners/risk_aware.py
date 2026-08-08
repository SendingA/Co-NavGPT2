"""Composable risk-aware decorator for global planners."""
from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Set, Tuple

import numpy as np

from utils.risk.frontier import UtilityWeights

from .base import (
    AgentFrontierMap,
    GlobalPlanner,
    GlobalPlannerContext,
    GlobalPlannerResult,
    goal_from_frontier,
)
from .risk_module import (
    SharedRiskAwareness,
    grid_line_cells,
    low_risk_fallback_goal,
)


def risk_utility_weights(
    nav_mode: str,
    frontier_weight: float,
    cost_utility_lambda: float = 0.5,
) -> UtilityWeights:
    """Return legacy mode weights for import compatibility.

    The composable safety path no longer uses this function: each planner now
    supplies its exact normal frontier preferences, and
    :class:`SharedRiskAwareness` adds the same risk costs to all of them.  The
    helper remains available so older integrations importing it do not break.
    """

    risk_weight = float(frontier_weight)
    if nav_mode == "nearest":
        return UtilityWeights(
            information_gain=0.0,
            distance=1.0,
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.0,
        )
    if nav_mode == "co_ut":
        return UtilityWeights(
            information_gain=1.0,
            distance=float(cost_utility_lambda),
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.0,
        )
    if nav_mode == "fill":
        return UtilityWeights(
            information_gain=1.0,
            distance=0.25,
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.75,
        )
    return UtilityWeights(
        information_gain=1.0,
        distance=0.35,
        risk=risk_weight,
        uncertainty=0.5,
        redundancy=0.75,
    )


class RiskAwareGlobalPlanner(GlobalPlanner):
    """Run the normal policy, then add one shared navigation-safety layer."""

    def __init__(self, planner: GlobalPlanner) -> None:
        self._planner = planner
        self.name = planner.name
        self.uses_shared_frontier_map = bool(
            getattr(planner, "uses_shared_frontier_map", False)
        )

    @staticmethod
    def _without_reserved_frontiers(
        agent_map: AgentFrontierMap,
        reserved: Set[Tuple[int, int]],
        reserved_frontier_cells: np.ndarray,
    ) -> Tuple[AgentFrontierMap, List[int]]:
        """Remove already assigned physical frontiers from a local namespace."""

        edge = np.asarray(agent_map.target_edge_map)
        if reserved_frontier_cells.shape != edge.shape:
            raise ValueError(
                "reserved frontier mask must match the agent map shape"
            )
        original_ids = []
        for frontier_id, point in enumerate(agent_map.target_points):
            point_reserved = (int(point[0]), int(point[1])) in reserved
            component_reserved = np.any(
                (edge == frontier_id + 1) & reserved_frontier_cells
            )
            if not point_reserved and not component_reserved:
                original_ids.append(frontier_id)
        relabelled = np.zeros(edge.shape, dtype=edge.dtype)
        for new_id, original_id in enumerate(original_ids):
            relabelled[edge == original_id + 1] = new_id + 1
        scores = (
            None
            if agent_map.target_score is None
            else [agent_map.target_score[index] for index in original_ids]
        )
        obstacle = np.asarray(agent_map.obstacle_map).copy()
        for row, col in reserved:
            if 0 <= row < obstacle.shape[0] and 0 <= col < obstacle.shape[1]:
                obstacle[row, col] = 1
        return (
            AgentFrontierMap(
                target_score=scores,
                target_edge_map=relabelled,
                target_points=[
                    agent_map.target_points[index] for index in original_ids
                ],
                obstacle_map=obstacle,
                explored_map=agent_map.explored_map,
                top_view_map=agent_map.top_view_map,
            ),
            original_ids,
        )

    @staticmethod
    def _local_context(
        context: GlobalPlannerContext,
        robot_id: int,
        agent_map: AgentFrontierMap,
    ) -> GlobalPlannerContext:
        actual_agent_id = (
            int(context.agent_ids[robot_id])
            if context.agent_ids is not None
            else int(robot_id)
        )
        return GlobalPlannerContext(
            target_score=agent_map.target_score,
            target_edge_map=agent_map.target_edge_map,
            target_points=agent_map.target_points,
            poses=[context.poses[robot_id]],
            agent_cells=[context.agent_cells[robot_id]],
            obstacle_map=agent_map.obstacle_map,
            explored_map=agent_map.explored_map,
            top_view_map=agent_map.top_view_map,
            goal_name=context.goal_name,
            local_step=context.local_step,
            navigation_step=context.navigation_step,
            num_agents=1,
            risk=context.risk,
            episode_index=context.episode_index,
            agent_ids=[actual_agent_id],
        )

    @staticmethod
    def _nearest_unique_free_goal(
        proposed,
        agent_map: AgentFrontierMap,
        reserved: Set[Tuple[int, int]],
        risk=None,
    ) -> List[int]:
        """Resolve a rare fallback collision inside the robot's own map."""

        free = (
            (np.asarray(agent_map.explored_map) > 0.0)
            & (np.asarray(agent_map.obstacle_map) <= 0.5)
        )
        if risk is not None:
            free &= ~np.asarray(risk.hard_unsafe, dtype=bool)
            free &= (
                np.asarray(risk.planning_risk, dtype=np.float32)
                <= float(risk.danger_threshold)
            )
        for row, col in reserved:
            if 0 <= row < free.shape[0] and 0 <= col < free.shape[1]:
                free[row, col] = False
        cells = np.argwhere(free)
        if cells.size == 0:
            return [int(proposed[0]), int(proposed[1])]
        target = np.asarray(proposed[:2], dtype=np.float64)
        chosen = cells[
            int(np.argmin(np.linalg.norm(cells - target[None, :], axis=1)))
        ]
        return [int(chosen[0]), int(chosen[1])]

    def plan_individual_maps(
        self,
        context: GlobalPlannerContext,
        agent_maps: Sequence[AgentFrontierMap],
    ) -> GlobalPlannerResult:
        """Run every non-GPT planner in the corresponding robot-local map."""

        if self.uses_shared_frontier_map:
            raise RuntimeError(
                "GPT global planning must keep the shared frontier map"
            )
        if len(agent_maps) != context.num_agents:
            raise ValueError(
                "agent_maps must contain exactly one map per robot"
            )

        offsets = []
        offset = 0
        for agent_map in agent_maps:
            offsets.append(offset)
            offset += len(agent_map.target_points)

        goals: List[List[int]] = []
        assignments = {}
        reports = []
        report_agent_ids = []
        reserved: Set[Tuple[int, int]] = set()
        reserved_frontier_cells = np.zeros(
            np.asarray(agent_maps[0].target_edge_map).shape,
            dtype=bool,
        )
        computed_step = None

        for robot_id, original_map in enumerate(agent_maps):
            local_map, original_ids = self._without_reserved_frontiers(
                original_map,
                reserved,
                reserved_frontier_cells,
            )
            local_context = self._local_context(
                context,
                robot_id,
                local_map,
            )
            local_result = self.plan(local_context)
            if len(local_result.goal_points) != 1:
                raise RuntimeError(
                    "single-agent planner must return exactly one goal"
                )
            goal = [
                int(local_result.goal_points[0][0]),
                int(local_result.goal_points[0][1]),
            ]
            if tuple(goal) in reserved:
                goal = self._nearest_unique_free_goal(
                    goal,
                    local_map,
                    reserved,
                    risk=local_context.risk,
                )
            reserved.add((goal[0], goal[1]))
            goals.append(goal)

            local_id = local_result.frontier_assignments.get(0)
            if (
                local_id is not None
                and 0 <= int(local_id) < len(original_ids)
            ):
                original_id = original_ids[int(local_id)]
                assignments[robot_id] = offsets[robot_id] + original_id
                reserved_frontier_cells |= (
                    np.asarray(original_map.target_edge_map)
                    == original_id + 1
                )
            else:
                assignments[robot_id] = None

            for report in local_result.frontier_reports:
                local_report_id = int(report.frontier_id)
                if not 0 <= local_report_id < len(original_ids):
                    continue
                original_id = original_ids[local_report_id]
                reports.append(
                    replace(
                        report,
                        frontier_id=offsets[robot_id] + original_id,
                    )
                )
                report_agent_ids.append(robot_id)
            if local_result.frontier_computed_step is not None:
                computed_step = context.navigation_step

        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=assignments,
            frontier_reports=reports,
            frontier_report_agent_ids=report_agent_ids,
            frontier_computed_step=computed_step,
        )

    def _risk_base_planner(self) -> GlobalPlanner:
        """Select the deterministic normal policy used before risk costs."""

        # GPT keeps its semantic refinement, but its deterministic safety
        # fallback is the normal co_ut planner configured by GPTGlobalPlanner.
        return getattr(self._planner, "_fallback", self._planner)

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if context.risk is None:
            return self._planner.plan(context)

        awareness = SharedRiskAwareness(context)
        reports = awareness.build_reports()

        if bool(getattr(self._planner, "uses_map_goal_sampling", False)):
            result = self._planner.plan_in_domain(
                context,
                awareness.safe_traversable_map(),
            )
            result.frontier_reports = list(reports)
            result.frontier_computed_step = context.navigation_step
            return result

        base_planner = self._risk_base_planner()
        base_preferences = base_planner.frontier_preferences(context)
        if base_preferences is None:
            raise RuntimeError(
                "{} must expose frontier_preferences for shared risk "
                "awareness".format(base_planner.name)
            )
        risk_result = awareness.assign_frontiers(
            base_preferences,
            reports=reports,
        )
        assignments = self._planner.refine_risk_assignments(
            context,
            risk_result.reports,
            risk_result.assignments,
            risk_result.hard_threshold,
        )

        agent_cells = [
            [int(cell[0]), int(cell[1])]
            for cell in context.agent_cells[: context.num_agents]
        ]
        goals = []
        normalized_assignments = {}
        for robot_id in range(context.num_agents):
            frontier_id = assignments.get(robot_id)
            if (
                frontier_id is not None
                and 0 <= int(frontier_id) < len(context.target_points)
            ):
                normalized_assignments[robot_id] = int(frontier_id)
                goals.append(goal_from_frontier(context, int(frontier_id)))
            else:
                normalized_assignments[robot_id] = None
                goals.append(
                    low_risk_fallback_goal(
                        agent_cells[robot_id],
                        context.obstacle_map,
                        context.explored_map,
                        context.risk.planning_risk,
                        context.risk.hard_unsafe,
                    )
                )

        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=normalized_assignments,
            frontier_reports=list(risk_result.reports),
            frontier_computed_step=context.navigation_step,
        )


__all__ = [
    "RiskAwareGlobalPlanner",
    "SharedRiskAwareness",
    "grid_line_cells",
    "low_risk_fallback_goal",
    "risk_utility_weights",
]
