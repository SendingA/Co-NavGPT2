"""Shared risk-aware decorator for all global frontier planners."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from utils.risk.frontier import (
    SeverityThresholds,
    UtilityWeights,
    assign_frontiers,
    build_frontier_risk_reports,
)

from .base import (
    GlobalPlanner,
    GlobalPlannerContext,
    GlobalPlannerResult,
    goal_from_frontier,
)


def grid_line_cells(start, goal, shape) -> List[List[int]]:
    """Return the legacy clipped one-cell-wide route proxy."""

    canvas = np.zeros(shape, dtype=np.uint8)
    start_row = int(np.clip(round(float(start[0])), 0, shape[0] - 1))
    start_col = int(np.clip(round(float(start[1])), 0, shape[1] - 1))
    goal_row = int(np.clip(round(float(goal[0])), 0, shape[0] - 1))
    goal_col = int(np.clip(round(float(goal[1])), 0, shape[1] - 1))
    cv2.line(
        canvas,
        (start_col, start_row),
        (goal_col, goal_row),
        color=1,
        thickness=1,
    )
    return np.argwhere(canvas > 0).astype(int).tolist()


def low_risk_fallback_goal(
    agent_cell,
    obstacle_map,
    explored_map,
    planning_risk,
    hard_unsafe,
) -> List[int]:
    """Select a nearby explored, navigable low-risk safety waypoint."""

    shape = np.asarray(planning_risk).shape
    obstacle = cv2.dilate(
        (np.asarray(obstacle_map) > 0.5).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    ).astype(bool)
    explored = np.asarray(explored_map) > 0.0
    hard = np.asarray(hard_unsafe, dtype=bool)
    start = np.asarray(agent_cell[:2], dtype=np.float64)
    start_cell = (
        int(np.clip(round(start[0]), 0, shape[0] - 1)),
        int(np.clip(round(start[1]), 0, shape[1] - 1)),
    )
    free = ~obstacle & ~hard
    if not free.any():
        return [start_cell[0], start_cell[1]]

    seed = start_cell
    if not free[seed]:
        free_cells = np.argwhere(free)
        seed = tuple(
            free_cells[
                int(
                    np.argmin(
                        np.linalg.norm(
                            free_cells - start[None, :],
                            axis=1,
                        )
                    )
                )
            ]
        )
    _, labels = cv2.connectedComponents(
        free.astype(np.uint8),
        connectivity=8,
    )
    reachable = labels == labels[seed]
    candidates = explored & reachable
    if not candidates.any():
        candidates = reachable

    cells = np.argwhere(candidates)
    distances = np.linalg.norm(cells - start[None, :], axis=1)
    nontrivial = distances >= 4.0
    if nontrivial.any():
        cells = cells[nontrivial]
        distances = distances[nontrivial]
    risk = np.asarray(planning_risk, dtype=np.float32)[
        cells[:, 0],
        cells[:, 1],
    ]
    distance_scale = max(float(distances.max()), 1.0)
    score = risk + 0.08 * distances / distance_scale
    best = cells[int(np.argmin(score))]
    return [int(best[0]), int(best[1])]


def risk_utility_weights(
    nav_mode: str,
    frontier_weight: float,
    cost_utility_lambda: float = 1.0,
) -> UtilityWeights:
    """Map legacy frontier policies onto the existing safety utility."""

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
    """Apply common risk scoring, then optionally let GPT refine it."""

    def __init__(self, planner: GlobalPlanner) -> None:
        self._planner = planner
        self.name = planner.name

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if context.risk is None:
            return self._planner.plan(context)

        risk = context.risk
        agent_cells = [
            [int(cell[0]), int(cell[1])]
            for cell in context.agent_cells[: context.num_agents]
        ]
        route_cells = []
        for frontier in context.target_points:
            nearest_cell = min(
                agent_cells,
                key=lambda cell: np.linalg.norm(
                    np.asarray(cell) - np.asarray(frontier)
                ),
            )
            route_cells.append(
                grid_line_cells(
                    nearest_cell,
                    frontier,
                    np.asarray(risk.planning_risk).shape,
                )
            )

        danger_threshold = float(risk.danger_threshold)
        safe_threshold = min(0.25, danger_threshold)
        moderate_threshold = max(safe_threshold, danger_threshold)
        hard_threshold = float(
            np.clip(
                risk.hard_frontier_threshold,
                moderate_threshold,
                1.0,
            )
        )
        thresholds = SeverityThresholds(
            safe_max=safe_threshold,
            moderate_max=moderate_threshold,
            hard_max=hard_threshold,
        )
        reports = build_frontier_risk_reports(
            context.target_edge_map,
            risk.planning_risk,
            risk.confidence,
            hard_unsafe_map=risk.hard_unsafe,
            frontier_points=context.target_points,
            route_cells=route_cells,
            route_is_proxy=True,
            thresholds=thresholds,
        )
        if bool(getattr(self._planner, "uses_map_goal_sampling", False)):
            result = self._planner.plan(context)
            result.frontier_reports = list(reports)
            result.frontier_computed_step = context.navigation_step
            return result

        fallback_name = str(
            getattr(self._planner, "risk_fallback_name", self.name)
        )
        deterministic = assign_frontiers(
            agent_cells,
            reports,
            information_gain=context.target_score,
            weights=risk_utility_weights(
                fallback_name,
                risk.frontier_weight,
                float(
                    getattr(
                        self._planner,
                        "cost_utility_lambda",
                        getattr(
                            getattr(self._planner, "_fallback", None),
                            "cost_utility_lambda",
                            1.0,
                        ),
                    )
                ),
            ),
            hard_risk_threshold=hard_threshold,
            allow_shared=True,
            redundancy_radius_cells=(
                1.0 / (float(risk.map_resolution_cm) / 100.0)
            ),
        )
        assignments = self._planner.refine_risk_assignments(
            context,
            reports,
            deterministic,
            hard_threshold,
        )

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
                        risk.planning_risk,
                        risk.hard_unsafe,
                    )
                )

        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=normalized_assignments,
            frontier_reports=list(reports),
            frontier_computed_step=context.navigation_step,
        )
