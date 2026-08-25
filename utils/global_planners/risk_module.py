"""Planner-independent risk awareness for global navigation goals.

The normal global planners own *what is useful*: nearest distance,
cost-utility, fill score, or random map-goal sampling.  This module owns only
*what is safe*.  Keeping those concerns separate guarantees that enabling a
zero-valued risk map preserves the corresponding normal planner's ordering.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
from typing import Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from utils.risk.frontier import (
    FrontierRiskReport,
    SeverityThresholds,
    build_frontier_risk_reports,
)

from .base import GlobalPlannerContext


def grid_line_cells(start, goal, shape) -> List[List[int]]:
    """Return a clipped one-cell-wide route proxy for risk reporting."""

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


def risk_aware_route_cells(
    start,
    goal,
    obstacle_map,
    explored_map,
    planning_risk,
    hard_unsafe,
    *,
    risk_alpha: float = 4.0,
) -> Optional[List[List[int]]]:
    """Find an explored free-space route used to score a frontier.

    The former one-cell straight-line proxy frequently crossed walls or a
    fire core even when the local planner could go around it.  This search
    uses the same eight-connected, risk-weighted edge objective as the local
    A* planner and forbids diagonal corner cutting.  ``None`` deliberately
    means that the caller should retain the conservative line *proxy* rather
    than pretending that an unreachable route is exact.
    """

    risk = np.asarray(planning_risk, dtype=np.float32)
    shape = risk.shape
    for name, value in (
        ("obstacle_map", obstacle_map),
        ("explored_map", explored_map),
        ("hard_unsafe", hard_unsafe),
    ):
        if np.asarray(value).shape != shape:
            raise ValueError(f"{name} shape must match planning_risk")

    obstacle = np.asarray(obstacle_map) > 0.5
    explored = np.asarray(explored_map) > 0.0
    hard = np.asarray(hard_unsafe, dtype=bool)
    traversable = explored & ~obstacle & ~hard

    def _cell(value):
        return (
            int(np.clip(round(float(value[0])), 0, shape[0] - 1)),
            int(np.clip(round(float(value[1])), 0, shape[1] - 1)),
        )

    source = _cell(start)
    target = _cell(goal)
    # Mapping lag can leave the robot's current cell just outside the latest
    # explored mask.  It is still a valid source unless physically blocked.
    if obstacle[source] or hard[source]:
        return None
    traversable[source] = True

    targets = []
    if traversable[target]:
        targets.append(target)
    else:
        # Frontier centroids can fall one or two cells into the unknown side
        # of a thick component.  Route to the nearest free approach cell;
        # the frontier footprint itself is still risk-scored separately.
        for radius in (1, 2):
            row0 = max(0, target[0] - radius)
            row1 = min(shape[0], target[0] + radius + 1)
            col0 = max(0, target[1] - radius)
            col1 = min(shape[1], target[1] + radius + 1)
            local = np.argwhere(traversable[row0:row1, col0:col1])
            if local.size:
                targets = [
                    (int(cell[0] + row0), int(cell[1] + col0))
                    for cell in local
                ]
                break
    if not targets:
        return None

    target_set = set(targets)
    target_array = np.asarray(targets, dtype=np.float32)

    def _heuristic(cell) -> float:
        delta = target_array - np.asarray(cell, dtype=np.float32)[None, :]
        return float(np.sqrt(np.sum(delta * delta, axis=1)).min())

    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    )
    alpha = max(0.0, float(risk_alpha))
    counter = itertools.count()
    queue = [(_heuristic(source), 0.0, next(counter), source)]
    costs = {source: 0.0}
    parents = {source: None}
    reached = None

    while queue:
        _, current_cost, _, current = heapq.heappop(queue)
        if current_cost > costs[current] + 1e-9:
            continue
        if current in target_set:
            reached = current
            break
        for drow, dcol, geometric in moves:
            nxt = (current[0] + drow, current[1] + dcol)
            if not (
                0 <= nxt[0] < shape[0]
                and 0 <= nxt[1] < shape[1]
                and traversable[nxt]
            ):
                continue
            if drow != 0 and dcol != 0:
                if not (
                    traversable[current[0] + drow, current[1]]
                    and traversable[current[0], current[1] + dcol]
                ):
                    continue
            mean_risk = 0.5 * (float(risk[current]) + float(risk[nxt]))
            candidate = current_cost + geometric * (
                1.0 + alpha * mean_risk
            )
            if candidate + 1e-9 >= costs.get(nxt, np.inf):
                continue
            costs[nxt] = candidate
            parents[nxt] = current
            heapq.heappush(
                queue,
                (
                    candidate + _heuristic(nxt),
                    candidate,
                    next(counter),
                    nxt,
                ),
            )

    if reached is None:
        return None
    route = []
    cursor = reached
    while cursor is not None:
        route.append([int(cursor[0]), int(cursor[1])])
        cursor = parents[cursor]
    route.reverse()
    return route


def low_risk_fallback_goal(
    agent_cell,
    obstacle_map,
    explored_map,
    planning_risk,
    hard_unsafe,
) -> List[int]:
    """Select a nearby explored, reachable, low-risk safety waypoint."""

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


@dataclass(frozen=True)
class RiskAwareAssignment:
    """Shared safety-layer output before goals are materialized."""

    assignments: Dict[int, Optional[int]]
    reports: List[FrontierRiskReport]
    hard_threshold: float


class SharedRiskAwareness:
    """Use continuous route risk only to break comparable task choices."""

    def __init__(self, context: GlobalPlannerContext) -> None:
        if context.risk is None:
            raise ValueError("SharedRiskAwareness requires context.risk")
        self.context = context
        self.risk = context.risk
        danger_threshold = float(self.risk.danger_threshold)
        safe_threshold = min(0.25, danger_threshold)
        moderate_threshold = max(safe_threshold, danger_threshold)
        self.hard_threshold = float(
            np.clip(
                self.risk.hard_frontier_threshold,
                moderate_threshold,
                1.0,
            )
        )
        self.thresholds = SeverityThresholds(
            safe_max=safe_threshold,
            moderate_max=moderate_threshold,
            hard_max=self.hard_threshold,
        )

    def build_reports(self) -> List[FrontierRiskReport]:
        """Build the common frontier and approximate-route hazard report."""

        agent_cells = [
            [int(cell[0]), int(cell[1])]
            for cell in self.context.agent_cells[: self.context.num_agents]
        ]
        route_cells = []
        route_is_proxy = []
        for frontier in self.context.target_points:
            nearest_cell = min(
                agent_cells,
                key=lambda cell: np.linalg.norm(
                    np.asarray(cell) - np.asarray(frontier)
                ),
            )
            route = risk_aware_route_cells(
                nearest_cell,
                frontier,
                self.context.obstacle_map,
                self.context.explored_map,
                self.risk.planning_risk,
                np.zeros_like(self.risk.hard_unsafe, dtype=bool),
                risk_alpha=self.risk.route_risk_alpha,
            )
            is_proxy = route is None
            if is_proxy:
                route = grid_line_cells(
                    nearest_cell,
                    frontier,
                    np.asarray(self.risk.planning_risk).shape,
                )
            route_cells.append(route)
            route_is_proxy.append(is_proxy)
        return build_frontier_risk_reports(
            self.context.target_edge_map,
            self.risk.planning_risk,
            self.risk.confidence,
            hard_unsafe_map=self.risk.hard_unsafe,
            frontier_points=self.context.target_points,
            route_cells=route_cells,
            route_is_proxy=route_is_proxy,
            thresholds=self.thresholds,
        )

    @staticmethod
    def _normalise_preferences(values: Sequence[float]) -> np.ndarray:
        """Map base-policy preferences to [0, 1] without changing ordering."""

        raw = np.asarray(values, dtype=np.float64)
        if raw.ndim != 1:
            raise ValueError("frontier preferences must be one-dimensional")
        finite = np.isfinite(raw)
        normalized = np.full(raw.shape, -np.inf, dtype=np.float64)
        if not finite.any():
            return normalized
        low = float(raw[finite].min())
        high = float(raw[finite].max())
        if high > low:
            normalized[finite] = (raw[finite] - low) / (high - low)
        else:
            # Equal preferences remain equal so the normal planner's
            # deterministic lowest-frontier-id tie break is preserved.
            normalized[finite] = 1.0
        return normalized

    def assign_frontiers(
        self,
        base_preferences: Mapping[int, Sequence[float]],
        reports: Optional[Sequence[FrontierRiskReport]] = None,
    ) -> RiskAwareAssignment:
        """Preserve task utility, using route risk only within a value tie.

        No frontier is rejected because of a hard mask, a continuous-risk
        threshold or epistemic uncertainty.  A lower-value frontier may replace
        the normal winner only when its base utility is within the configured
        relative regret tolerance.
        """

        ordered_reports = sorted(
            self.build_reports() if reports is None else reports,
            key=lambda report: report.frontier_id,
        )
        assignments: Dict[int, Optional[int]] = {}
        expected = len(ordered_reports)
        for robot_id in range(self.context.num_agents):
            preferences = np.asarray(
                base_preferences.get(robot_id, ()), dtype=np.float64
            )
            if len(preferences) != expected:
                raise ValueError(
                    "robot {} has {} frontier preferences; expected {}".format(
                        robot_id,
                        len(preferences),
                        expected,
                    )
                )
            finite_ids = np.flatnonzero(np.isfinite(preferences))
            if finite_ids.size == 0:
                assignments[robot_id] = None
                continue

            best_value = float(np.max(preferences[finite_ids]))
            typical_magnitude = float(np.median(np.abs(preferences[finite_ids])))
            value_scale = max(abs(best_value), typical_magnitude, 1.0)
            max_regret = (
                float(self.risk.frontier_value_tolerance) * value_scale
            )
            comparable = [
                int(ordinal)
                for ordinal in finite_ids
                if best_value - float(preferences[ordinal])
                <= max_regret + 1e-12
            ]
            winner = min(
                comparable,
                key=lambda ordinal: (
                    float(ordered_reports[ordinal].planning_risk),
                    -float(preferences[ordinal]),
                    int(ordered_reports[ordinal].frontier_id),
                ),
            )
            assignments[robot_id] = int(
                ordered_reports[winner].frontier_id
            )

        return RiskAwareAssignment(
            assignments=assignments,
            reports=list(ordered_reports),
            hard_threshold=self.hard_threshold,
        )

    def safe_traversable_map(self) -> np.ndarray:
        """Return the normal explored domain without risk rejection."""

        traversable = (
            (np.asarray(self.context.explored_map) > 0.0)
            & (np.asarray(self.context.obstacle_map) <= 0.5)
        )
        return traversable


__all__ = [
    "RiskAwareAssignment",
    "SharedRiskAwareness",
    "grid_line_cells",
    "low_risk_fallback_goal",
    "risk_aware_route_cells",
]
