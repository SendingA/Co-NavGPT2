"""Cost-Utility frontier assignment used by ``nav_mode=co_ut``."""
from __future__ import annotations

import numpy as np

from .base import (
    GlobalPlanner,
    GlobalPlannerContext,
    GlobalPlannerResult,
    goal_from_frontier,
    random_goal_result,
)


class CostUtilityGlobalPlanner(GlobalPlanner):
    """Maximise ``frontier_size - lambda * robot_distance`` per robot."""

    name = "co_ut"

    def __init__(self, cost_utility_lambda: float = 1.0) -> None:
        value = float(cost_utility_lambda)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "cost_utility_lambda must be finite and non-negative"
            )
        self.cost_utility_lambda = value

    @staticmethod
    def _frontier_sizes(context: GlobalPlannerContext) -> np.ndarray:
        """Return frontier connected-component sizes in map cells."""

        sizes = np.zeros(len(context.target_points), dtype=np.float64)
        labels = np.asarray(context.target_edge_map)
        for frontier_id in range(len(context.target_points)):
            if (
                context.target_score is not None
                and frontier_id < len(context.target_score)
            ):
                score = float(context.target_score[frontier_id])
                sizes[frontier_id] = score if np.isfinite(score) else 0.0
            else:
                sizes[frontier_id] = float(
                    np.count_nonzero(labels == frontier_id + 1)
                )
        return sizes

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if len(context.target_points) == 0:
            return random_goal_result(context)

        frontiers = np.asarray(context.target_points, dtype=np.float64)
        frontier_sizes = self._frontier_sizes(context)
        assignments = {}
        goals = []
        for robot_id in range(context.num_agents):
            robot = np.asarray(context.poses[robot_id][:2], dtype=np.float64)
            distances = np.linalg.norm(frontiers - robot[None, :], axis=1)
            utilities = (
                frontier_sizes - self.cost_utility_lambda * distances
            )
            frontier_id = int(np.argmax(utilities))
            assignments[robot_id] = frontier_id
            goals.append(goal_from_frontier(context, frontier_id))

        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=assignments,
        )


# Keep imports from earlier project revisions working while correcting the
# implementation behind ``nav_mode=co_ut``.
CooperativeGlobalPlanner = CostUtilityGlobalPlanner
