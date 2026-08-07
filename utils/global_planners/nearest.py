"""Nearest-frontier global planner."""
from __future__ import annotations

import numpy as np

from .base import (
    GlobalPlanner,
    GlobalPlannerContext,
    GlobalPlannerResult,
    goal_from_frontier,
    random_goal_result,
)


class NearestGlobalPlanner(GlobalPlanner):
    """Assign each robot its independently nearest frontier."""

    name = "nearest"

    def frontier_preferences(self, context: GlobalPlannerContext):
        frontiers = np.asarray(context.target_points, dtype=np.float64)
        if len(frontiers) == 0:
            return {
                robot_id: np.empty((0,), dtype=np.float64)
                for robot_id in range(context.num_agents)
            }
        return {
            robot_id: -np.linalg.norm(
                frontiers
                - np.asarray(
                    context.poses[robot_id][:2],
                    dtype=np.float64,
                )[None, :],
                axis=1,
            )
            for robot_id in range(context.num_agents)
        }

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if len(context.target_points) == 0:
            return random_goal_result(context)

        preferences = self.frontier_preferences(context)
        assignments = {}
        goals = []
        for robot_id in range(context.num_agents):
            frontier_id = int(np.argmax(preferences[robot_id]))
            assignments[robot_id] = frontier_id
            goals.append(goal_from_frontier(context, frontier_id))
        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=assignments,
        )
