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

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if len(context.target_points) == 0:
            return random_goal_result(context)

        assignments = {}
        goals = []
        for robot_id in range(context.num_agents):
            distances = [
                np.linalg.norm(
                    np.asarray(frontier)
                    - np.asarray(context.poses[robot_id][:2])
                )
                for frontier in context.target_points
            ]
            frontier_id = int(np.argmin(distances))
            assignments[robot_id] = frontier_id
            goals.append(goal_from_frontier(context, frontier_id))
        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=assignments,
        )
