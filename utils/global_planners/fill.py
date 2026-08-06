"""Highest-frontier-score global planner."""
from __future__ import annotations

import numpy as np

from .base import (
    GlobalPlanner,
    GlobalPlannerContext,
    GlobalPlannerResult,
    goal_from_frontier,
    random_goal_result,
)


class FillGlobalPlanner(GlobalPlanner):
    """Send each robot to the frontier with the highest available score."""

    name = "fill"

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if len(context.target_points) == 0:
            return random_goal_result(context)

        assignments = {}
        goals = []
        for robot_id in range(context.num_agents):
            best_idx = 0
            best_score = -1.0
            for frontier_id, frontier in enumerate(context.target_points):
                if (
                    context.target_score is not None
                    and frontier_id < len(context.target_score)
                ):
                    score = context.target_score[frontier_id]
                else:
                    score = 1.0 / (
                        1.0
                        + np.linalg.norm(
                            np.asarray(frontier)
                            - np.asarray(context.poses[robot_id][:2])
                        )
                    )
                if score > best_score:
                    best_score = float(score)
                    best_idx = frontier_id
            assignments[robot_id] = int(best_idx)
            goals.append(goal_from_frontier(context, best_idx))

        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=assignments,
        )
