"""Eight-connected, risk-aware A* local planner."""
from __future__ import annotations

import heapq
import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import GridCell, GridPlannerBase


class AStarPlanner(GridPlannerBase):
    """FMM-compatible A* backend over the same local navigation grid."""

    def _edge_cost(
        self, current: GridCell, nxt: GridCell, geometric_cost: float
    ) -> float:
        if self.risk_map is None or self.risk_alpha <= 0.0:
            return geometric_cost
        mean_risk = 0.5 * (
            float(self.risk_map[current]) + float(self.risk_map[nxt])
        )
        return geometric_cost * (1.0 + self.risk_alpha * mean_risk)

    def _plan_path(self, start: GridCell) -> Optional[List[GridCell]]:
        if self.goal_map is None or not np.any(self.goal_map):
            return None
        if self.goal_map[start]:
            return [start]

        heuristic = self.goal_distance()
        counter = itertools.count()
        queue = [(float(heuristic[start]), 0.0, next(counter), start)]
        cost_so_far: Dict[GridCell, float] = {start: 0.0}
        parents: Dict[GridCell, Optional[GridCell]] = {start: None}
        reached = None

        while queue:
            _, current_cost, _, current = heapq.heappop(queue)
            if current_cost > cost_so_far[current] + 1e-9:
                continue
            if self.goal_map[current]:
                reached = current
                break
            for nxt, geometric_cost in self.iter_neighbors(current):
                candidate = current_cost + self._edge_cost(
                    current, nxt, geometric_cost
                )
                if candidate + 1e-9 >= cost_so_far.get(nxt, np.inf):
                    continue
                cost_so_far[nxt] = candidate
                parents[nxt] = current
                priority = candidate + float(heuristic[nxt])
                heapq.heappush(
                    queue, (priority, candidate, next(counter), nxt)
                )

        if reached is None:
            return None
        path = []
        cell: Optional[GridCell] = reached
        while cell is not None:
            path.append(cell)
            cell = parents[cell]
        path.reverse()
        return path

    def get_short_term_goal(self, state):
        start = self._clip_cell(state)
        goal_distance = self.goal_distance()
        stop = bool(goal_distance[start] < float(self.step_size))
        path = self._plan_path(start)
        self.last_path = [] if path is None else path
        if path is None:
            return float(start[0]), float(start[1]), True, False
        if stop or len(path) == 1:
            return float(start[0]), float(start[1]), False, stop

        travelled = 0.0
        stg = path[-1]
        previous = path[0]
        for cell in path[1:]:
            travelled += float(np.linalg.norm(np.subtract(cell, previous)))
            stg = cell
            previous = cell
            if travelled >= float(self.step_size):
                break
        return float(stg[0]), float(stg[1]), False, stop
