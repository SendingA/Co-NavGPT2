"""Eight-connected, risk-aware A* local planner."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import GRID_ACTIONS, GridCell, GridPlannerBase


@dataclass(frozen=True)
class AStarPlanResult:
    """One complete A* search and all products derived from that search."""

    stg: Tuple[float, float]
    replan: bool
    stop: bool
    path: Tuple[GridCell, ...]
    searched: bool


@dataclass
class AStarPathCache:
    """Snapshot required to reuse an optimal A* path without semantic drift."""

    traversible: np.ndarray
    goal_map: np.ndarray
    risk_map: Optional[np.ndarray]
    risk_alpha: float
    goal_distance: np.ndarray
    path: Tuple[GridCell, ...]

    @classmethod
    def capture(
        cls,
        planner: "AStarPlanner",
        path: Sequence[GridCell],
    ) -> "AStarPathCache":
        return cls(
            traversible=np.asarray(planner.traversible).copy(),
            goal_map=np.asarray(planner.goal_map, dtype=bool).copy(),
            risk_map=(
                None
                if planner.risk_map is None
                else np.asarray(planner.risk_map, dtype=np.float32).copy()
            ),
            risk_alpha=float(planner.risk_alpha),
            goal_distance=planner.goal_distance().copy(),
            path=tuple(path),
        )

    def restore_goal_distance(self, planner: "AStarPlanner") -> bool:
        """Restore a goal-only heuristic even when the map has changed."""

        if not np.array_equal(self.goal_map, planner.goal_map):
            return False
        planner._goal_distance_cache = self.goal_distance
        return True

    def reusable_suffix(
        self,
        planner: "AStarPlanner",
        start: GridCell,
    ) -> Optional[Tuple[GridCell, ...]]:
        if float(planner.risk_alpha) != self.risk_alpha:
            return None
        if not np.array_equal(self.traversible, planner.traversible):
            return None
        if not np.array_equal(self.goal_map, planner.goal_map):
            return None
        if (self.risk_map is None) != (planner.risk_map is None):
            return None
        if (
            self.risk_map is not None
            and not np.array_equal(self.risk_map, planner.risk_map)
        ):
            return None
        try:
            start_index = self.path.index(start)
        except ValueError:
            return None
        return self.path[start_index:]


class AStarPlanner(GridPlannerBase):
    """FMM-compatible A* backend over the same local navigation grid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.search_count = 0

    def _edge_cost(
        self, current: GridCell, nxt: GridCell, geometric_cost: float
    ) -> float:
        if self.risk_map is None or self.risk_alpha <= 0.0:
            return geometric_cost
        mean_risk = 0.5 * (
            float(self.risk_map[current]) + float(self.risk_map[nxt])
        )
        return geometric_cost * (1.0 + self.risk_alpha * mean_risk)

    def _plan_path(
        self,
        start: GridCell,
        heuristic: Optional[np.ndarray] = None,
    ) -> Optional[List[GridCell]]:
        self.search_count += 1
        if self.goal_map is None or not np.any(self.goal_map):
            return None
        if self.goal_map[start]:
            return [start]

        if heuristic is None:
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

    def _coerce_reusable_path(
        self,
        start: GridCell,
        reusable_path: Optional[Sequence[GridCell]],
    ) -> Optional[List[GridCell]]:
        """Validate a previously optimal suffix before reusing it.

        The caller is responsible for supplying a suffix only when the
        traversibility, goal and risk-cost grids are unchanged.  This method
        still validates geometry so stale or malformed cache data cannot
        route through a newly invalid edge.
        """

        if reusable_path is None:
            return None
        path = [self._clip_cell(cell) for cell in reusable_path]
        if not path or path[0] != start:
            return None
        if self.goal_map is None or not self.goal_map[path[-1]]:
            return None
        domain = self.traversible > 0
        if any(not domain[cell] for cell in path):
            return None
        for current, nxt in zip(path, path[1:]):
            drow = nxt[0] - current[0]
            dcol = nxt[1] - current[1]
            if (drow, dcol) not in GRID_ACTIONS:
                return None
            if drow != 0 and dcol != 0:
                if (
                    not domain[current[0] + drow, current[1]]
                    or not domain[current[0], current[1] + dcol]
                ):
                    return None
        return path

    def _short_term_goal_from_path(
        self,
        path: Sequence[GridCell],
    ) -> GridCell:
        if len(path) <= 1:
            return path[0]
        travelled = 0.0
        stg = path[-1]
        previous = path[0]
        for cell in path[1:]:
            travelled += float(np.linalg.norm(np.subtract(cell, previous)))
            stg = cell
            previous = cell
            if travelled >= float(self.step_size):
                break
        return stg

    def sample_path(
        self,
        path: Sequence[GridCell],
        *,
        max_waypoints: int = 10,
    ) -> List[GridCell]:
        """Sample visualization waypoints from one already-computed path."""

        if not path:
            return []
        waypoints = [path[0]]
        if len(path) == 1 or max_waypoints <= 0:
            return waypoints

        travelled = 0.0
        previous = path[0]
        for cell in path[1:]:
            travelled += float(np.linalg.norm(np.subtract(cell, previous)))
            previous = cell
            if travelled + 1e-9 < float(self.step_size):
                continue
            waypoints.append(cell)
            # The legacy visualization loop restarted its distance budget at
            # every returned STG.  Reset here to preserve those waypoint
            # intervals while deriving all of them from this single path.
            travelled = 0.0
            if len(waypoints) >= max_waypoints + 1:
                break
        return waypoints

    def plan(
        self,
        state,
        *,
        reusable_path: Optional[Sequence[GridCell]] = None,
    ) -> AStarPlanResult:
        start = self._clip_cell(state)
        goal_distance = self.goal_distance()
        stop = bool(goal_distance[start] < float(self.step_size))
        path = self._coerce_reusable_path(start, reusable_path)
        searched = path is None
        if searched:
            path = self._plan_path(start, heuristic=goal_distance)
        self.last_path = [] if path is None else path
        if path is None:
            return AStarPlanResult(
                stg=(float(start[0]), float(start[1])),
                replan=True,
                stop=False,
                path=(),
                searched=searched,
            )
        if stop or len(path) == 1:
            stg = start
        else:
            stg = self._short_term_goal_from_path(path)
        return AStarPlanResult(
            stg=(float(stg[0]), float(stg[1])),
            replan=False,
            stop=stop,
            path=tuple(path),
            searched=searched,
        )

    def get_short_term_goal(self, state):
        result = self.plan(state)
        return result.stg[0], result.stg[1], result.replan, result.stop
