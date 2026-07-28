"""Shared grid and safety semantics for non-FMM local planners."""
from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import cv2
import numpy as np


GridCell = Tuple[int, int]

# Four cardinal moves first, followed by diagonals.  The ordering is part of
# the RL checkpoint contract and must remain stable.
GRID_ACTIONS: Tuple[GridCell, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


class GridPlannerBase:
    """Common traversibility, risk and emergency-escape implementation.

    FMM keeps its historical implementation in :mod:`utils.fmm_planner`.
    A* and RL inherit this class so they receive the same hard-hazard and
    multi-goal semantics without depending on scikit-fmm.
    """

    def __init__(
        self,
        traversible,
        *,
        scale: float = 1.0,
        step_size: int = 5,
        risk_map=None,
        risk_alpha: float = 0.0,
        hard_unsafe_mask=None,
    ) -> None:
        if float(scale) != 1.0:
            raise ValueError("A* and RL local planners currently require scale=1")
        values = np.asarray(traversible)
        if values.ndim != 2:
            raise ValueError("traversible must be a 2-D grid")

        self.scale = 1.0
        self.step_size = max(1, int(step_size))
        self.risk_alpha = max(0.0, float(risk_alpha))
        self.base_traversible = values.copy()
        self.base_traversible_mask = values != 0
        self.risk_map = self._optional_grid(
            risk_map, values.shape, "risk_map", dtype=np.float32
        )
        if self.risk_map is not None:
            self.risk_map = np.clip(
                np.nan_to_num(
                    self.risk_map, nan=0.0, posinf=1.0, neginf=0.0
                ),
                0.0,
                1.0,
            )
        hard = self._optional_grid(
            hard_unsafe_mask,
            values.shape,
            "hard_unsafe_mask",
            dtype=bool,
        )
        self.hard_unsafe_mask = (
            np.zeros(values.shape, dtype=bool) if hard is None else hard.copy()
        )
        self.traversible = (
            self.base_traversible_mask & ~self.hard_unsafe_mask
        ).astype(np.float32)
        self.goal_map: Optional[np.ndarray] = None
        self.last_path = []

    @staticmethod
    def _optional_grid(grid, shape, name, *, dtype):
        if grid is None:
            return None
        result = np.asarray(grid, dtype=dtype)
        if result.shape != shape:
            raise ValueError(
                "{} shape {} does not match traversible shape {}".format(
                    name, result.shape, shape
                )
            )
        return result

    @property
    def risk_enabled(self) -> bool:
        return bool(
            (self.risk_map is not None and self.risk_alpha > 0.0)
            or np.any(self.hard_unsafe_mask)
        )

    def set_goal(self, goal, auto_improve: bool = False) -> None:
        del auto_improve
        cell = self._clip_cell(goal)
        goal_map = np.zeros(self.traversible.shape, dtype=np.uint8)
        goal_map[cell] = 1
        self.set_multi_goal(goal_map)

    def set_multi_goal(self, goal_map) -> None:
        goal = np.asarray(goal_map)
        if goal.shape != self.traversible.shape:
            raise ValueError(
                "goal_map shape {} does not match traversible shape {}".format(
                    goal.shape, self.traversible.shape
                )
            )
        self.goal_map = goal == 1
        # The caller filters unsafe dilated goals before this method.  Admit
        # those selected goal cells exactly as the historical FMM planner does.
        self.traversible[self.goal_map] = 1.0
        self.hard_unsafe_mask[self.goal_map] = False

    def prepare_emergency_escape(self, start):
        """Open only the obstacle-free corridor out of a newly hard hazard."""
        source = self._clip_cell(start)
        if not self.hard_unsafe_mask[source]:
            return None

        queue = deque([source])
        parents = {source: None}
        escape = None
        while queue:
            cell = queue.popleft()
            if (
                not self.hard_unsafe_mask[cell]
                and self.base_traversible_mask[cell]
            ):
                escape = cell
                break
            for nxt, _ in self.iter_neighbors(
                cell, use_base_traversible=True, cardinal_only=True
            ):
                if nxt in parents:
                    continue
                parents[nxt] = cell
                queue.append(nxt)

        if escape is None:
            self.hard_unsafe_mask[source] = False
            self.traversible[source] = 1.0
            return np.zeros(self.traversible.shape, dtype=np.uint8)

        cell = escape
        while cell is not None:
            self.hard_unsafe_mask[cell] = False
            cell = parents[cell]
        self.traversible = (
            self.base_traversible_mask & ~self.hard_unsafe_mask
        ).astype(np.float32)

        goal = np.zeros(self.traversible.shape, dtype=np.uint8)
        goal[escape] = 1
        return goal

    def _clip_cell(self, cell: Sequence[float]) -> GridCell:
        return (
            int(np.clip(round(float(cell[0])), 0, self.traversible.shape[0] - 1)),
            int(np.clip(round(float(cell[1])), 0, self.traversible.shape[1] - 1)),
        )

    def iter_neighbors(
        self,
        cell: GridCell,
        *,
        use_base_traversible: bool = False,
        cardinal_only: bool = False,
    ) -> Iterator[Tuple[GridCell, float]]:
        domain = (
            self.base_traversible_mask
            if use_base_traversible
            else self.traversible > 0
        )
        actions: Iterable[GridCell] = (
            GRID_ACTIONS[:4] if cardinal_only else GRID_ACTIONS
        )
        height, width = domain.shape
        row, col = cell
        for drow, dcol in actions:
            nxt = (row + drow, col + dcol)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if not domain[nxt]:
                continue
            if drow != 0 and dcol != 0:
                # A diagonal cannot squeeze through two touching obstacles.
                if not domain[row + drow, col] or not domain[row, col + dcol]:
                    continue
                distance = float(np.sqrt(2.0))
            else:
                distance = 1.0
            yield nxt, distance

    def valid_action_mask(self, cell: GridCell) -> np.ndarray:
        valid = np.zeros(len(GRID_ACTIONS), dtype=bool)
        neighbor_cells = {
            nxt for nxt, _ in self.iter_neighbors(cell)
        }
        for index, (drow, dcol) in enumerate(GRID_ACTIONS):
            valid[index] = (cell[0] + drow, cell[1] + dcol) in neighbor_cells
        return valid

    def goal_distance(self) -> np.ndarray:
        if self.goal_map is None or not np.any(self.goal_map):
            return np.full(self.traversible.shape, np.inf, dtype=np.float32)
        return cv2.distanceTransform(
            (~self.goal_map).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
