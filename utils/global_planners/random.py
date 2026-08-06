"""Reproducible random long-term goal sampling."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from .base import GlobalPlanner, GlobalPlannerContext, GlobalPlannerResult


class RandomGlobalPlanner(GlobalPlanner):
    """Sample reachable goals from the explored free-space map.

    The generator is derived from the experiment seed, episode index and
    replan step. Repeating or resuming the same evaluation therefore produces
    the same long-term goal for an identical planning state.
    """

    name = "random"
    uses_map_goal_sampling = True

    def __init__(
        self,
        *,
        random_seed: int = 1,
        min_goal_distance_m: float = 1.0,
        map_resolution_cm: float = 5.0,
    ) -> None:
        if float(min_goal_distance_m) < 0.0:
            raise ValueError("min_goal_distance_m must be non-negative")
        if float(map_resolution_cm) <= 0.0:
            raise ValueError("map_resolution_cm must be positive")
        self.random_seed = int(random_seed)
        self.min_goal_distance_cells = (
            float(min_goal_distance_m) * 100.0 / float(map_resolution_cm)
        )

    def _rng(self, context: GlobalPlannerContext) -> np.random.Generator:
        words = [
            self.random_seed & 0xFFFFFFFF,
            int(context.episode_index) & 0xFFFFFFFF,
            int(context.navigation_step) & 0xFFFFFFFF,
            int(context.local_step) & 0xFFFFFFFF,
        ]
        return np.random.default_rng(np.random.SeedSequence(words))

    @staticmethod
    def _nearest_seed(
        traversable: np.ndarray,
        agent_cell,
    ) -> Optional[Tuple[int, int]]:
        cells = np.argwhere(traversable)
        if cells.size == 0:
            return None
        shape = traversable.shape
        row = int(np.clip(round(float(agent_cell[0])), 0, shape[0] - 1))
        col = int(np.clip(round(float(agent_cell[1])), 0, shape[1] - 1))
        if traversable[row, col]:
            return row, col
        nearest = cells[
            int(
                np.argmin(
                    np.linalg.norm(
                        cells - np.asarray([row, col])[None, :],
                        axis=1,
                    )
                )
            )
        ]
        return int(nearest[0]), int(nearest[1])

    def _traversable_map(self, context: GlobalPlannerContext) -> np.ndarray:
        traversable = (
            (np.asarray(context.explored_map) > 0.0)
            & (np.asarray(context.obstacle_map) <= 0.5)
        )
        if context.risk is not None:
            traversable &= ~np.asarray(context.risk.hard_unsafe, dtype=bool)
            traversable &= (
                np.asarray(context.risk.planning_risk, dtype=np.float32)
                <= float(context.risk.danger_threshold)
            )
        return traversable

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        traversable = self._traversable_map(context)
        _, labels = cv2.connectedComponents(
            traversable.astype(np.uint8),
            connectivity=8,
        )
        rng = self._rng(context)
        used: Set[Tuple[int, int]] = set()
        goals: List[List[int]] = []

        for robot_id in range(context.num_agents):
            agent_cell = context.agent_cells[robot_id]
            seed = self._nearest_seed(traversable, agent_cell)
            if seed is None:
                shape = traversable.shape
                goals.append(
                    [
                        int(
                            np.clip(
                                round(float(agent_cell[0])),
                                0,
                                shape[0] - 1,
                            )
                        ),
                        int(
                            np.clip(
                                round(float(agent_cell[1])),
                                0,
                                shape[1] - 1,
                            )
                        ),
                    ]
                )
                continue

            component = labels == labels[seed]
            candidates = np.argwhere(component)
            distances = np.linalg.norm(
                candidates
                - np.asarray(agent_cell[:2], dtype=np.float64)[None, :],
                axis=1,
            )
            long_term = distances >= self.min_goal_distance_cells
            if long_term.any():
                candidates = candidates[long_term]

            if used and len(candidates) > 1:
                unused = np.asarray(
                    [
                        cell
                        for cell in candidates
                        if (int(cell[0]), int(cell[1])) not in used
                    ],
                    dtype=np.int64,
                )
                if unused.size:
                    candidates = unused.reshape(-1, 2)

            chosen = candidates[int(rng.integers(0, len(candidates)))]
            goal = (int(chosen[0]), int(chosen[1]))
            used.add(goal)
            goals.append([goal[0], goal[1]])

        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments={
                robot_id: None for robot_id in range(context.num_agents)
            },
        )
