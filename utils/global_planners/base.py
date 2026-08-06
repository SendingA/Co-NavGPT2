"""Common request/result contract for global frontier planners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils.risk.frontier import FrontierRiskReport


GridPoint = Sequence[int]


@dataclass(frozen=True)
class RiskPlanningContext:
    """Risk inputs shared by every risk-aware global planner."""

    planning_risk: np.ndarray
    confidence: np.ndarray
    hard_unsafe: np.ndarray
    danger_threshold: float
    hard_frontier_threshold: float
    frontier_weight: float
    map_resolution_cm: float

    def __post_init__(self) -> None:
        shape = np.asarray(self.planning_risk).shape
        if len(shape) != 2:
            raise ValueError("planning_risk must be a 2-D map")
        for name, value in (
            ("confidence", self.confidence),
            ("hard_unsafe", self.hard_unsafe),
        ):
            if np.asarray(value).shape != shape:
                raise ValueError(
                    f"{name} shape {np.asarray(value).shape} does not match "
                    f"planning_risk shape {shape}"
                )
        if float(self.map_resolution_cm) <= 0.0:
            raise ValueError("map_resolution_cm must be positive")


@dataclass(frozen=True)
class GlobalPlannerContext:
    """All state needed to turn detected frontiers into robot goals.

    ``poses`` follows the existing visualization/global-planner convention,
    while ``agent_cells`` contains direct occupancy-grid cells used by the
    risk map. Keeping the two explicit prevents accidental coordinate mixing.
    """

    target_score: Optional[Sequence[float]]
    target_edge_map: np.ndarray
    target_points: Sequence[GridPoint]
    poses: Sequence[Sequence[float]]
    agent_cells: Sequence[GridPoint]
    obstacle_map: np.ndarray
    explored_map: np.ndarray
    top_view_map: np.ndarray
    goal_name: str
    local_step: int
    navigation_step: int
    num_agents: int
    risk: Optional[RiskPlanningContext] = None
    episode_index: int = 0

    def __post_init__(self) -> None:
        if int(self.num_agents) < 1:
            raise ValueError("num_agents must be at least one")
        if len(self.poses) < self.num_agents:
            raise ValueError("poses must contain one entry per robot")
        if len(self.agent_cells) < self.num_agents:
            raise ValueError("agent_cells must contain one entry per robot")
        map_shape = np.asarray(self.obstacle_map).shape
        if len(map_shape) != 2:
            raise ValueError("obstacle_map must be a 2-D map")
        if np.asarray(self.explored_map).shape != map_shape:
            raise ValueError(
                "explored_map shape must match obstacle_map shape"
            )
        if np.asarray(self.target_edge_map).shape != map_shape:
            raise ValueError(
                "target_edge_map shape must match obstacle_map shape"
            )
        top_view_shape = np.asarray(self.top_view_map).shape
        if top_view_shape[:2] != map_shape:
            raise ValueError(
                "top_view_map spatial shape must match obstacle_map shape"
            )
        if self.risk is not None:
            if np.asarray(self.risk.planning_risk).shape != map_shape:
                raise ValueError(
                    "risk-map shape must match obstacle_map shape"
                )


@dataclass
class GlobalPlannerResult:
    """Goals plus metadata consumed by navigation and risk logging."""

    goal_points: List[List[int]]
    frontier_assignments: Dict[int, Optional[int]] = field(
        default_factory=dict
    )
    frontier_reports: List[FrontierRiskReport] = field(default_factory=list)
    frontier_computed_step: Optional[int] = None


class GlobalPlanner(ABC):
    """Interface implemented by every global frontier planner."""

    name: str

    @abstractmethod
    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        """Assign one map goal to every robot."""

    def refine_risk_assignments(
        self,
        context: GlobalPlannerContext,
        reports: Sequence[FrontierRiskReport],
        fallback_assignments: Dict[int, Optional[int]],
        hard_risk_threshold: float,
    ) -> Dict[int, Optional[int]]:
        """Optionally refine deterministic risk assignments.

        Classical planners use the deterministic assignments unchanged.
        The GPT implementation overrides this hook and applies the existing
        VLM response guard before returning.
        """

        del context, reports, hard_risk_threshold
        return fallback_assignments


def random_goal_result(context: GlobalPlannerContext) -> GlobalPlannerResult:
    """Preserve the legacy uniform random-map fallback."""

    goals: List[List[int]] = []
    for _ in range(context.num_agents):
        action = np.random.rand(1, 2).squeeze() * (
            context.obstacle_map.shape[0] - 1
        )
        goals.append([int(action[0]), int(action[1])])
    return GlobalPlannerResult(
        goal_points=goals,
        frontier_assignments={
            robot_id: None for robot_id in range(context.num_agents)
        },
    )


def goal_from_frontier(
    context: GlobalPlannerContext,
    frontier_id: int,
) -> List[int]:
    point = context.target_points[int(frontier_id)]
    return [int(point[0]), int(point[1])]
