"""Common request/result contract for global frontier planners."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils.risk.frontier import FrontierRiskReport


GridPoint = Sequence[int]


@dataclass(frozen=True)
class AgentFrontierMap:
    """One robot's map and frontier namespace.

    Frontier IDs are local to this record. The team-level individual-map
    adapter flattens them only after each robot has made a choice, so two
    locally named frontier-zero candidates are never treated as one shared
    frontier.
    """

    target_score: Optional[Sequence[float]]
    target_edge_map: np.ndarray
    target_points: Sequence[GridPoint]
    obstacle_map: np.ndarray
    explored_map: np.ndarray
    top_view_map: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.obstacle_map).shape
        if len(shape) != 2:
            raise ValueError("agent obstacle_map must be a 2-D map")
        if np.asarray(self.explored_map).shape != shape:
            raise ValueError(
                "agent explored_map shape must match obstacle_map shape"
            )
        if np.asarray(self.target_edge_map).shape != shape:
            raise ValueError(
                "agent target_edge_map shape must match obstacle_map shape"
            )
        if np.asarray(self.top_view_map).shape[:2] != shape:
            raise ValueError(
                "agent top_view_map spatial shape must match obstacle_map"
            )
        if (
            self.target_score is not None
            and len(self.target_score) < len(self.target_points)
        ):
            raise ValueError(
                "agent target_score must contain every target point"
            )


def merge_agent_frontier_maps(
    agent_maps: Sequence[AgentFrontierMap],
):
    """Flatten local frontier labels for diagnostics and visualization only."""

    if not agent_maps:
        raise ValueError("agent_maps must not be empty")
    shape = np.asarray(agent_maps[0].target_edge_map).shape
    merged_edge = np.zeros(shape, dtype=np.int32)
    merged_points = []
    merged_scores = []
    all_scores_known = True
    offset = 0
    for agent_map in agent_maps:
        edge = np.asarray(agent_map.target_edge_map)
        if edge.shape != shape:
            raise ValueError("all agent frontier maps must share one shape")
        for local_id, point in enumerate(agent_map.target_points):
            merged_edge[edge == local_id + 1] = offset + local_id + 1
            merged_points.append([int(point[0]), int(point[1])])
            if agent_map.target_score is None:
                all_scores_known = False
            else:
                merged_scores.append(
                    float(agent_map.target_score[local_id])
                )
        offset += len(agent_map.target_points)
    return (
        merged_scores if all_scores_known else None,
        merged_edge,
        merged_points,
    )


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
    route_risk_alpha: float = 4.0

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
        if float(self.route_risk_alpha) < 0.0:
            raise ValueError("route_risk_alpha must be non-negative")


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
    agent_ids: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        if int(self.num_agents) < 1:
            raise ValueError("num_agents must be at least one")
        if len(self.poses) < self.num_agents:
            raise ValueError("poses must contain one entry per robot")
        if len(self.agent_cells) < self.num_agents:
            raise ValueError("agent_cells must contain one entry per robot")
        if (
            self.agent_ids is not None
            and len(self.agent_ids) < self.num_agents
        ):
            raise ValueError("agent_ids must contain one entry per robot")
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
    frontier_report_agent_ids: List[int] = field(default_factory=list)
    frontier_computed_step: Optional[int] = None


class GlobalPlanner(ABC):
    """Interface implemented by every global frontier planner."""

    name: str
    uses_shared_frontier_map: bool = False

    @abstractmethod
    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        """Assign one map goal to every robot."""

    def plan_individual_maps(
        self,
        context: GlobalPlannerContext,
        agent_maps: Sequence[AgentFrontierMap],
    ) -> GlobalPlannerResult:
        """Assign goals from robot-local maps.

        The factory's risk-aware adapter implements this for every non-GPT
        planner. Direct planner instances retain a clear failure mode instead
        of silently falling back to the shared map.
        """

        del context, agent_maps
        raise RuntimeError(
            f"{self.name} does not implement individual-map planning"
        )

    def frontier_preferences(
        self,
        context: GlobalPlannerContext,
    ) -> Optional[Dict[int, Sequence[float]]]:
        """Return the normal policy's per-robot frontier preferences.

        Classical frontier planners override this method.  A shared safety
        layer can then add risk costs without reimplementing or replacing the
        planner's normal objective.  Non-frontier policies such as random map
        sampling and GPT return ``None`` and use their dedicated adapters.
        """

        del context
        return None

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
