"""Composable risk-aware decorator for global planners."""
from __future__ import annotations

from utils.risk.frontier import UtilityWeights

from .base import (
    GlobalPlanner,
    GlobalPlannerContext,
    GlobalPlannerResult,
    goal_from_frontier,
)
from .risk_module import (
    SharedRiskAwareness,
    grid_line_cells,
    low_risk_fallback_goal,
)


def risk_utility_weights(
    nav_mode: str,
    frontier_weight: float,
    cost_utility_lambda: float = 1.0,
) -> UtilityWeights:
    """Return legacy mode weights for import compatibility.

    The composable safety path no longer uses this function: each planner now
    supplies its exact normal frontier preferences, and
    :class:`SharedRiskAwareness` adds the same risk costs to all of them.  The
    helper remains available so older integrations importing it do not break.
    """

    risk_weight = float(frontier_weight)
    if nav_mode == "nearest":
        return UtilityWeights(
            information_gain=0.0,
            distance=1.0,
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.0,
        )
    if nav_mode == "co_ut":
        return UtilityWeights(
            information_gain=1.0,
            distance=float(cost_utility_lambda),
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.0,
        )
    if nav_mode == "fill":
        return UtilityWeights(
            information_gain=1.0,
            distance=0.25,
            risk=risk_weight,
            uncertainty=0.5,
            redundancy=0.75,
        )
    return UtilityWeights(
        information_gain=1.0,
        distance=0.35,
        risk=risk_weight,
        uncertainty=0.5,
        redundancy=0.75,
    )


class RiskAwareGlobalPlanner(GlobalPlanner):
    """Run the normal policy, then add one shared navigation-safety layer."""

    def __init__(self, planner: GlobalPlanner) -> None:
        self._planner = planner
        self.name = planner.name

    def _risk_base_planner(self) -> GlobalPlanner:
        """Select the deterministic normal policy used before risk costs."""

        # GPT keeps its semantic refinement, but its deterministic safety
        # fallback is the normal co_ut planner configured by GPTGlobalPlanner.
        return getattr(self._planner, "_fallback", self._planner)

    def plan(self, context: GlobalPlannerContext) -> GlobalPlannerResult:
        if context.risk is None:
            return self._planner.plan(context)

        awareness = SharedRiskAwareness(context)
        reports = awareness.build_reports()

        if bool(getattr(self._planner, "uses_map_goal_sampling", False)):
            result = self._planner.plan_in_domain(
                context,
                awareness.safe_traversable_map(),
            )
            result.frontier_reports = list(reports)
            result.frontier_computed_step = context.navigation_step
            return result

        base_planner = self._risk_base_planner()
        base_preferences = base_planner.frontier_preferences(context)
        if base_preferences is None:
            raise RuntimeError(
                "{} must expose frontier_preferences for shared risk "
                "awareness".format(base_planner.name)
            )
        risk_result = awareness.assign_frontiers(
            base_preferences,
            reports=reports,
        )
        assignments = self._planner.refine_risk_assignments(
            context,
            risk_result.reports,
            risk_result.assignments,
            risk_result.hard_threshold,
        )

        agent_cells = [
            [int(cell[0]), int(cell[1])]
            for cell in context.agent_cells[: context.num_agents]
        ]
        goals = []
        normalized_assignments = {}
        for robot_id in range(context.num_agents):
            frontier_id = assignments.get(robot_id)
            if (
                frontier_id is not None
                and 0 <= int(frontier_id) < len(context.target_points)
            ):
                normalized_assignments[robot_id] = int(frontier_id)
                goals.append(goal_from_frontier(context, int(frontier_id)))
            else:
                normalized_assignments[robot_id] = None
                goals.append(
                    low_risk_fallback_goal(
                        agent_cells[robot_id],
                        context.obstacle_map,
                        context.explored_map,
                        context.risk.planning_risk,
                        context.risk.hard_unsafe,
                    )
                )

        return GlobalPlannerResult(
            goal_points=goals,
            frontier_assignments=normalized_assignments,
            frontier_reports=list(risk_result.reports),
            frontier_computed_step=context.navigation_step,
        )


__all__ = [
    "RiskAwareGlobalPlanner",
    "SharedRiskAwareness",
    "grid_line_cells",
    "low_risk_fallback_goal",
    "risk_utility_weights",
]
