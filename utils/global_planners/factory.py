"""Factory for global frontier planners."""
from __future__ import annotations

from .co_ut import CostUtilityGlobalPlanner
from .fill import FillGlobalPlanner
from .gpt import GPTGlobalPlanner
from .nearest import NearestGlobalPlanner
from .random import RandomGlobalPlanner
from .risk_aware import RiskAwareGlobalPlanner


GLOBAL_PLANNERS = ("nearest", "co_ut", "fill", "random", "gpt")


def create_global_planner(name, **kwargs):
    """Build one planner behind the shared risk-aware interface."""

    planner_name = str(name).strip().lower()
    cost_utility_lambda = kwargs.pop("cost_utility_lambda", 1.0)
    random_seed = kwargs.pop("random_seed", 1)
    random_goal_min_distance_m = kwargs.pop(
        "random_goal_min_distance_m",
        1.0,
    )
    map_resolution_cm = kwargs.pop("map_resolution_cm", 5.0)
    if planner_name == "nearest":
        planner = NearestGlobalPlanner()
    elif planner_name == "co_ut":
        planner = CostUtilityGlobalPlanner(cost_utility_lambda)
    elif planner_name == "fill":
        planner = FillGlobalPlanner()
    elif planner_name == "random":
        planner = RandomGlobalPlanner(
            random_seed=random_seed,
            min_goal_distance_m=random_goal_min_distance_m,
            map_resolution_cm=map_resolution_cm,
        )
    elif planner_name == "gpt":
        planner = GPTGlobalPlanner(
            cost_utility_lambda=cost_utility_lambda,
            **kwargs,
        )
    else:
        raise ValueError(
            "unknown global planner {!r}; choose one of {}".format(
                name,
                ", ".join(GLOBAL_PLANNERS),
            )
        )
    return RiskAwareGlobalPlanner(planner)
