"""Factory for the formal FMM, A* and RL local-planner baselines."""
from __future__ import annotations

from utils.fmm_planner import FMMPlanner
from utils.risk.config import RiskConfig

from .astar import AStarPlanner
from .rl import RLGridPlanner, load_rl_policy


LOCAL_PLANNERS = ("fmm", "astar", "rl")


def validate_local_planner_config(args):
    """Validate planner settings and return automatic risk awareness.

    Local awareness is deliberately not a separate experiment switch.  A
    sensed/oracle planning map makes every local planner risk-aware; risk-off
    and evaluator-only ``source=none`` runs remain risk-blind.
    """
    planner_name = str(getattr(args, "local_planner", "fmm")).lower()
    if planner_name not in LOCAL_PLANNERS:
        raise ValueError(
            "unknown local planner {!r}; choose one of {}".format(
                planner_name, ", ".join(LOCAL_PLANNERS)
            )
        )
    risk_config = RiskConfig.from_namespace(args)
    aware = risk_config.effective_source in {"oracle", "sensed"}
    crop_size = int(getattr(args, "rl_local_crop_size", 31))
    if crop_size < 5 or crop_size % 2 == 0:
        raise ValueError("--rl_local_crop_size must be an odd integer >= 5")
    if int(getattr(args, "rl_local_rollout_steps", 5)) < 1:
        raise ValueError("--rl_local_rollout_steps must be at least 1")
    if planner_name == "rl":
        load_rl_policy(
            getattr(args, "rl_local_checkpoint", None),
            device=getattr(args, "rl_local_device", "cpu"),
            expected_risk_aware=aware,
            expected_crop_size=crop_size,
        )
    return aware


def create_local_planner(
    name,
    traversible,
    *,
    risk_map=None,
    risk_alpha=0.0,
    hard_unsafe_mask=None,
    rl_checkpoint=None,
    rl_device="cpu",
    rl_deterministic=True,
    rl_crop_size=31,
    rl_rollout_steps=5,
    risk_aware=False,
):
    planner_name = str(name).strip().lower()
    common = {
        "risk_map": risk_map,
        "risk_alpha": risk_alpha,
        "hard_unsafe_mask": hard_unsafe_mask,
    }
    if planner_name == "fmm":
        return FMMPlanner(traversible, **common)
    if planner_name == "astar":
        return AStarPlanner(traversible, **common)
    if planner_name == "rl":
        return RLGridPlanner(
            traversible,
            checkpoint_path=rl_checkpoint,
            device=rl_device,
            deterministic=rl_deterministic,
            crop_size=rl_crop_size,
            rollout_steps=rl_rollout_steps,
            risk_aware=bool(risk_aware),
            **common,
        )
    raise ValueError(
        "unknown local planner {!r}; choose one of {}".format(
            name, ", ".join(LOCAL_PLANNERS)
        )
    )
