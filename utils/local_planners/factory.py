"""Factory for FMM, A*, map-RL and pretrained PointNav local planners."""
from __future__ import annotations

from utils.fmm_planner import FMMPlanner
from utils.risk.config import RiskConfig

from .astar import AStarPlanner
from .pointnav import (
    load_pointnav_policy_adapter,
    load_pointnav_runtime_spec,
)
from .rl import RLGridPlanner, load_rl_policy


LOCAL_PLANNERS = ("fmm", "astar", "rl", "pointnav")


def resolve_fmm_backend(args) -> str:
    """Resolve one auditable FMM backend for the active experiment.

    Historical normal runs remain navmesh-shortest-path-first. Fire-risk
    ablations deliberately use grid FMM for both evaluator-only ``none`` and
    planner-aware ``oracle``/``sensed`` conditions, so risk information cannot
    silently change the geometric execution backend.
    """

    requested = str(getattr(args, "fmm_backend", "auto")).strip().lower()
    if requested not in {"auto", "navmesh", "grid"}:
        raise ValueError("fmm_backend must be one of auto, navmesh or grid")
    if requested != "auto":
        return requested
    fire_risk_ablation = bool(
        int(getattr(args, "fire_world", 0))
        and int(getattr(args, "risk_enabled", 0))
    )
    return "grid" if fire_risk_ablation else "navmesh"


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
    resolve_fmm_backend(args)
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
    if planner_name == "pointnav":
        if float(getattr(args, "pointnav_goal_tolerance", 0.05)) < 0.0:
            raise ValueError("--pointnav_goal_tolerance must be non-negative")
        load_pointnav_runtime_spec(
            getattr(args, "pointnav_checkpoint", None),
            getattr(args, "pointnav_config", None),
            getattr(args, "pointnav_observation_mode", "auto"),
        )
    return aware


def create_local_planner(
    name,
    traversible=None,
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
    pointnav_checkpoint=None,
    pointnav_config=None,
    pointnav_device="cpu",
    pointnav_deterministic=True,
    pointnav_goal_tolerance=0.05,
    pointnav_env_action_map=None,
    pointnav_observation_mode="auto",
):
    planner_name = str(name).strip().lower()
    if planner_name == "pointnav":
        return load_pointnav_policy_adapter(
            pointnav_checkpoint,
            config_path=pointnav_config,
            device=pointnav_device,
            deterministic=pointnav_deterministic,
            goal_tolerance_m=pointnav_goal_tolerance,
            env_action_map=pointnav_env_action_map,
            observation_mode=pointnav_observation_mode,
        )
    if traversible is None:
        raise ValueError(
            "{} local planner requires a traversible grid".format(
                planner_name
            )
        )
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
