"""Habitat PointNav checkpoint adapter for frontier-local navigation.

The existing ``rl`` planner in this project is a map-crop PPO policy.  A
Habitat PointNav policy has a fundamentally different contract: it consumes
camera observations and a point goal, owns recurrent state, and emits Habitat
actions directly.  This module keeps that contract separate and derives every
runtime schema value from the checkpoint's training config.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


POINTNAV_ACTION_NAMES = (
    "stop",
    "move_forward",
    "turn_left",
    "turn_right",
)
SUPPORTED_POLICY_NAMES = {
    "PointNavBaselinePolicy",
    "PointNavResNetPolicy",
}
DEFAULT_OFFICIAL_POINTNAV_CHECKPOINT = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)
DEFAULT_OFFICIAL_POINTNAV_CONFIG = "ddppo_pointnav.yaml"
OFFICIAL_POINTNAV_PROFILES = {
    "gibson-4plus-resnet50.pth": {
        "observation_mode": "depth",
        "backbone": "resnet50",
        "hidden_size": 512,
    },
    "gibson-2plus-resnet50.pth": {
        "observation_mode": "depth",
        "backbone": "resnet50",
        "hidden_size": 512,
    },
    "gibson-2plus-se-resneXt50.pth": {
        "observation_mode": "depth",
        "backbone": "se_resneXt50",
        "hidden_size": 512,
    },
    "gibson-2plus-se-resneXt101-lstm1024.pth": {
        "observation_mode": "depth",
        "backbone": "se_resneXt101",
        "hidden_size": 1024,
    },
    "gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth": {
        "observation_mode": "rgb",
        "backbone": "se_resneXt50",
        "hidden_size": 512,
    },
    "gibson-0plus-mp3d-train-val-test-blind.pth": {
        "observation_mode": "blind",
        "backbone": "resnet50",
        "hidden_size": 512,
    },
}

_CHECKPOINT_CACHE: Dict[str, Mapping[str, Any]] = {}
_ADAPTER_CACHE: Dict[Tuple[Any, ...], "PointNavPolicyAdapter"] = {}


def _get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    return getattr(node, key, default)


def _items(node: Any):
    if node is None:
        return ()
    if isinstance(node, Mapping) or hasattr(node, "items"):
        return node.items()
    raise TypeError("configuration node does not provide mapping items()")


def _torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _rotation_matrix(rotation: Any) -> np.ndarray:
    """Return a 3x3 world-from-agent rotation matrix."""
    if isinstance(rotation, np.ndarray):
        matrix = np.asarray(rotation, dtype=np.float64)
        if matrix.shape == (3, 3):
            return matrix
    try:
        import quaternion

        return quaternion.as_rotation_matrix(rotation).astype(np.float64)
    except (ImportError, TypeError, ValueError) as exc:
        raise TypeError(
            "agent rotation must be a numpy-quaternion or a 3x3 matrix"
        ) from exc


def compute_pointgoal(
    source_position: Sequence[float],
    source_rotation: Any,
    goal_position: Sequence[float],
    *,
    goal_format: str = "POLAR",
    dimensionality: int = 2,
) -> np.ndarray:
    """Match Habitat's ``PointGoalSensor._compute_pointgoal`` exactly."""
    goal_format = str(goal_format).upper()
    dimensionality = int(dimensionality)
    if goal_format not in {"POLAR", "CARTESIAN"}:
        raise ValueError("PointNav goal_format must be POLAR or CARTESIAN")
    if dimensionality not in {2, 3}:
        raise ValueError("PointNav point-goal dimensionality must be 2 or 3")

    source = np.asarray(source_position, dtype=np.float64)
    goal = np.asarray(goal_position, dtype=np.float64)
    if source.shape != (3,) or goal.shape != (3,):
        raise ValueError("PointNav source and goal positions must be 3-D")
    direction_agent = _rotation_matrix(source_rotation).T @ (goal - source)

    if goal_format == "CARTESIAN":
        if dimensionality == 2:
            return np.asarray(
                [-direction_agent[2], direction_agent[0]],
                dtype=np.float32,
            )
        return direction_agent.astype(np.float32)

    rho_xz = float(
        np.hypot(-direction_agent[2], direction_agent[0])
    )
    phi = float(np.arctan2(direction_agent[0], -direction_agent[2]))
    if dimensionality == 2:
        return np.asarray([rho_xz, -phi], dtype=np.float32)

    rho = float(np.linalg.norm(direction_agent))
    theta = (
        0.0
        if rho <= np.finfo(np.float64).eps
        else float(np.arccos(np.clip(direction_agent[1] / rho, -1.0, 1.0)))
    )
    return np.asarray([rho, -phi, theta], dtype=np.float32)


def compute_episode_gps(
    agent_position: Sequence[float],
    episode_start_position: Sequence[float],
    episode_start_rotation: Any,
    *,
    dimensionality: int = 2,
) -> np.ndarray:
    """Match Habitat's episodic GPS coordinate convention."""
    position = np.asarray(agent_position, dtype=np.float64)
    origin = np.asarray(episode_start_position, dtype=np.float64)
    relative = _rotation_matrix(episode_start_rotation).T @ (
        position - origin
    )
    if int(dimensionality) == 2:
        return np.asarray([-relative[2], relative[0]], dtype=np.float32)
    if int(dimensionality) == 3:
        return relative.astype(np.float32)
    raise ValueError("PointNav GPS dimensionality must be 2 or 3")


def compute_compass(
    agent_rotation: Any,
    episode_start_rotation: Any,
) -> np.ndarray:
    """Match Habitat's episodic CompassSensor heading convention."""
    # R_agent_from_start is equivalent to
    # ``rotation_world_agent.inverse() * rotation_world_start``.
    relative = (
        _rotation_matrix(agent_rotation).T
        @ _rotation_matrix(episode_start_rotation)
    )
    heading_vector = relative @ np.asarray([0.0, 0.0, -1.0])
    phi = float(np.arctan2(heading_vector[0], -heading_vector[2]))
    return np.asarray([phi], dtype=np.float32)


def compute_heading(agent_rotation: Any) -> np.ndarray:
    """Match Habitat's absolute HeadingSensor convention."""
    heading_vector = (
        _rotation_matrix(agent_rotation).T
        @ np.asarray([0.0, 0.0, -1.0])
    )
    phi = float(np.arctan2(heading_vector[0], -heading_vector[2]))
    return np.asarray([phi], dtype=np.float32)


def frontier_grid_to_world(
    frontier_cell: Sequence[float],
    *,
    origins_grid: Sequence[float],
    map_resolution_cm: float,
    camera_local_y: float,
    initial_agent_position: Sequence[float],
    initial_sensor_rotation: Any,
) -> np.ndarray:
    """Convert the project's row/column frontier into Habitat world XYZ.

    This is the same Open3D-to-Habitat transform historically used by
    ``VLM_Agent.act``.  Keeping it in the adapter makes the coordinate
    convention explicit and testable before computing the policy point goal.
    """
    cell = np.asarray(frontier_cell, dtype=np.float64)
    origin = np.asarray(origins_grid, dtype=np.float64)
    if cell.shape != (2,) or origin.shape != (2,):
        raise ValueError("frontier_cell and origins_grid must be 2-D")
    scale = float(map_resolution_cm) / 100.0
    open3d_goal = np.asarray(
        [
            (cell[0] - origin[0]) * scale,
            float(camera_local_y),
            (cell[1] - origin[1]) * scale,
        ],
        dtype=np.float64,
    )
    rx = np.asarray(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    habitat_to_open3d = _rotation_matrix(initial_sensor_rotation) @ rx.T
    return (
        habitat_to_open3d @ open3d_goal
        + np.asarray(initial_agent_position, dtype=np.float64)
    ).astype(np.float32)


def world_to_frontier_grid(
    goal_world: Sequence[float],
    *,
    origins_grid: Sequence[float],
    map_resolution_cm: float,
    initial_agent_position: Sequence[float],
    initial_sensor_rotation: Any,
) -> np.ndarray:
    """Inverse XZ map projection for a controlled known world goal."""

    world = np.asarray(goal_world, dtype=np.float64)
    origin = np.asarray(origins_grid, dtype=np.float64)
    initial = np.asarray(initial_agent_position, dtype=np.float64)
    if world.shape != (3,) or initial.shape != (3,) or origin.shape != (2,):
        raise ValueError(
            "goal_world, initial_agent_position and origins_grid must be 3-D, "
            "3-D and 2-D respectively"
        )
    scale = float(map_resolution_cm) / 100.0
    if scale <= 0.0:
        raise ValueError("map_resolution_cm must be positive")
    rx = np.asarray(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    habitat_to_open3d = _rotation_matrix(initial_sensor_rotation) @ rx.T
    open3d_goal = habitat_to_open3d.T @ (world - initial)
    return np.asarray(
        [
            origin[0] + open3d_goal[0] / scale,
            origin[1] + open3d_goal[2] / scale,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ObservationField:
    uuid: str
    kind: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    low: float
    high: float
    config: Any


@dataclass(frozen=True)
class PointNavRuntimeSpec:
    config: Any
    policy_name: str
    policy_agent_name: str
    simulator_agent_name: str
    observation_fields: Tuple[ObservationField, ...]
    action_names: Tuple[str, ...]
    forward_step_size: float
    turn_angle: float
    rgb_sensor_config: Optional[Any]
    depth_sensor_config: Optional[Any]
    observation_mode: str
    official_profile: Optional[str]

    @property
    def field_map(self) -> Dict[str, ObservationField]:
        return {field.uuid: field for field in self.observation_fields}


def _load_checkpoint_payload(checkpoint_path: str) -> Mapping[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "PointNav checkpoint does not exist: {}".format(path)
        )
    key = str(path)
    if key not in _CHECKPOINT_CACHE:
        payload = _torch_load(path)
        if not isinstance(payload, Mapping):
            raise ValueError(
                "PointNav checkpoint must contain a dictionary payload"
            )
        _CHECKPOINT_CACHE[key] = payload
    return _CHECKPOINT_CACHE[key]


def _load_fallback_config(config_path: Optional[str]):
    try:
        from habitat_baselines.config.default import get_config
        import habitat_baselines
    except ImportError as exc:
        raise ImportError(
            "Habitat-Baselines 0.3.3 and its dependencies are required for "
            "--local_planner=pointnav. Install the matching checkout with "
            "`python -m pip install -e <HABITAT_LAB_ROOT>/habitat-baselines`."
        ) from exc

    if config_path:
        path = Path(config_path).expanduser()
        if path.is_file():
            return get_config(str(path.resolve()))
        packaged = (
            Path(habitat_baselines.__file__).resolve().parent
            / "config"
            / "pointnav"
            / config_path
        )
        if packaged.is_file():
            return get_config(str(packaged))
        raise FileNotFoundError(
            "PointNav policy config does not exist as a path or packaged "
            "Habitat-Baselines config: {}".format(config_path)
        )

    packaged = (
        Path(habitat_baselines.__file__).resolve().parent
        / "config"
        / "pointnav"
        / DEFAULT_OFFICIAL_POINTNAV_CONFIG
    )
    if not packaged.is_file():
        raise FileNotFoundError(
            "Habitat-Baselines is missing its official PointNav config: "
            "{}".format(packaged)
        )
    return get_config(str(packaged))


def load_pointnav_config(checkpoint_path: str, config_path: Optional[str]):
    payload = _load_checkpoint_payload(checkpoint_path)
    config = payload.get("config")
    if config is None:
        config = _load_fallback_config(config_path)
    return config


def _official_profile(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    return OFFICIAL_POINTNAV_PROFILES.get(
        Path(checkpoint_path).expanduser().name
    )


def _apply_official_profile(config: Any, profile: Mapping[str, Any]) -> None:
    """Apply the architecture metadata used by Habitat's official tests."""
    from contextlib import nullcontext

    try:
        from habitat.config.read_write import read_write
        from omegaconf import DictConfig
    except ImportError:
        context = nullcontext(config)
    else:
        context = (
            read_write(config)
            if isinstance(config, DictConfig)
            else nullcontext(config)
        )
    with context:
        ddppo = config.habitat_baselines.rl.ddppo
        ppo = config.habitat_baselines.rl.ppo
        ddppo.backbone = str(profile["backbone"])
        ddppo.rnn_type = "LSTM"
        ddppo.num_recurrent_layers = 2
        ppo.hidden_size = int(profile["hidden_size"])


def _sensor_uuid(sensor_key: str, sensor_type: str) -> Optional[str]:
    sensor_type = str(sensor_type)
    known = {
        "HabitatSimRGBSensor": "rgb",
        "HabitatSimDepthSensor": "depth",
        "PointGoalWithGPSCompassSensor": "pointgoal_with_gps_compass",
        "PointGoalSensor": "pointgoal",
        "GPSSensor": "gps",
        "CompassSensor": "compass",
        "HeadingSensor": "heading",
    }
    if sensor_type in known:
        return known[sensor_type]
    key = str(sensor_key).lower()
    for suffix in ("_sensor",):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key or None


def build_pointnav_runtime_spec(
    config: Any,
    *,
    observation_mode: Optional[str] = None,
    official_profile: Optional[str] = None,
) -> PointNavRuntimeSpec:
    habitat_cfg = _get(config, "habitat")
    baselines_cfg = _get(config, "habitat_baselines")
    if habitat_cfg is None or baselines_cfg is None:
        raise ValueError(
            "PointNav config must contain habitat and habitat_baselines roots"
        )

    policy_nodes = _get(_get(baselines_cfg, "rl"), "policy")
    policy_items = list(_items(policy_nodes))
    if not policy_items:
        raise ValueError("PointNav config has no habitat_baselines.rl.policy")
    policy_agent_name, policy_cfg = policy_items[0]
    policy_name = str(_get(policy_cfg, "name", ""))
    if policy_name not in SUPPORTED_POLICY_NAMES:
        raise ValueError(
            "unsupported PointNav policy {!r}; expected one of {}".format(
                policy_name, ", ".join(sorted(SUPPORTED_POLICY_NAMES))
            )
        )

    simulator = _get(habitat_cfg, "simulator")
    agents_order = list(_get(simulator, "agents_order", []))
    simulator_agents = _get(simulator, "agents")
    simulator_agent_name = (
        str(policy_agent_name)
        if _get(simulator_agents, str(policy_agent_name)) is not None
        else str(agents_order[0])
    )
    simulator_agent = _get(simulator_agents, simulator_agent_name)
    if simulator_agent is None:
        raise ValueError("PointNav config has no simulator agent definition")

    gym_obs_keys = _get(_get(habitat_cfg, "gym"), "obs_keys")
    obs_filter = None if gym_obs_keys is None else set(gym_obs_keys)
    fields = []
    rgb_config = None
    depth_config = None
    mode = None if observation_mode is None else str(observation_mode).lower()
    if mode not in {None, "depth", "rgb", "rgbd", "blind"}:
        raise ValueError(
            "PointNav observation_mode must be auto, depth, rgb, rgbd or blind"
        )
    for sensor_key, sensor_cfg in _items(
        _get(simulator_agent, "sim_sensors", {})
    ):
        kind = str(_get(sensor_cfg, "type", ""))
        uuid = _sensor_uuid(sensor_key, kind)
        if uuid is None or (obs_filter is not None and uuid not in obs_filter):
            continue
        height = int(_get(sensor_cfg, "height"))
        width = int(_get(sensor_cfg, "width"))
        if kind == "HabitatSimRGBSensor":
            rgb_config = sensor_cfg
            if mode in {"depth", "blind"}:
                continue
            fields.append(
                ObservationField(
                    uuid=uuid,
                    kind="rgb",
                    shape=(height, width, 3),
                    dtype=np.dtype(np.uint8),
                    low=0.0,
                    high=255.0,
                    config=sensor_cfg,
                )
            )
        elif kind == "HabitatSimDepthSensor":
            depth_config = sensor_cfg
            if mode in {"rgb", "blind"}:
                continue
            normalized = bool(_get(sensor_cfg, "normalize_depth", True))
            fields.append(
                ObservationField(
                    uuid=uuid,
                    kind="depth",
                    shape=(height, width, 1),
                    dtype=np.dtype(np.float32),
                    low=(
                        0.0
                        if normalized
                        else float(_get(sensor_cfg, "min_depth", 0.0))
                    ),
                    high=(
                        1.0
                        if normalized
                        else float(_get(sensor_cfg, "max_depth", 10.0))
                    ),
                    config=sensor_cfg,
                )
            )
        else:
            raise ValueError(
                "unsupported PointNav simulator observation sensor {!r}"
                .format(kind)
            )

    task = _get(habitat_cfg, "task")
    for sensor_key, sensor_cfg in _items(_get(task, "lab_sensors", {})):
        kind = str(_get(sensor_cfg, "type", ""))
        uuid = _sensor_uuid(sensor_key, kind)
        if uuid is None or (obs_filter is not None and uuid not in obs_filter):
            continue
        if kind in {"PointGoalWithGPSCompassSensor", "PointGoalSensor"}:
            dimensionality = int(_get(sensor_cfg, "dimensionality", 2))
            field_kind = (
                "integrated_pointgoal"
                if kind == "PointGoalWithGPSCompassSensor"
                else "pointgoal"
            )
            shape = (dimensionality,)
        elif kind == "GPSSensor":
            shape = (int(_get(sensor_cfg, "dimensionality", 2)),)
            field_kind = "gps"
        elif kind in {"CompassSensor", "HeadingSensor"}:
            shape = (1,)
            field_kind = kind.replace("Sensor", "").lower()
        else:
            raise ValueError(
                "unsupported PointNav lab observation sensor {!r}".format(
                    kind
                )
            )
        fields.append(
            ObservationField(
                uuid=uuid,
                kind=field_kind,
                shape=shape,
                dtype=np.dtype(np.float32),
                low=float(np.finfo(np.float32).min),
                high=float(np.finfo(np.float32).max),
                config=sensor_cfg,
            )
        )

    field_uuids = {field.uuid for field in fields}
    if not (
        {"pointgoal_with_gps_compass", "pointgoal"} & field_uuids
        or {"gps", "compass"} <= field_uuids
    ):
        raise ValueError(
            "PointNav checkpoint must use a point goal or GPS+compass"
        )

    action_names = tuple(str(name) for name, _ in _items(
        _get(task, "actions", {})
    ))
    if action_names != POINTNAV_ACTION_NAMES:
        raise ValueError(
            "PointNav action schema must be {}; checkpoint has {}".format(
                POINTNAV_ACTION_NAMES, action_names
            )
        )

    return PointNavRuntimeSpec(
        config=config,
        policy_name=policy_name,
        policy_agent_name=str(policy_agent_name),
        simulator_agent_name=simulator_agent_name,
        observation_fields=tuple(fields),
        action_names=action_names,
        forward_step_size=float(_get(simulator, "forward_step_size", 0.25)),
        turn_angle=float(_get(simulator, "turn_angle", 30.0)),
        rgb_sensor_config=rgb_config,
        depth_sensor_config=depth_config,
        observation_mode=(
            mode if mode is not None else (
                "rgbd"
                if {"rgb", "depth"} <= field_uuids
                else "rgb"
                if "rgb" in field_uuids
                else "depth"
                if "depth" in field_uuids
                else "blind"
            )
        ),
        official_profile=official_profile,
    )


def load_pointnav_runtime_spec(
    checkpoint_path: Optional[str],
    config_path: Optional[str] = None,
    observation_mode: str = "auto",
) -> PointNavRuntimeSpec:
    if not checkpoint_path:
        raise ValueError(
            "--pointnav_checkpoint is required when "
            "--local_planner=pointnav"
        )
    payload = _load_checkpoint_payload(checkpoint_path)
    embedded_config = payload.get("config") is not None
    config = load_pointnav_config(checkpoint_path, config_path)
    profile = _official_profile(checkpoint_path)
    if profile is not None and not embedded_config:
        _apply_official_profile(config, profile)
    requested_mode = str(observation_mode).lower()
    if requested_mode == "auto":
        selected_mode = (
            None
            if embedded_config
            else (
                str(profile["observation_mode"])
                if profile is not None
                else None
            )
        )
    else:
        selected_mode = requested_mode
    if not embedded_config and profile is None and selected_mode is None:
        raise ValueError(
            "PointNav weights have no embedded config and are not a known "
            "Habitat official model; pass --pointnav_config and "
            "--pointnav_observation_mode explicitly"
        )
    return build_pointnav_runtime_spec(
        config,
        observation_mode=selected_mode,
        official_profile=(
            Path(checkpoint_path).name if profile is not None else None
        ),
    )


def apply_pointnav_simulator_schema(
    task_config: Any,
    args: Any,
    spec: PointNavRuntimeSpec,
) -> None:
    """Add checkpoint cameras without changing the mapping RGB-D stream."""
    from omegaconf import OmegaConf

    visual_fields = [
        field
        for field in spec.observation_fields
        if field.kind in {"rgb", "depth"}
    ]
    visual_by_kind = {field.kind: field for field in visual_fields}
    if {"rgb", "depth"} <= set(visual_by_kind):
        rgb = visual_by_kind["rgb"].config
        depth = visual_by_kind["depth"].config
        geometry_fields = (
            "height",
            "width",
            "hfov",
            "position",
            "orientation",
        )
        mismatched = [
            key
            for key in geometry_fields
            if np.any(
                np.asarray(_get(rgb, key))
                != np.asarray(_get(depth, key))
            )
        ]
        if mismatched:
            raise ValueError(
                "PointNav RGB/depth cameras must be aligned; mismatched {}"
                .format(", ".join(mismatched))
            )

    simulator = task_config.habitat.simulator
    simulator.forward_step_size = float(spec.forward_step_size)
    # Habitat 0.3.3's structured SimulatorConfig declares turn_angle as an
    # integer even though checkpoint/config readers expose it numerically as
    # a float. Preserve the structured config's exact field type.
    simulator.turn_angle = int(round(spec.turn_angle))
    sensor_fields = (
        "type",
        "height",
        "width",
        "hfov",
        "position",
        "orientation",
        "sensor_subtype",
        "noise_model",
        "noise_model_kwargs",
        "min_depth",
        "max_depth",
        "normalize_depth",
    )
    for agent_cfg in simulator.agents.values():
        sensors = agent_cfg.sim_sensors
        if "rgb_sensor" not in sensors or "depth_sensor" not in sensors:
            raise ValueError(
                "ObjectNav agents need rgb_sensor and depth_sensor for "
                "the PointNav checkpoint"
            )
        for field in visual_fields:
            source_uuid = "pointnav_{}".format(field.uuid)
            sensor_key = "{}_sensor".format(source_uuid)
            payload = {
                key: _get(field.config, key)
                for key in sensor_fields
                if _get(field.config, key) is not None
            }
            payload["uuid"] = source_uuid
            sensors[sensor_key] = OmegaConf.create(payload)

    args.pointnav_observation_source_map = {
        field.uuid: "pointnav_{}".format(field.uuid)
        for field in visual_fields
    }
    args.turn_angle = float(spec.turn_angle)
    task_action_names = tuple(task_config.habitat.task.actions.keys())
    missing = [name for name in spec.action_names if name not in task_action_names]
    if missing:
        raise ValueError(
            "ObjectNav task is missing PointNav actions: {}".format(
                ", ".join(missing)
            )
        )
    args.pointnav_env_action_map = {
        name: task_action_names.index(name) for name in spec.action_names
    }


def _extract_policy_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state = payload.get("state_dict")
    if (
        state is None
        and all(isinstance(key, str) for key in payload.keys())
        and all(torch.is_tensor(value) for value in payload.values())
    ):
        state = payload
    if not isinstance(state, Mapping):
        raise ValueError("PointNav checkpoint is missing state_dict")
    keys = list(state.keys())
    for prefix in ("module.actor_critic.", "actor_critic.", "module."):
        if keys and all(str(key).startswith(prefix) for key in keys):
            return {
                str(key)[len(prefix):]: value for key, value in state.items()
            }
    return state


def _create_production_policy(
    checkpoint_path: str,
    spec: PointNavRuntimeSpec,
    *,
    device: torch.device,
):
    try:
        from gym import spaces
        from habitat_baselines.common.baseline_registry import baseline_registry
        from habitat_baselines.common.obs_transformers import (
            apply_obs_transforms_batch,
            apply_obs_transforms_obs_space,
            get_active_obs_transforms,
        )
        from habitat_baselines.utils.common import batch_obs
        # Imports register both supported policies.
        import habitat_baselines.rl.ddppo.policy.resnet_policy  # noqa: F401
        import habitat_baselines.rl.ppo.policy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Habitat-Baselines 0.3.3 and its dependencies are required for "
            "PointNav inference. Install the matching checkout with "
            "`python -m pip install -e <HABITAT_LAB_ROOT>/habitat-baselines`."
        ) from exc

    raw_space = spaces.Dict({
        field.uuid: spaces.Box(
            low=field.low,
            high=field.high,
            shape=field.shape,
            dtype=field.dtype,
        )
        for field in spec.observation_fields
    })
    transforms = get_active_obs_transforms(
        spec.config, agent_name=spec.policy_agent_name
    )
    policy_space = apply_obs_transforms_obs_space(raw_space, transforms)
    action_space = spaces.Discrete(len(spec.action_names))
    policy_cls = baseline_registry.get_policy(spec.policy_name)
    if policy_cls is None:
        raise ValueError(
            "Habitat-Baselines did not register policy {}".format(
                spec.policy_name
            )
        )
    policy = policy_cls.from_config(
        spec.config,
        policy_space,
        action_space,
        orig_action_space=action_space,
        agent_name=spec.policy_agent_name,
    )
    state = _extract_policy_state(
        _load_checkpoint_payload(checkpoint_path)
    )
    policy.load_state_dict(state, strict=True)
    policy.to(device)
    policy.eval()

    def batcher(observation):
        batch = batch_obs([observation], device=device)
        return apply_obs_transforms_batch(batch, transforms)

    return policy, batcher


@dataclass
class _RobotPolicyState:
    hidden: torch.Tensor
    prev_action: torch.Tensor
    mask: torch.Tensor
    local_goal_world: Optional[np.ndarray] = None
    goal_generation: int = 0


@dataclass(frozen=True)
class PointNavDecision:
    action: Optional[int]
    policy_action: int
    action_name: str
    local_goal_reached: bool
    request_global_replan: bool
    goal_changed: bool


class PointNavPolicyAdapter:
    """One shared policy with isolated recurrent/local-goal state per robot."""

    def __init__(
        self,
        policy: Any,
        spec: PointNavRuntimeSpec,
        *,
        device: Any = "cpu",
        deterministic: bool = True,
        goal_tolerance_m: float = 0.05,
        env_action_map: Optional[Mapping[str, int]] = None,
        batcher: Optional[Callable[[Mapping[str, np.ndarray]], Any]] = None,
    ) -> None:
        self.policy = policy
        self.spec = spec
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.goal_tolerance_m = max(0.0, float(goal_tolerance_m))
        self.env_action_map = dict(
            env_action_map
            if env_action_map is not None
            else {name: index for index, name in enumerate(spec.action_names)}
        )
        if set(self.env_action_map) != set(spec.action_names):
            raise ValueError(
                "PointNav environment action map must cover checkpoint actions"
            )
        self._batcher = batcher or self._simple_batch
        self._states: Dict[int, _RobotPolicyState] = {}

    def _simple_batch(self, observation):
        return {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in observation.items()
        }

    def _new_state(self) -> _RobotPolicyState:
        hidden_shape = tuple(int(v) for v in self.policy.hidden_state_shape)
        return _RobotPolicyState(
            hidden=torch.zeros(
                (1, *hidden_shape), dtype=torch.float32, device=self.device
            ),
            prev_action=torch.zeros(
                (1, 1), dtype=torch.long, device=self.device
            ),
            mask=torch.zeros(
                (1, 1), dtype=torch.bool, device=self.device
            ),
        )

    def _state(self, robot_id: int) -> _RobotPolicyState:
        robot_id = int(robot_id)
        if robot_id not in self._states:
            self._states[robot_id] = self._new_state()
        return self._states[robot_id]

    def reset_robot(self, robot_id: int) -> None:
        self._states[int(robot_id)] = self._new_state()

    def clear_local_goal(self, robot_id: int) -> None:
        state = self._state(robot_id)
        generation = state.goal_generation
        self._states[int(robot_id)] = self._new_state()
        self._states[int(robot_id)].goal_generation = generation

    def _set_local_goal(
        self, robot_id: int, goal_world: Sequence[float]
    ) -> bool:
        state = self._state(robot_id)
        goal = np.asarray(goal_world, dtype=np.float32)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("PointNav local goal must be a finite world XYZ")
        changed = (
            state.local_goal_world is None
            or float(np.linalg.norm(goal - state.local_goal_world))
            > self.goal_tolerance_m
        )
        if changed:
            generation = state.goal_generation + 1
            self._states[int(robot_id)] = self._new_state()
            state = self._states[int(robot_id)]
            state.goal_generation = generation
            state.local_goal_world = goal.copy()
        return changed

    def _build_observation(
        self,
        observations: Mapping[str, Any],
        agent_state: Any,
        goal_world: Sequence[float],
        episode_start_position: Sequence[float],
        episode_start_rotation: Any,
    ) -> Dict[str, np.ndarray]:
        output = {}
        position = np.asarray(agent_state.position, dtype=np.float32)
        rotation = agent_state.rotation
        goal = np.asarray(goal_world, dtype=np.float32)
        for field in self.spec.observation_fields:
            if field.kind in {"rgb", "depth"}:
                policy_uuid = "pointnav_{}".format(field.uuid)
                source_uuid = (
                    policy_uuid
                    if policy_uuid in observations
                    else field.uuid
                )
                if source_uuid not in observations:
                    raise KeyError(
                        "PointNav observation is missing {!r} "
                        "(or policy sensor {!r})".format(
                            field.uuid, policy_uuid
                        )
                    )
                value = np.asarray(observations[source_uuid])
            elif field.kind == "integrated_pointgoal":
                value = compute_pointgoal(
                    position,
                    rotation,
                    goal,
                    goal_format=_get(field.config, "goal_format", "POLAR"),
                    dimensionality=int(
                        _get(field.config, "dimensionality", 2)
                    ),
                )
            elif field.kind == "pointgoal":
                value = compute_pointgoal(
                    episode_start_position,
                    episode_start_rotation,
                    goal,
                    goal_format=_get(field.config, "goal_format", "POLAR"),
                    dimensionality=int(
                        _get(field.config, "dimensionality", 2)
                    ),
                )
            elif field.kind == "gps":
                value = compute_episode_gps(
                    position,
                    episode_start_position,
                    episode_start_rotation,
                    dimensionality=int(
                        _get(field.config, "dimensionality", 2)
                    ),
                )
            elif field.kind == "compass":
                value = compute_compass(rotation, episode_start_rotation)
            elif field.kind == "heading":
                value = compute_heading(rotation)
            else:
                raise ValueError(
                    "unsupported PointNav observation kind {!r}".format(
                        field.kind
                    )
                )

            if value.shape != field.shape:
                raise ValueError(
                    "PointNav {} shape {} does not match checkpoint {}".format(
                        field.uuid, value.shape, field.shape
                    )
                )
            if value.dtype != field.dtype:
                raise TypeError(
                    "PointNav {} dtype {} does not match checkpoint {}".format(
                        field.uuid, value.dtype, field.dtype
                    )
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    "PointNav {} contains non-finite values".format(field.uuid)
                )
            if np.any(value < field.low) or np.any(value > field.high):
                raise ValueError(
                    "PointNav {} is outside checkpoint observation bounds"
                    .format(field.uuid)
                )
            output[field.uuid] = value
        return output

    def act(
        self,
        *,
        robot_id: int,
        observations: Mapping[str, Any],
        agent_state: Any,
        goal_world: Sequence[float],
        episode_start_position: Sequence[float],
        episode_start_rotation: Any,
    ) -> PointNavDecision:
        goal_changed = self._set_local_goal(robot_id, goal_world)
        state = self._state(robot_id)
        policy_observation = self._build_observation(
            observations,
            agent_state,
            goal_world,
            episode_start_position,
            episode_start_rotation,
        )
        batch = self._batcher(policy_observation)
        with torch.inference_mode():
            action_data = self.policy.act(
                batch,
                state.hidden,
                state.prev_action,
                state.mask,
                deterministic=self.deterministic,
            )
        policy_action = int(action_data.env_actions.reshape(-1)[0].item())
        if not 0 <= policy_action < len(self.spec.action_names):
            raise ValueError(
                "PointNav policy emitted out-of-range action {}".format(
                    policy_action
                )
            )
        action_name = self.spec.action_names[policy_action]
        state.hidden = action_data.rnn_hidden_states.detach().clone()

        local_stop = action_name == "stop"
        if local_stop:
            # STOP belongs to the local PointNav goal.  Clear this robot's
            # frontier and recurrence; the caller must never pass STOP to the
            # ObjectNav environment.
            self.clear_local_goal(robot_id)
            return PointNavDecision(
                action=None,
                policy_action=policy_action,
                action_name=action_name,
                local_goal_reached=True,
                request_global_replan=True,
                goal_changed=goal_changed,
            )

        state.prev_action.fill_(policy_action)
        state.mask.fill_(True)
        return PointNavDecision(
            action=int(self.env_action_map[action_name]),
            policy_action=policy_action,
            action_name=action_name,
            local_goal_reached=False,
            request_global_replan=False,
            goal_changed=goal_changed,
        )

    def record_executed_action(self, robot_id: int, env_action: int) -> None:
        """Keep recurrent previous-action input equal to the executed action."""
        inverse = {value: key for key, value in self.env_action_map.items()}
        action_name = inverse.get(int(env_action))
        if action_name is None:
            raise ValueError(
                "executed action {} is absent from PointNav schema".format(
                    env_action
                )
            )
        state = self._state(robot_id)
        state.prev_action.fill_(self.spec.action_names.index(action_name))
        state.mask.fill_(True)

    def debug_robot_state(self, robot_id: int) -> Dict[str, Any]:
        state = self._state(robot_id)
        return {
            "hidden": state.hidden.detach().cpu().clone(),
            "prev_action": state.prev_action.detach().cpu().clone(),
            "mask": state.mask.detach().cpu().clone(),
            "local_goal_world": (
                None
                if state.local_goal_world is None
                else state.local_goal_world.copy()
            ),
            "goal_generation": int(state.goal_generation),
        }


def load_pointnav_policy_adapter(
    checkpoint_path: str,
    *,
    config_path: Optional[str] = None,
    device: Any = "cpu",
    deterministic: bool = True,
    goal_tolerance_m: float = 0.05,
    env_action_map: Optional[Mapping[str, int]] = None,
    observation_mode: str = "auto",
) -> PointNavPolicyAdapter:
    spec = load_pointnav_runtime_spec(
        checkpoint_path, config_path, observation_mode
    )
    action_items = tuple(sorted(dict(env_action_map or {}).items()))
    cache_key = (
        str(Path(checkpoint_path).expanduser().resolve()),
        None if config_path is None else str(
            Path(config_path).expanduser().resolve()
        ),
        str(device),
        bool(deterministic),
        float(goal_tolerance_m),
        action_items,
        str(observation_mode),
    )
    if cache_key not in _ADAPTER_CACHE:
        torch_device = torch.device(device)
        policy, batcher = _create_production_policy(
            checkpoint_path, spec, device=torch_device
        )
        _ADAPTER_CACHE[cache_key] = PointNavPolicyAdapter(
            policy,
            spec,
            device=torch_device,
            deterministic=deterministic,
            goal_tolerance_m=goal_tolerance_m,
            env_action_map=env_action_map,
            batcher=batcher,
        )
    return _ADAPTER_CACHE[cache_key]


def shield_pointnav_action(
    action: int,
    *,
    env_action_map: Mapping[str, int],
    current_cell: Sequence[float],
    relative_angle_deg: float,
    hard_unsafe_mask: Optional[np.ndarray],
    risk_map: Optional[np.ndarray],
    map_resolution_cm: float,
    forward_step_size_m: float,
    turn_angle_deg: float,
) -> int:
    """Veto a forward step that crosses a hard hazard.

    This shield deliberately stays outside the neural observation.  It
    therefore provides the risk-enabled baseline with a local safety veto
    without changing any checkpoint tensor or pretending the pretrained
    policy was trained with hazard channels.
    """
    action_map = dict(env_action_map)
    if int(action) != int(action_map["move_forward"]):
        return int(action)
    if hard_unsafe_mask is None:
        return int(action)
    hard = np.asarray(hard_unsafe_mask, dtype=bool)
    if hard.ndim != 2:
        raise ValueError("PointNav hard_unsafe_mask must be 2-D")
    risk = (
        np.zeros(hard.shape, dtype=np.float32)
        if risk_map is None
        else np.asarray(risk_map, dtype=np.float32)
    )
    if risk.shape != hard.shape:
        raise ValueError("PointNav risk map and hard mask shapes differ")

    start = np.asarray(current_cell, dtype=np.float64)
    cells_per_step = max(
        1.0,
        float(forward_step_size_m)
        / max(float(map_resolution_cm) / 100.0, 1e-6),
    )

    def ray(angle_deg: float):
        count = max(1, int(np.ceil(cells_per_step)))
        samples = []
        for distance in np.linspace(1.0, cells_per_step, count):
            row = int(round(
                start[0] + distance * np.sin(np.deg2rad(angle_deg))
            ))
            col = int(round(
                start[1] + distance * np.cos(np.deg2rad(angle_deg))
            ))
            row = int(np.clip(row, 0, hard.shape[0] - 1))
            col = int(np.clip(col, 0, hard.shape[1] - 1))
            samples.append((row, col))
        return samples

    forward = ray(float(relative_angle_deg))
    if not any(hard[cell] for cell in forward):
        return int(action)

    candidates = (
        (
            "turn_left",
            float(relative_angle_deg) - float(turn_angle_deg),
        ),
        (
            "turn_right",
            float(relative_angle_deg) + float(turn_angle_deg),
        ),
    )
    best_name = min(
        candidates,
        key=lambda item: (
            any(hard[cell] for cell in ray(item[1])),
            sum(float(risk[cell]) for cell in ray(item[1])),
            item[0],
        ),
    )[0]
    return int(action_map[best_name])
