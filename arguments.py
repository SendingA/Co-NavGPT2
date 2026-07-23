"""CLI for main.py / main_vec.py under Habitat-Lab 0.3.3.

The flag surface stayed almost identical to the Habitat 0.2.1 version so
users' existing shell scripts keep working; the migration is contained in
:func:`load_config` which now returns an ``omegaconf.DictConfig`` composed
by Hydra rather than the old YACS ``CfgNode``.

Three new groups were added in the 2026-07 migration:

* ``--num_humans`` and the humanoid data paths for the pedestrians spawned
  by :class:`envs.random_humanoid.RandomHumanoidWalker`.
* ``--robot_models_enabled`` / ``--robot_profiles`` for the visible robot
  URDF models loaded by :class:`envs.robot_models.RobotModelManager`.
* Explicit dataset / scene-dataset overrides so the same task_config can
  target Co-NavGPTv2's HM3D v2 install or Co-NavGPTv3's demo assets.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from omegaconf import OmegaConf

from habitat.config.default import get_config as habitat_get_config
from habitat.config.read_write import read_write


PROJECT_ROOT = Path(__file__).resolve().parent


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent-Semantic-Exploration")

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("-d", "--dump_location", type=str, default="./tmp",
                        help="where main.py writes logs / dumps. "
                             "output goes to <dump_location>/logs/<nav_mode>/"
                             " and <dump_location>/dump/<nav_mode>/")
    parser.add_argument("-v", "--visualize", type=int, default=0,
                        help="1: render observations + predicted semantic "
                             "map; opens an Open3D GUI in main.py")
    parser.add_argument("--print_images", type=int, default=0,
                        help="1: persist visualization frames to disk")

    # ------------------------------------------------------------------
    # Camera + scene config
    # ------------------------------------------------------------------
    parser.add_argument("-fw", "--frame_width", type=int, default=640)
    parser.add_argument("-fh", "--frame_height", type=int, default=480)
    parser.add_argument("--task_config", type=str,
                        default="multi_objectnav_hm3d.yaml",
                        help="path to config yaml under configs/. Passed to "
                             "habitat.config.default.get_config as a Hydra "
                             "config file.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional absolute/relative path to a Hydra "
                             "config file, overrides --task_config.")
    parser.add_argument("--hfov", type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="override habitat.dataset.data_path (e.g. "
                             "data/datasets/objectnav_hm3d_v2/{split}/{split}.json.gz)")
    parser.add_argument("--scenes_dir", type=str, default=None,
                        help="override habitat.dataset.scenes_dir")
    parser.add_argument("--scene_dataset", type=str, default=None,
                        help="override habitat.simulator.scene_dataset "
                             "(scene_dataset_config.json)")

    # ------------------------------------------------------------------
    # Multi-agent / parallel run
    # ------------------------------------------------------------------
    parser.add_argument("--num_local_steps", type=int, default=25,
                        help="steps between two global re-plans")
    parser.add_argument("-n", "--num_processes", type=int, default=1,
                        help="only honored by main_vec.py")
    parser.add_argument("--rank", type=int, default=0,
                        help="set automatically by main_vec.py per worker; "
                             "main.py keeps the default 0")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="Habitat-sim GPU device id")
    parser.add_argument("--num_agents", type=int, default=2,
                        help="number of robot agents in the simulator")
    parser.add_argument("--self_exclusion_radius", type=float, default=0.45,
                        help="metres: discard depth points within this XZ "
                             "radius of the camera during mapping so the "
                             "agent never treats its own visible robot URDF "
                             "(or the floor right under itself when looking "
                             "down) as an obstacle. 0 disables the filter.")

    # Habitat 3 humanoid pedestrians
    parser.add_argument("--num_humans", type=int, default=None,
                        help="number of kinematic humanoid pedestrians to "
                             "spawn alongside the robots. If omitted, use "
                             "conav.num_humans from the task config.")

    # Habitat 3 visible robot URDF models (Fetch / Spot / ...) rendered on
    # top of the classic ObjectNav navigation agents.
    parser.add_argument("--robot_models_enabled", type=int, default=0,
                        help="1: load Habitat3 articulated robot URDFs as "
                             "kinematic visual models synchronized to the "
                             "nav agents.")
    parser.add_argument("--robot_profiles", type=str, default=None,
                        help="Comma-separated robot profile names for the "
                             "visible URDFs: fetch, fetch_no_wheels, "
                             "fetch_suction, spot, stretch. Cycles through "
                             "the list if num_agents > len(profiles).")
    parser.add_argument("--robot_urdfs", type=str, default=None,
                        help="Optional comma-separated URDF path overrides; "
                             "kept parallel to --robot_profiles for asset "
                             "swapping without changing the robot class.")

    # ------------------------------------------------------------------
    # Mapping / perception
    # ------------------------------------------------------------------
    parser.add_argument("--map_resolution", type=int, default=5,
                        help="cm per occupancy grid cell")
    parser.add_argument("--map_size_cm", type=int, default=2400,
                        help="occupancy map side length (cm)")
    parser.add_argument("--map_height_cm", type=int, default=130,
                        help="top-down map slice height (cm)")
    parser.add_argument("--sem_threshold", type=float, default=0.85,
                        help="semantic detection confidence above which "
                             "the goal is considered found")
    # ------------------------------------------------------------------
    # Global planner
    # ------------------------------------------------------------------
    parser.add_argument("--nav_mode", type=str, default="gpt",
                        choices=["nearest", "co_ut", "fill", "gpt"],
                        help="global frontier policy. nearest=closest, "
                             "co_ut=cooperative assignment, fill=highest "
                             "frontier score, gpt=GPT-4o decision.")
    parser.add_argument("--fill_mode", type=int, default=0,
                        help="1: when an agent revisits the same frontier, "
                             "mark its area as obstacle and re-detect")
    parser.add_argument("--gpt_type", type=int, default=2,
                        help="1: gpt-3.5-turbo  2: gpt-4o (default)  "
                             "3: gpt-4o-mini  (only used when nav_mode=gpt)")

    # ------------------------------------------------------------------
    # Fire-scene observation suite (voxel RGB + Thermal + noisy depth +
    # radar / lidar). The suite is constructed automatically when
    # --fire_world=1.
    # ------------------------------------------------------------------
    parser.add_argument("--fire_apply_to_obs", type=int, default=1)
    parser.add_argument("--smoke_density", type=float, default=0.6)
    parser.add_argument("--fire_dump_dir", type=str,
                        default="./outputs/fire_sensors")
    parser.add_argument("--fire_save_every", type=int, default=1)
    parser.add_argument("--fire_save_npz", type=int, default=0)
    parser.add_argument("--fire_show_window", type=int, default=0)
    parser.add_argument("--lidar_360", type=int, default=0)
    parser.add_argument("--lidar_resolution", type=int, default=320)

    # Smoke-scene perception switches
    parser.add_argument("--depth_use_clean", type=int, default=-1,
                        help="Which depth is written back into observations "
                             "FOR MAPPING. -1 (default) = auto: use clean "
                             "depth whenever a FireWorld scene is active, "
                             "because the smoke-degraded depth clips distant "
                             "walls/floors to the smoke layer (Jin "
                             "visibility) and the mapper would then bake the "
                             "whole flame region as an obstacle wall. Set 1 "
                             "to force clean depth, 0 to force the "
                             "smoke-degraded depth (only if you WANT smoke to "
                             "occlude the map). NOTE: this flag no longer "
                             "affects the RGB/thermal fire render or the "
                             "dashboard — those always ray-march against the "
                             "pristine geometric depth, so flame/smoke look "
                             "identical for 0 and 1.")
    parser.add_argument("--use_thermal_perception", type=int, default=1)

    # FireWorld runtime
    parser.add_argument("--fire_world", type=int, default=0)
    parser.add_argument("--fire_world_plan_id", type=str, default=None)
    parser.add_argument("--fire_world_scenes_root", type=str, default="scenes")
    parser.add_argument("--fire_world_out_root", type=str,
                        default="outputs/fire_world")
    parser.add_argument("--fire_clock_mode", type=str, default="wallclock",
                        choices=["wallclock", "step"])
    parser.add_argument("--fire_speedup", type=float, default=1.0)
    parser.add_argument("--fire_steps_per_unit", type=int, default=5)
    parser.add_argument("--fire_seconds_per_unit", type=float, default=2.0)
    parser.add_argument("--fire_world_smoke_k_ext", type=float, default=4.0)
    parser.add_argument("--fire_world_n_steps", type=int, default=24)
    parser.add_argument("--fire_world_render_scale", type=float, default=0.5)
    parser.add_argument("--fire_fast", type=int, default=1,
                        help="1 (default): navigation-speed fire rendering "
                             "- disables the procedural flame flicker/wisp "
                             "noise and lowers ray-march steps + render "
                             "scale for a ~5-15x speedup. Set 0 for the "
                             "pretty teleop-demo look (much slower).")
    parser.add_argument("--fire_flame_noise", type=float, default=None,
                        help="Override flame procedural-noise strength "
                             "[0..1.5]. None = decided by --fire_fast "
                             "(0 when fast, 0.55 otherwise).")

    # ------------------------------------------------------------------
    # Dynamic risk assessment.  Thresholds below define a normalized
    # simulator hazard index for navigation experiments; they are not a
    # physiological survival model.  ``risk_enabled=0`` is intentionally the
    # default so existing ObjectNav runs keep their original behavior.
    # ------------------------------------------------------------------
    parser.add_argument("--risk_enabled", type=int, default=0,
                        help="1: enable dynamic fire/smoke/temperature risk "
                             "assessment and hazard-aware planning")
    parser.add_argument("--risk_source", type=str, default="sensed",
                        choices=["none", "oracle", "sensed"],
                        help="planner risk source. sensed is the benchmark "
                             "setting; oracle is a privileged upper bound")
    parser.add_argument("--risk_weight_temperature", type=float, default=0.60)
    parser.add_argument("--risk_weight_smoke", type=float, default=0.40)
    parser.add_argument("--risk_temperature_ambient_c", type=float,
                        default=25.0,
                        help="ambient temperature retained in physical "
                             "sensor/evaluator outputs (deg C)")
    parser.add_argument("--risk_temperature_reference_c", type=float,
                        default=35.0,
                        help="temperature at which normalized heat risk "
                             "starts to rise (deg C)")
    parser.add_argument("--risk_temperature_hazard_c", type=float,
                        default=150.0,
                        help="temperature mapped to normalized heat risk 1 "
                             "(deg C; benchmark calibration, not a medical "
                             "survival threshold)")
    parser.add_argument("--risk_temperature_hard_c", type=float,
                        default=250.0,
                        help="temperature that makes a planner cell hard "
                             "unsafe (deg C)")
    parser.add_argument("--risk_flame_hard_threshold", type=float,
                        default=0.20)
    parser.add_argument("--risk_flame_safety_distance_m", type=float,
                        default=0.45,
                        help="hard-unsafe dilation around flame cells (m)")
    parser.add_argument("--risk_danger_threshold", type=float, default=0.55)
    parser.add_argument("--risk_critical_threshold", type=float, default=0.80)
    parser.add_argument("--risk_decay_tau_s", type=float, default=20.0,
                        help="time constant for stale sensed hazard evidence")
    parser.add_argument("--risk_confidence_decay_tau_s", type=float,
                        default=30.0)
    parser.add_argument("--risk_unknown_risk_prior", type=float, default=0.25,
                        help="planner cost assigned to unobserved cells")
    parser.add_argument("--risk_uncertainty_weight", type=float, default=0.25)
    parser.add_argument("--risk_sensor_stride", type=int, default=4,
                        help="pixel stride for sensed risk back-projection")
    parser.add_argument("--risk_floor_min_offset_m", type=float, default=0.0)
    parser.add_argument("--risk_floor_max_offset_m", type=float, default=1.50)
    parser.add_argument("--risk_smoke_source", type=str,
                        default="appearance_depth",
                        choices=["appearance_depth", "privileged_transmittance"],
                        help="smoke evidence source; transmittance is an "
                             "explicit privileged ablation")
    parser.add_argument("--risk_geometry_depth_source", type=str,
                        default="clean",
                        choices=["clean", "smoke"],
                        help="depth used to place thermal/smoke evidence: "
                             "clean is the smoke-robust radar/depth geometry "
                             "surrogate used by main; smoke is the degraded "
                             "vision-depth ablation")
    parser.add_argument("--risk_alpha", type=float, default=4.0,
                        help="hazard strength in FMM speed=1/(1+alpha*risk)")
    parser.add_argument("--risk_frontier_weight", type=float, default=2.0)
    parser.add_argument("--risk_hard_frontier_threshold", type=float,
                        default=0.80)
    parser.add_argument("--risk_dump_dir", type=str,
                        default="./outputs/risk_assessment")
    parser.add_argument("--risk_save_every", type=int, default=10,
                        help="save one risk-map snapshot every N navigation "
                             "steps; 0 disables step images")
    parser.add_argument("--risk_max_floor_deviation_m", type=float,
                        default=0.75,
                        help="fail fast if any robot leaves the current-floor "
                             "risk-map band; multi-floor risk maps are not "
                             "silently collapsed")
    parser.add_argument("--risk_run_id", type=str, default="default",
                        help="subdirectory used to isolate risk artifacts")
    parser.add_argument("--risk_rank", type=int, default=0,
                        help="artifact rank id for parallel launchers")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    # ``args.turn_angle`` was populated from the YACS config in the H2
    # code path. Migration keeps it as a Python attribute so downstream
    # planners (VLM_Agent, ffm_act, ShortestPathFollowerCompat) don't
    # need to reach into the DictConfig.  Populated by :func:`load_config`.
    args.turn_angle = 30

    return args


# ---------------------------------------------------------------------------
# Hydra config loading
# ---------------------------------------------------------------------------
def load_config(args: argparse.Namespace):
    """Compose the Habitat 3.3 Hydra config and apply CLI overrides.

    * Reads ``configs/<task_config>`` (or ``--config`` if given).
    * Injects ``habitat.simulator.habitat_sim_v0.gpu_device_id`` from
      ``args.gpu_id``.
    * Replicates the first agent template into ``args.num_agents``
      entries under ``habitat.simulator.agents`` and rewrites
      ``agents_order`` to match. The default agent stays ``main_agent``.
    * Adds a ``conav`` DictConfig group holding humanoid and robot-model
      knobs so envs/random_humanoid.py + envs/robot_models.py can pull
      everything from a single object.
    """
    overrides: List[str] = [
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={args.gpu_id}",
    ]
    if args.dataset_path is not None:
        overrides.append(
            f"habitat.dataset.data_path={_project_path(args.dataset_path)}"
        )
    if args.scenes_dir is not None:
        overrides.append(
            f"habitat.dataset.scenes_dir={_project_path(args.scenes_dir)}"
        )
    if args.scene_dataset is not None:
        overrides.append(
            f"habitat.simulator.scene_dataset={_project_path(args.scene_dataset)}"
        )

    config_path = args.config
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / args.task_config)
    config_path = str(Path(config_path).expanduser())

    config = habitat_get_config(config_path, overrides=overrides)

    with read_write(config):
        config.habitat.seed = args.seed
        _apply_camera_geometry(config, args)
        _set_num_robot_agents(config, args.num_agents)

        _apply_conav_overrides(config, args)
        _resolve_conav_paths(config)
        _resolve_dataset_paths(config)

    # Propagate turn angle back to the CLI namespace so the planners can
    # read it without touching DictConfig again.
    args.turn_angle = float(config.habitat.simulator.turn_angle)
    return config


def voxel_smoke_kwargs(args) -> dict:
    """Return kwargs for utils.fire_sensors.config.VoxelSmokeConfig that
    honour --fire_fast / --fire_flame_noise.

    Fast mode (default) trades the procedural flame flicker/wisp noise
    and high ray-march resolution for a large speedup — the noise is
    pure eye-candy that does nothing for navigation. Pretty mode
    (``--fire_fast 0``) restores the teleop-demo defaults.
    """
    fast = bool(int(getattr(args, "fire_fast", 1)))

    if fast:
        n_steps = min(int(args.fire_world_n_steps), 10)
        render_scale = min(float(args.fire_world_render_scale), 0.35)
        default_noise = 0.0
    else:
        n_steps = int(args.fire_world_n_steps)
        render_scale = float(args.fire_world_render_scale)
        default_noise = 0.55

    noise = getattr(args, "fire_flame_noise", None)
    noise = default_noise if noise is None else float(noise)
    # When noise is off, kill all three noise terms; when on, use the
    # canonical teleop ratios (edge_break/color_jitter/smoke scale with it).
    if noise <= 0.0:
        flame_noise = edge_break = color_jitter = smoke_noise = 0.0
    else:
        flame_noise = noise
        edge_break = 0.8 * (noise / 0.55)
        color_jitter = 0.25 * (noise / 0.55)
        smoke_noise = 0.30 * (noise / 0.55)

    return {
        "n_steps": n_steps,
        "smoke_k_ext": float(args.fire_world_smoke_k_ext),
        "render_scale": render_scale,
        "thermal_color_blend": 0.85,
        "flame_noise_strength": flame_noise,
        "flame_edge_break": edge_break,
        "flame_color_jitter": color_jitter,
        "smoke_noise_strength": smoke_noise,
    }


def humanoid_kwargs(config, seed: int) -> dict:
    """Bundle up kwargs for :class:`envs.random_humanoid.RandomHumanoidWalker`."""
    return {
        "num_humans": int(config.conav.num_humans),
        "urdf_path": (
            config.conav.human_urdfs
            if "human_urdfs" in config.conav
            else config.conav.human_urdf
        ),
        "motion_data_path": (
            config.conav.human_motion_data_paths
            if "human_motion_data_paths" in config.conav
            else config.conav.human_motion_data
        ),
        "seed": seed,
        "walk_speed": float(config.conav.human_walk_speed),
        "turn_speed": float(config.conav.human_turn_speed),
        "goal_radius": float(config.conav.human_goal_radius),
        "target_radius": float(config.conav.human_target_radius),
        "min_spawn_distance": float(config.conav.min_spawn_distance),
        "motion_dt": float(config.conav.human_motion_dt),
        "use_controller_root_motion": bool(
            config.conav.human_use_controller_root_motion
        ),
    }


def robot_model_kwargs(config, num_agents: int) -> dict:
    """Bundle up kwargs for :class:`envs.robot_models.RobotModelManager`."""
    robot_model_urdfs = (
        config.conav.robot_model_urdfs
        if "robot_model_urdfs" in config.conav
        and len(config.conav.robot_model_urdfs) > 0
        else None
    )
    return {
        "num_robots": num_agents,
        "profiles": (
            config.conav.robot_model_profiles
            if "robot_model_profiles" in config.conav
            else "fetch"
        ),
        "urdf_paths": robot_model_urdfs,
        "enabled": bool(config.conav.get("robot_models_enabled", False)),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _apply_camera_geometry(config, args: argparse.Namespace) -> None:
    sim_cfg = config.habitat.simulator
    hfov = int(round(float(args.hfov)))  # H3.3 typing: hfov is int
    for agent_cfg in sim_cfg.agents.values():
        for sensor_cfg in agent_cfg.sim_sensors.values():
            if hasattr(sensor_cfg, "width"):
                sensor_cfg.width = int(args.frame_width)
            if hasattr(sensor_cfg, "height"):
                sensor_cfg.height = int(args.frame_height)
            if hasattr(sensor_cfg, "hfov"):
                sensor_cfg.hfov = hfov


def _set_num_robot_agents(config, num_robots: int) -> None:
    if num_robots <= 0:
        raise ValueError("num_agents must be positive.")

    sim_cfg = config.habitat.simulator
    template_name = sim_cfg.agents_order[0]
    template = OmegaConf.to_container(sim_cfg.agents[template_name], resolve=True)

    sim_cfg.agents.clear()
    sim_cfg.agents_order.clear()
    for idx in range(num_robots):
        name = "main_agent" if idx == 0 else f"agent_{idx}"
        sim_cfg.agents[name] = OmegaConf.create(template)
        sim_cfg.agents_order.append(name)
    sim_cfg.default_agent_id = 0


def _apply_conav_overrides(config, args: argparse.Namespace) -> None:
    """CLI values win over conav.* defaults from the yaml."""
    conav = config.conav
    conav.num_robots = int(args.num_agents)
    if args.num_humans is not None:
        conav.num_humans = int(args.num_humans)

    if args.robot_models_enabled:
        conav.robot_models_enabled = bool(args.robot_models_enabled)
    if args.robot_profiles is not None:
        conav.robot_model_profiles = _csv_items(args.robot_profiles)
    if args.robot_urdfs is not None:
        conav.robot_model_urdfs = _csv_items(args.robot_urdfs)


def _resolve_conav_paths(config) -> None:
    conav = config.conav
    if "human_urdfs" in conav:
        conav.human_urdfs = _project_paths(conav.human_urdfs)
    elif "human_urdf" in conav:
        conav.human_urdf = _project_path(conav.human_urdf)

    if "human_motion_data_paths" in conav:
        conav.human_motion_data_paths = _project_paths(
            conav.human_motion_data_paths
        )
    elif "human_motion_data" in conav:
        conav.human_motion_data = _project_path(conav.human_motion_data)

    if (
        "robot_model_urdfs" in conav
        and len(conav.robot_model_urdfs) > 0
    ):
        conav.robot_model_urdfs = _project_paths(conav.robot_model_urdfs)


def _resolve_dataset_paths(config) -> None:
    config.habitat.dataset.data_path = _project_path(
        config.habitat.dataset.data_path
    )
    config.habitat.dataset.scenes_dir = _project_path(
        config.habitat.dataset.scenes_dir
    )
    if "scene_dataset" in config.habitat.simulator:
        config.habitat.simulator.scene_dataset = _project_path(
            config.habitat.simulator.scene_dataset
        )


def _project_path(path_value) -> str:
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _project_paths(value):
    if isinstance(value, str):
        return _project_path(value)
    return [_project_path(str(item)) for item in value]


def _csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
