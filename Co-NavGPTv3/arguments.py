"""Argument parsing and Habitat config preparation for Co-NavGPTv3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

from habitat.config.default import get_config as habitat_get_config
from habitat.config.read_write import read_write


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Habitat3 HM3D ObjectNav with synchronized robots and random humanoids."
    )
    parser.add_argument(
        "--task_config",
        default="objectnav_hm3d_multi_humans.yaml",
        help="Config yaml name under Co-NavGPTv3/configs.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional full path to the Habitat3 demo config.",
    )
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--num-humans", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--robot-models",
        dest="robot_models_enabled",
        action="store_true",
        help="Enable visible articulated robot URDF models.",
    )
    parser.add_argument(
        "--no-robot-models",
        dest="robot_models_enabled",
        action="store_false",
        help="Disable visible articulated robot URDF models.",
    )
    parser.set_defaults(robot_models_enabled=None)
    parser.add_argument(
        "--robot-profiles",
        default=None,
        help="Comma-separated visual robot profiles, e.g. fetch,fetch_no_wheels.",
    )
    parser.add_argument(
        "--robot-urdfs",
        default=None,
        help="Comma-separated custom URDF paths. Profiles still choose robot classes.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Override ObjectNav dataset path, e.g. data/datasets/objectnav_hm3d_v2/{split}/{split}.json.gz.",
    )
    parser.add_argument(
        "--scenes-dir",
        default=None,
        help="Override HM3D scenes directory.",
    )
    parser.add_argument(
        "--scene-dataset",
        default=None,
        help="Override Habitat-Sim scene_dataset config.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    overrides = [f"habitat.simulator.habitat_sim_v0.gpu_device_id={args.gpu_id}"]
    if args.dataset_path is not None:
        overrides.append(f"habitat.dataset.data_path={_project_path(args.dataset_path)}")
    if args.scenes_dir is not None:
        overrides.append(f"habitat.dataset.scenes_dir={_project_path(args.scenes_dir)}")
    if args.scene_dataset is not None:
        overrides.append(
            f"habitat.simulator.scene_dataset={_project_path(args.scene_dataset)}"
        )

    config_path = args.config
    if config_path is None:
        config_path = str(ROOT / "configs" / args.task_config)

    config = habitat_get_config(config_path, overrides=overrides)
    with read_write(config):
        config.habitat.seed = args.seed
        config.habitat.environment.max_episode_steps = args.max_steps

        if args.num_robots is not None:
            config.conav.num_robots = args.num_robots
        if args.num_humans is not None:
            config.conav.num_humans = args.num_humans
        if args.robot_models_enabled is not None:
            config.conav.robot_models_enabled = args.robot_models_enabled
        if args.robot_profiles is not None:
            config.conav.robot_model_profiles = _csv_items(args.robot_profiles)
        if args.robot_urdfs is not None:
            config.conav.robot_model_urdfs = _csv_items(args.robot_urdfs)

        _set_num_robot_agents(config, int(config.conav.num_robots))
        config.habitat.dataset.data_path = _project_path(
            config.habitat.dataset.data_path
        )
        config.habitat.dataset.scenes_dir = _project_path(
            config.habitat.dataset.scenes_dir
        )
        config.habitat.simulator.scene_dataset = _project_path(
            config.habitat.simulator.scene_dataset
        )

        if "human_urdfs" in config.conav:
            config.conav.human_urdfs = _project_paths(config.conav.human_urdfs)
        else:
            config.conav.human_urdf = _project_path(config.conav.human_urdf)

        if "human_motion_data_paths" in config.conav:
            config.conav.human_motion_data_paths = _project_paths(
                config.conav.human_motion_data_paths
            )
        else:
            config.conav.human_motion_data = _project_path(
                config.conav.human_motion_data
            )

        if (
            "robot_model_urdfs" in config.conav
            and len(config.conav.robot_model_urdfs) > 0
        ):
            config.conav.robot_model_urdfs = _project_paths(
                config.conav.robot_model_urdfs
            )

    return config


def humanoid_kwargs(config, seed: int) -> dict:
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
        "enabled": (
            bool(config.conav.robot_models_enabled)
            if "robot_models_enabled" in config.conav
            else False
        ),
    }


def _project_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def _project_paths(value):
    if isinstance(value, str):
        return _project_path(value)
    return [_project_path(str(item)) for item in value]


def _csv_items(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _set_num_robot_agents(config, num_robots: int) -> None:
    if num_robots <= 0:
        raise ValueError("num_robots must be positive.")

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
