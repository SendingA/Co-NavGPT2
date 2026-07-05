#!/usr/bin/env python3
"""Keyboard teleoperation for Habitat-Lab 0.3.3 agents.

Usage:
    python scripts/keyboard_teleop.py \
        --task-config configs/multi_objectnav_hm3d.yaml \
        --num-agents 1 --agent-id 0 --scene-id Nfvxx8J5NCo --show-depth 1

Controls:
    W  move forward
    A  turn left
    D  turn right
    S  stop
    Q  look down
    E  look up
    ESC quit
"""
from __future__ import annotations

import argparse
import sys
import traceback

import cv2
import numpy as np

import habitat
from habitat import Env

# Reuse the main project's Hydra config loader so all flags stay
# consistent with main.py.
sys.path.insert(0, "/home/liushe10/Co-NavGPT2")
from arguments import get_args, load_config  # noqa: E402  after sys.path fix


def _override_num_agents(config, num_agents: int) -> None:
    """Rewrite ``habitat.simulator.agents_order`` for a smaller crew.

    ``load_config`` replicates the main template into ``args.num_agents``
    agents; this teleop overrides it further if the caller wants fewer
    (or more) agents than the default config's ``num_robots``.
    """
    from omegaconf import OmegaConf

    with habitat.config.read_write(config):
        sim_cfg = config.habitat.simulator
        template_name = sim_cfg.agents_order[0]
        template = OmegaConf.to_container(
            sim_cfg.agents[template_name], resolve=True
        )
        sim_cfg.agents.clear()
        sim_cfg.agents_order.clear()
        for idx in range(num_agents):
            name = "main_agent" if idx == 0 else f"agent_{idx}"
            sim_cfg.agents[name] = OmegaConf.create(template)
            sim_cfg.agents_order.append(name)
        sim_cfg.default_agent_id = 0
        config.conav.num_robots = int(num_agents)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for Habitat 0.3.3 agents"
    )
    parser.add_argument("--task-config", required=True,
                        help="Hydra yaml under configs/")
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--agent-id", type=int, default=0)
    parser.add_argument("--scene-id", type=str, default=None,
                        help="filter episodes by scene name substring")
    parser.add_argument("--show-depth", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    # arguments.load_config takes a full Namespace, so build a
    # compatible one out of the teleop flags.
    fake = get_args()
    fake.task_config = args.task_config.replace("configs/", "")
    fake.config = args.task_config if "/" in args.task_config else None
    fake.num_agents = args.num_agents
    fake.num_humans = 0
    fake.gpu_id = args.gpu_id
    fake.robot_models_enabled = 0
    fake.robot_profiles = None
    fake.robot_urdfs = None
    fake.dataset_path = None
    fake.scenes_dir = None
    fake.scene_dataset = None
    config = load_config(fake)

    if args.num_agents != int(config.conav.num_robots):
        _override_num_agents(config, args.num_agents)

    # Discrete action map — H3.3 uses lowercase singleton names.
    action_dict = {"stop": 0, "move_forward": 1, "turn_left": 2,
                   "turn_right": 3, "look_up": 4, "look_down": 5}
    default_stop = 0

    env = Env(config=config)

    if args.scene_id is not None:
        matching = [ep for ep in env.episodes if args.scene_id in ep.scene_id]
        if not matching:
            print(f"No episodes found for scene {args.scene_id}")
            return
        env.current_episode = matching[0]
        print(f"Selected episode: {env.current_episode.episode_id} "
              f"in scene {args.scene_id}")

    observations = env.reset()
    if not isinstance(observations, list):
        observations = [observations]

    window_name = "Habitat Teleop (RGB+Depth)" if args.show_depth else "Habitat Teleop (RGB)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Controls: W/A/S/D move+turn, Q/E look, ESC quit")

    key_map = {
        ord("w"): "move_forward",
        ord("a"): "turn_left",
        ord("d"): "turn_right",
        ord("s"): "stop",
        ord("q"): "look_down",
        ord("e"): "look_up",
    }

    try:
        while True:
            obs = observations[args.agent_id]
            rgb = obs["rgb"][:, :, [2, 1, 0]]

            if args.show_depth and "depth" in obs:
                depth = obs["depth"]
                depth_map = depth[:, :, 0] if depth.ndim == 3 else np.squeeze(depth)
                depth_norm = np.clip((depth_map / 5.0) * 255.0, 0, 255).astype(np.uint8)
                depth_col = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                if depth_col.shape[:2] != rgb.shape[:2]:
                    depth_col = cv2.resize(
                        depth_col,
                        (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                cv2.imshow(window_name, np.hstack((rgb, depth_col)))
            else:
                cv2.imshow(window_name, rgb)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break

            action_name = key_map.get(key)
            if action_name is None:
                continue

            action_idx = action_dict.get(action_name)
            if action_idx is None:
                print(f"Action '{action_name}' not available")
                continue

            actions = [default_stop for _ in range(args.num_agents)]
            actions[args.agent_id] = action_idx

            try:
                observations = env.step(actions)
                if not isinstance(observations, list):
                    observations = [observations]
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                print(f"Error executing step with actions={actions}: {exc}")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
