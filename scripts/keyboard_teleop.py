#!/usr/bin/env python3
"""Keyboard teleoperation for Habitat env (W/A/S/D) with live visualization.

Controls:
  W: MOVE_FORWARD
  A: TURN_LEFT
  D: TURN_RIGHT
  S: STOP (no backward action in default ObjectNav action space)
  Q: LOOK_UP    (if enabled in action space)
  E: LOOK_DOWN  (if enabled in action space)
  ESC: quit
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# Prefer local habitat fork in this repo.
ROOT = Path(__file__).resolve().parents[1]
LOCAL_HABITAT = ROOT / "multi-robot-setting"
if LOCAL_HABITAT.exists() and str(LOCAL_HABITAT) not in sys.path:
    sys.path.insert(0, str(LOCAL_HABITAT))

from habitat import Env
from habitat.config.default import get_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WASD teleop for Co-Nav Habitat env")
    parser.add_argument("--task-config", type=str, default="configs/multi_objectnav_hm3d.yaml")
    parser.add_argument("--num-agents", type=int, default=1, help="Total agents in env")
    parser.add_argument("--agent-id", type=int, default=0, help="Which agent to control")
    parser.add_argument("--episode-idx", type=int, default=0, help="Dataset episode index (0-based)")
    parser.add_argument(
        "--scene-id",
        type=str,
        default="",
        help="Optional scene filter (e.g. Nfvxx8J5NCo). If set, first matching episode is used.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--show-depth", type=int, default=1, help="1 to visualize depth next to RGB")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means unlimited")
    return parser.parse_args()


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb[:, :, ::-1]


def _depth_to_vis(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth.squeeze(-1)
    depth = depth.astype(np.float32)
    valid = depth > 0
    if np.any(valid):
        dmin, dmax = np.percentile(depth[valid], 2), np.percentile(depth[valid], 98)
        if dmax <= dmin:
            dmax = dmin + 1e-3
        depth_n = np.clip((depth - dmin) / (dmax - dmin), 0, 1)
    else:
        depth_n = np.zeros_like(depth, dtype=np.float32)
    depth_u8 = (depth_n * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def _select_obs(observations, agent_id: int):
    if isinstance(observations, list):
        return observations[agent_id]
    return observations


def _render(observations, agent_id: int, step: int, action_name: str, show_depth: bool) -> np.ndarray:
    obs = _select_obs(observations, agent_id)
    rgb = _to_bgr(obs["rgb"])
    panel = rgb

    if show_depth and "depth" in obs:
        depth_vis = _depth_to_vis(obs["depth"])
        if depth_vis.shape[:2] != rgb.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]))
        panel = np.hstack([rgb, depth_vis])

    info = f"step={step} action={action_name} | W/A/S/D control, Q/E look, ESC quit"
    cv2.putText(panel, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()

    config = get_config(config_paths=[args.task_config])
    config.defrost()
    config.SIMULATOR.NUM_AGENTS = max(1, int(args.num_agents))
    config.SIMULATOR.AGENTS = [f"AGENT_{i}" for i in range(config.SIMULATOR.NUM_AGENTS)]
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    env = Env(config=config)

    if args.agent_id < 0 or args.agent_id >= config.SIMULATOR.NUM_AGENTS:
        raise ValueError(f"agent-id must be in [0, {config.SIMULATOR.NUM_AGENTS - 1}]")

    if args.scene_id:
        matches = [
            i for i, ep in enumerate(env.episodes)
            if args.scene_id in ep.scene_id
        ]
        if not matches:
            raise ValueError(f"No episode found for scene-id contains: {args.scene_id}")
        selected_idx = matches[0]
    else:
        if args.episode_idx < 0 or args.episode_idx >= len(env.episodes):
            raise ValueError(f"episode-idx out of range: 0..{len(env.episodes)-1}")
        selected_idx = args.episode_idx

    # Pin exact episode.
    env.current_episode = env.episodes[selected_idx]
    env._episode_from_iter_on_reset = False

    observations = env.reset()

    action_names: List[str] = list(config.TASK.POSSIBLE_ACTIONS)
    action_to_idx: Dict[str, int] = {name: i for i, name in enumerate(action_names)}

    if "STOP" not in action_to_idx:
        raise RuntimeError(f"Action space has no STOP. Actions: {action_names}")

    key_to_action: Dict[int, str] = {
        ord("w"): "MOVE_FORWARD",
        ord("a"): "TURN_LEFT",
        ord("d"): "TURN_RIGHT",
        ord("s"): "STOP",
        ord("q"): "LOOK_UP",
        ord("e"): "LOOK_DOWN",
    }

    print("\nTeleop started")
    print(f"episode_idx={selected_idx}, scene={env.current_episode.scene_id}")
    print("controls: W forward, A left, D right, S stop, Q/E look, ESC quit")

    step = 0
    last_action = "STOP"

    try:
        while not env.episode_over:
            frame = _render(observations, args.agent_id, step, last_action, bool(args.show_depth))
            cv2.imshow("Co-Nav Keyboard Teleop", frame)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

            if key not in key_to_action:
                continue

            action_name = key_to_action[key]
            if action_name not in action_to_idx:
                print(f"[skip] action '{action_name}' not in this config")
                continue

            action_idx = action_to_idx[action_name]
            actions = [action_to_idx["STOP"] for _ in range(config.SIMULATOR.NUM_AGENTS)]
            actions[args.agent_id] = action_idx

            observations = env.step(actions)
            step += 1
            last_action = action_name

            if args.max_steps > 0 and step >= args.max_steps:
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, exiting teleop.")

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print(f"Finished. steps={step}, episode_over={env.episode_over}")


if __name__ == "__main__":
    main()
