#!/usr/bin/env python3
"""Keyboard teleoperation with FireWorld runtime overlay.

Drives one agent through a Habitat scene while a precomputed FireWorld
timeline is rendered onto the agent's RGB / Thermal observations. The
fire advances with the **robot step counter** (not wall clock), so you
can watch the scenario evolve as you walk through it.

Example::

    python scripts/keyboard_teleop_fire.py \\
        --task-config configs/multi_objectnav_hm3d.yaml \\
        --scene-id Nfvxx8J5NCo \\
        --plan-id 83679a07b632 \\
        --steps-per-unit 5 --seconds-per-unit 2.0 \\
        --depth_use_clean 1

Controls (window must be focused for keys to register):
  W: forward   A: turn left   D: turn right
  Q: look down E: look up     S: stop (no movement, fire still ticks)
  R: reset to t_sim=0          ESC: quit
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from habitat import Env  # noqa: E402
from habitat.config.default import get_config  # noqa: E402

from utils.fire_world.controller import FireWorldController  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-config", type=str, required=True)
    p.add_argument("--num-agents", type=int, default=1)
    p.add_argument("--agent-id", type=int, default=0)
    p.add_argument("--scene-id", type=str, required=True,
                   help="HM3D short id, e.g. Nfvxx8J5NCo. The script "
                        "filters the episode dataset to one matching scene.")
    p.add_argument("--plan-id", type=str, required=True,
                   help="FireWorld plan id (12-hex). The propagation "
                        "outputs/fire_world/<scene>/<plan_id>/timeline.npz "
                        "must already exist.")
    p.add_argument("--scenes-root", default="scenes")
    p.add_argument("--out-root", default="outputs/fire_world")
    p.add_argument("--steps-per-unit", type=int, default=5)
    p.add_argument("--seconds-per-unit", type=float, default=2.0)
    p.add_argument("--smoke-k-ext", type=float, default=4.0)
    p.add_argument("--n-steps", type=int, default=24)
    p.add_argument("--depth_use_clean", type=int, default=1,
                   help="1: keep Habitat's clean depth (recommended). "
                        "0: keep whatever the simulator produced.")
    p.add_argument("--save-frames-to", type=str, default=None,
                   help="Optional directory to dump per-step PNGs.")
    return p.parse_args()


def colorize_depth(d: np.ndarray, max_d: float = 5.0) -> np.ndarray:
    """Habitat depth (HxWx1 in [0,1] or HxW in metres) -> BGR colormap."""
    if d.ndim == 3:
        d = d[..., 0]
    if d.dtype != np.float32:
        d = d.astype(np.float32)
    if d.max() <= 1.0 + 1e-3:
        d = d * float(max_d)
    norm = np.clip(d / float(max_d) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def overlay_label(img: np.ndarray, lines: list) -> None:
    """Stack short label lines on top of the image (small black bar)."""
    bar_h = 22 * len(lines) + 6
    cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (24, 24, 24), -1)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (8, 18 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    # ---------- config ----------
    config = get_config(config_paths=[args.task_config])
    config.defrost()
    config.SIMULATOR.NUM_AGENTS = args.num_agents
    config.SIMULATOR.AGENTS = [f"AGENT_{i}" for i in range(args.num_agents)]
    config.freeze()
    action_list = list(config.TASK.POSSIBLE_ACTIONS)
    action_dict = {name: idx for idx, name in enumerate(action_list)}
    default_stop = action_dict.get("STOP", 0)

    # ---------- env + episode pick ----------
    env = Env(config=config)
    matching = [ep for ep in env.episodes if args.scene_id in ep.scene_id]
    if not matching:
        print(f"[teleop] no episodes for scene_id={args.scene_id}")
        return
    env.current_episode = matching[0]
    print(f"[teleop] scene={args.scene_id} ep={env.current_episode.episode_id}")

    # ---------- FireWorld controller ----------
    fw_args = SimpleNamespace(
        fire_world=1,
        fire_world_plan_id=args.plan_id,
        fire_world_scenes_root=args.scenes_root,
        fire_world_out_root=args.out_root,
        fire_steps_per_unit=args.steps_per_unit,
        fire_seconds_per_unit=args.seconds_per_unit,
        fire_world_smoke_k_ext=args.smoke_k_ext,
        fire_world_n_steps=args.n_steps,
        frame_width=config.SIMULATOR.RGB_SENSOR.WIDTH,
        frame_height=config.SIMULATOR.RGB_SENSOR.HEIGHT,
        hfov=config.SIMULATOR.RGB_SENSOR.HFOV,
    )
    ctrl = FireWorldController.from_args(fw_args, config)
    print(f"[teleop] {ctrl.describe()}")

    # ---------- driving loop ----------
    obs = env.reset()
    window = "FireWorld Teleop"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 480)

    out_dir = Path(args.save_frames_to) if args.save_frames_to else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Local step counter; the real env doesn't expose one.
    robot_step = 0
    base_t0 = float(ctrl.fw.times[0])
    max_d = float(config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH)
    normalize = bool(getattr(config.SIMULATOR.DEPTH_SENSOR, "NORMALIZE_DEPTH", True))

    key_map = {
        ord("w"): "MOVE_FORWARD",
        ord("a"): "TURN_LEFT",
        ord("d"): "TURN_RIGHT",
        ord("s"): "STOP",
        ord("q"): "LOOK_DOWN",
        ord("e"): "LOOK_UP",
    }

    print("Controls:  W/A/D move/turn  S stop  Q/E look down/up  R reset  ESC quit")

    try:
        while True:
            agent_state = env.sim.get_agent_state(args.agent_id)
            sensors = ctrl.render_for_agent(
                obs[args.agent_id], agent_state,
                robot_step=robot_step,
                max_depth_m=max_d,
                normalize_depth=normalize,
            )
            rgb_smoky = sensors["rgb_smoke"]      # uint8 RGB
            therm_bgr = sensors["thermal_image"]  # uint8 BGR
            depth_panel = colorize_depth(
                obs[args.agent_id]["depth"], max_d=max_d
            )

            t_sim = sensors["t_sim_s"]
            T_mean = float(np.mean(sensors["transmittance"]))
            flame_frac = float(np.mean(sensors["thermal_flame_mask"]))

            rgb_bgr = cv2.cvtColor(rgb_smoky, cv2.COLOR_RGB2BGR)
            label_lines = [
                f"step={robot_step:>4d}  t_sim={t_sim:>6.1f}s  "
                f"T_mean={T_mean:.2f}  flame={flame_frac:.1%}",
                f"clock: {ctrl.clock.steps_per_unit} steps/unit, "
                f"{ctrl.clock.seconds_per_unit:.2f} s/unit",
            ]
            overlay_label(rgb_bgr, label_lines)

            panels = [rgb_bgr, therm_bgr, depth_panel]
            # Match heights
            target_h = panels[0].shape[0]
            for i, p in enumerate(panels):
                if p.shape[0] != target_h:
                    panels[i] = cv2.resize(
                        p, (int(p.shape[1] * target_h / p.shape[0]), target_h),
                    )
            grid = np.hstack(panels)

            cv2.imshow(window, grid)

            if out_dir is not None:
                cv2.imwrite(str(out_dir / f"step_{robot_step:05d}.png"), grid)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("r"):
                obs = env.reset()
                robot_step = 0
                print("[teleop] reset; t_sim back to 0")
                continue
            action_name = key_map.get(key)
            if action_name is None:
                continue
            action_idx = action_dict.get(action_name)
            if action_idx is None:
                print(f"[teleop] action {action_name} not available in this task")
                continue
            actions = [default_stop for _ in range(args.num_agents)]
            actions[args.agent_id] = action_idx
            try:
                obs = env.step(actions)
            except Exception:
                traceback.print_exc()
                break
            robot_step += 1
            print(f"  step={robot_step:>4d}  action={action_name:<12s}  "
                  f"t_sim={ctrl.clock.t_sim_for_step(robot_step):>6.1f}s")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
