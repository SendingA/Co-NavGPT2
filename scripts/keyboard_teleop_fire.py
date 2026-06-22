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

from utils.fire_world.scene import FireScene  # noqa: E402
from utils.fire_pipeline import step_fire_observation  # noqa: E402
from utils.fire_sensors import FireSensorConfig, FireSensorSuite  # noqa: E402
from utils.fire_sensors.config import VoxelSmokeConfig  # noqa: E402
from utils.general_utils import get_camera_K  # noqa: E402


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
    p.add_argument("--n-steps", type=int, default=16,
                   help="Ray-march samples per pixel. 12-16 = fast "
                        "teleop, 24-32 = sharper but slower.")
    p.add_argument("--render-scale", type=float, default=0.5,
                   help="Volume integrator render scale. 0.5 cuts cost "
                        "by ~4x with negligible visual loss; 1.0 = full "
                        "camera resolution.")
    p.add_argument("--depth_use_clean", type=int, default=1,
                   help="1: keep Habitat's clean depth (recommended). "
                        "0: keep whatever the simulator produced.")
    p.add_argument("--save-frames-to", type=str, default=None,
                   help="Optional directory to dump per-step PNGs.")

    # ------------------------------------------------------------------
    # Beer-Lambert sensor suite is always on under the new architecture
    # (the suite is the observation layer of the fire scene). The
    # remaining flags below tune *what* the suite does.
    # ------------------------------------------------------------------
    p.add_argument("--enable-suite", type=int, default=1,
                   help="Deprecated: kept for back-compat. The suite is "
                        "always on now; toggle individual modalities via "
                        "--smoke-density / --compound-rgb / etc.")
    p.add_argument("--smoke-density", type=float, default=0.6,
                   help="Beer-Lambert smoke density [0,1] applied to the "
                        "suite's noisy depth model and (when "
                        "--compound-rgb=1) to the global RGB pass.")
    p.add_argument("--compound-rgb", type=int, default=0,
                   help="1: stack a global Beer-Lambert pass on top of "
                        "the FireWorld voxel RGB so areas outside the "
                        "active fire room still feel smoky.")
    p.add_argument("--show-dashboard", type=int, default=1,
                   help="1: open a second window with the suite's 2x4 "
                        "dashboard (RGB clean/smoke, Depth clean/smoke, "
                        "Thermal, Radar, LIDAR).")
    p.add_argument("--save-npz", type=int, default=0,
                   help="1: dump raw .npz per step alongside PNGs.")
    p.add_argument("--flame-smoke-passthrough", type=float, default=0.85,
                   dest="flame_smoke_passthrough",
                   help="Fraction of smoke extinction the flame radiation "
                        "ignores. 0=flame is eaten by smoke just like the "
                        "scene; 1=smoke is invisible to flame. Realistic "
                        "values are 0.7-0.9 (Starr & Lattimer 2014).")
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

    # ---------- FireScene (the world model) ----------
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
    scene = FireScene.from_args(fw_args, config)
    print(f"[teleop] {scene.describe()}")

    # ---------- FireSensorSuite (the observation layer) ----------
    suite_cfg = FireSensorConfig(
        max_depth_m=float(config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH),
        hfov_deg=float(config.SIMULATOR.DEPTH_SENSOR.HFOV),
        smoke_density=float(args.smoke_density),
        save_npz=bool(int(args.save_npz)),
        rgb_source="voxel",
        thermal_source="voxel",
        compound_rgb=bool(int(args.compound_rgb)),
        voxel=VoxelSmokeConfig(
            n_steps=int(args.n_steps),
            smoke_k_ext=float(args.smoke_k_ext),
            render_scale=float(args.render_scale),
            flame_smoke_passthrough=float(args.flame_smoke_passthrough),
            # Force INFERNO-blended thermal so it visually differs from
            # the JET-colored depth panel sitting next to it.
            thermal_color_blend=1.0,
        ),
    )
    K = get_camera_K(
        config.SIMULATOR.RGB_SENSOR.WIDTH,
        config.SIMULATOR.RGB_SENSOR.HEIGHT,
        config.SIMULATOR.RGB_SENSOR.HFOV,
    )
    dump_dir = (
        args.save_frames_to if args.save_frames_to else "./outputs/teleop_fire"
    )
    suite = FireSensorSuite(
        cfg=suite_cfg,
        dump_dir=os.path.join(str(dump_dir), "suite"),
        save_every=1,
        seed=0,
        scene=scene,
        camera_K=K,
    )
    print(f"[teleop] suite enabled (rgb_source=voxel, thermal_source=voxel, "
          f"compound_rgb={bool(int(args.compound_rgb))}, "
          f"depth_use_clean={bool(int(args.depth_use_clean))})")

    # `step_fire_observation` reads these flags by name.
    pipeline_args = SimpleNamespace(
        depth_use_clean=int(args.depth_use_clean),
        use_thermal_perception=1,
        fire_apply_to_obs=1,
    )

    # ---------- driving loop ----------
    obs = env.reset()
    window = "FireWorld Teleop"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 480)

    dashboard_window = None
    if int(args.show_dashboard):
        dashboard_window = "FireWorld Teleop - Sensor Dashboard"
        cv2.namedWindow(dashboard_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(dashboard_window, 1600, 720)

    out_dir = Path(args.save_frames_to) if args.save_frames_to else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Local step counter; the real env doesn't expose one.
    robot_step = 0
    max_d = float(config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH)

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

            sensors = step_fire_observation(
                observations=obs[args.agent_id],
                suite=suite,
                agent_state=agent_state,
                robot_step=robot_step,
                config=config,
                args=pipeline_args,
            )
            assert sensors is not None  # suite is always on here

            suite.save_step(
                sensors,
                episode=0,
                step=robot_step,
                agent_id=args.agent_id,
            )

            rgb_used = obs[args.agent_id]["rgb"]   # what the nav stack sees
            therm_bgr = sensors["thermal_image"]   # INFERNO-blended LWIR

            # Always show the suite's *degraded* depth here so the
            # panel reflects what the depth sensor would actually see
            # in smoke. Independent of --depth_use_clean (that flag
            # only controls what gets written back into observations).
            depth_smoke = sensors.get("depth_smoke", obs[args.agent_id]["depth"])
            if depth_smoke.ndim == 3:
                depth_smoke = depth_smoke[..., 0]
            depth_used_panel = colorize_depth(depth_smoke, max_d=max_d)

            t_sim = float(sensors.get("t_sim_s", 0.0))
            T_mean = float(np.mean(sensors.get("transmittance", np.ones((1, 1)))))
            flame_frac = float(np.mean(sensors.get("thermal_flame_mask", np.zeros((1, 1)))))

            rgb_bgr = cv2.cvtColor(rgb_used, cv2.COLOR_RGB2BGR)
            label_lines = [
                f"step={robot_step:>4d}  t_sim={t_sim:>6.1f}s  "
                f"T_mean={T_mean:.2f}  flame={flame_frac:.1%}",
                f"clock: {scene.clock.steps_per_unit} steps/unit, "
                f"{scene.clock.seconds_per_unit:.2f} s/unit  "
                f"compound={'on' if int(args.compound_rgb) else 'off'}  "
                f"clean_depth={'on' if int(args.depth_use_clean) else 'off'}  "
                f"flame_passthrough={float(args.flame_smoke_passthrough):.2f}",
            ]
            overlay_label(rgb_bgr, label_lines)

            # Mark the panels so users know exactly what they're looking at.
            depth_panel = depth_used_panel.copy()
            cv2.putText(depth_panel, "Depth (smoke-degraded)", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240),
                        1, cv2.LINE_AA)
            therm_panel = therm_bgr.copy()
            cv2.putText(therm_panel, "Thermal IR", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240),
                        1, cv2.LINE_AA)

            panels = [rgb_bgr, depth_panel, therm_panel]
            target_h = panels[0].shape[0]
            for i, p in enumerate(panels):
                if p.shape[0] != target_h:
                    panels[i] = cv2.resize(
                        p, (int(p.shape[1] * target_h / p.shape[0]), target_h),
                    )
            grid = np.hstack(panels)

            cv2.imshow(window, grid)

            if dashboard_window is not None and "dashboard" in sensors:
                cv2.imshow(dashboard_window, sensors["dashboard"])

            if out_dir is not None:
                cv2.imwrite(str(out_dir / f"step_{robot_step:05d}.png"), grid)
                if "dashboard" in sensors:
                    cv2.imwrite(
                        str(out_dir / f"step_{robot_step:05d}_dashboard.png"),
                        sensors["dashboard"],
                    )

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
                  f"t_sim={scene.t_sim_for_step(robot_step):>6.1f}s")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
