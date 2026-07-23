#!/usr/bin/env python3
"""All-in-one keyboard teleop for Co-NavGPT2 on Habitat-Lab 0.3.3.

Combines everything the migrated project ships:

* Habitat 3 multi-agent navigation (drive one selected robot with WASD).
* Optional humanoid pedestrians (Habitat 3 KinematicHumanoid + oracle
  walker) that wander the scene on the navmesh.
* Optional visible robot URDF overlays (Fetch / Spot / Stretch / ...)
  synced to the ObjectNav navigation agents.
* Optional FireWorld voxel-timeline overlay + FireSensorSuite (voxel
  RGB, smoke-degraded depth, thermal, radar, dashboard). Fire time
  advances by wall clock so it evolves even when you are idle.

The unified window shows clean/smoke RGB-D, thermal, LIDAR, radar BEV,
radar range-azimuth, and radar range-elevation for the *active* robot.
Robot, fire-clock, and control state is kept in a top HUD. Press Tab (or
the number keys) to switch active robot when ``--num-agents`` is bigger
than one.

Example (single robot + humanoids + visible Spot + fire)::

    python scripts/keyboard_teleop_full.py \\
        --num-agents 1 --num-humans 2 \\
        --robot-models-enabled 1 --robot-profiles spot \\
        --plan-id 83679a07b632 --scene-id Nfvxx8J5NCo

Example (two robots + fire only, no humans)::

    python scripts/keyboard_teleop_full.py \\
        --num-agents 2 --plan-id 83679a07b632 --scene-id Nfvxx8J5NCo

Controls (window must have focus):

  W / A / D  : move forward / turn left / turn right (active robot)
  S / SPACE  : stop (no movement, fire keeps ticking in wallclock mode)
  Q / E      : look down / up (active robot)
  Tab        : cycle active robot
  1..N       : jump to robot N-1 as the active robot
  P          : pause/resume the fire wall-clock (only in wallclock mode)
  R          : reset the episode (rewind fire, respawn humans + robots)
  ESC        : quit
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import habitat  # noqa: E402
from habitat import Env  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from arguments import (  # noqa: E402
    load_config,
    humanoid_kwargs,
    robot_model_kwargs,
)
from envs import RandomHumanoidWalker, RobotModelManager  # noqa: E402
from utils.fire_world.scene import FireScene  # noqa: E402
from utils.fire_pipeline import step_fire_observation  # noqa: E402
from utils.fire_sensors import FireSensorConfig, FireSensorSuite  # noqa: E402
from utils.fire_sensors.config import VoxelSmokeConfig  # noqa: E402
from utils.fire_sensors.dashboard import (  # noqa: E402
    colorize_depth as dashboard_colorize_depth,
    render_dashboard,
)
from utils.general_utils import get_camera_K  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-config", default="multi_objectnav_hm3d.yaml",
                   help="Hydra yaml name under configs/.")
    p.add_argument("--config", default=None,
                   help="Full path to a Hydra yaml (overrides --task-config).")

    # scene selection
    p.add_argument("--scene-id", default=None,
                   help="Substring match to pick the first episode whose "
                        "scene_id contains it. Omit to use the first "
                        "episode of the dataset.")

    # robot agents
    p.add_argument("--num-agents", type=int, default=1,
                   help="Number of ObjectNav robot agents.")
    p.add_argument("--agent-id", type=int, default=0,
                   help="Initial active robot index (Tab switches).")

    # humanoid pedestrians
    p.add_argument("--num-humans", type=int, default=0,
                   help="Number of KinematicHumanoid pedestrians to spawn "
                        "on the navmesh.")

    # visible robot URDF models
    p.add_argument("--robot-models-enabled", type=int, default=0,
                   dest="robot_models_enabled",
                   help="1: attach Habitat 3 articulated robot URDFs to "
                        "each nav agent for visualisation.")
    p.add_argument("--robot-profiles", default=None,
                   help="Comma-separated profile names cycled across "
                        "agents, e.g. spot,fetch.")
    p.add_argument("--robot-urdfs", default=None,
                   help="Optional comma-separated URDF path overrides.")

    # fire world (all optional; the fire pipeline turns on when --plan-id is set)
    p.add_argument("--plan-id", default=None,
                   help="FireWorld plan id (12-hex). Omit to disable fire.")
    p.add_argument("--scenes-root", default="scenes")
    p.add_argument("--out-root", default="outputs/fire_world")
    p.add_argument("--clock-mode", default="wallclock",
                   choices=["wallclock", "step"])
    p.add_argument("--speedup", type=float, default=1.0)
    p.add_argument("--steps-per-unit", type=int, default=5)
    p.add_argument("--seconds-per-unit", type=float, default=2.0)
    p.add_argument("--smoke-density", type=float, default=0.6)
    p.add_argument("--smoke-k-ext", type=float, default=4.0)
    p.add_argument("--n-steps", type=int, default=16,
                   help="Ray-march samples per pixel in the voxel renderer.")
    p.add_argument("--render-scale", type=float, default=0.5)
    p.add_argument("--fast", type=int, default=0,
                   help="1: disable procedural flame flicker/wisp noise "
                        "and lower ray-march cost for a much higher teleop "
                        "frame-rate (default 0 keeps the pretty look).")
    p.add_argument("--flame-smoke-passthrough", type=float, default=0.95)
    p.add_argument("--show-dashboard", type=int, default=1,
                   help="Deprecated compatibility flag. The unified sensor "
                        "dashboard is always shown in the teleop window.")
    p.add_argument("--depth-use-clean", type=int, default=1,
                   help="1: write Habitat's clean depth back into obs even "
                        "when the fire suite computed a smoky one.")
    p.add_argument("--use-thermal", type=int, default=1)

    # misc
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-frames-to", default=None,
                   help="Optional per-step PNG dump directory.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading (reuses arguments.load_config so we stay in sync).
# We build the Namespace by hand instead of calling arguments.get_args()
# so the teleop's flags don't collide with main.py's flag surface.
# ---------------------------------------------------------------------------
def load_teleop_config(args: argparse.Namespace):
    fake = argparse.Namespace(
        task_config=(args.task_config or "multi_objectnav_hm3d.yaml").replace(
            "configs/", ""
        ),
        config=args.config,
        num_agents=args.num_agents,
        num_humans=args.num_humans,
        gpu_id=args.gpu_id,
        seed=args.seed,
        robot_models_enabled=args.robot_models_enabled,
        robot_profiles=args.robot_profiles,
        robot_urdfs=args.robot_urdfs,
        dataset_path=None,
        scenes_dir=None,
        scene_dataset=None,
        # Camera geometry — mirror the defaults from arguments.get_args
        frame_width=640,
        frame_height=480,
        hfov=79.0,
        # Consumed by arguments.load_config
        turn_angle=30,
    )
    return load_config(fake)


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------
def compose_view(*,
                 rgb_clean: np.ndarray,
                 rgb_smoke: np.ndarray,
                 depth_clean: np.ndarray,
                 depth_smoke: np.ndarray,
                 thermal: np.ndarray,
                 lidar: Optional[np.ndarray] = None,
                 radar_bev: Optional[np.ndarray] = None,
                 radar_az: Optional[np.ndarray] = None,
                 radar_el: Optional[np.ndarray] = None,
                 status_lines: Optional[List[str]] = None,
                 max_d: float = 5.0,
                 dashboard_size=(2000, 900),
                 title: str = "Co-NavGPT2 Teleoperation") -> np.ndarray:
    """Compose the same multimodal dashboard used by ``main.py``.

    The 2x4 grid contains both clean/smoke camera products and the radar
    bird's-eye/range-azimuth views.  Range-elevation spans the full row below
    it, while robot and clock metadata live in the header above the images.
    """

    def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return cv2.cvtColor(
            np.ascontiguousarray(arr[..., :3]), cv2.COLOR_RGB2BGR
        )

    def _metric_depth(depth: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(finite.max()) <= 1.0 + 1e-3:
            arr = arr * float(max_d)
        return arr

    panels = {
        "rgb": _rgb_to_bgr(rgb_clean),
        "rgb_smoke": _rgb_to_bgr(rgb_smoke),
        "depth": dashboard_colorize_depth(_metric_depth(depth_clean), max_d),
        "depth_smoke": dashboard_colorize_depth(
            _metric_depth(depth_smoke), max_d
        ),
        "thermal": thermal,
        "lidar": lidar,
        "radar": radar_bev,
        "radar_az": radar_az,
    }
    return render_dashboard(
        panels,
        size=dashboard_size,
        title=title,
        extra_panel=radar_el,
        extra_label="Radar Range-Elev",
        header_lines=status_lines,
    )


# ---------------------------------------------------------------------------
# Fire suite setup
# ---------------------------------------------------------------------------
def maybe_build_fire(args: argparse.Namespace, config, num_agents: int):
    if not args.plan_id:
        return None, None

    fw_args = SimpleNamespace(
        fire_world=1,
        fire_world_plan_id=args.plan_id,
        fire_world_scenes_root=args.scenes_root,
        fire_world_out_root=args.out_root,
        fire_clock_mode=args.clock_mode,
        fire_speedup=args.speedup,
        fire_steps_per_unit=args.steps_per_unit,
        fire_seconds_per_unit=args.seconds_per_unit,
        fire_world_smoke_k_ext=args.smoke_k_ext,
        fire_world_n_steps=args.n_steps,
    )
    scene = FireScene.from_args(fw_args, config)
    print(f"[fire_world] {scene.describe()}")

    main_agent = config.habitat.simulator.agents_order[0]
    depth_cfg = config.habitat.simulator.agents[main_agent].sim_sensors.depth_sensor
    rgb_cfg = config.habitat.simulator.agents[main_agent].sim_sensors.rgb_sensor

    suite_cfg = FireSensorConfig(
        max_depth_m=float(depth_cfg.max_depth),
        hfov_deg=float(depth_cfg.hfov),
        smoke_density=float(args.smoke_density),
        save_npz=False,
        voxel=VoxelSmokeConfig(
            n_steps=(min(int(args.n_steps), 10) if int(args.fast)
                     else int(args.n_steps)),
            smoke_k_ext=float(args.smoke_k_ext),
            render_scale=(min(float(args.render_scale), 0.35) if int(args.fast)
                          else float(args.render_scale)),
            flame_smoke_passthrough=float(args.flame_smoke_passthrough),
            thermal_color_blend=0.85,
            flame_noise_strength=(0.0 if int(args.fast) else 0.55),
            flame_edge_break=(0.0 if int(args.fast) else 0.8),
            flame_color_jitter=(0.0 if int(args.fast) else 0.25),
            smoke_noise_strength=(0.0 if int(args.fast) else 0.30),
        ),
    )
    K = get_camera_K(int(rgb_cfg.width), int(rgb_cfg.height), float(rgb_cfg.hfov))

    dump_dir = args.save_frames_to or "./outputs/teleop_fire"
    suites = [
        FireSensorSuite(
            cfg=suite_cfg,
            dump_dir=os.path.join(dump_dir, f"agent_{i}"),
            save_every=1,
            seed=args.seed + i,
            scene=scene,
            camera_K=K,
        )
        for i in range(num_agents)
    ]
    return scene, suites


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    config = load_teleop_config(args)

    # ---------- env ----------
    env = Env(config=config)
    if args.scene_id is not None:
        matching = [ep for ep in env.episodes if args.scene_id in ep.scene_id]
        if not matching:
            print(f"[teleop] no episodes for scene_id={args.scene_id}")
            return
        env.current_episode = matching[0]
        print(f"[teleop] scene={args.scene_id} ep={env.current_episode.episode_id}")

    num_agents = int(config.conav.num_robots)
    active_agent = max(0, min(args.agent_id, num_agents - 1))

    # ---------- humanoid walker + visible robot models ----------
    walker = RandomHumanoidWalker(
        sim=env.sim,
        **humanoid_kwargs(config, args.seed),
    )
    robot_models = RobotModelManager(
        sim=env.sim,
        **robot_model_kwargs(config, num_agents),
    )

    # ``fire_scene`` / ``fire_suites`` are constructed **after** the first
    # env.reset() below because Env(config=...) initialises the scene id
    # to the first episode of the dataset iterator; only reset() honours
    # the ``env.current_episode = matching[0]`` override we may have set
    # via ``--scene-id``. Building the FireScene before reset would read
    # the wrong scene name and look for the wrong plan.json.
    fire_scene = None
    fire_suites = None

    pipeline_args = SimpleNamespace(
        depth_use_clean=int(args.depth_use_clean),
        use_thermal_perception=int(args.use_thermal),
        fire_apply_to_obs=1,
    )

    # ---------- window setup ----------
    main_window = "Co-NavGPT2 Teleop - Unified Sensor Dashboard"
    cv2.namedWindow(main_window, cv2.WINDOW_NORMAL)

    # ---------- action map ----------
    # Habitat-Lab 0.3.3 uses lowercase HabitatSimActions singleton values;
    # our patched sim registers 0..5 = stop/move_forward/turn_left/
    # turn_right/look_up/look_down.
    action_dict = {"stop": 0, "move_forward": 1, "turn_left": 2,
                   "turn_right": 3, "look_up": 4, "look_down": 5}
    default_stop = 0
    key_map = {
        ord("w"): "move_forward",
        ord("a"): "turn_left",
        ord("d"): "turn_right",
        ord("s"): "stop",
        ord(" "): "stop",
        ord("q"): "look_down",
        ord("e"): "look_up",
    }

    main_agent_name = config.habitat.simulator.agents_order[0]
    depth_cfg = config.habitat.simulator.agents[main_agent_name].sim_sensors.depth_sensor
    max_d = float(depth_cfg.max_depth)

    def reset_episode() -> list:
        obs = env.reset()
        walker.reset()
        robot_models.reset()
        # walker.reset() spawns humanoids without stepping physics; make
        # them visible in the observations by asking for a fresh
        # sensor read via env.sim.step(None).
        if walker.num_humans > 0 or robot_models.enabled:
            obs = env.sim.step(None)
        if fire_scene is not None:
            fire_scene.clock.start()
        if not isinstance(obs, list):
            obs = [obs]
        return obs

    print("Controls:  W/A/D move/turn  S/SPACE stop  Q/E look  Tab switch  P pause  R reset  ESC quit")

    observations = reset_episode()

    # ---------- fire (built after first reset so scene id is right) ----------
    fire_scene, fire_suites = maybe_build_fire(args, config, num_agents)
    if fire_scene is not None:
        # First-frame observation was captured before the fire clock
        # was started; restart the clock now so t_sim=0 lines up with
        # what we're about to render.
        fire_scene.clock.start()
        print(f"           fire clock={fire_scene.clock.mode} "
              f"speedup={fire_scene.clock.speedup:.2f}")
    robot_step = 0
    POLL_MS = 50   # ~20 Hz redraw so wallclock fire visibly evolves
    save_dir = Path(args.save_frames_to) if args.save_frames_to else None
    last_saved_step = -1
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            # ------ fire pipeline (must run before we composite the view) ------
            if fire_suites is not None:
                agent_state = env.sim.get_agent_state(active_agent)
                sensors = step_fire_observation(
                    observations=observations[active_agent],
                    suite=fire_suites[active_agent],
                    agent_state=agent_state,
                    robot_step=robot_step,
                    config=config,
                    args=pipeline_args,
                    walker=walker,
                )
            else:
                sensors = None

            # ------ what to show ------
            active_obs = observations[active_agent]
            rgb_clean = sensors.get(
                "rgb", active_obs.get("_fire_clean_rgb", active_obs["rgb"])
            ) if sensors else active_obs["rgb"]
            depth_clean_raw = sensors.get(
                "depth_clean",
                active_obs.get("_fire_clean_depth_raw", active_obs["depth"]),
            ) if sensors else active_obs["depth"]
            rgb_smoke = sensors.get("rgb_smoke", rgb_clean) if sensors else rgb_clean
            depth_smoke = sensors.get(
                "depth_smoke", depth_clean_raw
            ) if sensors else depth_clean_raw
            therm_bgr = sensors.get("thermal_image") if sensors else None
            lidar_img = sensors.get("lidar_image") if sensors else None
            radar_bev = sensors.get("radar_image_bev") if sensors else None
            radar_az = sensors.get("radar_image_az") if sensors else None
            radar_el = sensors.get("radar_image_el") if sensors else None

            status_lines = [
                f"Active Robot: {active_agent + 1}/{num_agents}  |  "
                f"Teleop Step: {robot_step}  |  Humans: {walker.num_humans}  |  "
                f"Robot Models: {'ON' if robot_models.enabled else 'OFF'}",
            ]
            if fire_scene is not None:
                # This timestamp belongs to the sensor frame shown below.  In
                # wall-clock mode, sampling the clock again here could make the
                # HUD slightly newer than the rendered smoke/thermal frame.
                t_sim = float(sensors.get(
                    "t_sim_s", fire_scene.t_sim(robot_step)
                )) if sensors else float(fire_scene.t_sim(robot_step))
                paused = (fire_scene.clock.mode == "wallclock" and
                          fire_scene.clock._paused_at is not None)
                if fire_scene.clock.mode == "wallclock":
                    clock_detail = (
                        f"Fire Clock: WALLCLOCK  |  "
                        f"{'PAUSED' if paused else 'RUNNING'}  |  "
                        f"t_sim: {t_sim:.1f} s  |  "
                        f"Speedup: {fire_scene.clock.speedup:.2f}x"
                    )
                else:
                    clock_detail = (
                        f"Fire Clock: STEP  |  t_sim: {t_sim:.1f} s  |  "
                        f"{args.steps_per_unit} steps/unit  |  "
                        f"{args.seconds_per_unit:.2f} s/unit"
                    )
                status_lines.append(clock_detail)
                pause_key = "  P=pause" if fire_scene.clock.mode == "wallclock" else ""
            else:
                status_lines.append("Fire: DISABLED")
                pause_key = ""
            status_lines.append(
                "Controls: W/A/D=move/turn  S/Space=stop  Q/E=look  "
                f"Tab/1-9=switch{pause_key}  R=reset  Esc=quit"
            )

            grid = compose_view(
                rgb_clean=rgb_clean,
                rgb_smoke=rgb_smoke,
                depth_clean=depth_clean_raw,
                depth_smoke=depth_smoke,
                thermal=therm_bgr,
                lidar=lidar_img,
                radar_bev=radar_bev,
                radar_az=radar_az,
                radar_el=radar_el,
                status_lines=status_lines,
                max_d=max_d,
                dashboard_size=(
                    tuple(fire_suites[active_agent].cfg.dashboard_size)
                    if fire_suites is not None else (2000, 900)
                ),
                title=(
                    "Co-NavGPT2 Teleoperation | Fire-Scene Sensors"
                    if fire_scene is not None
                    else "Co-NavGPT2 Teleoperation"
                ),
            )

            cv2.imshow(main_window, grid)

            # Save the frame after it is rendered for this exact step.  The
            # previous implementation incremented robot_step first and saved
            # the older image under the newer step number.
            if save_dir is not None and robot_step != last_saved_step:
                cv2.imwrite(str(save_dir / f"step_{robot_step:05d}.png"), grid)
                last_saved_step = robot_step

            # ------ key handling ------
            key = cv2.waitKey(POLL_MS) & 0xFF
            if key == 0xFF:
                continue
            if key == 27:  # ESC
                break
            if key == 9:  # Tab
                active_agent = (active_agent + 1) % num_agents
                print(f"[teleop] active robot -> {active_agent}")
                continue
            if ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < num_agents:
                    active_agent = idx
                    print(f"[teleop] active robot -> {active_agent}")
                continue
            if key == ord("p") and fire_scene is not None and \
                    fire_scene.clock.mode == "wallclock":
                if fire_scene.clock._paused_at is None:
                    fire_scene.clock.pause()
                    print(f"[teleop] paused at t_sim={fire_scene.t_sim():.1f}s")
                else:
                    fire_scene.clock.resume()
                    print(f"[teleop] resumed at t_sim={fire_scene.t_sim():.1f}s")
                continue
            if key == ord("r"):
                observations = reset_episode()
                robot_step = 0
                last_saved_step = -1
                print("[teleop] reset; fire clock rewound")
                continue

            action_name = key_map.get(key)
            if action_name is None:
                continue
            action_idx = action_dict.get(action_name, default_stop)

            # ------ step env ------
            walker.step()  # advance humans on the navmesh

            # This is a free-roam teleop, not an ObjectNav rollout: a
            # STOP key (or ObjectNav success/stop logic) must NOT end
            # the episode. Route STOP to a physics-only sim step so the
            # agent holds still while the fire keeps evolving.
            if action_idx == default_stop:
                observations = env.sim.step(None)
            else:
                # If a previous action already ended the ObjectNav
                # episode (success / stop), transparently reset so the
                # teleop keeps roaming instead of crashing on the
                # "Episode over" assert.
                if env.episode_over:
                    print("[teleop] episode ended (ObjectNav success/stop); "
                          "auto-resetting to keep roaming")
                    observations = reset_episode()
                    robot_step = 0
                    last_saved_step = -1
                    continue
                actions = [default_stop for _ in range(num_agents)]
                actions[active_agent] = action_idx
                try:
                    observations = env.step(actions)
                except AssertionError:
                    observations = reset_episode()
                    robot_step = 0
                    last_saved_step = -1
                    continue
                except Exception:
                    traceback.print_exc()
                    break
            if not isinstance(observations, list):
                observations = [observations]
            robot_models.step()  # sync visible URDFs to nav agents
            robot_step += 1

            print(f"  step={robot_step:>4d}  agent={active_agent}  "
                  f"action={action_name}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
