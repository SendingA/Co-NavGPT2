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
  V          : save every sensor panel and the dashboard
  Mouse      : click "SAVE SENSOR PANELS" in the top-right corner
  ESC        : quit
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

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
    p.add_argument(
        "--lidar-360",
        "--lidar_360",
        dest="lidar_360",
        type=int,
        choices=(0, 1),
        default=1,
        help="1: install four 90-degree depth sensors and stitch a true "
             "360-degree LiDAR scan (default: 1).",
    )
    p.add_argument(
        "--lidar-resolution",
        "--lidar_resolution",
        dest="lidar_resolution",
        type=int,
        default=320,
        help="Width and height of each of the four LiDAR depth slices.",
    )

    # misc
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-frames-to", default=None,
                   help="Optional per-step PNG dump directory.")
    p.add_argument(
        "--snapshot-dir",
        default=None,
        help="Directory for manual all-sensor snapshots. Defaults to "
             "--save-frames-to when set, otherwise "
             "outputs/teleop_sensor_snapshots.",
    )
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


SNAPSHOT_BUTTON_LABEL = "SAVE SENSOR PANELS  [V]"


def snapshot_button_rect(image_shape) -> Tuple[int, int, int, int]:
    """Return the clickable snapshot-button rectangle as x1, y1, x2, y2."""

    height, width = image_shape[:2]
    margin = max(8, min(16, width // 100))
    button_width = min(340, max(180, width // 4))
    button_height = min(42, max(30, height // 24))
    return (
        max(0, width - button_width - margin),
        margin,
        max(0, width - margin),
        min(height - 1, margin + button_height),
    )


def point_in_rect(
    x: int,
    y: int,
    rect: Tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= int(x) <= x2 and y1 <= int(y) <= y2


def window_point_to_image(
    x: int,
    y: int,
    *,
    window_size: Tuple[int, int],
    image_shape,
) -> Tuple[int, int]:
    """Map a click from a resized OpenCV window back to image pixels."""

    window_width, window_height = window_size
    image_height, image_width = image_shape[:2]
    if window_width <= 0 or window_height <= 0:
        return int(x), int(y)
    return (
        int(round(float(x) * image_width / window_width)),
        int(round(float(y) * image_height / window_height)),
    )


def draw_snapshot_button(
    dashboard: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Draw the mouse-accessible save button without changing panel layout."""

    rendered = np.asarray(dashboard).copy()
    rect = snapshot_button_rect(rendered.shape)
    x1, y1, x2, y2 = rect
    cv2.rectangle(rendered, (x1, y1), (x2, y2), (35, 112, 62), -1)
    cv2.rectangle(rendered, (x1, y1), (x2, y2), (120, 255, 165), 2)
    font_scale = 0.55 if (x2 - x1) >= 280 else 0.42
    cv2.putText(
        rendered,
        SNAPSHOT_BUTTON_LABEL,
        (x1 + 12, y1 + int((y2 - y1) * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return rendered, rect


def _snapshot_scene_name(scene_id: Optional[str]) -> str:
    if scene_id is None:
        return "unknown_scene"
    name = Path(str(scene_id)).stem.replace(".basis", "")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "unknown_scene"


def save_sensor_snapshot(
    *,
    root_dir: Path,
    scene_id: Optional[str],
    agent_id: int,
    robot_step: int,
    rgb_clean: np.ndarray,
    rgb_smoke: np.ndarray,
    depth_clean: np.ndarray,
    depth_smoke: np.ndarray,
    thermal: Optional[np.ndarray],
    lidar: Optional[np.ndarray],
    radar_bev: Optional[np.ndarray],
    radar_az: Optional[np.ndarray],
    radar_el: Optional[np.ndarray],
    dashboard: np.ndarray,
    max_depth_m: float,
    lidar_is_360: bool,
) -> Path:
    """Save each currently displayed sensor product into one directory."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_dir = (
        Path(root_dir)
        / _snapshot_scene_name(scene_id)
        / f"agent_{int(agent_id)}"
        / f"step_{int(robot_step):05d}_{timestamp}"
    )
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    def metric_depth(depth: np.ndarray) -> np.ndarray:
        arr = np.asarray(depth)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(finite.max()) <= 1.0 + 1e-3:
            arr = arr * float(max_depth_m)
        return arr

    products: Dict[str, Optional[np.ndarray]] = {
        "rgb_clean.png": cv2.cvtColor(
            np.ascontiguousarray(np.asarray(rgb_clean)[..., :3]),
            cv2.COLOR_RGB2BGR,
        ),
        "depth_clean.png": dashboard_colorize_depth(
            metric_depth(depth_clean), float(max_depth_m)
        ),
        "thermal.png": thermal,
        "lidar_bev.png": lidar,
        "rgb_smoke.png": cv2.cvtColor(
            np.ascontiguousarray(np.asarray(rgb_smoke)[..., :3]),
            cv2.COLOR_RGB2BGR,
        ),
        "depth_smoke.png": dashboard_colorize_depth(
            metric_depth(depth_smoke), float(max_depth_m)
        ),
        "radar_bev.png": radar_bev,
        "radar_range_azimuth.png": radar_az,
        "radar_range_elevation.png": radar_el,
        "dashboard.png": dashboard,
    }

    saved = []
    missing = []
    for filename, image in products.items():
        if image is None:
            missing.append(filename)
            continue
        path = snapshot_dir / filename
        if not cv2.imwrite(str(path), np.asarray(image)):
            raise RuntimeError(f"failed to write sensor snapshot {path}")
        saved.append(filename)

    manifest = {
        "scene_id": None if scene_id is None else str(scene_id),
        "agent_id": int(agent_id),
        "robot_step": int(robot_step),
        "captured_at": datetime.now().astimezone().isoformat(),
        "lidar_is_360": bool(lidar_is_360),
        "saved_files": saved,
        "missing_files": missing,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot_dir


# ---------------------------------------------------------------------------
# Fire/sensor suite setup
# ---------------------------------------------------------------------------
def maybe_build_fire(args: argparse.Namespace, config, num_agents: int):
    scene = None
    if args.plan_id:
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
        smoke_density=(
            float(args.smoke_density) if scene is not None else 0.0
        ),
        save_npz=False,
        voxel=VoxelSmokeConfig(
            n_steps=(min(int(args.n_steps), 10) if int(args.fast)
                     else int(args.n_steps)),
            smoke_k_ext=float(args.smoke_k_ext),
            render_scale=(min(float(args.render_scale), 0.35) if int(args.fast)
                          else float(args.render_scale)),
            flame_smoke_passthrough=float(args.flame_smoke_passthrough),
            thermal_color_blend=0.85,
            flame_noise_strength=(0.0 if int(args.fast) else 0.75),
            flame_edge_break=(0.0 if int(args.fast) else 1.05),
            flame_color_jitter=(0.0 if int(args.fast) else 0.32),
            smoke_noise_strength=(0.0 if int(args.fast) else 0.24),
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
    if int(args.lidar_resolution) <= 0:
        raise ValueError("--lidar-resolution must be positive")

    if int(args.lidar_360):
        from utils.fire_sensors.lidar_360 import (
            LIDAR_DEPTH_UUIDS,
            install_lidar_depth_sensors,
        )

        with habitat.config.read_write(config):
            install_lidar_depth_sensors(
                config,
                resolution=int(args.lidar_resolution),
                num_agents=int(args.num_agents),
            )
        print(
            "[lidar_360] installed four surround sensors per agent: "
            f"{LIDAR_DEPTH_UUIDS}"
        )

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
    ui_state = {
        "snapshot_requested": False,
        "snapshot_button_rect": None,
        "dashboard_shape": None,
    }

    def on_mouse(event, x, y, flags, userdata):
        del flags, userdata
        rect = ui_state["snapshot_button_rect"]
        image_shape = ui_state["dashboard_shape"]
        try:
            _, _, window_width, window_height = cv2.getWindowImageRect(
                main_window
            )
            image_x, image_y = window_point_to_image(
                x,
                y,
                window_size=(window_width, window_height),
                image_shape=image_shape,
            )
        except (cv2.error, TypeError):
            image_x, image_y = int(x), int(y)
        if (
            event == cv2.EVENT_LBUTTONUP
            and rect is not None
            and image_shape is not None
            and point_in_rect(image_x, image_y, rect)
        ):
            ui_state["snapshot_requested"] = True

    cv2.setMouseCallback(main_window, on_mouse)

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

    print(
        "Controls:  W/A/D move/turn  S/SPACE stop  Q/E look  Tab switch  "
        "P pause  R reset  V save sensors  ESC quit"
    )

    observations = reset_episode()

    # ---------- sensors/fire (built after reset so scene id is right) --------
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
    snapshot_dir = Path(
        args.snapshot_dir
        or args.save_frames_to
        or "outputs/teleop_sensor_snapshots"
    )
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
                f"Tab/1-9=switch{pause_key}  R=reset  V=save  Esc=quit"
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

            grid, button_rect = draw_snapshot_button(grid)
            ui_state["snapshot_button_rect"] = button_rect
            ui_state["dashboard_shape"] = grid.shape
            cv2.imshow(main_window, grid)

            # Save the frame after it is rendered for this exact step.  The
            # previous implementation incremented robot_step first and saved
            # the older image under the newer step number.
            if save_dir is not None and robot_step != last_saved_step:
                cv2.imwrite(str(save_dir / f"step_{robot_step:05d}.png"), grid)
                last_saved_step = robot_step

            # ------ key handling ------
            key = cv2.waitKey(POLL_MS) & 0xFF
            snapshot_requested = bool(ui_state["snapshot_requested"])
            ui_state["snapshot_requested"] = False
            if key in (ord("v"), ord("V")):
                snapshot_requested = True
            if snapshot_requested:
                current_scene = getattr(
                    getattr(env, "current_episode", None),
                    "scene_id",
                    args.scene_id,
                )
                saved_to = save_sensor_snapshot(
                    root_dir=snapshot_dir,
                    scene_id=current_scene,
                    agent_id=active_agent,
                    robot_step=robot_step,
                    rgb_clean=rgb_clean,
                    rgb_smoke=rgb_smoke,
                    depth_clean=depth_clean_raw,
                    depth_smoke=depth_smoke,
                    thermal=therm_bgr,
                    lidar=lidar_img,
                    radar_bev=radar_bev,
                    radar_az=radar_az,
                    radar_el=radar_el,
                    dashboard=grid,
                    max_depth_m=max_d,
                    lidar_is_360=bool(
                        sensors.get("lidar_is_360", False)
                    ) if sensors else False,
                )
                print(f"[teleop] saved sensor panels -> {saved_to}")
                continue
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
