"""Glue between the navigation loop and the FireWorld runtime.

Time semantics
--------------
Wall-clock time is meaningless in a Habitat simulation. We instead let
the user fix two integers:

  * ``steps_per_unit``       (e.g. 5) - how many robot steps make one
                             "fire-time unit"
  * ``seconds_per_unit``     (e.g. 2 s) - how much of the precomputed
                             timeline is consumed per fire-time unit

So if the agent has taken ``N`` env.step() calls, the rendering layer
queries the timeline at::

    t_sim = (N // steps_per_unit) * seconds_per_unit  +  base_t0_s

and the timeline is clamped at its maximum so it stays at the burnt-out
state once the agent has been around long enough.

Pose conversion
---------------
Habitat reports the depth sensor's world position and rotation. The
camera looks along its local -Z axis (OpenGL convention). The
FireWorldRenderer wants a 3x3 ``R_cam2world`` whose columns are the
camera's right/up/forward-as-(-Z) axes in world space - exactly what
quaternion.as_rotation_matrix() returns for Habitat agent state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import quaternion  # noqa: F401  (used by habitat-sim Python bindings)
    _HAS_QUAT = True
except Exception:
    _HAS_QUAT = False

from utils.general_utils import get_camera_K
from utils.fire_world.runtime import FireWorld, FireWorldRenderer


# ---------------------------------------------------------------------------
# Time mapping
# ---------------------------------------------------------------------------
@dataclass
class FireClock:
    """Translates per-robot-step counters into FireWorld timeline seconds."""

    steps_per_unit: int = 5
    seconds_per_unit: float = 2.0
    base_t0_s: float = 0.0

    def t_sim_for_step(self, robot_step: int) -> float:
        units = max(0, int(robot_step)) // max(1, int(self.steps_per_unit))
        return float(self.base_t0_s + units * self.seconds_per_unit)


def _habitat_agent_state_to_cam(agent_state) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cam_pos_world, R_cam2world) for the agent's depth sensor.

    R_cam2world has columns = camera's right / up / -forward in world
    coordinates. This matches what FireWorldRenderer._build_pixel_rays
    expects: a point at ``(x_cam, y_cam, z_cam=-d)`` projected via
    ``R @ pt + cam_pos`` lands at ``cam_pos + d * (-z_axis)`` along the
    camera's forward direction.
    """
    sensor_state = agent_state.sensor_states.get("depth", agent_state)
    pos = np.asarray(sensor_state.position, dtype=np.float64)
    rot = sensor_state.rotation
    if hasattr(rot, "x"):  # numpy.quaternion
        if not _HAS_QUAT:
            raise RuntimeError("habitat returned a quaternion but the "
                               "`quaternion` module is not importable")
        import quaternion as q
        R = q.as_rotation_matrix(rot)
    else:
        R = np.asarray(rot, dtype=np.float64)
        if R.shape == (4,):
            # (w, x, y, z) Habitat-lab ordering -> rotation matrix.
            w, x, y, z = R
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
    return pos, R


# ---------------------------------------------------------------------------
# Top-level controller plugged into main.py
# ---------------------------------------------------------------------------
@dataclass
class FireWorldController:
    """One per-run object that feeds the runtime renderer for every agent."""

    fw: FireWorld
    renderer: FireWorldRenderer
    clock: FireClock

    @classmethod
    def from_args(
        cls,
        args,
        config,
    ) -> "FireWorldController":
        if not getattr(args, "fire_world_plan_id", None):
            raise ValueError(
                "FireWorldController requires --fire_world_plan_id to point "
                "at a plan.json under scenes/<scene>/plans/."
            )

        # Resolve the scene id from the simulator config (Habitat sets
        # SCENE to the absolute glb path on each reset).
        scene_glb = config.SIMULATOR.SCENE
        scene_short = (
            scene_glb.split("/")[-1]
                     .replace(".basis.glb", "")
                     .replace(".glb", "")
        )

        scenes_root = Path(args.fire_world_scenes_root)
        out_root = Path(args.fire_world_out_root)
        plan_path = scenes_root / scene_short / "plans" / f"{args.fire_world_plan_id}.json"
        if not plan_path.exists():
            raise FileNotFoundError(
                f"plan not found: {plan_path}. Build inventory.json + "
                f"plan.json + run propagation for scene {scene_short} first."
            )

        fw = FireWorld.load(scene_short, args.fire_world_plan_id, out_root=out_root)

        K = get_camera_K(
            args.frame_width, args.frame_height, args.hfov,
        )
        renderer = FireWorldRenderer(
            fw=fw,
            camera_K=K,
            max_depth_m=float(config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH),
            n_steps=int(args.fire_world_n_steps),
            smoke_k_ext=float(args.fire_world_smoke_k_ext),
        )
        clock = FireClock(
            steps_per_unit=int(args.fire_steps_per_unit),
            seconds_per_unit=float(args.fire_seconds_per_unit),
            base_t0_s=float(fw.times[0]),
        )
        return cls(fw=fw, renderer=renderer, clock=clock)

    # ------------------------------------------------------------------
    def render_for_agent(
        self,
        observations: Dict[str, np.ndarray],
        agent_state,
        robot_step: int,
        max_depth_m: float,
        normalize_depth: bool,
    ) -> Dict[str, np.ndarray]:
        """Render the active fire frame from the agent's pose and return
        a dict in the same shape that ``FireSensorSuite.process`` returns.

        ``observations`` provides the **clean** RGB and metric depth.
        """
        rgb = np.asarray(observations["rgb"])[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        depth_raw = np.asarray(observations["depth"])
        depth_m = depth_raw[..., 0] if depth_raw.ndim == 3 else depth_raw
        depth_m = depth_m.astype(np.float32)
        if normalize_depth:
            depth_m = depth_m * float(max_depth_m)
        cam_pos, R = _habitat_agent_state_to_cam(agent_state)
        t_sim = self.clock.t_sim_for_step(robot_step)
        out = self.renderer.render(
            rgb_clean=rgb,
            depth_m=depth_m,
            cam_pos_world=cam_pos.astype(np.float32),
            R_cam2world=R.astype(np.float32),
            t_sim=t_sim,
        )
        return {
            # raw clean
            "rgb": rgb,
            "depth_clean": depth_m.astype(np.float32),
            # rendered
            "rgb_smoke": out["image"],
            "depth_smoke": depth_m.astype(np.float32),  # geometry unchanged
            "transmittance": out["transmittance"],
            "thermal_image": out["thermal_image"],
            "thermal_temperature": out["thermal_temperature"],
            "thermal_flame_mask": out["flame_mask"],
            # bookkeeping for the dashboard / logger
            "t_sim_s": float(t_sim),
            "robot_step": int(robot_step),
        }

    # Quick descriptor for log lines.
    def describe(self) -> str:
        return (
            f"FireWorldController(scene={self.fw.scene_id} "
            f"plan={self.fw.plan_id} "
            f"timeline=[{float(self.fw.times[0]):.0f}, "
            f"{float(self.fw.times[-1]):.0f}]s @ {len(self.fw.times)} frames; "
            f"clock: {self.clock.steps_per_unit} steps/unit, "
            f"{self.clock.seconds_per_unit:.2f} s/unit)"
        )
