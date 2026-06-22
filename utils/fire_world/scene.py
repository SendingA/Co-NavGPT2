"""``FireScene``: world-model facade for the runtime navigation loop.

After this refactor, the responsibility split is::

    utils.fire_world  -> *what* burns where, and *when* (data + clock)
    utils.fire_sensors -> *how* the robot sees the burn (rendering / degradation)

This module owns the world side. ``FireScene`` aggregates:

* a :class:`FireWorld` voxel timeline (flame/smoke/temperature),
* a :class:`FireClock` mapping robot steps to fire-time seconds, and
* the Habitat agent-state -> ``(cam_pos, R_cam2world)`` conversion.

Sensors take a ``FireScene`` and ask it for ``query(t_sim)`` plus
``camera_pose(agent_state)``; they don't need any other knowledge of
how the timeline is laid out or how robot steps translate to seconds.

Time semantics
--------------
Wall-clock time is meaningless in a Habitat simulation. Two integers
fix the mapping::

    steps_per_unit       e.g. 5 - robot steps that elapse for 1 fire-time unit
    seconds_per_unit     e.g. 2 s - timeline seconds consumed per unit

So if the agent has taken ``N`` env.step() calls::

    t_sim = (N // steps_per_unit) * seconds_per_unit  +  base_t0_s

Beyond the last frame the timeline clamps to its burnt-out state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from utils.fire_world.runtime import FireWorld

try:
    import quaternion  # noqa: F401
    _HAS_QUAT = True
except Exception:  # pragma: no cover - the project ships habitat-sim
    _HAS_QUAT = False


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


# ---------------------------------------------------------------------------
# Habitat pose helper
# ---------------------------------------------------------------------------
def habitat_agent_state_to_cam(agent_state) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(cam_pos_world, R_cam2world)`` for the agent's depth sensor.

    ``R_cam2world`` columns = camera right / up / -forward in world
    coords, so a point at ``(x_cam, y_cam, z_cam=-d)`` lands at
    ``cam_pos + d * (-z_axis)`` in world space — exactly what the
    voxel renderer expects.
    """
    sensor_state = agent_state.sensor_states.get("depth", agent_state)
    pos = np.asarray(sensor_state.position, dtype=np.float64)
    rot = sensor_state.rotation
    if hasattr(rot, "x"):  # numpy.quaternion
        if not _HAS_QUAT:
            raise RuntimeError(
                "habitat returned a quaternion but the `quaternion` "
                "module is not importable"
            )
        import quaternion as q
        R = q.as_rotation_matrix(rot)
    else:
        R = np.asarray(rot, dtype=np.float64)
        if R.shape == (4,):
            w, x, y, z = R
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
    return pos, R


# ---------------------------------------------------------------------------
# Top-level scene object
# ---------------------------------------------------------------------------
@dataclass
class FireScene:
    """World-model facade. One instance per Habitat episode."""

    fw: FireWorld
    clock: FireClock

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls, args, config) -> "FireScene":
        """Build a scene matching the CLI flags from ``arguments.py``.

        Resolves the scene id from ``config.SIMULATOR.SCENE`` (Habitat
        rewrites this on every reset) and locates the precomputed
        timeline npz on disk.
        """
        if not getattr(args, "fire_world_plan_id", None):
            raise ValueError(
                "FireScene.from_args requires --fire_world_plan_id to point "
                "at a plan.json under scenes/<scene>/plans/."
            )
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
        clock = FireClock(
            steps_per_unit=int(args.fire_steps_per_unit),
            seconds_per_unit=float(args.fire_seconds_per_unit),
            base_t0_s=float(fw.times[0]),
        )
        return cls(fw=fw, clock=clock)

    # ------------------------------------------------------------------
    # Pass-throughs to FireWorld so downstream code only depends on FireScene
    # ------------------------------------------------------------------
    @property
    def origin(self) -> np.ndarray:
        return self.fw.origin

    @property
    def voxel_m(self) -> float:
        return self.fw.voxel_m

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.fw.shape

    @property
    def ambient_c(self) -> float:
        return self.fw.ambient_c

    @property
    def scene_id(self) -> str:
        return self.fw.scene_id

    @property
    def plan_id(self) -> str:
        return self.fw.plan_id

    def query(self, t_sim: float):
        return self.fw.query(t_sim)

    # ------------------------------------------------------------------
    # Time + pose helpers used by sensors
    # ------------------------------------------------------------------
    def t_sim_for_step(self, robot_step: int) -> float:
        return self.clock.t_sim_for_step(robot_step)

    def camera_pose(self, agent_state) -> Tuple[np.ndarray, np.ndarray]:
        return habitat_agent_state_to_cam(agent_state)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return (
            f"FireScene(scene={self.fw.scene_id} "
            f"plan={self.fw.plan_id} "
            f"timeline=[{float(self.fw.times[0]):.0f}, "
            f"{float(self.fw.times[-1]):.0f}]s @ {len(self.fw.times)} frames; "
            f"clock: {self.clock.steps_per_unit} steps/unit, "
            f"{self.clock.seconds_per_unit:.2f} s/unit)"
        )
