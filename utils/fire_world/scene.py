"""``FireScene``: world-model facade for the runtime navigation loop.

After this refactor, the responsibility split is::

    utils.fire_world  -> *what* burns where, and *when* (data + clock)
    utils.fire_sensors -> *how* the robot sees the burn (rendering / degradation)

This module owns the world side. ``FireScene`` aggregates:

* a :class:`FireWorld` voxel timeline (flame/smoke/temperature),
* a :class:`FireClock` mapping wall-clock time (or robot steps) to
  FireWorld timeline seconds, and
* the Habitat agent-state -> ``(cam_pos, R_cam2world)`` conversion.

Sensors take a ``FireScene`` and ask it for ``query(t_sim)`` plus
``camera_pose(agent_state)``; they don't need any other knowledge of
how the timeline is laid out or how time flows.

Time semantics
--------------
Two clock modes are supported:

* ``mode="wallclock"`` (default) - time advances with real wall-clock,
  scaled by ``speedup`` (fire-seconds per real-second). Fire and
  smoke evolve continuously regardless of how slowly or quickly the
  agent decides to move. Call ``clock.start()`` once per episode to
  reset the origin; ``clock.pause()`` / ``clock.resume()`` are
  available for evaluation pauses (e.g. while waiting for an LLM).
* ``mode="step"`` - the legacy mapping ``t_sim = floor(N / s) * tau``
  used when reproducibility tied to discrete step counts is required
  (e.g. for benchmarking).

Beyond the last frame the timeline clamps to its burnt-out state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
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
    """Translates real time (or robot steps) into FireWorld timeline seconds.

    Wall-clock mode is the default: every call to :meth:`t_sim` returns
    ``base_t0_s + (now - origin) * speedup``, where ``origin`` is
    captured by :meth:`start` (called automatically on the first
    :meth:`t_sim` call if not done explicitly). ``speedup`` is
    fire-seconds per real-second; 1.0 means real-time, 5.0 means the
    fire evolves five times faster than wall clock.

    Step mode is kept for back-compat / reproducible benchmarking: the
    timeline advances ``seconds_per_unit`` whenever ``robot_step``
    crosses another ``steps_per_unit`` boundary, and is independent of
    wall clock entirely.
    """

    # ----------- mode -----------
    mode: str = "wallclock"        # "wallclock" | "step"

    # ----------- wall-clock parameters -----------
    speedup: float = 1.0           # fire-seconds per real-second

    # ----------- step parameters -----------
    steps_per_unit: int = 5
    seconds_per_unit: float = 2.0

    # ----------- timeline origin -----------
    base_t0_s: float = 0.0

    # ----------- internal wall-clock state (do not set directly) -----------
    _wall_origin: Optional[float] = field(default=None, repr=False)
    _paused_at: Optional[float] = field(default=None, repr=False)
    _pause_offset: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Reset the wall-clock origin to 'now'.

        Idempotent and safe to call from anywhere; e.g. ``main.py``
        invokes it once per Habitat episode reset so each episode
        starts at the same fire-time. No-op in step mode.
        """
        if self.mode == "wallclock":
            self._wall_origin = time.monotonic()
            self._paused_at = None
            self._pause_offset = 0.0

    def pause(self) -> None:
        """Pause the wall clock so subsequent ``t_sim()`` calls keep
        returning the same value until :meth:`resume` is called."""
        if self.mode == "wallclock" and self._paused_at is None:
            self._paused_at = time.monotonic()

    def resume(self) -> None:
        if self.mode == "wallclock" and self._paused_at is not None:
            self._pause_offset += time.monotonic() - self._paused_at
            self._paused_at = None

    # ------------------------------------------------------------------
    def t_sim(self, robot_step: int = 0) -> float:
        """Return the current fire-time in seconds.

        ``robot_step`` is ignored in wallclock mode and only consulted
        in step mode. The argument is kept so existing call sites that
        do ``clock.t_sim(robot_step=k)`` keep working in both modes.
        """
        if self.mode == "step":
            return self.t_sim_for_step(robot_step)
        # wallclock
        if self._wall_origin is None:
            self.start()
        now = self._paused_at if self._paused_at is not None else time.monotonic()
        elapsed = max(0.0, now - float(self._wall_origin) - self._pause_offset)
        return float(self.base_t0_s + elapsed * float(self.speedup))

    # Legacy alias kept so callers using the older name keep working.
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

        Resolves the scene id from the config (Habitat rewrites the
        active scene on every reset) and locates the precomputed
        timeline npz on disk. Supports both Habitat-Lab 0.3.3
        (``config.habitat.simulator.scene``) and legacy 0.2.1 YACS
        (``config.SIMULATOR.SCENE``).
        """
        if not getattr(args, "fire_world_plan_id", None):
            raise ValueError(
                "FireScene.from_args requires --fire_world_plan_id to point "
                "at a plan.json under scenes/<scene>/plans/."
            )
        if hasattr(config, "habitat"):
            scene_glb = config.habitat.simulator.scene
        else:
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
        mode = str(getattr(args, "fire_clock_mode", "wallclock")).lower()
        clock = FireClock(
            mode=mode,
            speedup=float(getattr(args, "fire_speedup", 1.0)),
            steps_per_unit=int(getattr(args, "fire_steps_per_unit", 5)),
            seconds_per_unit=float(getattr(args, "fire_seconds_per_unit", 2.0)),
            base_t0_s=float(fw.times[0]),
        )
        # Wallclock mode: lock t=0 to the moment the FireScene is built
        # (one per episode reset). Step mode: this is a no-op.
        clock.start()
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
    def t_sim(self, robot_step: int = 0) -> float:
        """Current fire-time in seconds. In wallclock mode (default) the
        argument is ignored. Kept for back-compat with the step API."""
        return self.clock.t_sim(robot_step)

    # Back-compat alias used by older code paths.
    def t_sim_for_step(self, robot_step: int) -> float:
        return self.clock.t_sim_for_step(robot_step)

    def camera_pose(self, agent_state) -> Tuple[np.ndarray, np.ndarray]:
        return habitat_agent_state_to_cam(agent_state)

    # ------------------------------------------------------------------
    def describe(self) -> str:
        if self.clock.mode == "wallclock":
            tail = (f"clock: wallclock x{self.clock.speedup:.2f} "
                    f"(fire-s per real-s)")
        else:
            tail = (f"clock: {self.clock.steps_per_unit} steps/unit, "
                    f"{self.clock.seconds_per_unit:.2f} s/unit")
        return (
            f"FireScene(scene={self.fw.scene_id} "
            f"plan={self.fw.plan_id} "
            f"timeline=[{float(self.fw.times[0]):.0f}, "
            f"{float(self.fw.times[-1]):.0f}]s @ {len(self.fw.times)} frames; "
            f"{tail})"
        )
