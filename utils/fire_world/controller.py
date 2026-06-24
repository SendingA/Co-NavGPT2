"""Backwards-compatible entry point used by older code paths.

After the architecture refactor, ``main.py`` / the keyboard teleop
script construct :class:`utils.fire_world.scene.FireScene` directly and
let :class:`utils.fire_sensors.FireSensorSuite` observe it. This module
keeps the old ``FireWorldController`` / ``FireClock`` names alive so
external scripts and tests don't break.

What you'll find here:

* ``FireClock``               -> re-export from :mod:`utils.fire_world.scene`
* ``FireScene``               -> re-export
* ``FireWorldController``     -> ``FireScene`` + a legacy
                                 ``render_for_agent`` method that runs
                                 the voxel ray-march via the sensor-side
                                 :func:`utils.fire_sensors.voxel_render.volumetric_composite`
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from utils.fire_world.scene import (
    FireClock,
    FireScene,
    habitat_agent_state_to_cam,
)
from utils.fire_world.runtime import FireWorld

__all__ = [
    "FireClock",
    "FireScene",
    "FireWorldController",
    "habitat_agent_state_to_cam",
]


# ---------------------------------------------------------------------------
# Top-level controller (legacy API)
# ---------------------------------------------------------------------------
class FireWorldController:
    """Legacy wrapper kept for back-compat.

    The new code path is::

        scene = FireScene.from_args(args, config)
        suite = FireSensorSuite(cfg=..., scene=scene, camera_K=...)
        sensors = suite.process(rgb, depth_m,
                                agent_state=state, robot_step=k)

    Old call sites that still expect ``ctrl.render_for_agent(...)`` keep
    working through the method below.
    """

    # Legacy positional/keyword fields. The previous version of this
    # class used (fw, renderer, clock) as init args. We accept them for
    # back-compat and synthesise a FireScene + camera_K + params on the
    # fly.
    def __init__(self, scene=None, camera_K=None, _params=None,
                 fw=None, renderer=None, clock=None):
        from utils.fire_sensors.voxel_render import VoxelRenderParams
        if scene is None and fw is not None:
            if clock is None:
                clock = FireClock(steps_per_unit=5, seconds_per_unit=2.0,
                                  base_t0_s=float(fw.times[0]))
            scene = FireScene(fw=fw, clock=clock)
        if scene is None:
            raise ValueError(
                "FireWorldController needs either `scene` or `fw` (legacy)."
            )
        if _params is None:
            _params = VoxelRenderParams()
            if renderer is not None and getattr(renderer, "params", None) is not None:
                _params = renderer.params
            elif renderer is not None:
                # legacy FireWorldRenderer with attribute-style params
                _params.render_scale = getattr(renderer, "render_scale",
                                               _params.render_scale)
                _params.n_steps = getattr(renderer, "n_steps", _params.n_steps)
                _params.smoke_k_ext = getattr(renderer, "smoke_k_ext",
                                              _params.smoke_k_ext)
        if camera_K is None and renderer is not None:
            camera_K = getattr(renderer, "camera_K", None)
        self.scene = scene
        self.camera_K = camera_K
        self._params = _params

    @property
    def fw(self) -> FireWorld:
        return self.scene.fw

    @property
    def clock(self) -> FireClock:
        return self.scene.clock

    @property
    def renderer(self):
        """Legacy attribute. Returns a small object whose mutable
        attributes (``render_scale``, ``n_steps``, ``smoke_k_ext``)
        are forwarded onto the cached VoxelRenderParams.
        """
        return _ParamsView(self._params)

    @classmethod
    def from_args(cls, args, config) -> "FireWorldController":
        from utils.general_utils import get_camera_K
        from utils.fire_sensors.voxel_render import VoxelRenderParams

        scene = FireScene.from_args(args, config)
        K = get_camera_K(args.frame_width, args.frame_height, args.hfov)
        params = VoxelRenderParams(
            max_depth_m=float(config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH),
            n_steps=int(args.fire_world_n_steps),
            smoke_k_ext=float(args.fire_world_smoke_k_ext),
            render_scale=float(getattr(args, "fire_world_render_scale", 0.5)),
        )
        return cls(scene=scene, camera_K=K, _params=params)

    # ------------------------------------------------------------------
    def render_for_agent(
        self,
        observations: Dict[str, np.ndarray],
        agent_state,
        robot_step: int,
        max_depth_m: float,
        normalize_depth: bool,
    ) -> Dict[str, np.ndarray]:
        """Render the active fire frame from the agent's pose.

        Returns the same dict shape as the original
        ``FireWorldController.render_for_agent`` so callers that haven't
        migrated to :class:`FireSensorSuite` yet keep working.
        """
        from utils.fire_sensors.voxel_render import volumetric_composite

        rgb = np.asarray(observations["rgb"])[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        depth_raw = np.asarray(observations["depth"])
        depth_m = depth_raw[..., 0] if depth_raw.ndim == 3 else depth_raw
        depth_m = depth_m.astype(np.float32)
        if normalize_depth:
            depth_m = depth_m * float(max_depth_m)

        cam_pos, R = self.scene.camera_pose(agent_state)
        # Use the unified ``t_sim()`` API so wall-clock mode kicks in
        # automatically when the scene is configured for it. Step mode
        # falls back to the legacy step->seconds mapping.
        t_sim = self.scene.t_sim(robot_step)
        flame_field, smoke_field, temp_field = self.scene.query(t_sim)

        out = volumetric_composite(
            rgb_clean=rgb,
            depth_m=depth_m,
            cam_pos_world=cam_pos.astype(np.float32),
            R_cam2world=R.astype(np.float32),
            flame_field=flame_field,
            smoke_field=smoke_field,
            temp_field=temp_field,
            origin=self.scene.origin,
            voxel_m=self.scene.voxel_m,
            grid_shape=self.scene.shape,
            ambient_c=self.scene.ambient_c,
            camera_K=self.camera_K,
            params=self._params,
        )
        return {
            "rgb": rgb,
            "depth_clean": depth_m.astype(np.float32),
            "rgb_smoke": out["image"],
            "depth_smoke": depth_m.astype(np.float32),
            "transmittance": out["transmittance"],
            "thermal_image": out["thermal_image"],
            "thermal_temperature": out["thermal_temperature"],
            "thermal_flame_mask": out["flame_mask"],
            "t_sim_s": float(t_sim),
            "robot_step": int(robot_step),
        }

    def describe(self) -> str:
        return self.scene.describe()


# ---------------------------------------------------------------------------
class _ParamsView:
    """Tiny proxy so ``ctrl.renderer.render_scale = 1.0`` keeps working
    after the refactor — it now mutates the underlying VoxelRenderParams.
    """

    def __init__(self, params):
        self._params = params

    def __getattr__(self, name):
        return getattr(self._params, name)

    def __setattr__(self, name, value):
        if name == "_params":
            object.__setattr__(self, name, value)
        else:
            setattr(self._params, name, value)
