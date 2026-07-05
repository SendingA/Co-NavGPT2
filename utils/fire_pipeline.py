"""Tiny glue between :class:`FireSensorSuite` and the Habitat obs dict.

After the architecture refactor the heavy lifting is split clearly:

* :mod:`utils.fire_world` owns the *world model* (voxel timeline +
  clock + camera-pose helper).
* :mod:`utils.fire_sensors` owns the *observation layer* (Beer-Lambert
  RGB / noisy depth / radar / lidar / thermal / voxel observer / dashboard).

The runtime navigation loop only needs to:

1. Build a :class:`utils.fire_world.scene.FireScene` (optional) and a
   :class:`utils.fire_sensors.FireSensorSuite` bound to it.
2. Each step, call ``suite.process(rgb, depth_m, agent_state=..., robot_step=...)``.
3. Patch the returned dict back into the agent's observation dict via
   :func:`apply_fire_step`.

This module is the helper for step 3.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from utils.smoke_perception import apply_clean_depth_and_thermal


def _depth_to_metric(depth_raw: np.ndarray, normalize: bool, max_d: float) -> np.ndarray:
    """Habitat depth -> metric float32, HxW."""
    arr = np.asarray(depth_raw)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32, copy=False)
    if normalize:
        arr = arr * float(max_d)
    return arr


def _resolve_depth_config(config):
    """Return (max_depth_m, normalize_depth) for either H2 YACS or H3 DictConfig."""
    # Habitat-Lab 0.3.3: DictConfig with lowercase keys.
    if hasattr(config, "habitat"):
        try:
            main_agent_name = config.habitat.simulator.agents_order[0]
            depth_cfg = config.habitat.simulator.agents[
                main_agent_name
            ].sim_sensors.depth_sensor
            return float(depth_cfg.max_depth), bool(
                getattr(depth_cfg, "normalize_depth", True)
            )
        except (AttributeError, KeyError):
            pass
    # Legacy YACS fallback (kept for unit tests / offline scripts that
    # still hand a YACS config to the helper).
    if hasattr(config, "SIMULATOR") and hasattr(config.SIMULATOR, "DEPTH_SENSOR"):
        return (
            float(config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH),
            bool(getattr(config.SIMULATOR.DEPTH_SENSOR, "NORMALIZE_DEPTH", True)),
        )
    return 5.0, True


def step_fire_observation(
    *,
    observations: Dict[str, np.ndarray],
    suite: Optional[Any],
    agent_state,
    robot_step: int,
    config,
    args,
) -> Optional[Dict[str, np.ndarray]]:
    """Run the sensor suite on one frame and patch ``observations`` in place.

    Returns the suite's output dict (handy for ``suite.save_step`` and
    GUI viewers) or ``None`` if the suite is disabled.
    """
    if suite is None:
        return None

    max_d, normalize = _resolve_depth_config(config)
    use_clean_depth = bool(int(getattr(args, "depth_use_clean", 0)))
    apply_smoky_rgb = bool(int(getattr(args, "fire_apply_to_obs", 1)))
    # When the voxel sensor is producing the thermal channel we want it
    # in obs by default. With Beer-Lambert / HSV we keep the legacy
    # opt-in behaviour to avoid surprising existing callers.
    use_thermal_default = 1 if suite.scene is not None else 0
    use_thermal = bool(int(getattr(args, "use_thermal_perception", use_thermal_default)))

    rgb_clean = np.asarray(observations["rgb"])[..., :3]
    if rgb_clean.dtype != np.uint8:
        rgb_clean = np.clip(rgb_clean, 0, 255).astype(np.uint8)
    depth_raw = np.asarray(observations["depth"])
    depth_m = _depth_to_metric(depth_raw, normalize, max_d)

    sensors = suite.process(
        rgb_clean, depth_m,
        obs=observations,
        agent_state=agent_state,
        robot_step=int(robot_step),
    )

    apply_clean_depth_and_thermal(
        observations,
        sensors,
        clean_depth_raw=depth_raw,
        use_clean_depth=use_clean_depth,
        use_thermal=use_thermal,
        apply_smoky_rgb=apply_smoky_rgb,
        normalize_depth=normalize,
        max_depth_m=max_d,
    )
    return sensors


# Back-compat alias for the previous helper name.
def compose_fire_step(
    *,
    observations,
    agent_state,
    robot_step,
    fire_world_ctrl=None,
    fire_suite=None,
    config,
    args,
):
    """Legacy entry point kept for the previous compose_fire_step API.

    The new code path constructs the suite once with ``scene=...`` and
    calls :func:`step_fire_observation` per step. We keep this wrapper
    so callers built before the refactor (and tests / downstream
    scripts) keep working: it lazily re-binds the suite to the
    controller's scene if needed and forwards.
    """
    if fire_suite is None and fire_world_ctrl is None:
        return None
    suite = fire_suite
    if suite is not None and fire_world_ctrl is not None and suite.scene is None:
        # If the caller constructed the suite without a scene but is
        # also passing a controller, bind them now.
        from utils.general_utils import get_camera_K
        K = getattr(fire_world_ctrl, "camera_K", None) or get_camera_K(
            args.frame_width, args.frame_height, args.hfov,
        )
        suite.bind_scene(fire_world_ctrl.scene, camera_K=K)
    return step_fire_observation(
        observations=observations,
        suite=suite,
        agent_state=agent_state,
        robot_step=robot_step,
        config=config,
        args=args,
    )
