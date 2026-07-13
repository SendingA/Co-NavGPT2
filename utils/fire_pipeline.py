"""Tiny glue between :class:`FireSensorSuite` and the Habitat obs dict.

After the architecture refactor the heavy lifting is split clearly:

* :mod:`utils.fire_world` owns the *world model* (voxel timeline +
  clock + camera-pose helper).
* :mod:`utils.fire_sensors` owns the *observation layer* (voxel RGB /
  Thermal observer + noisy depth / radar / lidar / dashboard).

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

# Emit the human-thermal overlay failure only once to avoid log spam.
_HUMAN_THERMAL_WARNED = False


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
    walker: Optional[Any] = None,
    humans: Optional[Any] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Run the sensor suite on one frame and patch ``observations`` in place.

    When ``walker`` (or an explicit ``humans`` list) is passed, each
    live humanoid is projected into the current camera and added to
    the thermal image + temperature map as a ~+9 C blob so it stays
    visible against the ambient background — this is what a real
    FLIR-style IR camera would see for a person walking through the
    scene.

    Returns the suite's output dict (handy for ``suite.save_step`` and
    GUI viewers) or ``None`` if the suite is disabled.
    """
    if suite is None:
        return None

    max_d, normalize = _resolve_depth_config(config)
    # depth_use_clean: -1 = auto. The smoke-degraded depth clips distant
    # surfaces to the Jin visibility layer, so in a flame/smoke region
    # the mapper would treat the whole area as a near-field obstacle
    # wall (the "floor fire = obstacle" bug). Default to clean depth for
    # mapping whenever a fire scene is active; only honour a smoky depth
    # when the user explicitly asks for it with --depth_use_clean 0.
    _duc = int(getattr(args, "depth_use_clean", -1))
    if _duc < 0:
        use_clean_depth = bool(suite.scene is not None)
    else:
        use_clean_depth = bool(_duc)
    apply_smoky_rgb = bool(int(getattr(args, "fire_apply_to_obs", 1)))
    # The voxel sensor always produces the thermal channel, so inject it
    # into observations by default (opt out with --use_thermal_perception 0).
    use_thermal = bool(int(getattr(args, "use_thermal_perception", 1)))

    rgb_clean = np.asarray(observations["rgb"])[..., :3]
    if rgb_clean.dtype != np.uint8:
        rgb_clean = np.clip(rgb_clean, 0, 255).astype(np.uint8)

    # ---- Pristine clean depth for the RGB/thermal ray-march -------------
    # The volumetric renderer terminates every camera ray at the depth
    # surface, so the fire is only integrated *between the camera and
    # that surface*. The smoke-degraded LIDAR depth clips smoky pixels to
    # the Jin-visibility layer (~0.5 m), which — if it ever becomes the
    # render's ray length — stops the rays short of the flame/smoke voxels
    # and wipes the fire out of the RGB/thermal image.
    #
    # That coupling is wrong: the RGB camera's rays physically terminate
    # at true geometry, not at the LIDAR's smoke-clipped range. Smoke
    # degradation belongs only to the depth *sensor* output. So we always
    # ray-march against the pristine geometric depth, independent of
    # --depth_use_clean (which now controls *only* the depth written back
    # for mapping).
    #
    # We stash the pristine depth on the observation dict so it survives
    # the write-back and stays clean across frames that reuse the same
    # dict (e.g. teleop idle polls). A fresh env.step() returns a new dict
    # without the stash, so the simulator's clean depth is re-captured.
    if "_fire_clean_depth_raw" in observations:
        depth_raw = np.asarray(observations["_fire_clean_depth_raw"])
    else:
        depth_raw = np.asarray(observations["depth"])
        observations["_fire_clean_depth_raw"] = depth_raw
    depth_m = _depth_to_metric(depth_raw, normalize, max_d)

    sensors = suite.process(
        rgb_clean, depth_m,
        obs=observations,
        agent_state=agent_state,
        robot_step=int(robot_step),
    )

    # ---- Add humanoid thermal signatures ---------------------------------
    if humans is None and walker is not None:
        try:
            from utils.fire_sensors.humans_thermal import humans_from_walker
            humans = humans_from_walker(walker)
        except Exception:
            humans = None
    if humans:
        try:
            from utils.fire_sensors.humans_thermal import (
                add_humans_to_thermal_image,
                project_humans_to_thermal,
            )
            camera_K = getattr(
                suite.voxel_sensor,
                "camera_K",
                getattr(suite, "camera_K", None),
            )
            if camera_K is not None:
                # Expose the raw pixel mask so downstream detectors
                # (or GPT reasoning) can distinguish person from flame.
                H = int(sensors["thermal_temperature"].shape[0])
                W = int(sensors["thermal_temperature"].shape[1])
                human_mask = project_humans_to_thermal(
                    humans=humans,
                    agent_state=agent_state,
                    camera_K=camera_K,
                    image_hw=(H, W),
                    depth_m=depth_m,
                    max_depth_m=float(max_d),
                )
                thermal_img_new, thermal_temp_new = add_humans_to_thermal_image(
                    thermal_image_bgr=sensors["thermal_image"],
                    thermal_temperature=sensors["thermal_temperature"],
                    humans=humans,
                    agent_state=agent_state,
                    camera_K=camera_K,
                    depth_m=depth_m,
                    max_depth_m=float(max_d),
                    color_blend=float(getattr(
                        getattr(getattr(suite, "cfg", None), "voxel", None),
                        "thermal_color_blend",
                        1.0,
                    ) if getattr(suite, "cfg", None) is not None else 1.0),
                )
                sensors["thermal_image"] = thermal_img_new
                sensors["thermal_temperature"] = thermal_temp_new
                # Binary 0/1 mask of pixels that are "human warm" so
                # downstream code can e.g. add a 'person' detection
                # sourced from thermal (mirrors thermal_flame_mask).
                sensors["thermal_human_mask"] = (
                    human_mask > 0.5
                ).astype(np.float32)
        except Exception as exc:
            # Human thermal overlay is best-effort; never break the
            # main fire pipeline over a rendering issue. But do surface
            # the failure once so a real bug (e.g. an intrinsics type
            # mismatch) doesn't silently wipe humans from the IR image.
            global _HUMAN_THERMAL_WARNED
            if not _HUMAN_THERMAL_WARNED:
                _HUMAN_THERMAL_WARNED = True
                import traceback
                print(
                    "[fire_pipeline] human thermal overlay disabled after "
                    f"error: {exc!r}\n" + traceback.format_exc()
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
