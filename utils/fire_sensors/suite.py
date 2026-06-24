"""Top-level orchestrator that runs every sensor and writes outputs.

The suite runs the *observation layer* of the fire pipeline. It owns
the per-sensor configurations and, optionally, a bound
:class:`utils.fire_world.scene.FireScene` that the voxel-driven sensors
can query.

Sensor source selection (controlled by ``cfg.rgb_source`` /
``cfg.thermal_source``)
-----------------------------------------------------------------
* ``"beer_lambert"`` (or ``"hsv"`` for thermal): the legacy
  density-driven smoke filter / HSV thermal. No fire scene needed.
* ``"voxel"``: ask the voxel sensor for the smoky RGB / thermal,
  using the bound :class:`FireScene`. If no scene is bound this
  silently falls back to Beer-Lambert / HSV so callers that toggle
  ``--fire_world=0`` keep working.
* ``"auto"`` (default): voxel when a scene is bound, otherwise
  Beer-Lambert / HSV.

Output dict
-----------
Same keys as before (``rgb``, ``rgb_smoke``, ``depth_clean``,
``depth_smoke``, ``thermal_image``, ``thermal_flame_mask``,
``radar_*``, ``lidar_*``, ``dashboard``, ...). New optional keys
``t_sim_s`` and ``robot_step`` are populated when the voxel renderer
is the RGB source so loggers can read them straight off the dict.

Public surface kept stable for ``main.py`` / ``main_vec.py``::

    suite = FireSensorSuite(cfg, dump_dir, save_every, seed,
                            scene=scene_or_none, camera_K=K_or_none)
    out   = suite.process(rgb, depth_m,
                          agent_state=agent_state, robot_step=k)
    suite.save_step(out, episode, step, agent_id)
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np

from .config import FireSensorConfig
from .dashboard import colorize_depth, render_dashboard
from .sensors import (
    LidarSensor,
    RadarSensor,
    SmokeDepthSensor,
    SmokeRGBSensor,
    ThermalSensor,
    VoxelSmokeSensor,
)


def _resolve_source(setting: str, has_scene: bool) -> str:
    s = (setting or "auto").lower()
    if s == "auto":
        return "voxel" if has_scene else "beer_lambert"
    if s in ("hsv", "luma"):
        return "beer_lambert"
    return s


def _global_beer_lambert(
    rgb_uint8: np.ndarray,
    depth_m: np.ndarray,
    smoke_density: float,
    smoke_color_rgb,
    smoke_k_max: float,
    *,
    flame_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a global Beer-Lambert pass on top of an existing RGB.

    Used when ``cfg.compound_rgb=True`` so the voxel-rendered RGB still
    feels environmentally smoky outside the active fire room.

    The optional ``flame_mask`` (float32 in [0, 1]) restores flame
    pixels so the second pass doesn't repaint them with the fog colour.
    Without this guard, dense compound fog can wash a small flame back
    out of the picture entirely.
    """
    if smoke_density <= 0.0:
        return rgb_uint8.copy()
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    k = float(np.clip(smoke_density, 0.0, 1.0)) * float(smoke_k_max)
    T = np.exp(-k * depth_m).astype(np.float32)[..., None]
    T = np.clip(T, 0.0, 1.0)
    smoke = np.asarray(smoke_color_rgb, dtype=np.float32).reshape(1, 1, 3)
    fogged = rgb_uint8.astype(np.float32) * T + smoke * (1.0 - T)
    if flame_mask is not None and flame_mask.size > 0:
        # Soft-blend the original (un-fogged) voxel RGB back in over the
        # flame mask. flame=1 -> keep flame pixel as the voxel renderer
        # produced it; flame=0 -> let the global fog do its thing.
        m = np.clip(flame_mask, 0.0, 1.0).astype(np.float32)[..., None]
        fogged = fogged * (1.0 - m) + rgb_uint8.astype(np.float32) * m
    return np.clip(fogged, 0, 255).astype(np.uint8)


class FireSensorSuite:
    """Bundle that runs every sensor, builds a dashboard and writes to disk."""

    def __init__(
        self,
        cfg: Optional[FireSensorConfig] = None,
        dump_dir: str = "./outputs/fire_sensors",
        save_every: int = 1,
        seed: int = 0,
        *,
        scene=None,
        camera_K=None,
    ) -> None:
        self.cfg = cfg or FireSensorConfig()
        self.dump_dir = dump_dir
        self.save_every = max(1, int(save_every))
        self._rng = np.random.default_rng(seed)
        os.makedirs(self.dump_dir, exist_ok=True)

        # Beer-Lambert / HSV / active sensors share a single RNG.
        self.rgb_sensor = SmokeRGBSensor(self.cfg, self._rng)
        self.depth_sensor = SmokeDepthSensor(self.cfg, self._rng)
        self.radar_sensor = RadarSensor(self.cfg, self._rng)
        self.thermal_sensor = ThermalSensor(self.cfg, self._rng)
        self.lidar_sensor = LidarSensor(self.cfg, self._rng)

        # Voxel observer of the FireScene. Built lazily so callers that
        # don't use it don't pay the camera-K validation cost.
        self.scene = scene
        self.camera_K = camera_K
        self.voxel_sensor: Optional[VoxelSmokeSensor] = None
        if scene is not None:
            if camera_K is None:
                raise ValueError(
                    "FireSensorSuite needs camera_K when a fire scene is bound; "
                    "build it with utils.general_utils.get_camera_K(W, H, hfov)."
                )
            self.voxel_sensor = VoxelSmokeSensor(
                self.cfg, self._rng, camera_K=camera_K, scene=scene,
            )

        # Latest dashboard so a GUI thread can pick it up.
        self.last_dashboard: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def bind_scene(self, scene, camera_K=None) -> None:
        """Attach (or replace) the fire scene this suite observes."""
        self.scene = scene
        if camera_K is not None:
            self.camera_K = camera_K
        if scene is None:
            self.voxel_sensor = None
            return
        if self.camera_K is None:
            raise ValueError("camera_K must be provided once when binding a scene")
        if self.voxel_sensor is None:
            self.voxel_sensor = VoxelSmokeSensor(
                self.cfg, self._rng, camera_K=self.camera_K, scene=scene,
            )
        else:
            self.voxel_sensor.bind_scene(scene)

    # ------------------------------------------------------------------
    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        obs: Optional[Dict[str, np.ndarray]] = None,
        *,
        agent_state=None,
        robot_step: int = 0,
    ) -> Dict[str, np.ndarray]:
        rgb_source = _resolve_source(self.cfg.rgb_source, has_scene=self.scene is not None)
        thermal_source = _resolve_source(
            self.cfg.thermal_source, has_scene=self.scene is not None
        )
        # Voxel sensors need an agent_state. Fall back gracefully if it
        # wasn't supplied (e.g. unit tests with mocked input).
        if rgb_source == "voxel" and (
            self.voxel_sensor is None or agent_state is None
        ):
            rgb_source = "beer_lambert"
        if thermal_source == "voxel" and (
            self.voxel_sensor is None or agent_state is None
        ):
            thermal_source = "hsv"

        # ---- Run RGB / Thermal source first ----
        voxel_out = None
        if "voxel" in (rgb_source, thermal_source):
            voxel_out = self.voxel_sensor.process(
                rgb, depth_m,
                agent_state=agent_state, robot_step=int(robot_step),
            )

        if rgb_source == "voxel":
            rgb_smoke = voxel_out["image"]
            transmittance = voxel_out["transmittance"]
            flame_mask_voxel = voxel_out.get("flame_mask")
            if self.cfg.compound_rgb:
                # Soft-dilate the flame mask so the warm halo around
                # the flame core also resists the global fog pass.
                fm = flame_mask_voxel
                if fm is not None and fm.size > 0:
                    try:
                        fm = cv2.GaussianBlur(fm.astype(np.float32),
                                              (21, 21), 0)
                        m = float(fm.max())
                        if m > 1e-6:
                            fm = np.clip(fm / m, 0.0, 1.0)
                    except Exception:
                        pass
                rgb_smoke = _global_beer_lambert(
                    rgb_smoke, depth_m,
                    smoke_density=self.cfg.smoke.smoke_density,
                    smoke_color_rgb=self.cfg.smoke.smoke_color_rgb,
                    smoke_k_max=self.cfg.smoke.smoke_k_max,
                    flame_mask=fm,
                )
            rgb_out = {
                "image": rgb_smoke,
                "transmittance": transmittance,
                "flame_mask": flame_mask_voxel,
            }
        else:
            rgb_out = self.rgb_sensor.process(rgb, depth_m)

        if thermal_source == "voxel":
            thermal_out = {
                "image": voxel_out["thermal_image"],
                "temperature_c": voxel_out["thermal_temperature"],
                "flame_mask": voxel_out["flame_mask"],
            }
        else:
            thermal_out = self.thermal_sensor.process(rgb, depth_m)

        depth_out = self.depth_sensor.process(
            rgb, depth_m,
            transmittance=(voxel_out["transmittance"] if voxel_out is not None else None),
        )
        # Both radar and lidar consume the **clean** depth/rgb because:
        #  - mmWave is largely unaffected by smoke (paper Table 3),
        #  - LIDAR's smoke degradation is modeled inside its own module.
        radar_out = self.radar_sensor.process(rgb, depth_m)
        lidar_out = self.lidar_sensor.process(rgb, depth_m, obs=obs)

        d_clean_2d = depth_m[..., 0] if depth_m.ndim == 3 else depth_m
        d_smoke = depth_out["depth"]
        d_smoke_2d = d_smoke[..., 0] if d_smoke.ndim == 3 else d_smoke

        title_extra = ""
        if voxel_out is not None and "t_sim_s" in voxel_out:
            title_extra = (
                f"  step={int(voxel_out['robot_step'])}  "
                f"t_sim={float(voxel_out['t_sim_s']):.1f}s"
            )
        panels = {
            "rgb":         cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            "rgb_smoke":   cv2.cvtColor(rgb_out["image"], cv2.COLOR_RGB2BGR),
            "depth":       colorize_depth(d_clean_2d, self.cfg.max_depth_m),
            "depth_smoke": colorize_depth(d_smoke_2d, self.cfg.max_depth_m),
            "thermal":     thermal_out["image"],
            "lidar":       lidar_out["image"],
            "radar":       radar_out["image_bev"],
            "radar_az":    radar_out["image_az"],
        }
        title_src = (
            "FireWorld voxel + Beer-Lambert" if rgb_source == "voxel"
            else "Beer-Lambert"
        )
        dashboard = render_dashboard(
            panels,
            size=self.cfg.dashboard_size,
            title=f"Fire-Scene Sensors ({title_src}){title_extra}",
            extra_panel=radar_out["image_el"],
        )
        self.last_dashboard = dashboard

        out = {
            # raw inputs
            "rgb": rgb,
            "depth_clean": depth_m.astype(np.float32),
            # smoke-affected
            "rgb_smoke": rgb_out["image"],
            "depth_smoke": d_smoke.astype(np.float32),
            "transmittance": rgb_out.get("transmittance"),
            # thermal
            "thermal_image": thermal_out["image"],
            "thermal_temperature": thermal_out["temperature_c"],
            "thermal_flame_mask": thermal_out["flame_mask"],
            # lidar
            "lidar_points": lidar_out["points"],
            "lidar_image": lidar_out["image"],
            # radar
            "radar_heatmap": radar_out["heatmap"],
            "radar_image_az": radar_out["image_az"],
            "radar_image_el": radar_out["image_el"],
            "radar_image_bev": radar_out["image_bev"],
            "radar_points": radar_out["points"],
            "radar_points_3d": radar_out["points_3d"],
            # composite
            "dashboard": dashboard,
            # provenance
            "rgb_source": rgb_source,
            "thermal_source": thermal_source,
        }
        if voxel_out is not None and "t_sim_s" in voxel_out:
            out["t_sim_s"] = float(voxel_out["t_sim_s"])
            out["robot_step"] = int(voxel_out["robot_step"])
        return out

    # ------------------------------------------------------------------
    def save_step(
        self,
        outputs: Dict[str, np.ndarray],
        episode: int,
        step: int,
        agent_id: int = 0,
    ) -> None:
        if step % self.save_every != 0:
            return
        sub = os.path.join(
            self.dump_dir, f"ep_{episode:04d}", f"agent_{agent_id}"
        )
        os.makedirs(sub, exist_ok=True)
        tag = f"step_{step:05d}"

        cv2.imwrite(
            os.path.join(sub, f"{tag}_rgb.png"),
            cv2.cvtColor(outputs["rgb"], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(sub, f"{tag}_rgb_smoke.png"),
            cv2.cvtColor(outputs["rgb_smoke"], cv2.COLOR_RGB2BGR),
        )

        d_clean = outputs["depth_clean"]
        if d_clean.ndim == 3:
            d_clean = d_clean[..., 0]
        d_smoke = outputs["depth_smoke"]
        if d_smoke.ndim == 3:
            d_smoke = d_smoke[..., 0]
        cv2.imwrite(
            os.path.join(sub, f"{tag}_depth.png"),
            colorize_depth(d_clean, self.cfg.max_depth_m),
        )
        cv2.imwrite(
            os.path.join(sub, f"{tag}_depth_smoke.png"),
            colorize_depth(d_smoke, self.cfg.max_depth_m),
        )

        cv2.imwrite(os.path.join(sub, f"{tag}_thermal.png"),
                    outputs["thermal_image"])
        cv2.imwrite(os.path.join(sub, f"{tag}_lidar_bev.png"),
                    outputs["lidar_image"])
        cv2.imwrite(os.path.join(sub, f"{tag}_radar_bev.png"),
                    outputs["radar_image_bev"])
        cv2.imwrite(os.path.join(sub, f"{tag}_radar_az.png"),
                    outputs["radar_image_az"])
        cv2.imwrite(os.path.join(sub, f"{tag}_radar_el.png"),
                    outputs["radar_image_el"])

        if self.cfg.save_dashboard and "dashboard" in outputs:
            cv2.imwrite(
                os.path.join(sub, f"{tag}_dashboard.png"), outputs["dashboard"]
            )

        if self.cfg.save_npz:
            np.savez_compressed(
                os.path.join(sub, f"{tag}_arrays.npz"),
                rgb=outputs["rgb"],
                rgb_smoke=outputs["rgb_smoke"],
                depth_clean=d_clean,
                depth_smoke=d_smoke,
                lidar_points=outputs["lidar_points"],
                radar_heatmap=outputs["radar_heatmap"],
                radar_points=outputs["radar_points"],
                radar_points_3d=outputs["radar_points_3d"],
                thermal_temperature=outputs["thermal_temperature"],
                thermal_flame_mask=outputs["thermal_flame_mask"],
            )
