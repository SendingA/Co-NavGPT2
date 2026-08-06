"""Top-level orchestrator that runs every sensor and writes outputs.

The suite runs the *observation layer* of the fire pipeline. It owns
the per-sensor configurations and, optionally, a bound
:class:`utils.fire_world.scene.FireScene` that the voxel-driven sensors
can query.

RGB & Thermal source
--------------------
The smoky RGB and thermal images are produced exclusively by the
voxel renderer (:class:`VoxelSmokeSensor`), which ray-marches the bound
:class:`utils.fire_world.scene.FireScene`. When no scene is bound the
suite returns a clean RGB passthrough with ambient thermal, so the
depth / radar / lidar modalities stay usable. Depth / radar / lidar
model their own smoke degradation from the shared smoke config.

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
    VoxelSmokeSensor,
)


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

        # Depth / radar / lidar share a single RNG.
        self.depth_sensor = SmokeDepthSensor(self.cfg, self._rng)
        self.radar_sensor = RadarSensor(self.cfg, self._rng)
        self.lidar_sensor = LidarSensor(self.cfg, self._rng)

        # Voxel observer of the FireScene produces the smoky RGB + thermal.
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
    @staticmethod
    def _passthrough_rgb_thermal(
        rgb: np.ndarray, depth_m: np.ndarray, robot_step: int
    ) -> Dict[str, np.ndarray]:
        """Clean RGB + ambient thermal for the no-scene case.

        Keeps depth / radar / lidar usable (e.g. the 360° stitching
        smoke test) without a bound FireScene.
        """
        H, W = (depth_m.shape[:2] if depth_m.ndim >= 2 else rgb.shape[:2])
        zeros = np.zeros((H, W), dtype=np.float32)
        return {
            "image": rgb.copy(),
            "transmittance": np.ones((H, W), dtype=np.float32),
            "flame_mask": zeros,
            "thermal_image": np.zeros((H, W, 3), dtype=np.uint8),
            "thermal_temperature": np.full((H, W), 25.0, dtype=np.float32),
            "t_sim_s": 0.0,
            "robot_step": int(robot_step),
        }

    # ------------------------------------------------------------------
    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        obs: Optional[Dict[str, np.ndarray]] = None,
        *,
        agent_state=None,
        robot_step: int = 0,
        t_sim_s: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        # ---- Voxel RGB + Thermal (single ray-march produces both) ----
        # A bound FireScene is the normal path. With no scene the voxel
        # sensor (or the fallback below) returns a clean passthrough:
        # no smoke, no flame, ambient thermal.
        voxel_out = (
            self.voxel_sensor.process(
                rgb, depth_m,
                agent_state=agent_state, robot_step=int(robot_step),
                t_sim_s=t_sim_s,
            )
            if self.voxel_sensor is not None
            else self._passthrough_rgb_thermal(rgb, depth_m, robot_step)
        )
        rgb_out = {
            "image": voxel_out["image"],
            "transmittance": voxel_out["transmittance"],
            "flame_mask": voxel_out.get("flame_mask"),
        }
        thermal_out = {
            "image": voxel_out["thermal_image"],
            "temperature_c": voxel_out["thermal_temperature"],
            "flame_mask": voxel_out["flame_mask"],
        }

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
        dashboard = render_dashboard(
            panels,
            size=self.cfg.dashboard_size,
            title=f"Fire-Scene Sensors (FireWorld voxel){title_extra}",
            extra_panel=radar_out["image_el"],
        )
        self.last_dashboard = dashboard

        out = {
            # raw inputs
            "rgb": rgb,
            "depth_clean": depth_m.astype(np.float32),
            "sensor_max_depth_m": float(self.cfg.max_depth_m),
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
            # Optional for compatibility with custom/test LiDAR backends
            # implementing the pre-360 output contract.
            "lidar_is_360": bool(lidar_out.get("is_360", False)),
            # radar
            "radar_heatmap": radar_out["heatmap"],
            "radar_image_az": radar_out["image_az"],
            "radar_image_el": radar_out["image_el"],
            "radar_image_bev": radar_out["image_bev"],
            "radar_points": radar_out["points"],
            "radar_points_3d": radar_out["points_3d"],
            # composite
            "dashboard": dashboard,
            # renderer diagnostics (useful for benchmark manifests)
            "fire_render_backend": voxel_out.get(
                "render_backend", "passthrough"
            ),
            "fire_render_device": voxel_out.get(
                "render_device", "cpu"
            ),
        }
        for source_key, output_key in (
            ("render_frame_index", "fire_render_frame_index"),
            ("render_cache_hits", "fire_render_cache_hits"),
            ("render_cache_uploads", "fire_render_cache_uploads"),
        ):
            if source_key in voxel_out:
                out[output_key] = int(voxel_out[source_key])
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
