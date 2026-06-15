"""Top-level orchestrator that runs every sensor and writes outputs.

Public surface kept stable for ``main.py`` / ``main_vec.py``::

    suite = FireSensorSuite(cfg, dump_dir, save_every, seed)
    out   = suite.process(rgb, depth_m)
    suite.save_step(out, episode, step, agent_id)

The dashboard now contains 8 panels (2x4) covering RGB / Depth (both
clean and smoky), Thermal IR, LIDAR BEV, Radar BEV (3D) and Radar
range-azimuth, plus an auxiliary panel for the radar range-elevation
heatmap stacked underneath. The latest dashboard is also exposed as
``suite.last_dashboard`` so a GUI thread can grab it for live display.
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
)


class FireSensorSuite:
    """Bundle that runs every sensor, builds a dashboard and writes to disk."""

    def __init__(
        self,
        cfg: Optional[FireSensorConfig] = None,
        dump_dir: str = "./outputs/fire_sensors",
        save_every: int = 1,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg or FireSensorConfig()
        self.dump_dir = dump_dir
        self.save_every = max(1, int(save_every))
        self._rng = np.random.default_rng(seed)
        os.makedirs(self.dump_dir, exist_ok=True)

        # All sensors share the same RNG so a single seed controls the
        # full pipeline.
        self.rgb_sensor = SmokeRGBSensor(self.cfg, self._rng)
        self.depth_sensor = SmokeDepthSensor(self.cfg, self._rng)
        self.radar_sensor = RadarSensor(self.cfg, self._rng)
        self.thermal_sensor = ThermalSensor(self.cfg, self._rng)
        self.lidar_sensor = LidarSensor(self.cfg, self._rng)

        # Latest 2x4 dashboard so a GUI thread can pick it up.
        self.last_dashboard: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        obs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        rgb_out = self.rgb_sensor.process(rgb, depth_m)
        depth_out = self.depth_sensor.process(rgb, depth_m)
        thermal_out = self.thermal_sensor.process(rgb, depth_m)
        # Both radar and lidar consume the **clean** depth/rgb because:
        #  - mmWave is largely unaffected by smoke (paper Table 3),
        #  - LIDAR's smoke degradation is modeled inside its own module.
        radar_out = self.radar_sensor.process(rgb, depth_m)
        lidar_out = self.lidar_sensor.process(rgb, depth_m, obs=obs)

        d_clean_2d = depth_m[..., 0] if depth_m.ndim == 3 else depth_m
        d_smoke = depth_out["depth"]
        d_smoke_2d = d_smoke[..., 0] if d_smoke.ndim == 3 else d_smoke

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
            title="Fire-Scene Sensors (RGB / Depth / Thermal / LIDAR / Radar)",
            extra_panel=radar_out["image_el"],
        )
        self.last_dashboard = dashboard

        return {
            # raw inputs
            "rgb": rgb,
            "depth_clean": depth_m.astype(np.float32),
            # smoke-affected
            "rgb_smoke": rgb_out["image"],
            "depth_smoke": d_smoke.astype(np.float32),
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
        }

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
