"""LIDAR simulator: 3D point cloud from a (logically 360°) depth source.

Two acquisition paths are supported:

* **True 360°** when the agent has the four ``lidar_depth_{front,
  left,back,right}`` sensors (see ``utils/fire_sensors/lidar_360.py``).
  The four 90° HFOV depth slices are back-projected, rotated by their
  yaw and concatenated, yielding a full surround scan.
* **Forward-only fallback** when those UUIDs are absent. The forward
  depth is back-projected with the configured HFOV. Useful for legacy
  configs and for the offline smoke test.

In both cases the post-processing chain (range-/density-dependent
Gaussian noise, smoke-layer clipping at Jin visibility V=2.3/k, and
density-dependent dropout) is identical and follows
Starr & Lattimer 2014, Fig. 5.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import BaseSensor, density_to_k
from ..lidar_360 import LIDAR_DEPTH_UUIDS, stitch_lidar_360


def _depth_to_points_3d(
    depth_m: np.ndarray,
    hfov_deg: float,
    max_range_m: float,
    stride: int = 2,
) -> np.ndarray:
    """Back-project a single depth image to a (N, 3) cloud.

    Sensor frame: X forward, Y left, Z up.
    """
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    H, W = depth_m.shape
    fx = (W / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[::stride, ::stride]
    valid = (z > 0) & (z < max_range_m)
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]

    xc = (xs - cx) * z / fx
    yc = (ys - cy) * z / fy
    return np.stack([z, -xc, -yc], axis=-1).astype(np.float32)


class LidarSensor(BaseSensor):
    name = "lidar"

    # ------------------------------------------------------------------
    # Acquisition strategy
    # ------------------------------------------------------------------
    def _acquire_points(
        self,
        depth_m: np.ndarray,
        obs: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Return the *clean* point cloud before noise/dropout."""
        if obs is not None and any(u in obs for u in LIDAR_DEPTH_UUIDS):
            cloud = stitch_lidar_360(
                obs,
                max_range_m=self.cfg.lidar.max_range_m,
                stride=self.cfg.lidar.stride,
                normalize_depth=True,  # matches our yaml
                min_depth_m=0.0,
                depth_norm_max_m=self.cfg.max_depth_m,
            )
            if cloud is not None and cloud.size > 0:
                return cloud
        # Fallback: forward depth only.
        return _depth_to_points_3d(
            depth_m,
            hfov_deg=self.cfg.hfov_deg,
            max_range_m=self.cfg.lidar.max_range_m,
            stride=self.cfg.lidar.stride,
        )

    # ------------------------------------------------------------------
    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        obs: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        l_cfg = cfg.lidar
        s_cfg = cfg.smoke
        density = float(np.clip(s_cfg.smoke_density, 0.0, 1.0))

        pts = self._acquire_points(depth_m, obs)
        is_360 = obs is not None and any(u in obs for u in LIDAR_DEPTH_UUIDS)

        if pts.size > 0:
            # 1) range-dependent Gaussian noise
            r = np.linalg.norm(pts, axis=1)
            sigma = (
                l_cfg.sigma_base_m
                + l_cfg.sigma_range_m * r
                + l_cfg.sigma_smoke_m * density * r
            )
            pts = pts + self.rng.normal(
                0.0, sigma[:, None], size=pts.shape
            ).astype(np.float32)

            # 2) smoke-layer clip (Starr & Lattimer Fig. 5)
            if l_cfg.clip_to_smoke and density > 0.0:
                k = density_to_k(density, s_cfg.smoke_k_max)
                visibility = 2.3 / max(k, 1e-3)
                r2 = np.linalg.norm(pts, axis=1)
                far = r2 > visibility
                if np.any(far):
                    scale = visibility / np.maximum(r2[far], 1e-3)
                    pts[far] = pts[far] * scale[:, None]
                    pts[far] += self.rng.normal(
                        0.0, 0.05, size=(int(far.sum()), 3)
                    ).astype(np.float32)

            # 3) random dropout
            p_drop = l_cfg.dropout_max * density
            if p_drop > 0:
                keep = self.rng.random(pts.shape[0]) >= p_drop
                pts = pts[keep]

        from ..bev import points_to_bev
        title = ("LIDAR (360°)  " if is_360 else "LIDAR (fwd only)  ") + f"{len(pts)} pts"
        bev = points_to_bev(
            pts,
            size=l_cfg.bev_size_px,
            extent_m=l_cfg.max_range_m,
            z_range=l_cfg.z_color_range_m,
            title=title,
        )

        return {
            "points": pts.astype(np.float32),
            "image": bev,
            "is_360": is_360,
        }
