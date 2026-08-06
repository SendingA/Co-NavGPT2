"""mmWave radar simulator: 3D point cloud + range-az / range-el heatmaps.

Two operating modes are provided via ``RadarConfig.mode``:

* ``raw``     — classical sinc-blurred range-azimuth heatmap from the
                forward depth strip (mid-elevation only). This is the
                "input" to a typical radar processing pipeline and matches
                Fig. 1 (left) of RadarHD 2023 (arXiv:2206.09273).
* ``learned`` *(default)* — bypasses the learning step entirely and
                returns a lidar-like 3D point cloud whose accuracy and
                resolution match what RadarHD reports **after** training
                (~24 cm Hausdorff median error, see Fig. 5 of the paper).
                Smoke is treated as transparent because mmWave is largely
                unaffected by it (Starr & Lattimer 2014: <6% range error
                in dense smoke).

Both modes always emit:
  * ``image``     – range-azimuth heatmap (BGR uint8) for legacy callers
  * ``image_az``  – alias of ``image`` (for clarity in the dashboard)
  * ``image_el``  – range-elevation heatmap (BGR uint8)
  * ``image_bev`` – top-down BEV of the 3D point cloud
  * ``heatmap``   – range-azimuth float32 [0,1]
  * ``points``    – (N, 2) projection of points (legacy)
  * ``points_3d`` – (N, 3) point cloud
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .base import BaseSensor
from ..bev import add_metric_axes, points_to_bev, points_to_range_elevation


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------


def _depth_to_az_heatmap(depth_m: np.ndarray, cfg) -> np.ndarray:
    """Classical (range, azimuth) heatmap from a horizontal depth strip."""
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]

    H, W = depth_m.shape
    fx = (W / 2.0) / np.tan(np.deg2rad(cfg.hfov_deg) / 2.0)

    band = depth_m[H // 2 - 8 : H // 2 + 8, :]
    band = np.where(band > 0, band, np.inf)
    col_d = band.min(axis=0)
    col_d = np.where(np.isfinite(col_d), col_d, 0.0)

    cols = np.arange(W) - W / 2.0
    az_deg = np.rad2deg(np.arctan2(cols, fx))

    rcfg = cfg.radar
    R, A = rcfg.range_bins, rcfg.az_bins
    az_min, az_max = -rcfg.az_fov_deg, rcfg.az_fov_deg
    r_max = rcfg.max_range_m

    heat = np.zeros((R, A), dtype=np.float32)
    for d, a in zip(col_d, az_deg):
        if d <= 0 or d > r_max or a < az_min or a > az_max:
            continue
        ri = int(np.clip(d / r_max * (R - 1), 0, R - 1))
        ai = int(np.clip((a - az_min) / (az_max - az_min) * (A - 1), 0, A - 1))
        heat[ri, ai] += 1.0 / max(d * d, 0.5)

    if heat.max() > 0:
        ksize = max(3, int(rcfg.sinc_sigma_bins * 6) | 1)
        blurred = cv2.GaussianBlur(heat, (ksize, 1), sigmaX=rcfg.sinc_sigma_bins)
        wide = cv2.GaussianBlur(
            heat, (max(3, ksize * 3) | 1, 1), sigmaX=rcfg.sinc_sigma_bins * 4
        )
        heat = blurred + 0.25 * wide
        heat = heat / (heat.max() + 1e-6)
    return heat


def _depth_to_points_3d(depth_m: np.ndarray, cfg, stride: int = 4) -> np.ndarray:
    """Same back-projection as the LIDAR module but with a coarser stride.

    Sensor frame: X forward, Y left, Z up.
    """
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    H, W = depth_m.shape
    fx = (W / 2.0) / np.tan(np.deg2rad(cfg.hfov_deg) / 2.0)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[::stride, ::stride]
    valid = (z > 0) & (z < cfg.radar.max_range_m)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    xc = (xs - cx) * z / fx
    yc = (ys - cy) * z / fy
    return np.stack([z, -xc, -yc], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------


class RadarSensor(BaseSensor):
    name = "radar"

    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        rcfg = cfg.radar

        # ---------- 1) classical range-azimuth heatmap ----------
        heat_az = _depth_to_az_heatmap(depth_m, cfg)
        if rcfg.noise_std > 0:
            heat_az = heat_az + self.rng.normal(
                0.0, rcfg.noise_std, size=heat_az.shape
            ).astype(np.float32)
            heat_az = np.clip(heat_az, 0.0, 1.0)

        u8_az = (heat_az * 255).astype(np.uint8)
        image_az_raw = cv2.applyColorMap(u8_az, cv2.COLORMAP_INFERNO)
        image_az = add_metric_axes(
            image_az_raw,
            x_label="Azimuth [deg]",
            y_label="Range [m]",
            x_limits=(-rcfg.az_fov_deg, rcfg.az_fov_deg),
            # Range bins are stored from near (top) to far (bottom).
            y_limits=(0.0, rcfg.max_range_m),
            plot_size=(rcfg.bev_size_px, rcfg.bev_size_px),
        )

        # ---------- 2) generate point cloud ----------
        if rcfg.mode == "raw":
            # threshold the heatmap (legacy 2D points only)
            mask = heat_az > rcfg.threshold
            ri, ai = np.where(mask)
            R, A = rcfg.range_bins, rcfg.az_bins
            az_min, az_max = -rcfg.az_fov_deg, rcfg.az_fov_deg
            r_vals = ri.astype(np.float32) / max(R - 1, 1) * rcfg.max_range_m
            a_vals = (
                az_min
                + ai.astype(np.float32) / max(A - 1, 1) * (az_max - az_min)
            )
            a_rad = np.deg2rad(a_vals)
            xy = np.stack(
                [r_vals * np.cos(a_rad), r_vals * np.sin(a_rad)], axis=-1
            )
            # promote to 3D at z=0 (raw mode has no elevation info)
            z = np.zeros((xy.shape[0], 1), dtype=np.float32)
            pts3d = np.concatenate([xy, z], axis=1).astype(np.float32)
        else:
            # "learned" mode: bypass the network, return a lidar-like
            # cloud derived from the clean depth (radar is unaffected by
            # smoke - paper Table 3) and add a Gaussian noise whose std
            # matches RadarHD's reported post-training Hausdorff error.
            pts3d = _depth_to_points_3d(depth_m, cfg, stride=rcfg.learned_stride)
            if pts3d.size > 0 and rcfg.learned_noise_m > 0.0:
                pts3d = pts3d + self.rng.normal(
                    0.0, rcfg.learned_noise_m, size=pts3d.shape
                ).astype(np.float32)
            # subsample so the cloud density looks like a real lidar slice
            if pts3d.size > 0 and rcfg.learned_target_points > 0:
                n = pts3d.shape[0]
                if n > rcfg.learned_target_points:
                    idx = self.rng.choice(
                        n, rcfg.learned_target_points, replace=False
                    )
                    pts3d = pts3d[idx]

        # ---------- 3) range-elevation heatmap from 3D points ----------
        image_el_raw = points_to_range_elevation(
            pts3d,
            range_bins=rcfg.range_bins,
            elev_bins=max(32, rcfg.az_bins),
            max_range_m=rcfg.max_range_m,
            elev_fov_deg=rcfg.elev_fov_deg,
        )
        image_el = add_metric_axes(
            image_el_raw,
            x_label="Elevation [deg]",
            y_label="Range [m]",
            x_limits=(-rcfg.elev_fov_deg, rcfg.elev_fov_deg),
            y_limits=(0.0, rcfg.max_range_m),
            # Match the very wide full-width dashboard row so labels are not
            # stretched horizontally by the final compositor.
            plot_size=(rcfg.bev_size_px * 6, rcfg.bev_size_px // 2),
        )

        # ---------- 4) BEV preview ----------
        image_bev_raw = points_to_bev(
            pts3d,
            size=rcfg.bev_size_px,
            extent_m=rcfg.max_range_m,
            z_range=(-1.0, 2.0),
            title=f"Radar 3D  {len(pts3d)} pts",
        )
        image_bev = add_metric_axes(
            image_bev_raw,
            x_label="Lateral Y [m]  (left +)",
            y_label="Forward X [m]",
            # Pixel-left is sensor-left (+Y); pixel-top is forward (+X).
            x_limits=(rcfg.max_range_m, -rcfg.max_range_m),
            y_limits=(rcfg.max_range_m, -rcfg.max_range_m),
        )

        return {
            "heatmap": heat_az.astype(np.float32),
            # primary image in dashboard for "radar" panel = BEV cloud
            "image": image_bev,
            "image_az": image_az,
            "image_el": image_el,
            "image_bev": image_bev,
            "points": pts3d[:, :2].astype(np.float32),
            "points_3d": pts3d.astype(np.float32),
        }
