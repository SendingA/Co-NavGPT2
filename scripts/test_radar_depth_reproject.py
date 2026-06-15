"""Re-project the radar point cloud back into a depth image.

Goal: visualise how much information the radar simulator preserves vs the
clean depth, by inverting the same pinhole back-projection used inside
``utils/fire_sensors/sensors/radar.py``.

Pipeline (single forward frame):
    clean depth  ─►  RadarSensor.process()  ─►  pts3d (radar cloud)
    pts3d        ─►  pinhole forward projection  ─►  re-projected depth

The script saves a side-by-side panel (clean depth | radar-reprojected
depth | error map) under ``outputs/radar_depth_recon/``.

Run::

    python scripts/test_radar_depth_reproject.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from utils.fire_sensors import FireSensorConfig  # noqa: E402
from utils.fire_sensors.sensors.radar import RadarSensor  # noqa: E402
from utils.fire_sensors.dashboard import colorize_depth  # noqa: E402


def make_synthetic_room(h: int = 240, w: int = 320, max_d: float = 5.0) -> np.ndarray:
    """A coarse depth scene: front wall, side wall, floor, one box."""
    depth = np.full((h, w), max_d, dtype=np.float32)
    # floor: depth grows with row
    rows = np.arange(h)[:, None]
    floor = np.clip(2.0 + (rows / h) * 3.0, 0.5, max_d)
    floor = np.broadcast_to(floor, (h, w)).copy()
    # left side wall closer
    side = np.linspace(1.5, 4.0, w).astype(np.float32)[None, :]
    side = np.broadcast_to(side, (h, w)).copy()
    # combine
    depth = np.minimum(floor, side)
    # a box in the middle
    depth[110:170, 130:200] = 1.6
    depth[140:170, 80:130] = 2.4
    return depth.astype(np.float32)


def points_to_depth_image(
    pts3d: np.ndarray,
    H: int,
    W: int,
    hfov_deg: float,
    max_d: float,
) -> np.ndarray:
    """Forward-project a (N, 3) cloud (X fwd, Y left, Z up) into a depth img."""
    if pts3d.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    fx = (W / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    # invert the convention used in radar's _depth_to_points_3d:
    #   pts = [z, -xc, -yc] where xc=(u-cx)*z/fx, yc=(v-cy)*z/fy
    z = pts3d[:, 0]
    xc = -pts3d[:, 1]
    yc = -pts3d[:, 2]
    valid = z > 1e-3
    z, xc, yc = z[valid], xc[valid], yc[valid]
    if z.size == 0:
        return np.zeros((H, W), dtype=np.float32)

    u = (xc * fx / z + cx).astype(np.int32)
    v = (yc * fy / z + cy).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < max_d)
    u, v, z = u[inb], v[inb], z[inb]

    out = np.full((H, W), 0.0, dtype=np.float32)
    # take min depth at each pixel (closest return wins)
    # use np.minimum.at which broadcasts over duplicated indices
    big = np.full_like(out, np.inf)
    np.minimum.at(big, (v, u), z)
    big[~np.isfinite(big)] = 0.0
    return big


def main() -> int:
    H, W = 240, 320
    max_d = 5.0
    depth = make_synthetic_room(H, W, max_d)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)  # radar ignores rgb

    cfg = FireSensorConfig(
        max_depth_m=max_d,
        hfov_deg=79.0,
        smoke_density=0.0,
    )
    rng = np.random.default_rng(0)
    radar = RadarSensor(cfg, rng)
    out = radar.process(rgb, depth)

    pts3d = out["points_3d"]
    print(f"radar emits {len(pts3d)} 3D points "
          f"(mode={cfg.radar.mode}, az_fov=±{cfg.radar.az_fov_deg}°, "
          f"elev_fov=±{cfg.radar.elev_fov_deg}°, max_range={cfg.radar.max_range_m} m)")

    recon = points_to_depth_image(pts3d, H, W, hfov_deg=cfg.hfov_deg, max_d=max_d)

    # ----- diagnostics -------------------------------------------------
    valid = (depth > 0) & (recon > 0)
    err = np.zeros_like(depth)
    err[valid] = np.abs(depth[valid] - recon[valid])
    coverage = float(valid.sum()) / float(depth.size)
    if valid.any():
        rmse = float(np.sqrt(np.mean(err[valid] ** 2)))
        mae = float(np.mean(err[valid]))
    else:
        rmse = mae = float("nan")
    print(f"coverage (radar pixels covered) = {coverage*100:.1f}%")
    print(f"depth MAE on covered pixels    = {mae:.3f} m")
    print(f"depth RMSE on covered pixels   = {rmse:.3f} m")

    # ----- visualise ---------------------------------------------------
    clean = colorize_depth(depth, max_d)
    radar_img = colorize_depth(recon, max_d)
    err_norm = np.clip(err / 0.5, 0, 1)  # 0..0.5 m mapped to colormap
    err_u8 = (err_norm * 255).astype(np.uint8)
    err_img = cv2.applyColorMap(err_u8, cv2.COLORMAP_TURBO)

    label = lambda img, text: cv2.putText(
        img.copy(), text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    panel = np.concatenate(
        [
            label(clean, "Clean depth"),
            label(radar_img, "Radar -> reprojected depth"),
            label(err_img, "abs error (0..0.5 m)"),
        ],
        axis=1,
    )

    out_dir = os.path.join(ROOT, "outputs", "radar_depth_recon")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "panel.png")
    cv2.imwrite(out_path, panel)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
