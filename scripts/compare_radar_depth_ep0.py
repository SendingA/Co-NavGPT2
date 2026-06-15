"""Compare radar vs clean depth on the dumped run ep_0000 of agent_0.

For each saved step_*_arrays.npz under
``outputs/fire_sensors/agent_0/ep_0000/agent_0/`` we:

  1. Read ``depth_clean``.
  2. Re-run :class:`RadarSensor` on that depth to recover the 3D radar
     point cloud (the legacy npz only stored the 2D projection).
  3. Forward-project the radar cloud back into a depth image using the
     same pinhole intrinsics.
  4. Stitch a 4-up panel
     [clean depth | radar-reprojected depth | abs error | radar BEV]
     and aggregate per-step metrics.

Outputs go to ``outputs/radar_vs_depth_ep0/``:

  * ``compare_step_XXXXX.png`` for each NPZ
  * ``summary.csv`` with coverage / MAE / RMSE per step
  * ``summary.txt`` with overall mean stats

Run::

    python scripts/compare_radar_depth_ep0.py
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from utils.fire_sensors import FireSensorConfig  # noqa: E402
from utils.fire_sensors.sensors.radar import RadarSensor  # noqa: E402
from utils.fire_sensors.dashboard import colorize_depth  # noqa: E402

DUMP_DIR = os.path.join(
    ROOT, "outputs", "fire_sensors", "agent_0", "ep_0000", "agent_0"
)
OUT_DIR = os.path.join(ROOT, "outputs", "radar_vs_depth_ep0")
MAX_DEPTH_M = 5.0
HFOV_DEG = 79.0


def points_to_depth_image(
    pts3d: np.ndarray, H: int, W: int, hfov_deg: float, max_d: float
) -> np.ndarray:
    if pts3d.size == 0:
        return np.zeros((H, W), dtype=np.float32)
    fx = (W / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    # invert the convention used in radar's _depth_to_points_3d:
    # pts = [z, -xc, -yc] with xc=(u-cx)*z/fx, yc=(v-cy)*z/fy
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
    big = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(big, (v, u), z)
    big[~np.isfinite(big)] = 0.0
    return big


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 0), 4, cv2.LINE_AA,
    )
    cv2.putText(
        out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(DUMP_DIR, "step_*_arrays.npz")))
    if not npz_files:
        print(f"no NPZ dumps under {DUMP_DIR}")
        return 1
    print(f"found {len(npz_files)} npz dumps")

    cfg = FireSensorConfig(
        max_depth_m=MAX_DEPTH_M, hfov_deg=HFOV_DEG, smoke_density=0.0,
    )
    rng = np.random.default_rng(0)
    radar = RadarSensor(cfg, rng)

    rows = []
    coverages = []
    maes = []
    rmses = []

    for path in npz_files:
        step = int(os.path.basename(path).split("_")[1])
        data = np.load(path)
        depth = data["depth_clean"].astype(np.float32)
        rgb = data["rgb"]
        H, W = depth.shape

        out = radar.process(rgb, depth)
        pts3d = out["points_3d"]

        recon = points_to_depth_image(
            pts3d, H, W, hfov_deg=HFOV_DEG, max_d=MAX_DEPTH_M
        )

        valid = (depth > 0) & (recon > 0)
        cov = float(valid.sum()) / float(depth.size)
        if valid.any():
            err = np.abs(depth[valid] - recon[valid])
            mae = float(err.mean())
            rmse = float(np.sqrt((err ** 2).mean()))
        else:
            mae = rmse = float("nan")

        rows.append((step, len(pts3d), cov, mae, rmse))
        coverages.append(cov)
        if not np.isnan(mae):
            maes.append(mae)
            rmses.append(rmse)

        # ---- panel ----
        clean_img = colorize_depth(depth, MAX_DEPTH_M)
        recon_img = colorize_depth(recon, MAX_DEPTH_M)
        err_full = np.zeros_like(depth)
        err_full[valid] = np.abs(depth[valid] - recon[valid])
        err_norm = np.clip(err_full / 0.5, 0, 1)
        err_u8 = (err_norm * 255).astype(np.uint8)
        err_img = cv2.applyColorMap(err_u8, cv2.COLORMAP_TURBO)

        radar_bev_path = os.path.join(
            DUMP_DIR, f"step_{step:05d}_radar_bev.png"
        )
        if os.path.exists(radar_bev_path):
            bev = cv2.imread(radar_bev_path)
            bev = cv2.resize(bev, (W, H), interpolation=cv2.INTER_AREA)
        else:
            bev = np.zeros_like(clean_img)

        panel = np.concatenate(
            [
                label(clean_img, f"step {step}  clean depth"),
                label(recon_img, f"radar reproj  cov={cov*100:.1f}% pts={len(pts3d)}"),
                label(err_img, f"abs err  MAE={mae:.2f}m  RMSE={rmse:.2f}m"),
                label(bev, "radar BEV (saved)"),
            ],
            axis=1,
        )
        cv2.imwrite(
            os.path.join(OUT_DIR, f"compare_step_{step:05d}.png"), panel
        )

    # ---- summary CSV ----
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "radar_points_3d", "coverage", "mae_m", "rmse_m"])
        w.writerows(rows)

    overall = (
        f"steps analysed     : {len(rows)}\n"
        f"mean coverage      : {np.mean(coverages)*100:.2f}%  "
        f"(min {np.min(coverages)*100:.2f} max {np.max(coverages)*100:.2f})\n"
        f"mean MAE           : {np.mean(maes):.3f} m\n"
        f"mean RMSE          : {np.mean(rmses):.3f} m\n"
        f"radar FOV az/elev  : ±{cfg.radar.az_fov_deg:.0f}° / "
        f"±{cfg.radar.elev_fov_deg:.0f}°\n"
        f"radar max range    : {cfg.radar.max_range_m:.1f} m\n"
        f"radar mode         : {cfg.radar.mode}\n"
    )
    print(overall)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
