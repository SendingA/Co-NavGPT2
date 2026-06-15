"""Common helpers to render 3D point clouds.

Two views are provided:

* :func:`points_to_bev` - top-down (XY) bird's-eye view, color by height (Z).
* :func:`points_to_range_elevation` - 2D (range, elevation) heatmap.

Both return a uint8 BGR image suitable for ``cv2.imwrite`` or being
plugged into the dashboard.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def points_to_bev(
    points: np.ndarray,
    size: int = 480,
    extent_m: float = 10.0,
    z_range: Tuple[float, float] = (-0.6, 1.6),
    point_radius: int = 1,
    bg_color: Tuple[int, int, int] = (15, 15, 15),
    title: str = "",
) -> np.ndarray:
    """Top-down BEV image. X is forward, Y is left.

    Args:
        points:    (N, 2) or (N, 3). Sensor frame (X forward, Y left, Z up).
        size:      output image side length in pixels.
        extent_m:  map shows [-extent_m, extent_m] m around the sensor.
        z_range:   (z_min, z_max) for height-based coloring (only used
                   when points has a Z column).
    """
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)
    if points is None or len(points) == 0:
        cv2.circle(img, (size // 2, size // 2), 3, (0, 255, 255), -1)
        if title:
            cv2.putText(
                img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return img

    pts = np.asarray(points, dtype=np.float32)
    has_z = pts.shape[1] >= 3

    # World->pixel: x forward (up in image), y left (left in image)
    px = (size / 2 - pts[:, 1] / extent_m * (size / 2)).astype(np.int32)
    py = (size / 2 - pts[:, 0] / extent_m * (size / 2)).astype(np.int32)
    keep = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    px, py = px[keep], py[keep]

    if has_z:
        z = pts[keep, 2]
        z_norm = np.clip(
            (z - z_range[0]) / max(z_range[1] - z_range[0], 1e-3), 0.0, 1.0
        )
        # build a small palette via colormap
        cmap_in = (z_norm * 255).astype(np.uint8).reshape(-1, 1)
        colors = cv2.applyColorMap(cmap_in, cv2.COLORMAP_TURBO).reshape(-1, 3)
        for x, y, c in zip(px, py, colors):
            cv2.circle(img, (int(x), int(y)), point_radius,
                       (int(c[0]), int(c[1]), int(c[2])), -1)
    else:
        for x, y in zip(px, py):
            cv2.circle(img, (int(x), int(y)), point_radius, (50, 220, 50), -1)

    # ego marker + ring grid
    cv2.circle(img, (size // 2, size // 2), 4, (0, 255, 255), -1)
    for r_m in (2.0, 5.0, 10.0):
        if r_m > extent_m:
            continue
        rpx = int(r_m / extent_m * (size / 2))
        cv2.circle(img, (size // 2, size // 2), rpx, (60, 60, 60), 1)

    if title:
        cv2.putText(
            img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return img


def points_to_range_elevation(
    points: np.ndarray,
    range_bins: int = 256,
    elev_bins: int = 64,
    max_range_m: float = 10.0,
    elev_fov_deg: float = 30.0,
) -> np.ndarray:
    """Build a (range, elevation) heatmap from a (N, 3) point cloud.

    Returns a uint8 BGR image (INFERNO colormap).
    """
    heat = np.zeros((range_bins, elev_bins), dtype=np.float32)
    if points is not None and len(points) > 0:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        rng = np.sqrt(x ** 2 + y ** 2)
        # elevation: 0 at horizon, +up
        with np.errstate(invalid="ignore"):
            elev_deg = np.rad2deg(np.arctan2(z, np.maximum(rng, 1e-3)))

        ri = np.clip((rng / max_range_m * (range_bins - 1)).astype(np.int32),
                     0, range_bins - 1)
        ai = np.clip(
            ((elev_deg + elev_fov_deg)
             / (2 * elev_fov_deg) * (elev_bins - 1)).astype(np.int32),
            0, elev_bins - 1,
        )
        valid = (rng > 0) & (rng < max_range_m) & (np.abs(elev_deg) < elev_fov_deg)
        for r, a in zip(ri[valid], ai[valid]):
            heat[r, a] += 1.0

        if heat.max() > 0:
            heat = heat / heat.max()
            heat = cv2.GaussianBlur(heat, (3, 3), 0)

    u8 = (heat * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
