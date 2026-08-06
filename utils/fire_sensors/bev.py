"""Common helpers to render 3D point clouds.

Two views are provided:

* :func:`points_to_bev` - top-down (XY) bird's-eye view, color by height (Z).
* :func:`points_to_range_elevation` - 2D (range, elevation) heatmap.

Both return a uint8 BGR image suitable for ``cv2.imwrite`` or being
plugged into the dashboard.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _format_axis_tick(value: float) -> str:
    """Format physical tick values compactly for a small dashboard panel."""
    if abs(value) < 5e-7:
        value = 0.0
    if np.isclose(value, round(value), atol=1e-6):
        return f"{value:.0f}"
    return f"{value:.1f}"


def _put_vertical_text(
    image: np.ndarray,
    text: str,
    *,
    center_y: int,
    x: int = 5,
    font_scale: float = 0.48,
    color: Tuple[int, int, int] = (225, 225, 225),
) -> None:
    """Draw a 90-degree counter-clockwise label on a BGR image."""
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    patch = np.zeros(
        (text_h + baseline + 8, text_w + 8, 3), dtype=np.uint8
    )
    cv2.putText(
        patch,
        text,
        (4, text_h + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        1,
        cv2.LINE_AA,
    )
    rotated = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    y = max(0, center_y - rotated.shape[0] // 2)
    y2 = min(image.shape[0], y + rotated.shape[0])
    x2 = min(image.shape[1], x + rotated.shape[1])
    if y2 <= y or x2 <= x:
        return
    crop = rotated[: y2 - y, : x2 - x]
    mask = np.any(crop != 0, axis=2)
    target = image[y:y2, x:x2]
    target[mask] = crop[mask]


def add_metric_axes(
    image: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
    plot_size: Optional[Tuple[int, int]] = None,
    x_tick_count: int = 5,
    y_tick_count: int = 5,
) -> np.ndarray:
    """Add labelled physical axes around a sensor preview.

    ``x_limits`` are the values at the left and right image edges.
    ``y_limits`` are the values at the top and bottom image edges.  This
    explicit image-edge convention preserves the existing radar orientation:
    range bin zero stays at the top of the range-angle heatmaps.

    Args:
        image: BGR/gray sensor preview.
        x_label: Horizontal-axis label including physical units.
        y_label: Vertical-axis label including physical units.
        x_limits: Physical values at the left/right plot edges.
        y_limits: Physical values at the top/bottom plot edges.
        plot_size: Optional ``(width, height)`` used to make small heatmaps
            readable before they enter the dashboard.
    """
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if x_tick_count < 2 or y_tick_count < 2:
        raise ValueError("axis tick counts must be at least two")

    if image.ndim == 2:
        plot = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        plot = image.copy()
    else:
        raise ValueError("image must be gray or three-channel BGR")

    if plot_size is not None:
        plot_w, plot_h = int(plot_size[0]), int(plot_size[1])
        if plot_w <= 0 or plot_h <= 0:
            raise ValueError("plot_size dimensions must be positive")
        plot = cv2.resize(
            plot, (plot_w, plot_h), interpolation=cv2.INTER_NEAREST
        )
    else:
        plot_h, plot_w = plot.shape[:2]

    left_margin = 78
    right_margin = 18
    top_margin = 30
    bottom_margin = 62
    canvas = np.full(
        (
            top_margin + plot_h + bottom_margin,
            left_margin + plot_w + right_margin,
            3,
        ),
        10,
        dtype=np.uint8,
    )

    x_positions = np.linspace(0, plot_w - 1, x_tick_count)
    y_positions = np.linspace(0, plot_h - 1, y_tick_count)
    grid_color = (60, 60, 60)
    for x_pos in x_positions:
        cv2.line(
            plot,
            (int(round(x_pos)), 0),
            (int(round(x_pos)), plot_h - 1),
            grid_color,
            1,
            cv2.LINE_AA,
        )
    for y_pos in y_positions:
        cv2.line(
            plot,
            (0, int(round(y_pos))),
            (plot_w - 1, int(round(y_pos))),
            grid_color,
            1,
            cv2.LINE_AA,
        )

    x0, y0 = left_margin, top_margin
    canvas[y0 : y0 + plot_h, x0 : x0 + plot_w] = plot
    axis_color = (205, 205, 205)
    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + plot_w - 1, y0 + plot_h - 1),
        axis_color,
        1,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    tick_scale = 0.42
    x_values = np.linspace(x_limits[0], x_limits[1], x_tick_count)
    for x_pos, value in zip(x_positions, x_values):
        x = x0 + int(round(x_pos))
        cv2.line(
            canvas,
            (x, y0 + plot_h),
            (x, y0 + plot_h + 5),
            axis_color,
            1,
        )
        text = _format_axis_tick(float(value))
        (text_w, _), _ = cv2.getTextSize(text, font, tick_scale, 1)
        cv2.putText(
            canvas,
            text,
            (x - text_w // 2, y0 + plot_h + 20),
            font,
            tick_scale,
            axis_color,
            1,
            cv2.LINE_AA,
        )

    y_values = np.linspace(y_limits[0], y_limits[1], y_tick_count)
    for y_pos, value in zip(y_positions, y_values):
        y = y0 + int(round(y_pos))
        cv2.line(canvas, (x0 - 5, y), (x0, y), axis_color, 1)
        text = _format_axis_tick(float(value))
        (text_w, text_h), _ = cv2.getTextSize(text, font, tick_scale, 1)
        cv2.putText(
            canvas,
            text,
            (x0 - text_w - 9, y + text_h // 2),
            font,
            tick_scale,
            axis_color,
            1,
            cv2.LINE_AA,
        )

    label_scale = 0.48
    (x_label_w, _), _ = cv2.getTextSize(x_label, font, label_scale, 1)
    cv2.putText(
        canvas,
        x_label,
        (
            x0 + max(0, (plot_w - x_label_w) // 2),
            canvas.shape[0] - 10,
        ),
        font,
        label_scale,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    _put_vertical_text(
        canvas,
        y_label,
        center_y=y0 + plot_h // 2,
        x=5,
        font_scale=label_scale,
    )
    return canvas


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
