"""Compose a single image with all modalities side by side.

Layout (2 rows x 4 cols)::

    +-----------+-----------+-----------+-----------+
    | RGB clean | Depth clr | Thermal   | LIDAR BEV |
    +-----------+-----------+-----------+-----------+
    | RGB smoke | Depth smk | Radar BEV | Radar R-A |
    +-----------+-----------+-----------+-----------+

There is also an optional auxiliary panel for the radar range-elevation
heatmap that gets stacked under the grid when ``include_aux`` is True.

Returns a single uint8 BGR image suitable for ``cv2.imwrite`` or
``cv2.imshow``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img


def _resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    bar_h = max(22, h // 18)
    cv2.rectangle(out, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (8, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def colorize_depth(depth_m: np.ndarray, max_depth: float) -> np.ndarray:
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    d = np.clip(depth_m, 0.0, max_depth)
    u8 = (d / max(max_depth, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


# Each tuple is (panel_key, label).
DEFAULT_PANELS = [
    ("rgb",         "RGB (clean)"),
    ("depth",       "Depth (clean)"),
    ("thermal",     "Thermal IR"),
    ("lidar",       "LIDAR BEV"),
    ("rgb_smoke",   "RGB (smoke)"),
    ("depth_smoke", "Depth (smoke)"),
    ("radar",       "Radar BEV (3D)"),
    ("radar_az",    "Radar Range-Az"),
]


def render_dashboard(
    panels: Dict[str, np.ndarray],
    size: Tuple[int, int] = (2000, 900),
    title: str = "",
    extra_panel: np.ndarray = None,
    extra_label: str = "Radar Range-Elev",
) -> np.ndarray:
    """Render a 2x4 grid + optional auxiliary panel below."""
    W, H = size
    cols = 4
    rows = 2
    cell_w, cell_h = W // cols, H // rows

    rows_imgs = []
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            key, label = DEFAULT_PANELS[r * cols + c]
            img = panels.get(key)
            if img is None:
                tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            else:
                tile = _resize(_to_bgr(img), (cell_w, cell_h))
            tile = _put_label(tile, label)
            row_imgs.append(tile)
        rows_imgs.append(np.hstack(row_imgs))
    grid = np.vstack(rows_imgs)

    if extra_panel is not None:
        aux_h = max(180, H // 4)
        aux = _resize(_to_bgr(extra_panel), (grid.shape[1], aux_h))
        aux = _put_label(aux, extra_label)
        grid = np.vstack([grid, aux])

    if title:
        title_h = 30
        bar = np.zeros((title_h, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            bar,
            title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        grid = np.vstack([bar, grid])

    return grid
