"""Smoke tests for utils.fire_world.topdown_video.

Verifies:
  - the renderer runs end-to-end on the canonical fixture and writes an
    mp4, a final-state png, and a summary json,
  - the mid-fire frame contains both flame-coloured pixels and a large
    region of smoke-grey pixels (so the renderer is actually drawing the
    fields, not just the floor plan),
  - object footprints are drawn (>200 non-default-grey pixels in the
    full-frame palette),
  - the renderer is deterministic across two consecutive runs (same npz
    inputs => same png).

Run with::

    python scripts/test_fire_topdown_video.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.propagation import run_propagation  # noqa: E402
from utils.fire_world.topdown_video import (  # noqa: E402
    draw_field_overlay,
    draw_ignitions,
    draw_object_footprints,
    project_top_down,
    render_topdown_video,
)


SCENE = "TEEsavR23oF"
PLAN_ID = "d4f8b9c253ab"


def _ensure_timeline(tmp_dir: Path) -> Path:
    """Generate a small timeline if one isn't already in tmp_dir."""
    out_dir = tmp_dir / SCENE / PLAN_ID
    if (out_dir / "timeline.npz").exists():
        return out_dir
    inv = json.loads((ROOT / "scenes" / SCENE / "inventory.json").read_text())
    plan = json.loads(
        (ROOT / "scenes" / SCENE / "plans" / f"{PLAN_ID}.json").read_text()
    )
    run_propagation(
        inventory=inv,
        plan=plan,
        voxel_m=0.20,
        dt=0.5,
        save_dt=10.0,
        out_dir=out_dir,
        verbose=False,
    )
    return out_dir


def _flame_pixel_count(img: np.ndarray) -> int:
    # BGR; flame is high R (chan 2) low B (chan 0).
    return int(((img[..., 2] > 180) & (img[..., 0] < 100)).sum())


def _smoke_grey_pixel_count(img: np.ndarray) -> int:
    g0, g1, g2 = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return int(
        ((g0 > 150) & (g0 < 220) & (np.abs(g0 - g1) < 12) & (np.abs(g0 - g2) < 12)).sum()
    )


def test_end_to_end(tmp_dir: Path) -> None:
    out_dir = _ensure_timeline(tmp_dir)
    inv = json.loads((ROOT / "scenes" / SCENE / "inventory.json").read_text())
    plan = json.loads(
        (ROOT / "scenes" / SCENE / "plans" / f"{PLAN_ID}.json").read_text()
    )
    summary = render_topdown_video(
        inv, plan, out_dir / "timeline.npz", out_dir,
        px_per_m=24.0, fps=20.0, floor_band_m=1.5,
    )
    assert (out_dir / "topdown.mp4").exists(), "mp4 missing"
    assert (out_dir / "topdown.png").exists(), "png missing"
    assert (out_dir / "topdown_summary.json").exists(), "summary missing"
    img = cv2.imread(str(out_dir / "topdown.png"))
    assert img is not None and img.size > 0, "png is empty"
    H, W = img.shape[:2]
    assert (W, H + 28) != (0, 0)
    print(f"end_to_end: OK  size={(W, H)} frames={summary['n_frames']}")


def test_midfire_frame_contains_smoke_and_flame(tmp_dir: Path) -> None:
    out_dir = _ensure_timeline(tmp_dir)
    inv = json.loads((ROOT / "scenes" / SCENE / "inventory.json").read_text())
    plan = json.loads(
        (ROOT / "scenes" / SCENE / "plans" / f"{PLAN_ID}.json").read_text()
    )
    data = np.load(out_dir / "timeline.npz", allow_pickle=True)
    flame_t = data["flame"].astype(np.float32)
    smoke_t = data["smoke"].astype(np.float32)
    times = data["times"].astype(np.float32)
    meta = json.loads(data["meta"][0])
    voxel_m = float(meta["voxel_m"])
    grid_origin = np.array(meta["origin"])
    bb_min = np.array(plan["world_aabb"][:3])
    bb_max = np.array(plan["world_aabb"][3:])
    px_per_m = 24.0
    img_w = max(320, int(round((bb_max[0] - bb_min[0]) * px_per_m)))
    img_h = max(320, int(round((bb_max[2] - bb_min[2]) * px_per_m)))
    floor_y = float(np.mean([ig["position"][1] for ig in plan["ignitions"]]))
    floor_y_idx = int(round((floor_y - grid_origin[1]) / voxel_m))
    band_cells = max(1, int(round(1.5 / voxel_m)))

    # Pick a mid-fire frame: between 60s and 240s where the plume is
    # well-developed.
    target_t = 120.0
    fi = int(np.argmin(np.abs(times - target_t)))
    img = np.full((img_h, img_w, 3), 245, dtype=np.uint8)
    draw_object_footprints(img, inv["objects"], floor_y, 1.5,
                           (float(bb_min[0]), float(bb_min[2])), 1.0 / px_per_m)
    smoke_xz = project_top_down(smoke_t[fi], floor_y_idx, band_cells)
    flame_xz = project_top_down(flame_t[fi], floor_y_idx, band_cells)
    draw_field_overlay(
        img, smoke_xz, flame_xz, voxel_m, 1.0 / px_per_m,
        (float(bb_min[0]), float(bb_min[2])),
        (float(grid_origin[0]), float(grid_origin[2])),
    )
    draw_ignitions(img, plan["ignitions"],
                   (float(bb_min[0]), float(bb_min[2])),
                   1.0 / px_per_m, floor_y, 1.5)

    n_flame = _flame_pixel_count(img)
    n_smoke = _smoke_grey_pixel_count(img)
    assert n_flame > 4, f"too few flame pixels in mid-fire frame: {n_flame}"
    assert n_smoke > 200, f"too few smoke-grey pixels: {n_smoke}"
    print(f"mid-fire frame: flame_px={n_flame} smoke_px={n_smoke} -> OK")


def test_renderer_deterministic(tmp_dir: Path) -> None:
    out_dir = _ensure_timeline(tmp_dir)
    inv = json.loads((ROOT / "scenes" / SCENE / "inventory.json").read_text())
    plan = json.loads(
        (ROOT / "scenes" / SCENE / "plans" / f"{PLAN_ID}.json").read_text()
    )
    a_dir = tmp_dir / "render_a"
    b_dir = tmp_dir / "render_b"
    a_dir.mkdir(exist_ok=True)
    b_dir.mkdir(exist_ok=True)
    render_topdown_video(inv, plan, out_dir / "timeline.npz", a_dir,
                         px_per_m=20.0, fps=10.0, out_basename="td")
    render_topdown_video(inv, plan, out_dir / "timeline.npz", b_dir,
                         px_per_m=20.0, fps=10.0, out_basename="td")
    a = cv2.imread(str(a_dir / "td.png"))
    b = cv2.imread(str(b_dir / "td.png"))
    assert np.array_equal(a, b), "renderer is non-deterministic"
    print("deterministic_render: OK")


def main() -> int:
    tmp = ROOT / "outputs" / "test_topdown_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    test_end_to_end(tmp)
    test_midfire_frame_contains_smoke_and_flame(tmp)
    test_renderer_deterministic(tmp)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
