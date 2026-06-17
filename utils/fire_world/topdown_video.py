"""Top-down validation video for a fire-world timeline.

Stage-4 of the fire_world pipeline. Reads ``timeline.npz`` (stage 3) +
``inventory.json`` (stage 1) + ``plan.json`` (stage 2) and renders an
mp4 plus a final-state PNG so a human can quickly tell whether the
scenario is plausible.

Layout of each frame:
    ┌─────────────────────────────────────────────────┐
    │ scene  plan  fire_type/intensity   t = NNNNs    │
    ├─────────────────────────────────────────────────┤
    │                                                 │
    │   top-down projection (Y collapsed by max):     │
    │     - object aabbs as light grey rectangles     │
    │       coloured per high-flammability category   │
    │     - smoke as gray alpha overlay               │
    │     - flame as orange-red intensity             │
    │     - ignitions marked with numbered crosses    │
    │                                                 │
    └─────────────────────────────────────────────────┘

Multi-floor scenes: we render only the floor band around the ignitions
(``floor_y +/- floor_band_m``). The plan templates already constrain
ignitions to a single floor, so this is sufficient for validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
CATEGORY_COLOR_BGR: Dict[str, Tuple[int, int, int]] = {
    "bed":         (180, 130, 200),
    "sofa":        (210, 130, 170),
    "couch":       (210, 130, 170),
    "chair":       (110, 200, 220),
    "tv_monitor":  (200, 220, 100),
    "tv":          (200, 220, 100),
    "monitor":     (200, 220, 100),
    "toilet":      (170, 200, 180),
    "plant":       (110, 220, 130),
    "potted plant": (110, 220, 130),
    "_default":    (160, 160, 160),
}


def category_color(cat: str) -> Tuple[int, int, int]:
    return CATEGORY_COLOR_BGR.get(cat.lower(), CATEGORY_COLOR_BGR["_default"])


def world_xz_to_pixel(
    xz: np.ndarray,
    origin_xz: Tuple[float, float],
    voxel_xz: float,
    img_h: int,
) -> np.ndarray:
    """Map world (x, z) to image (px, py) with z-axis flipped so 'up' on
    the image is the +x world direction (top-down convention)."""
    px = ((xz[..., 0] - origin_xz[0]) / voxel_xz).astype(int)
    py = img_h - 1 - ((xz[..., 1] - origin_xz[1]) / voxel_xz).astype(int)
    return np.stack([px, py], axis=-1)


def draw_object_footprints(
    img: np.ndarray,
    objects: List[Dict],
    floor_y: float,
    floor_band_m: float,
    origin_xz: Tuple[float, float],
    voxel_xz: float,
    skip_structural: bool = True,
) -> None:
    H, W = img.shape[:2]
    for obj in objects:
        if skip_structural and bool(obj.get("structural", False)):
            continue
        bb_min = np.asarray(obj["aabb_min"])
        bb_max = np.asarray(obj["aabb_max"])
        # Drop objects whose vertical centre is far from the active floor.
        cy = 0.5 * (bb_min[1] + bb_max[1])
        if abs(cy - floor_y) > floor_band_m + 0.5:
            continue
        rect = world_xz_to_pixel(
            np.array([[bb_min[0], bb_min[2]], [bb_max[0], bb_max[2]]]),
            origin_xz, voxel_xz, H,
        )
        x0, y0 = int(rect[0, 0]), int(rect[0, 1])
        x1, y1 = int(rect[1, 0]), int(rect[1, 1])
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W - 1, x1); y1 = min(H - 1, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        color = category_color(obj["category"])
        sub = img[y0:y1, x0:x1].copy()
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=-1)
        cv2.addWeighted(img[y0:y1, x0:x1], 0.35, sub, 0.65, 0, dst=img[y0:y1, x0:x1])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=1)


def draw_walls(
    img: np.ndarray,
    walls_voxel: np.ndarray,
    voxel_world_origin: Tuple[float, float, float],
    voxel_m: float,
    floor_y_idx: int,
    band_cells: int,
    image_origin_xz: Tuple[float, float],
    px_per_m: float,
) -> None:
    """Draw wall slabs at the active floor by collapsing the vertical band
    into a 2D mask, then overlaying a dark grey footprint."""
    Nx, Ny, Nz = walls_voxel.shape
    y0 = max(0, floor_y_idx - band_cells)
    y1 = min(Ny, floor_y_idx + band_cells + 1)
    band = walls_voxel[:, y0:y1, :]
    wall_xz = band.any(axis=1).astype(np.uint8) * 255  # (Nx, Nz)

    H, W = img.shape[:2]
    grid_w = int(round(Nx * voxel_m * px_per_m))
    grid_h = int(round(Nz * voxel_m * px_per_m))
    if grid_w <= 0 or grid_h <= 0:
        return
    wall_img = cv2.resize(wall_xz.T, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
    wall_img = np.flipud(wall_img)

    dx = int(round((voxel_world_origin[0] - image_origin_xz[0]) * px_per_m))
    dz = int(round((voxel_world_origin[2] - image_origin_xz[1]) * px_per_m))
    x0 = dx
    y0 = H - dz - grid_h
    x1 = x0 + grid_w
    y1 = y0 + grid_h
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(W, x1); dy1 = min(H, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    fw = dx1 - dx0; fh = dy1 - dy0
    mask = wall_img[sy0:sy0 + fh, sx0:sx0 + fw] > 0
    sub = img[dy0:dy1, dx0:dx1]
    sub[mask] = (60, 60, 60)  # dark grey walls


def project_top_down(
    field: np.ndarray,
    floor_y_idx: int,
    band_cells: int,
) -> np.ndarray:
    """Project (Nx, Ny, Nz) -> (Nx, Nz) by max over Y within the band.

    Returned image is oriented (rows = -z, cols = x) so a top-down view
    rotated to image space; the caller flips Y in ``world_xz_to_pixel``
    to keep both consistent.
    """
    Nx, Ny, Nz = field.shape
    y0 = max(0, floor_y_idx - band_cells)
    y1 = min(Ny, floor_y_idx + band_cells + 1)
    proj = field[:, y0:y1, :].max(axis=1)  # (Nx, Nz)
    return proj


def draw_field_overlay(
    img: np.ndarray,
    smoke_xz: np.ndarray,
    flame_xz: np.ndarray,
    voxel_xz_world: float,
    voxel_xz_image: float,
    image_origin_xz: Tuple[float, float],
    grid_origin_xz: Tuple[float, float],
) -> None:
    """Composite smoke+flame on top of the floor-plan image.

    The voxel grid uses ``voxel_xz_world`` per cell; the image uses
    ``voxel_xz_image`` per pixel. We resize the projection from voxel
    cells to pixel space using bilinear interpolation, then composite.
    """
    H, W = img.shape[:2]
    Nx, Nz = smoke_xz.shape
    grid_w = int(round(Nx * voxel_xz_world / voxel_xz_image))
    grid_h = int(round(Nz * voxel_xz_world / voxel_xz_image))
    if grid_w <= 0 or grid_h <= 0:
        return

    # cv2 expects (W, H) as dsize.
    smoke_img = cv2.resize(smoke_xz.T, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
    flame_img = cv2.resize(flame_xz.T, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR)
    # Flip vertically so +x in world maps to -y in image (top-down convention).
    smoke_img = np.flipud(smoke_img)
    flame_img = np.flipud(flame_img)

    # Place into image at offset corresponding to (grid_origin - image_origin).
    dx = int(round((grid_origin_xz[0] - image_origin_xz[0]) / voxel_xz_image))
    dz = int(round((grid_origin_xz[1] - image_origin_xz[1]) / voxel_xz_image))
    # Image y-axis is image_h-1 - z, so the destination top-left is:
    x0 = dx
    y0 = H - dz - grid_h
    x1 = x0 + grid_w
    y1 = y0 + grid_h

    # Clip.
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0);  dy0 = max(0, y0)
    dx1 = min(W, x1);  dy1 = min(H, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    fw = dx1 - dx0; fh = dy1 - dy0
    smoke_clip = smoke_img[sy0:sy0 + fh, sx0:sx0 + fw]
    flame_clip = flame_img[sy0:sy0 + fh, sx0:sx0 + fw]
    sub = img[dy0:dy1, dx0:dx1].astype(np.float32)

    # Smoke: alpha blend a gray colour.
    smoke_alpha = np.clip(smoke_clip, 0.0, 1.0)
    smoke_col = np.array([180, 180, 180], dtype=np.float32)
    sub = sub * (1.0 - smoke_alpha[..., None] * 0.85) + smoke_col * (smoke_alpha[..., None] * 0.85)

    # Flame: hot orange→yellow, additive.
    flame_alpha = np.clip(flame_clip, 0.0, 1.0)
    flame_col_lo = np.array([0, 60, 220], dtype=np.float32)   # BGR (deep orange)
    flame_col_hi = np.array([0, 220, 255], dtype=np.float32)  # BGR (bright yellow)
    flame_col = (
        (1.0 - flame_alpha[..., None]) * flame_col_lo
        + flame_alpha[..., None] * flame_col_hi
    )
    sub = sub * (1.0 - flame_alpha[..., None]) + flame_col * flame_alpha[..., None]

    img[dy0:dy1, dx0:dx1] = np.clip(sub, 0, 255).astype(np.uint8)


def draw_ignitions(
    img: np.ndarray,
    ignitions: List[Dict],
    origin_xz: Tuple[float, float],
    voxel_xz: float,
    floor_y: float,
    floor_band_m: float,
) -> None:
    H = img.shape[0]
    for i, ig in enumerate(ignitions):
        pos = ig["position"]
        if abs(pos[1] - floor_y) > floor_band_m + 0.5:
            continue
        px = int(round((pos[0] - origin_xz[0]) / voxel_xz))
        py = H - 1 - int(round((pos[2] - origin_xz[1]) / voxel_xz))
        cv2.drawMarker(img, (px, py), (40, 40, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
        cv2.circle(img, (px, py), 3, (40, 40, 255), -1)
        cv2.putText(img, f"#{i + 1}", (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2)
        cv2.putText(img, f"#{i + 1}", (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_header(img: np.ndarray, lines: List[str]) -> None:
    H, W = img.shape[:2]
    bar_h = 28
    cv2.rectangle(img, (0, 0), (W, bar_h), (30, 30, 30), -1)
    text = "  |  ".join(lines)
    cv2.putText(img, text, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def render_topdown_video(
    inventory: Dict,
    plan: Dict,
    timeline_npz: Path,
    out_dir: Path,
    px_per_m: float = 32.0,
    fps: float = 30.0,
    floor_band_m: float = 1.5,
    out_basename: str = "topdown",
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(timeline_npz, allow_pickle=True)
    flame_t = data["flame"].astype(np.float32)
    smoke_t = data["smoke"].astype(np.float32)
    times = data["times"].astype(np.float32)
    meta_blob = data["meta"][0] if "meta" in data.files else None
    meta = json.loads(meta_blob) if meta_blob else {}

    voxel_m = float(meta.get("voxel_m", 0.15))
    grid_origin = np.array(meta["origin"], dtype=np.float64) if "origin" in meta else \
        np.array(plan["world_aabb"][:3], dtype=np.float64)

    # Image origin: just take the world AABB so the floor plan covers the
    # full scene rather than only the voxel grid (which sometimes shrinks).
    bb_min = np.array(plan["world_aabb"][:3])
    bb_max = np.array(plan["world_aabb"][3:])
    image_origin_xz = (float(bb_min[0]), float(bb_min[2]))
    extent_x = float(bb_max[0] - bb_min[0])
    extent_z = float(bb_max[2] - bb_min[2])
    img_w = max(320, int(round(extent_x * px_per_m)))
    img_h = max(320, int(round(extent_z * px_per_m)))
    voxel_xz_image = 1.0 / px_per_m

    # Determine the active floor: average y of ignitions.
    floor_y = float(np.mean([ig["position"][1] for ig in plan["ignitions"]])) \
        if plan["ignitions"] else float(0.5 * (bb_min[1] + bb_max[1]))
    floor_y_idx = int(round((floor_y - grid_origin[1]) / voxel_m))
    floor_y_idx = int(np.clip(floor_y_idx, 0, flame_t.shape[2] - 1))
    band_cells = max(1, int(round(floor_band_m / voxel_m)))

    # Floor-plan background (constant across frames).
    floor_plan = np.full((img_h, img_w, 3), 245, dtype=np.uint8)
    # Walls (if available) form the strongest visual anchor for the floor
    # plan; draw them first so object footprints and overlays read above.
    walls_path = (inventory.get("structural") or {}).get("wall_voxel_path")
    if walls_path:
        try:
            walls_vox = np.load(walls_path)
            struct_origin = (inventory["structural"].get("origin")
                             or list(grid_origin))
            struct_voxel = float(inventory["structural"].get("voxel_m", voxel_m))
            wall_floor_y_idx = int(round((floor_y - struct_origin[1]) / struct_voxel))
            wall_band = max(1, int(round(floor_band_m / struct_voxel)))
            draw_walls(
                floor_plan, walls_vox,
                voxel_world_origin=tuple(struct_origin),
                voxel_m=struct_voxel,
                floor_y_idx=int(np.clip(wall_floor_y_idx, 0, walls_vox.shape[1] - 1)),
                band_cells=wall_band,
                image_origin_xz=image_origin_xz,
                px_per_m=px_per_m,
            )
        except Exception as e:
            print(f"[topdown_video] failed to draw walls from {walls_path}: {e}")

    # Object footprints: prefer schema-v2 ``instances`` (everything
    # inventory tracks), fall back to legacy ``objects``. Structural
    # items are rendered via draw_walls already, so we skip them here.
    items_for_footprints = (
        inventory.get("instances") or inventory.get("objects", [])
    )
    draw_object_footprints(floor_plan, items_for_footprints, floor_y, floor_band_m,
                           image_origin_xz, voxel_xz_image,
                           skip_structural=True)

    # Video writer.
    out_mp4 = out_dir / f"{out_basename}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4), fourcc, fps, (img_w, img_h + 28))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out_mp4}")

    n_frames = flame_t.shape[0]
    last_frame: Optional[np.ndarray] = None
    for fi in range(n_frames):
        frame = floor_plan.copy()
        smoke_xz = project_top_down(smoke_t[fi], floor_y_idx, band_cells)
        flame_xz = project_top_down(flame_t[fi], floor_y_idx, band_cells)
        draw_field_overlay(
            frame,
            smoke_xz, flame_xz,
            voxel_xz_world=voxel_m,
            voxel_xz_image=voxel_xz_image,
            image_origin_xz=image_origin_xz,
            grid_origin_xz=(float(grid_origin[0]), float(grid_origin[2])),
        )
        draw_ignitions(frame, plan["ignitions"], image_origin_xz, voxel_xz_image,
                       floor_y, floor_band_m)

        # Header banner above the floor plan.
        out_frame = np.zeros((img_h + 28, img_w, 3), dtype=np.uint8)
        out_frame[28:, :, :] = frame
        draw_header(out_frame, [
            f"scene {plan['scene_id']}",
            f"plan {plan['plan_id']}",
            f"{plan['fire_type']} / {plan['intensity']}",
            f"t = {float(times[fi]):.0f}s",
            f"floor y={floor_y:.2f} +/- {floor_band_m:.1f}m",
        ])
        writer.write(out_frame)
        last_frame = out_frame

    writer.release()

    # Final PNG snapshot.
    out_png = out_dir / f"{out_basename}.png"
    if last_frame is not None:
        cv2.imwrite(str(out_png), last_frame)

    # Tiny summary JSON for reviewers.
    summary = {
        "scene_id": plan["scene_id"],
        "plan_id": plan["plan_id"],
        "fire_type": plan["fire_type"],
        "intensity": plan["intensity"],
        "n_frames": int(n_frames),
        "duration_s": float(plan.get("duration_s", float(times[-1]) if len(times) else 0.0)),
        "fps": float(fps),
        "speedup": float(fps * float(meta.get("save_dt", 1.0))),
        "px_per_m": float(px_per_m),
        "image_size": [int(img_w), int(img_h + 28)],
        "floor_y": float(floor_y),
        "floor_band_m": float(floor_band_m),
        "out_mp4": str(out_mp4),
        "out_png": str(out_png),
    }
    (out_dir / f"{out_basename}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Render a top-down validation video for a fire scenario."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--plan_id", required=True)
    parser.add_argument("--scenes_root", default="scenes")
    parser.add_argument("--out_root", default="outputs/fire_world")
    parser.add_argument("--px_per_m", type=float, default=32.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--floor_band_m", type=float, default=1.5)
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    inv = json.loads((scenes_root / args.scene / "inventory.json").read_text())
    plan = json.loads(
        (scenes_root / args.scene / "plans" / f"{args.plan_id}.json").read_text()
    )
    out_dir = Path(args.out_root) / args.scene / args.plan_id
    timeline = out_dir / "timeline.npz"
    if not timeline.exists():
        raise FileNotFoundError(
            f"timeline.npz not found at {timeline}. Run propagation first."
        )
    res = render_topdown_video(
        inv, plan, timeline, out_dir,
        px_per_m=args.px_per_m, fps=args.fps, floor_band_m=args.floor_band_m,
    )
    print(f"[topdown_video] wrote {res['out_mp4']}")
    print(f"  size={res['image_size']}  frames={res['n_frames']}  "
          f"speedup={res['speedup']:.1f}x  duration={res['duration_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
