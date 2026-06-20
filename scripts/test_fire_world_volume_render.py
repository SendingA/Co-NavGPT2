"""Verify the volumetric flame renderer produces a 'hot core / cool edge'
colour gradient (instead of the old monolithic neon block).

Concretely:
  * stretch a single ignition over a small voxel grid,
  * render a head-on camera into a clean flat image,
  * sample the centre pixel (should be near-white emission core),
  * sample a pixel on the flame edge (should be deep orange / red),
  * sample a pixel outside the flame (should be ~ rgb_clean).

This catches regressions where the renderer goes back to a single
flat colour or where trilinear sampling silently flips to nearest.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.runtime import FireWorld, FireWorldRenderer  # noqa: E402


def make_synth_world(voxel_m: float = 0.10):
    """A 4x4x4 m air block with a Gaussian flame blob at the centre."""
    Nx = Ny = Nz = 40
    flame = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    smoke = np.zeros_like(flame)
    temp = np.full_like(flame, 25.0)
    cx, cy, cz = Nx // 2, Ny // 2, Nz // 2
    xx, yy, zz = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij")
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    flame = np.exp(-(r / 4.0) ** 2)               # Gaussian-shaped flame
    flame = (flame / flame.max()).astype(np.float32)
    smoke = np.exp(-(r / 8.0) ** 2) * 0.4         # softer smoke around it
    smoke = smoke.astype(np.float32)
    temp = (25.0 + 800.0 * flame).astype(np.float32)

    fw = FireWorld(
        flame=flame[None].astype(np.float16),
        smoke=smoke[None].astype(np.float16),
        temp=temp[None].astype(np.float16),
        times=np.array([0.0], dtype=np.float32),
        voxel_m=voxel_m,
        origin=np.array([-Nx * voxel_m / 2,
                         -Ny * voxel_m / 2,
                         -Nz * voxel_m / 2], dtype=np.float64),
        shape=(Nx, Ny, Nz),
        ambient_c=25.0,
        scene_id="synth",
        plan_id="synth",
    )
    return fw, (Nx, Ny, Nz)


def main() -> int:
    fw, _ = make_synth_world()
    H, W = 256, 256
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    fx = (W / 2.0) / np.tan(np.deg2rad(60.0 / 2.0))
    K = SimpleNamespace(cx=cx, cy=cy, fx=fx, fy=fx)

    renderer = FireWorldRenderer(
        fw=fw, camera_K=K, max_depth_m=6.0, n_steps=64,
        smoke_k_ext=2.0, flame_emission_gain=4.0,
    )
    cam_pos = np.array([0.0, 0.0, 4.0], dtype=np.float32)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    rgb_clean = np.full((H, W, 3), 80, dtype=np.uint8)
    depth = np.full((H, W), 6.0, dtype=np.float32)

    out = renderer.render(rgb_clean, depth, cam_pos, R, t_sim=0.0)
    img = out["image"]

    # Centre 4x4 patch at flame core.
    cx_px, cy_px = W // 2, H // 2
    core = img[cy_px - 2:cy_px + 2, cx_px - 2:cx_px + 2].mean(axis=(0, 1))
    # Edge: 30 px out from centre.
    edge = img[cy_px - 2:cy_px + 2, cx_px + 28:cx_px + 32].mean(axis=(0, 1))
    # Background: corner 16x16 patch.
    bg = img[:16, :16].mean(axis=(0, 1))

    print(f"core RGB={core.round(1).tolist()}  "
          f"edge RGB={edge.round(1).tolist()}  "
          f"bg RGB={bg.round(1).tolist()}")
    print(f"T_mean={float(out['transmittance'].mean()):.3f}")

    # Core should be brighter than edge, edge brighter than bg.
    if core.mean() <= edge.mean():
        print("FAIL: flame core is not brighter than its edge")
        return 1
    if edge.mean() <= bg.mean():
        print("FAIL: flame edge is not brighter than the background")
        return 1
    # Core should be warm-white (R, G, B all high).
    if not (core[0] > 200 and core[1] > 150 and core[2] > 80):
        print("FAIL: flame core is not warm-white "
              f"(expected R>200,G>150,B>80, got {core})")
        return 1
    # Edge should be redder than greener.
    if edge[0] <= edge[1]:
        print("FAIL: flame edge is not red-dominated "
              f"(R={edge[0]} G={edge[1]})")
        return 1
    print("flame core / edge / bg gradient: OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
