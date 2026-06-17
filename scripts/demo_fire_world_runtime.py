"""End-to-end demo: a synthetic robot 'walks' through a fire scenario and
its RGB / Thermal observations are composed from the FireWorld voxel
timeline (stage 5 hook)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.runtime import FireWorld, FireWorldRenderer  # noqa: E402


def synthetic_clean_rgb(h: int, w: int) -> np.ndarray:
    """A simple "clean" reference image so we can see how smoke degrades it.

    Uses an image gradient so the agent's RGB is non-uniform; a real run
    would feed Habitat's clean RGB sensor here.
    """
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    rgb = np.stack([
        (200 - 80 * yy + 40 * xx),
        (180 - 30 * yy + 80 * xx),
        (160 + 60 * yy + 20 * xx),
    ], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def make_camera_K(w: int, h: int, hfov_deg: float = 79.0) -> SimpleNamespace:
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    fx = (w / 2.0) / np.tan(np.deg2rad(hfov_deg / 2.0))
    return SimpleNamespace(cx=cx, cy=cy, fx=fx, fy=fx)


def look_at(eye: np.ndarray, target: np.ndarray, up=np.array([0.0, 1.0, 0.0])):
    """Right-handed look-at: returns ``R_cam2world`` so that the camera
    looks along (target - eye) with +Y_world as roughly up.
    """
    fwd = target - eye
    fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
    right = np.cross(fwd, up)
    right = right / max(np.linalg.norm(right), 1e-6)
    cam_up = np.cross(right, fwd)
    # Camera convention: +X right, +Y up, -Z forward.
    R = np.stack([right, cam_up, -fwd], axis=1)  # columns
    return R


def pick_camera_path(plan: dict, n: int = 6) -> np.ndarray:
    """Sample ``n`` waypoints around the first ignition for visual variety."""
    ig = plan["ignitions"][0]
    pos = np.asarray(ig["position"], dtype=np.float64)
    # Circular path at fixed radius around the ignition, eye height 1.5 m.
    radius = 2.5
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    waypoints = []
    for a in angles:
        eye = pos + np.array([radius * np.cos(a), 0.5, radius * np.sin(a)])
        eye[1] = pos[1] + 0.6
        waypoints.append(eye)
    return np.asarray(waypoints)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--plan_id", required=True)
    p.add_argument("--scenes_root", default="scenes")
    p.add_argument("--out_root", default="outputs/fire_world")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--hfov", type=float, default=79.0)
    p.add_argument("--max_depth_m", type=float, default=5.0)
    p.add_argument("--n_steps", type=int, default=24)
    p.add_argument("--smoke_k_ext", type=float, default=1.5)
    p.add_argument("--n_views", type=int, default=8)
    p.add_argument("--t_sim", type=float, default=120.0,
                   help="Simulation time (s) at which to render. Use a "
                        "negative value to render at peak smoke (auto).")
    args = p.parse_args(argv)

    scenes_root = Path(args.scenes_root)
    plan = json.loads(
        (scenes_root / args.scene / "plans" / f"{args.plan_id}.json").read_text()
    )
    fw = FireWorld.load(args.scene, args.plan_id, out_root=Path(args.out_root))
    K = make_camera_K(args.width, args.height, hfov_deg=args.hfov)

    renderer = FireWorldRenderer(
        fw=fw,
        camera_K=K,
        max_depth_m=args.max_depth_m,
        n_steps=args.n_steps,
        smoke_k_ext=args.smoke_k_ext,
    )

    # Pick a t_sim near peak smoke if the user didn't pin one.
    if args.t_sim < 0:
        smoke_total = fw.smoke.astype(np.float32).reshape(fw.smoke.shape[0], -1).sum(axis=1)
        t_sim = float(fw.times[int(np.argmax(smoke_total))])
        print(f"[demo] auto t_sim={t_sim:.0f}s (peak smoke)")
    else:
        t_sim = float(args.t_sim)

    # Build a circular camera path around the first ignition.
    eyes = pick_camera_path(plan, n=args.n_views)
    target = np.asarray(plan["ignitions"][0]["position"], dtype=np.float64)

    out_dir = Path(args.out_root) / args.scene / args.plan_id / "runtime_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_clean = synthetic_clean_rgb(args.height, args.width)
    depth_const = np.full((args.height, args.width), args.max_depth_m, dtype=np.float32)

    print(f"[demo] rendering {len(eyes)} views @ t={t_sim:.0f}s "
          f"-> {out_dir}")
    panels = []
    t0 = time.time()
    for i, eye in enumerate(eyes):
        R = look_at(eye, target)
        out = renderer.render(
            rgb_clean=rgb_clean,
            depth_m=depth_const,
            cam_pos_world=eye,
            R_cam2world=R,
            t_sim=t_sim,
        )
        rgb_smoky = out["image"]
        therm = out["thermal_image"]

        # Annotate transmittance and ignition distance.
        T_mean = float(out["transmittance"].mean())
        flame_frac = float(out["flame_mask"].mean())
        d_to_fire = float(np.linalg.norm(eye - target))
        cv2.putText(rgb_smoky, f"view {i + 1}/{len(eyes)}  "
                    f"d_to_fire={d_to_fire:.1f}m  "
                    f"T_mean={T_mean:.2f}  "
                    f"flame={flame_frac:.2%}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
        cv2.putText(rgb_smoky, f"view {i + 1}/{len(eyes)}  "
                    f"d_to_fire={d_to_fire:.1f}m  "
                    f"T_mean={T_mean:.2f}  "
                    f"flame={flame_frac:.2%}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)

        cv2.imwrite(str(out_dir / f"view_{i:02d}_rgb.png"),
                    cv2.cvtColor(rgb_smoky, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"view_{i:02d}_thermal.png"), therm)
        panels.append(np.hstack([cv2.cvtColor(rgb_smoky, cv2.COLOR_RGB2BGR), therm]))
        print(f"  view {i + 1}: T_mean={T_mean:.2f} flame_px={flame_frac:.2%}")
    elapsed = time.time() - t0

    grid = np.vstack(panels)
    cv2.imwrite(str(out_dir / "all_views.png"), grid)
    print(f"[demo] {len(eyes)} views rendered in {elapsed:.1f}s")
    print(f"[demo] saved: {out_dir}/view_*.png and all_views.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
