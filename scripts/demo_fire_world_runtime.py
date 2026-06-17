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


def pick_camera_path(plan: dict, n: int = 6,
                     radius: float = 2.5, eye_height_off: float = 1.0,
                     look_height_off: float = 0.0,
                     world_aabb: np.ndarray = None) -> np.ndarray:
    """Sample ``n`` waypoints around the first ignition.

    The camera is placed on a circle of ``radius`` metres around the
    ignition, ``eye_height_off`` metres above its centre. ``look_height_off``
    biases the look-at target above/below the ignition (positive looks
    upward toward the ceiling plume).

    If ``world_aabb`` is supplied, eye points are clamped to stay inside
    the building so the ray-march does not start from outside the scene.
    """
    ig = plan["ignitions"][0]
    pos = np.asarray(ig["position"], dtype=np.float64)
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    waypoints = []
    for a in angles:
        eye = pos + np.array([radius * np.cos(a), eye_height_off, radius * np.sin(a)])
        if world_aabb is not None:
            margin = 0.5
            eye = np.maximum(eye, np.asarray(world_aabb[:3]) + margin)
            eye = np.minimum(eye, np.asarray(world_aabb[3:]) - margin)
        waypoints.append(eye)
    targets = np.tile(pos + np.array([0.0, look_height_off, 0.0]), (n, 1))
    return np.asarray(waypoints), targets


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
    p.add_argument("--radius_m", type=float, default=2.5,
                   help="Distance of the orbiting camera from the first ignition.")
    p.add_argument("--eye_height_off", type=float, default=0.6,
                   help="Camera height above the first ignition (m). 0.6 m "
                        "puts a typical robot eye at ~1.4 m floor height.")
    p.add_argument("--look_up_off", type=float, default=0.6,
                   help="Bias the look-at target above the ignition (m); "
                        "increase this to look at the ceiling plume.")
    p.add_argument("--t_sim", type=float, default=120.0,
                   help="Simulation time (s) at which to render. Use a "
                        "negative value to render at peak smoke (auto).")
    p.add_argument("--time_lapse", type=int, default=0,
                   help="If >0, render this many frames at evenly spaced "
                        "times instead of an orbiting camera. Useful to "
                        "see fire propagation from a fixed viewpoint.")
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
    inv = json.loads((scenes_root / args.scene / "inventory.json").read_text())
    eyes, targets = pick_camera_path(
        plan, n=args.n_views,
        radius=args.radius_m,
        eye_height_off=args.eye_height_off,
        look_height_off=args.look_up_off,
        world_aabb=np.asarray(inv["world_aabb"]),
    )

    out_dir = Path(args.out_root) / args.scene / args.plan_id / "runtime_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_clean = synthetic_clean_rgb(args.height, args.width)
    depth_const = np.full((args.height, args.width), args.max_depth_m, dtype=np.float32)

    # ------------------------------------------------------------------
    # Time-lapse mode: a single fixed camera, sweep over simulation time.
    # ------------------------------------------------------------------
    if args.time_lapse > 0:
        # Fixed eye = first orbit position so the path is reproducible.
        eye = eyes[0]
        target = targets[0]
        R = look_at(eye, target)
        ts_render = np.linspace(float(fw.times[0]),
                                float(fw.times[-1]),
                                int(args.time_lapse))
        print(f"[demo] time-lapse: {len(ts_render)} frames, "
              f"t in [{ts_render[0]:.0f}, {ts_render[-1]:.0f}]s")
        panels = []
        t0 = time.time()
        for i, t in enumerate(ts_render):
            out = renderer.render(
                rgb_clean=rgb_clean,
                depth_m=depth_const,
                cam_pos_world=eye,
                R_cam2world=R,
                t_sim=float(t),
            )
            rgb_smoky = out["image"]
            therm = out["thermal_image"]
            T_mean = float(out["transmittance"].mean())
            flame_frac = float(out["flame_mask"].mean())
            label = (f"t={t:.0f}s  T_mean={T_mean:.2f}  flame={flame_frac:.2%}")
            cv2.putText(rgb_smoky, label, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2)
            cv2.putText(rgb_smoky, label, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
            cv2.imwrite(str(out_dir / f"timelapse_{i:03d}_rgb.png"),
                        cv2.cvtColor(rgb_smoky, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"timelapse_{i:03d}_thermal.png"), therm)
            panels.append(np.hstack([cv2.cvtColor(rgb_smoky, cv2.COLOR_RGB2BGR), therm]))
            print(f"  t={t:>5.0f}s  T_mean={T_mean:.2f} flame={flame_frac:.2%}")
        # Stitch into one tall image and an mp4.
        grid = np.vstack(panels)
        cv2.imwrite(str(out_dir / "timelapse_grid.png"), grid)
        h, w = panels[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out_dir / "timelapse.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (w, h),
        )
        for fr in panels:
            writer.write(fr)
        writer.release()
        elapsed = time.time() - t0
        print(f"[demo] time-lapse done in {elapsed:.1f}s -> "
              f"{out_dir}/timelapse.mp4 (+ per-frame PNGs)")
        return 0

    # ------------------------------------------------------------------
    # Orbit mode: many cameras around the fire at one fixed time.
    # ------------------------------------------------------------------

    print(f"[demo] rendering {len(eyes)} views @ t={t_sim:.0f}s "
          f"-> {out_dir}")
    panels = []
    t0 = time.time()
    for i, eye in enumerate(eyes):
        target = targets[i]
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
