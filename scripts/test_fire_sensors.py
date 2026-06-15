"""Smoke-test for utils.fire_sensors without booting Habitat.

Two scenarios:

1. Forward-only depth (legacy path).
2. Synthetic 4 yaw-rotated depth slices that mimic what the
   ``lidar_depth_*`` sensors would produce in Habitat. Used to verify
   the 360° stitching logic.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.fire_sensors import FireSensorConfig, FireSensorSuite  # noqa: E402
from utils.fire_sensors.lidar_360 import LIDAR_DEPTH_UUIDS  # noqa: E402


def make_scene(h: int = 240, w: int = 320):
    rgb = np.full((h, w, 3), 110, dtype=np.uint8)
    depth = np.linspace(0.5, 4.5, h, dtype=np.float32)[:, None]
    depth = np.repeat(depth, w, axis=1)
    cy, cx = h // 2 + 30, w // 2 - 20
    yy, xx = np.ogrid[:h, :w]
    flame = (yy - cy) ** 2 + (xx - cx) ** 2 < 25 ** 2
    rgb[flame] = (245, 90, 20)
    core = (yy - cy) ** 2 + (xx - cx) ** 2 < 8 ** 2
    rgb[core] = (255, 240, 150)
    return rgb, depth


def make_360_obs(side: int = 160, max_depth: float = 5.0):
    """Build a fake observation dict with the 4 lidar_depth_* slices.

    Each slice is a different uniform distance so the stitched cloud
    clearly spans the full 360°. Depth is normalised to [0, 1] to match
    NORMALIZE_DEPTH=True.
    """
    distances_m = {
        "lidar_depth_front": 2.0,
        "lidar_depth_left":  3.0,
        "lidar_depth_back":  1.5,
        "lidar_depth_right": 4.0,
    }
    obs = {}
    for uuid, d in distances_m.items():
        slice_norm = np.full((side, side, 1), d / max_depth, dtype=np.float32)
        obs[uuid] = slice_norm
    return obs, max_depth


def main() -> int:
    rgb, depth = make_scene()
    cfg = FireSensorConfig(
        max_depth_m=5.0,
        hfov_deg=79.0,
        smoke_density=0.7,
        save_npz=True,
    )

    # ---- Scenario 1: forward-only depth -----------------------------
    out_dir1 = os.path.join(ROOT, "outputs", "fire_sensors_smoke_test")
    suite1 = FireSensorSuite(cfg=cfg, dump_dir=out_dir1, save_every=1, seed=0)
    out1 = suite1.process(rgb, depth)
    suite1.save_step(out1, episode=0, step=0, agent_id=0)
    print(f"[fwd]  LIDAR pts={out1['lidar_points'].shape[0]} "
          f"(is_360 ~ False)")

    # ---- Scenario 2: 360° via 4 lidar_depth_* slices ----------------
    obs, max_d = make_360_obs(side=128, max_depth=5.0)
    cfg2 = FireSensorConfig(
        max_depth_m=max_d,
        hfov_deg=79.0,
        smoke_density=0.0,           # turn smoke off so we measure stitching
        save_npz=False,
    )
    out_dir2 = os.path.join(ROOT, "outputs", "fire_sensors_smoke_test_360")
    suite2 = FireSensorSuite(cfg=cfg2, dump_dir=out_dir2, save_every=1, seed=0)
    out2 = suite2.process(rgb, depth, obs=obs)
    suite2.save_step(out2, episode=0, step=0, agent_id=0)

    pts = out2["lidar_points"]
    n = pts.shape[0]
    print(f"[360]  LIDAR pts={n}")

    # Sanity: cloud should cover roughly all 4 quadrants in (X, Y).
    quadrants = {
        "+X+Y": int(((pts[:, 0] > 0.3) & (pts[:, 1] > 0.3)).sum()),
        "+X-Y": int(((pts[:, 0] > 0.3) & (pts[:, 1] < -0.3)).sum()),
        "-X+Y": int(((pts[:, 0] < -0.3) & (pts[:, 1] > 0.3)).sum()),
        "-X-Y": int(((pts[:, 0] < -0.3) & (pts[:, 1] < -0.3)).sum()),
    }
    print("  quadrant coverage:", quadrants)
    assert all(v > 0 for v in quadrants.values()), \
        f"Stitched 360° cloud missing quadrants: {quadrants}"

    # Sanity: distance-to-origin should match the synthetic distances.
    rng = np.linalg.norm(pts, axis=1)
    print(f"  range stats: min={rng.min():.2f}m  med={np.median(rng):.2f}m  max={rng.max():.2f}m")

    expected_files = [
        "step_00000_rgb.png",
        "step_00000_rgb_smoke.png",
        "step_00000_depth.png",
        "step_00000_depth_smoke.png",
        "step_00000_thermal.png",
        "step_00000_lidar_bev.png",
        "step_00000_radar_bev.png",
        "step_00000_radar_az.png",
        "step_00000_radar_el.png",
        "step_00000_dashboard.png",
    ]
    for f in expected_files:
        path1 = os.path.join(out_dir1, "ep_0000", "agent_0", f)
        path2 = os.path.join(out_dir2, "ep_0000", "agent_0", f)
        assert os.path.isfile(path1), f"missing {path1}"
        assert os.path.isfile(path2), f"missing {path2}"

    print("All assertions passed (forward fallback + 360° stitching).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
