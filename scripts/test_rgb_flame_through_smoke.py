"""Smoke test: SmokeRGBSensor must keep flame pixels visible through smoke.

Builds a synthetic dim background with a bright orange flame disk and a
yellow core at 4 m depth, runs SmokeRGBSensor at high smoke density, and
checks that the mean luminance inside the flame region is significantly
higher than the surrounding smoky background.

Run with::

    python scripts/test_rgb_flame_through_smoke.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.fire_sensors.config import FireSensorConfig  # noqa: E402
from utils.fire_sensors.sensors.rgb_smoke import SmokeRGBSensor  # noqa: E402


def make_scene(h: int = 240, w: int = 320, depth_m: float = 4.0):
    rgb = np.full((h, w, 3), 60, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    flame = (yy - cy) ** 2 + (xx - cx) ** 2 < 30 ** 2
    rgb[flame] = (245, 90, 20)
    core = (yy - cy) ** 2 + (xx - cx) ** 2 < 10 ** 2
    rgb[core] = (255, 240, 150)
    depth = np.full((h, w), depth_m, dtype=np.float32)
    return rgb, depth, flame


def main() -> int:
    rgb, depth, flame_bool = make_scene()

    cfg = FireSensorConfig(max_depth_m=5.0, smoke_density=0.9)
    sensor = SmokeRGBSensor(cfg, np.random.default_rng(0))
    out = sensor.process(rgb, depth)

    img = out["image"].astype(np.float32)
    flame_rgb = img[flame_bool].mean(axis=0)
    bg_rgb = img[~flame_bool].mean(axis=0)
    print(f"flame mean RGB={flame_rgb.round(1)}")
    print(f"bg    mean RGB={bg_rgb.round(1)}")

    # Without flame penetration, density=0.9 at 4 m drives transmittance to
    # ~2.5e-4 and the flame region collapses to the gray smoke color
    # (≈180,180,180). Our fix must keep the flame region clearly orange:
    # red >> green and red >> blue, and red noticeably higher than smoke red.
    if flame_rgb[0] - flame_rgb[2] < 30:
        print("FAIL: flame region lost its red dominance under smoke")
        return 1
    if flame_rgb[0] - bg_rgb[0] < 15:
        print("FAIL: flame red channel not brighter than smoky background")
        return 1
    if "flame_mask" not in out or out["flame_mask"].sum() <= 0:
        print("FAIL: missing or empty flame_mask")
        return 1

    # Sanity: at density=0 the output should be ~identical to the input.
    cfg0 = FireSensorConfig(max_depth_m=5.0, smoke_density=0.0)
    sensor0 = SmokeRGBSensor(cfg0, np.random.default_rng(0))
    out0 = sensor0.process(rgb, depth)
    diff = np.abs(out0["image"].astype(np.int16) - rgb.astype(np.int16)).mean()
    if diff > 1.0:
        print(f"FAIL: density=0 should leave RGB unchanged, mean diff={diff}")
        return 1
    print("OK: flame visible through dense smoke; clean pass-through preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
