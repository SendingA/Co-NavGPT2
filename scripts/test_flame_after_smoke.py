"""End-to-end check: HSV flame detector still fires on smoke-attenuated RGB.

Pipeline: synthetic flame -> SmokeRGBSensor (dense smoke) -> detect_flames_hsv.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.detection_segmentation import detect_flames_hsv  # noqa: E402
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
    return rgb, depth


def main() -> int:
    rgb, depth = make_scene()
    cfg = FireSensorConfig(max_depth_m=5.0, smoke_density=0.9)
    sensor = SmokeRGBSensor(cfg, np.random.default_rng(0))
    smoky = sensor.process(rgb, depth)["image"]

    boxes, masks, scores = detect_flames_hsv(smoky)
    print(f"detected {len(boxes)} flame region(s) in smoky RGB")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"  [{i}] xyxy={b.tolist()} score={float(s):.3f}")

    if len(boxes) == 0:
        print("FAIL: HSV detector lost the flame after smoke filtering")
        return 1
    print("OK: flame still detectable under dense smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
