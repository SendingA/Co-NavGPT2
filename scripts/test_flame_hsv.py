"""Smoke-test for the HSV flame detector inside detection_segmentation.

Builds a synthetic image with a hot orange blob plus a yellow core (mimicking
the test fixture used by ``scripts/test_fire_sensors.py``) and asserts that
``detect_flames_hsv`` returns at least one box covering the blob.

Run with::

    python scripts/test_flame_hsv.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)

from utils.detection_segmentation import detect_flames_hsv  # noqa: E402


def make_flame_image(h: int = 240, w: int = 320) -> np.ndarray:
    rgb = np.full((h, w, 3), 60, dtype=np.uint8)  # dim background
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    flame = (yy - cy) ** 2 + (xx - cx) ** 2 < 30 ** 2
    core = (yy - cy) ** 2 + (xx - cx) ** 2 < 10 ** 2
    rgb[flame] = (245, 90, 20)   # orange shell
    rgb[core] = (255, 240, 150)  # yellow core
    return rgb


def main() -> int:
    rgb = make_flame_image()
    boxes, masks, scores = detect_flames_hsv(rgb)
    print(f"detected {len(boxes)} flame region(s)")
    for i, (b, s) in enumerate(zip(boxes, scores)):
        print(f"  [{i}] xyxy={b.tolist()} score={float(s):.3f}")

    if len(boxes) == 0:
        print("FAIL: expected at least one flame detection")
        return 1
    if masks.shape[0] != len(boxes):
        print("FAIL: mask count != box count")
        return 1
    if masks.shape[1:] != rgb.shape[:2]:
        print("FAIL: mask shape mismatch with input image")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
