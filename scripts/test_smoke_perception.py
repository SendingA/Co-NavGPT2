"""Smoke tests for utils.smoke_perception.

Three things are covered end-to-end:

1. ``apply_clean_depth_and_thermal`` patches ``observations`` correctly:
   - smoky RGB replaces obs['rgb'],
   - clean depth (Habitat raw) is preserved when use_clean_depth=True,
   - thermal_image / thermal_flame_mask are injected.

2. ``dehaze_with_depth`` recovers a non-fog RGB whose mean colour is no
   longer dominated by the smoke gray.

3. ``thermal_mask_to_detections`` produces YOLO-shaped outputs that can be
   merged into ``Object_Detection_and_Segmentation``.

Run with::

    python scripts/test_smoke_perception.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.fire_sensors.config import FireSensorConfig  # noqa: E402
from utils.fire_sensors.sensors.rgb_smoke import SmokeRGBSensor  # noqa: E402
from utils.fire_sensors.sensors.thermal import ThermalSensor  # noqa: E402
from utils.smoke_perception import (  # noqa: E402
    apply_clean_depth_and_thermal,
    dehaze_with_depth,
    thermal_mask_to_detections,
)


def make_scene(h: int = 240, w: int = 320, depth_m: float = 4.0):
    rgb = np.full((h, w, 3), 80, dtype=np.uint8)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    flame = (yy - cy) ** 2 + (xx - cx) ** 2 < 30 ** 2
    rgb[flame] = (245, 90, 20)
    core = (yy - cy) ** 2 + (xx - cx) ** 2 < 10 ** 2
    rgb[core] = (255, 240, 150)
    depth = np.full((h, w), depth_m, dtype=np.float32)
    return rgb, depth


def test_apply_helper() -> None:
    rgb, depth = make_scene()
    cfg = FireSensorConfig(max_depth_m=5.0, smoke_density=0.7)
    rng = np.random.default_rng(0)
    smoke = SmokeRGBSensor(cfg, rng).process(rgb, depth)
    therm = ThermalSensor(cfg, rng).process(rgb, depth)
    sensors = {
        "rgb_smoke": smoke["image"],
        "depth_smoke": depth + rng.normal(0, 0.1, depth.shape).astype(np.float32),
        "thermal_image": therm["image"],
        "thermal_flame_mask": therm["flame_mask"],
    }

    obs = {
        "rgb": rgb,
        "depth": (depth / 5.0)[..., None].astype(np.float32),
    }
    clean_raw = obs["depth"].copy()

    apply_clean_depth_and_thermal(
        obs,
        sensors,
        clean_depth_raw=clean_raw,
        use_clean_depth=True,
        use_thermal=True,
        apply_smoky_rgb=True,
        normalize_depth=True,
        max_depth_m=5.0,
    )
    assert np.array_equal(obs["rgb"], sensors["rgb_smoke"]), "rgb not patched"
    assert np.array_equal(obs["depth"], clean_raw), "clean depth must be preserved"
    assert "thermal_flame_mask" in obs, "thermal mask not injected"
    assert obs["thermal_flame_mask"].sum() > 0, "thermal mask is empty"
    print("apply_clean_depth_and_thermal: OK")


def test_dehaze() -> None:
    rgb, depth = make_scene()
    cfg = FireSensorConfig(max_depth_m=5.0, smoke_density=0.7)
    smoke = SmokeRGBSensor(cfg, np.random.default_rng(0)).process(rgb, depth)
    smoky = smoke["image"]

    # Dehazed RGB: a non-flame patch should drift away from the smoke
    # gray color towards the dim background.
    dehazed = dehaze_with_depth(smoky, depth)
    bg_box = (slice(0, 40), slice(0, 40))
    smoky_bg = smoky[bg_box].astype(np.float32).mean(axis=(0, 1))
    dehazed_bg = dehazed[bg_box].astype(np.float32).mean(axis=(0, 1))
    print(f"smoky bg RGB={smoky_bg.round(1)} -> dehazed bg RGB={dehazed_bg.round(1)}")

    # The smoky background sits near (180,180,180); after dehaze it should
    # come down towards the original (80,80,80).
    assert dehazed_bg.mean() < smoky_bg.mean() - 20, \
        "dehazing failed to remove smoke gray"
    print("dehaze_with_depth: OK")


def test_thermal_to_detections() -> None:
    mask = np.zeros((240, 320), dtype=np.float32)
    mask[100:140, 150:190] = 1.0
    boxes, masks, scores = thermal_mask_to_detections(mask, target_hw=(240, 320))
    assert boxes.shape == (1, 4) and scores.shape == (1,)
    assert masks.shape == (1, 240, 320)
    assert int(boxes[0][0]) == 150 and int(boxes[0][2]) == 190
    print(f"thermal -> {boxes.tolist()} score {scores.tolist()}")
    print("thermal_mask_to_detections: OK")


def main() -> int:
    test_apply_helper()
    test_dehaze()
    test_thermal_to_detections()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
