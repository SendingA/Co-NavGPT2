"""Sanity test: smoky RGB has soft, low-frequency texture (not TV snow).

We feed a uniform wall + flame + depth=4 m through the smoke sensor and
compare the high-frequency residual (image minus a Gaussian blur) under:
  * the new spatial-turbulence path (default config), and
  * a control with turbulence and low-freq noise both disabled.

The new path must keep most of the energy in the **low** frequencies, i.e.
high-pass residual std should be small relative to the overall image std.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.fire_sensors.config import FireSensorConfig  # noqa: E402
from utils.fire_sensors.sensors.rgb_smoke import SmokeRGBSensor  # noqa: E402


def make_scene(h=240, w=320, depth_m=4.0):
    rgb = np.full((h, w, 3), 80, dtype=np.uint8)
    cy, cx = 60, 80
    yy, xx = np.ogrid[:h, :w]
    flame = (yy - cy) ** 2 + (xx - cx) ** 2 < 22 ** 2
    rgb[flame] = (245, 90, 20)
    core = (yy - cy) ** 2 + (xx - cx) ** 2 < 8 ** 2
    rgb[core] = (255, 240, 150)
    depth = np.full((h, w), depth_m, dtype=np.float32)
    return rgb, depth


def hf_ratio(img: np.ndarray, region=None) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(g, (15, 15), 0)
    hf = g - blur
    if region is not None:
        hf = hf[region]
        g = g[region]
    return float(hf.std()) / max(float(g.std()), 1e-6)


def main() -> int:
    rgb, depth = make_scene()

    cfg_new = FireSensorConfig(max_depth_m=5.0, smoke_density=0.7)
    sensor_new = SmokeRGBSensor(cfg_new, np.random.default_rng(0))
    out_new = sensor_new.process(rgb, depth)["image"]

    # Control: kill turbulence + low-freq noise to isolate the structure.
    cfg_ctrl = FireSensorConfig(max_depth_m=5.0, smoke_density=0.7)
    cfg_ctrl.smoke.smoke_turbulence_strength = 0.0
    cfg_ctrl.smoke.smoke_lowfreq_noise_std = 0.0
    sensor_ctrl = SmokeRGBSensor(cfg_ctrl, np.random.default_rng(0))
    out_ctrl = sensor_ctrl.process(rgb, depth)["image"]

    # Measure HF outside the flame region (the flame edge is a real strong
    # gradient and would inflate the metric for the wrong reason).
    wall = (slice(120, 240), slice(150, 320))
    r_new = hf_ratio(out_new, region=wall)
    r_ctrl = hf_ratio(out_ctrl, region=wall)
    # Reference: emulate the OLD pixel-wise iid noise behaviour by adding
    # white noise to the control image. The old sigma was ~4*density (=2.8).
    rng_ref = np.random.default_rng(0)
    out_old_emu = (
        out_ctrl.astype(np.float32)
        + rng_ref.normal(0.0, 4.0 * 0.7, size=out_ctrl.shape)
    )
    out_old_emu = np.clip(out_old_emu, 0, 255).astype(np.uint8)
    r_old = hf_ratio(out_old_emu, region=wall)
    print(f"high-freq energy ratio  new={r_new:.4f}  ctrl(no turb/noise)={r_ctrl:.4f}  old-pixel-iid-emu={r_old:.4f}")

    # The new image must be substantially smoother than the old pixel-iid
    # version, and have less high-frequency relative energy than the
    # previous default produced.
    if r_new >= r_old * 0.6:
        print("FAIL: new texture is not noticeably smoother than old TV-snow look")
        return 1

    # Sanity: turbulence should still introduce some macro variation, so
    # the image should NOT be identical to the control.
    diff = np.abs(out_new.astype(np.int16) - out_ctrl.astype(np.int16)).mean()
    print(f"mean abs diff vs control = {diff:.2f}")
    if diff < 1.5:
        print("FAIL: turbulence had no visible effect")
        return 1

    # Flame must still be visible.
    flame_box = (slice(40, 80), slice(60, 100))
    flame_rgb = out_new[flame_box].astype(np.float32).mean(axis=(0, 1))
    print(f"flame box mean RGB={flame_rgb.round(1)}")
    if flame_rgb[0] - flame_rgb[2] < 25:
        print("FAIL: flame lost its red dominance")
        return 1

    print("OK: smoky RGB has structured, low-frequency texture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
