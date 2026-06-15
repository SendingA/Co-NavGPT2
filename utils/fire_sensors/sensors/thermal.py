"""FLIR-style long-wave IR thermal camera.

Mimics Fig. 10 of Starr & Lattimer 2014: grayscale image with strong
edge contrast, scene structure preserved, flames saturated near white,
and a bright halo around objects adjacent to the flame from radiative
heating. Thermal cameras at 7.5–14 µm are largely unaffected by smoke,
so the synth uses the **clean** RGB.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .base import BaseSensor


def _detect_flame_mask(rgb: np.ndarray, t_cfg) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(
        hsv, np.array(t_cfg.flame_hsv_low1), np.array(t_cfg.flame_hsv_high1)
    )
    mask2 = cv2.inRange(
        hsv, np.array(t_cfg.flame_hsv_low2), np.array(t_cfg.flame_hsv_high2)
    )
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


class ThermalSensor(BaseSensor):
    name = "thermal"

    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        t = self.cfg.thermal

        flame = _detect_flame_mask(rgb, t)
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # 1) flame contribution
        flame_field = (
            (t.flame_c - t.ambient_c) * np.maximum(luma, 0.4) * flame
        )

        # 2) halo around the flame (radiative heating of nearby surfaces)
        ksize = max(3, t.halo_ksize | 1)
        halo = cv2.GaussianBlur(flame_field, (ksize, ksize), 0) * t.halo_gain

        # 3) scene structure proxy: emissivity / thermal-mass differences
        scene_field = (luma - 0.5) * t.scene_contrast_c

        # 4) Laplacian edge boost - thermal cameras outline boundaries
        edges = np.abs(cv2.Laplacian(luma, cv2.CV_32F, ksize=3))
        if edges.max() > 1e-6:
            edges = edges / edges.max()
        edge_field = edges * t.edge_gain * t.scene_contrast_c

        temperature = t.ambient_c + scene_field + edge_field + halo + flame_field

        # 5) auto-stretch + gamma to mimic FLIR auto-gain
        t_lo = float(np.percentile(temperature, 2))
        t_hi = float(max(np.percentile(temperature, 99.5), t.flame_c * 0.6))
        norm = np.clip((temperature - t_lo) / max(t_hi - t_lo, 1e-3), 0.0, 1.0)
        norm = np.power(norm, t.gamma)
        gray_u8 = (norm * 255).astype(np.uint8)

        image_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        if t.color_blend > 0.0:
            ir = cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)
            a = float(np.clip(t.color_blend, 0.0, 1.0))
            image_bgr = cv2.addWeighted(image_bgr, 1 - a, ir, a, 0)

        return {
            "image": image_bgr,
            "temperature_c": temperature.astype(np.float32),
            "flame_mask": flame.astype(np.float32),
        }
