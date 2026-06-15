"""RGB camera under smoke (Beer-Lambert depth-aware fog).

Visible-light cameras attenuate quickly in smoke. Starr & Lattimer 2014
report visible-band cameras start to lose detail at V≈8 m and become
unusable below ~1 m. We blend the original RGB with a gray smoke color
according to per-pixel transmittance T = exp(-k*d), with k tied to the
density knob via Jin's visibility equation V = 2.3/k.

Flames are *self-luminous* and their visible-band radiation (especially
red/orange wavelengths around 620-740 nm) is far less attenuated by
smoke than reflected ambient light because:
  - emission overpowers extinction at the source pixel,
  - Mie scattering of long-wavelength light is weaker, and
  - hot soot in the flame envelope itself glows.
We therefore detect flame pixels in HSV, restore their transmittance
(``flame_smoke_passthrough``) and bleed an orange glow into the
surrounding smoke (``flame_glow_gain`` / ``flame_color_bleed``). This
makes flames remain visible through medium-dense smoke, matching the
behaviour observed in real fireground footage.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .base import BaseSensor


def density_to_k(density: float, k_max: float) -> float:
    return float(np.clip(density, 0.0, 1.0)) * k_max


def _flame_mask(rgb: np.ndarray, s_cfg) -> np.ndarray:
    """HSV-based flame mask in [0, 1] (float32, shape ``(H, W)``)."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(
        hsv, np.array(s_cfg.flame_hsv_low1), np.array(s_cfg.flame_hsv_high1)
    )
    mask2 = cv2.inRange(
        hsv, np.array(s_cfg.flame_hsv_low2), np.array(s_cfg.flame_hsv_high2)
    )
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _flame_glow(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Soft glow halo around the flame mask, shape ``(H, W)`` float32."""

    if mask.max() <= 0.0:
        return np.zeros_like(mask)
    k = max(3, int(ksize) | 1)
    glow = cv2.GaussianBlur(mask, (k, k), 0)
    m = float(glow.max())
    if m > 1e-6:
        glow = glow / m
    return glow.astype(np.float32)


class SmokeRGBSensor(BaseSensor):
    name = "rgb_smoke"

    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        s = self.cfg.smoke
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]

        k = density_to_k(s.smoke_density, s.smoke_k_max)
        d = np.clip(depth_m, 0.0, self.cfg.max_depth_m)
        transmittance = np.exp(-k * d).astype(np.float32)  # (H, W)

        # ----- Flame radiation: keeps flames visible through smoke -----
        flame = _flame_mask(rgb, s)
        glow = _flame_glow(flame, s.flame_glow_ksize)

        passthrough = float(np.clip(s.flame_smoke_passthrough, 0.0, 1.0))
        glow_gain = float(np.clip(s.flame_glow_gain, 0.0, 1.0))

        # Boost transmittance: flame core -> ~1.0, halo -> blend.
        boost = np.maximum(flame * passthrough, glow * glow_gain * passthrough)
        # T' = T + (1 - T) * boost  (monotonic, stays in [T, 1])
        transmittance_eff = transmittance + (1.0 - transmittance) * boost
        transmittance_eff = np.clip(transmittance_eff, 0.0, 1.0)[..., None]

        smoke = np.array(s.smoke_color_rgb, dtype=np.float32).reshape(1, 1, 3)

        # Tint the smoke with the flame color near flames. Pull the local
        # flame color from the RGB itself (mean over the flame mask) so the
        # glow matches the actual fire palette in the scene.
        if flame.sum() > 0.0 and s.flame_color_bleed > 0.0:
            flame_pixels = rgb.astype(np.float32).reshape(-1, 3)[
                flame.reshape(-1) > 0.5
            ]
            if flame_pixels.shape[0] > 0:
                fire_color = flame_pixels.mean(axis=0).reshape(1, 1, 3)
                bleed = float(np.clip(s.flame_color_bleed, 0.0, 1.0))
                glow_w = (glow * bleed)[..., None].astype(np.float32)
                smoke_local = smoke * (1.0 - glow_w) + fire_color * glow_w
            else:
                smoke_local = smoke
        else:
            smoke_local = smoke

        out = (
            rgb.astype(np.float32) * transmittance_eff
            + smoke_local * (1.0 - transmittance_eff)
        )

        # Subtle drifting noise to mimic moving smoke particles. Suppress the
        # noise on flame pixels so the bright core stays clean.
        if s.smoke_density > 0.0:
            noise_scale = 1.0 - 0.85 * np.maximum(flame, glow * 0.5)
            noise = self.rng.normal(0.0, 4.0 * s.smoke_density, size=out.shape)
            out = out + noise * noise_scale[..., None]

        out_u8 = np.clip(out, 0, 255).astype(np.uint8)
        return {
            "image": out_u8,
            "transmittance": transmittance_eff[..., 0].astype(np.float32),
            "flame_mask": flame.astype(np.float32),
        }
