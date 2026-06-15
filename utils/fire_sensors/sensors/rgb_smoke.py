"""RGB camera under smoke (Beer-Lambert depth-aware fog with turbulence).

Visible-light cameras attenuate quickly in smoke. Starr & Lattimer 2014
report visible-band cameras start to lose detail at V≈8 m and become
unusable below ~1 m. We model it as

    I = J * T + A * (1 - T),    T = exp(-k_eff(x, y) * d(x, y))

where ``k_eff`` is the local extinction coefficient. Real smoke is **not**
spatially uniform: it billows in low-frequency clouds and drifts slowly in
time. We therefore modulate the global ``k`` by a multi-octave
Gaussian-filtered random field with frame-to-frame persistence. This kills
the "frosted-glass / TV-snow" look produced by pixel-wise iid noise and
recovers the soft, structured appearance of real smoke.

Flames are *self-luminous* and their visible-band radiation (especially
red/orange wavelengths around 620-740 nm) is far less attenuated by smoke
than reflected ambient light because:
  - emission overpowers extinction at the source pixel,
  - Mie scattering of long-wavelength light is weaker, and
  - hot soot in the flame envelope itself glows.
We detect flame pixels in HSV, restore their transmittance
(``flame_smoke_passthrough``) and bleed an orange glow into the surrounding
smoke (``flame_glow_gain`` / ``flame_color_bleed``).
"""
from __future__ import annotations

from typing import Dict, Optional

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


def _turbulence_field(
    rng: np.random.Generator,
    shape,
    scales,
) -> np.ndarray:
    """Multi-octave Gaussian-blurred white noise, normalised to ~ N(0, 1).

    The result is a low-frequency, locally-correlated random field. Stack
    several octaves to get a fractal-like, billowing pattern.
    """
    H, W = shape
    field = np.zeros((H, W), dtype=np.float32)
    weight = 0.0
    for sigma in scales:
        s = max(1, int(sigma))
        k = max(3, (6 * s) | 1)  # kernel size that comfortably covers sigma
        n = rng.standard_normal((H, W)).astype(np.float32)
        n = cv2.GaussianBlur(n, (k, k), s)
        # Re-normalise so each octave has unit std before stacking.
        std = float(n.std())
        if std > 1e-6:
            n /= std
        w = 1.0 / float(s)  # smaller scale -> finer detail -> smaller weight
        field += w * n
        weight += w
    if weight > 0.0:
        field /= weight
    # Final normalisation so the stacked field has ~ unit std.
    std = float(field.std())
    if std > 1e-6:
        field /= std
    return field


class SmokeRGBSensor(BaseSensor):
    name = "rgb_smoke"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Persistent turbulence field for temporal coherence across frames.
        self._turb_field: Optional[np.ndarray] = None

    def _step_turbulence(self, shape) -> np.ndarray:
        s = self.cfg.smoke
        new_field = _turbulence_field(self.rng, shape, s.smoke_turbulence_scales)
        if (
            self._turb_field is None
            or self._turb_field.shape != new_field.shape
        ):
            self._turb_field = new_field
        else:
            a = float(np.clip(s.smoke_turbulence_persist, 0.0, 0.999))
            self._turb_field = a * self._turb_field + np.sqrt(1.0 - a * a) * new_field
        return self._turb_field

    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        s = self.cfg.smoke
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]

        H, W = depth_m.shape
        d = np.clip(depth_m, 0.0, self.cfg.max_depth_m)
        k_base = density_to_k(s.smoke_density, s.smoke_k_max)

        # ----- Spatial turbulence: low-frequency density modulation -----
        if k_base > 0.0 and s.smoke_turbulence_strength > 0.0:
            turb = self._step_turbulence((H, W))  # ~N(0, 1)
            strength = float(np.clip(s.smoke_turbulence_strength, 0.0, 1.5))
            # exp gives strictly positive multiplier; clip keeps it sane.
            k_mult = np.exp(strength * turb).astype(np.float32)
            k_mult = np.clip(k_mult, 0.3, 3.0)
            k_field = k_base * k_mult
        else:
            k_field = np.full((H, W), k_base, dtype=np.float32)

        transmittance = np.exp(-k_field * d).astype(np.float32)

        # ----- Flame radiation: keeps flames visible through smoke ------
        flame = _flame_mask(rgb, s)
        glow = _flame_glow(flame, s.flame_glow_ksize)

        passthrough = float(np.clip(s.flame_smoke_passthrough, 0.0, 1.0))
        glow_gain = float(np.clip(s.flame_glow_gain, 0.0, 1.0))
        boost = np.maximum(flame * passthrough, glow * glow_gain * passthrough)
        transmittance_eff = transmittance + (1.0 - transmittance) * boost
        transmittance_eff = np.clip(transmittance_eff, 0.0, 1.0)[..., None]

        smoke = np.array(s.smoke_color_rgb, dtype=np.float32).reshape(1, 1, 3)

        # ----- Tint smoke around flames with the local fire colour -----
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

        # ----- Low-frequency brightness fluctuation on the smoke layer --
        # Adds the impression of slowly drifting plumes without producing
        # the white-noise frosted-glass artefact of the previous version.
        if k_base > 0.0 and s.smoke_lowfreq_noise_std > 0.0:
            ksize = max(3, int(s.smoke_lowfreq_noise_ksize) | 1)
            n = self.rng.standard_normal((H, W)).astype(np.float32)
            n = cv2.GaussianBlur(n, (ksize, ksize), 0)
            std = float(n.std())
            if std > 1e-6:
                n /= std
            n = (n * float(s.smoke_lowfreq_noise_std))[..., None]
            smoke_local = smoke_local + n  # broadcast over RGB

        out = (
            rgb.astype(np.float32) * transmittance_eff
            + smoke_local * (1.0 - transmittance_eff)
        )
        out_u8 = np.clip(out, 0, 255).astype(np.uint8)

        return {
            "image": out_u8,
            "transmittance": transmittance_eff[..., 0].astype(np.float32),
            "flame_mask": flame.astype(np.float32),
            "smoke_density_field": k_field.astype(np.float32),
        }
