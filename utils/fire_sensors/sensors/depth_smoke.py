"""Depth (LIDAR-like) sensor under smoke.

LIDAR distance measurements degrade significantly when smoke visibility
drops below ~4 m and below ~1 m the LIDAR returns the smoke layer
itself instead of the wall (Starr & Lattimer 2014, Fig. 5).

We model this with three effects:

  1. Range- and density-dependent Gaussian noise.
  2. Cm-level quantization.
  3. Random dropout that grows with smoke density.
  4. Beyond Jin visibility V=2.3/k, returns are clipped to the smoke
     layer (with small jitter).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import BaseSensor
from .rgb_smoke import density_to_k


class SmokeDepthSensor(BaseSensor):
    name = "depth_smoke"

    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        d_cfg = self.cfg.depth
        s_cfg = self.cfg.smoke

        squeeze = False
        if depth_m.ndim == 3:
            squeeze = True
            depth = depth_m[..., 0].astype(np.float32)
        else:
            depth = depth_m.astype(np.float32)

        density = float(np.clip(s_cfg.smoke_density, 0.0, 1.0))

        # 1) range- and smoke-dependent Gaussian noise
        sigma = (
            d_cfg.sigma_base_m
            + d_cfg.sigma_range_m * depth
            + d_cfg.sigma_smoke_m * density * depth
        )
        depth = depth + self.rng.normal(0.0, sigma).astype(np.float32)

        # 2) quantization
        if d_cfg.quant_m > 0:
            depth = np.round(depth / d_cfg.quant_m) * d_cfg.quant_m

        # 3) random dropout
        p_drop = d_cfg.dropout_max * density
        if p_drop > 0:
            mask = self.rng.random(depth.shape) < p_drop
            depth[mask] = 0.0

        # 4) below visibility, see only the smoke layer
        if d_cfg.clip_to_smoke and density > 0.0:
            k = density_to_k(density, s_cfg.smoke_k_max)
            visibility = 2.3 / max(k, 1e-3)
            far = depth > visibility
            n_far = int(far.sum())
            if n_far > 0:
                depth[far] = visibility + self.rng.normal(
                    0.0, 0.05, size=n_far
                ).astype(np.float32)

        depth = np.clip(depth, 0.0, self.cfg.max_depth_m)
        if squeeze:
            depth = depth[..., None]
        return {"depth": depth.astype(np.float32)}
