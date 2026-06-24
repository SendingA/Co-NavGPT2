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

Per-pixel vs global density:
    The legacy code applied a single global ``smoke_density`` across
    the whole frame, so depth degraded the same in a smoke-free room
    as in the kitchen on fire. When the suite is fed a per-pixel
    ``transmittance`` map (from the voxel renderer), we instead use
    ``local_density = clip(1 - T_pixel, 0, 1)`` so a clean room shows a
    clean depth and a smoky one degrades.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import BaseSensor
from .rgb_smoke import density_to_k


class SmokeDepthSensor(BaseSensor):
    name = "depth_smoke"

    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        transmittance: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run the smoke-aware depth degradation.

        ``transmittance`` (HxW float in [0,1]) is the per-pixel optical
        transmittance produced by the voxel renderer. When given, all
        smoke-driven effects scale by ``1 - T`` *per pixel*; when absent
        we fall back to the global ``smoke_density`` knob.
        """
        d_cfg = self.cfg.depth
        s_cfg = self.cfg.smoke

        squeeze = False
        if depth_m.ndim == 3:
            squeeze = True
            depth = depth_m[..., 0].astype(np.float32)
        else:
            depth = depth_m.astype(np.float32)

        # Spatial density map.
        if transmittance is not None and transmittance.size:
            T = np.asarray(transmittance, dtype=np.float32)
            if T.ndim == 3:
                T = T[..., 0]
            if T.shape != depth.shape:
                # Cheap nearest-neighbour fit when the renderer ran at a
                # different resolution.
                ix = np.linspace(0, T.shape[0] - 1, depth.shape[0]).astype(int)
                iy = np.linspace(0, T.shape[1] - 1, depth.shape[1]).astype(int)
                T = T[ix[:, None], iy[None, :]]
            density_map = np.clip(1.0 - T, 0.0, 1.0)
        else:
            density_map = np.full(
                depth.shape, float(np.clip(s_cfg.smoke_density, 0.0, 1.0)),
                dtype=np.float32,
            )
        density_scalar = float(np.clip(s_cfg.smoke_density, 0.0, 1.0))

        # 1) range- and smoke-dependent Gaussian noise (per-pixel density).
        sigma = (
            d_cfg.sigma_base_m
            + d_cfg.sigma_range_m * depth
            + d_cfg.sigma_smoke_m * density_map * depth
        )
        depth = depth + self.rng.normal(0.0, sigma).astype(np.float32)

        # 2) quantization
        if d_cfg.quant_m > 0:
            depth = np.round(depth / d_cfg.quant_m) * d_cfg.quant_m

        # 3) random dropout (per-pixel density).
        if d_cfg.dropout_max > 0:
            p_drop = d_cfg.dropout_max * density_map
            mask = self.rng.random(depth.shape) < p_drop
            depth[mask] = 0.0

        # 4) below local visibility, clip the return to the smoke layer.
        # ``visibility = 2.3 / k`` per pixel; depths beyond it are
        # replaced with the smoke layer + jitter.
        if d_cfg.clip_to_smoke:
            k_map = density_map * float(s_cfg.smoke_k_max)
            visibility = np.where(
                k_map > 1e-3,
                2.3 / np.maximum(k_map, 1e-3),
                self.cfg.max_depth_m,
            ).astype(np.float32)
            far = depth > visibility
            n_far = int(far.sum())
            if n_far > 0:
                depth[far] = visibility[far] + self.rng.normal(
                    0.0, 0.05, size=n_far
                ).astype(np.float32)

        depth = np.clip(depth, 0.0, self.cfg.max_depth_m)
        if squeeze:
            depth = depth[..., None]
        return {"depth": depth.astype(np.float32)}
