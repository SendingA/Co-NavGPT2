"""Observable smoke and confidence proxies for the sensed-risk pipeline.

The FireWorld smoke field and renderer transmittance are privileged simulator
state.  The primary ``sensed`` mode therefore uses only the smoke-degraded RGB
and depth products available to the robot.  This module intentionally returns
a conservative *visibility-risk proxy*; it does not estimate toxic gas or CO.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def estimate_smoke_from_appearance_depth(
    rgb_smoke: np.ndarray,
    depth_smoke_m: np.ndarray,
    *,
    max_depth_m: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-pixel smoke visibility loss and observation confidence.

    The estimator combines desaturation, loss of local contrast and invalid or
    clipped depth.  It is deliberately lightweight and deterministic so it can
    serve as a transparent benchmark baseline.  A learned estimator can later
    replace it without changing the downstream ``RiskEvidence`` contract.
    """

    rgb = np.asarray(rgb_smoke)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb_smoke must have shape (H, W, >=3)")
    rgb = rgb[..., :3].astype(np.float32)
    if float(np.nanmax(rgb, initial=0.0)) > 1.5:
        rgb /= 255.0
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 1.0)

    depth = np.asarray(depth_smoke_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != rgb.shape[:2]:
        raise ValueError("depth_smoke_m must share the RGB image shape")

    value = np.max(rgb, axis=2)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    saturation = chroma / np.maximum(value, 1e-3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
    local_sq_mean = cv2.GaussianBlur(gray * gray, (0, 0), sigmaX=3.0)
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

    # Dense smoke tends to be desaturated and locally low-contrast.  Smooth
    # clean walls are ambiguous, so the resulting high proxy is paired with a
    # lower confidence rather than presented as a certain physical reading.
    desaturation = np.clip(1.0 - saturation / 0.45, 0.0, 1.0)
    contrast_loss = np.exp(-local_std / 0.055).astype(np.float32)
    brightness_gate = np.clip((gray - 0.08) / 0.45, 0.0, 1.0)
    appearance_smoke = desaturation * contrast_loss * brightness_gate

    valid_depth = np.isfinite(depth) & (depth > 0.0)
    clipped_depth = valid_depth & (depth >= 0.98 * float(max_depth_m))
    depth_degradation = (~valid_depth).astype(np.float32)
    depth_degradation[clipped_depth] = np.maximum(
        depth_degradation[clipped_depth], 0.35
    )

    smoke = np.clip(
        0.80 * appearance_smoke + 0.20 * depth_degradation,
        0.0,
        1.0,
    ).astype(np.float32)
    confidence = np.clip(
        valid_depth.astype(np.float32) * (1.0 - 0.55 * smoke),
        0.05,
        1.0,
    ).astype(np.float32)
    confidence[~valid_depth] = 0.05
    return smoke, confidence

