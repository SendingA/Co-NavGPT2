"""Helpers for the smoke-scene perception path.

Three jobs live here:

1. ``apply_clean_depth_and_thermal``:
    Stitch the per-step output of the fire-sensor suite back into the
    Habitat ``observations`` dict so the downstream agent sees:
        - the smoke-attenuated RGB,
        - the *clean* (un-degraded) depth, when ``--depth_use_clean=1``,
        - the thermal grayscale image and flame mask, when
          ``--use_thermal_perception=1``.

2. ``dehaze_with_depth``:
    Depth-aware inverse Beer-Lambert + CLAHE that recovers a YOLO-friendly
    RGB from the smoky one when an accurate depth is available.

3. ``estimate_smoke_k``:
    Cheap MLE of the smoke extinction coefficient ``k`` from the smoky RGB
    and the clean depth, used by ``dehaze_with_depth`` if not given.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1) Observation patcher
# ---------------------------------------------------------------------------
def apply_clean_depth_and_thermal(
    observations: Dict[str, np.ndarray],
    sensors: Dict[str, np.ndarray],
    *,
    clean_depth_raw: np.ndarray,
    use_clean_depth: bool,
    use_thermal: bool,
    apply_smoky_rgb: bool,
    normalize_depth: bool,
    max_depth_m: float,
) -> None:
    """Patch ``observations`` in-place with the smoky RGB / clean depth /
    thermal channels produced by the fire-sensor suite.

    Args:
        observations: per-agent obs dict from Habitat. Mutated in place.
        sensors: dict returned by ``FireSensorSuite.process``.
        clean_depth_raw: the original (unmodified) depth tensor as it came
            out of Habitat. Used verbatim when ``use_clean_depth`` is True.
        use_clean_depth: keep the clean depth, ignore the smoky one.
        use_thermal: inject thermal_image / thermal_flame_mask.
        apply_smoky_rgb: replace observations['rgb'] with rgb_smoke.
        normalize_depth, max_depth_m: control re-normalisation when
            switching from a metric depth back to the [0,1] tensor that
            Habitat hands out.
    """
    if apply_smoky_rgb and "rgb_smoke" in sensors:
        observations["rgb"] = sensors["rgb_smoke"]

    if use_clean_depth:
        observations["depth"] = clean_depth_raw
    else:
        d_smoke = sensors.get("depth_smoke")
        if d_smoke is not None:
            if normalize_depth:
                d_smoke = np.clip(d_smoke / max_depth_m, 0.0, 1.0)
            if clean_depth_raw.ndim == 3 and d_smoke.ndim == 2:
                d_smoke = d_smoke[..., None]
            observations["depth"] = d_smoke.astype(clean_depth_raw.dtype)

    if use_thermal:
        if "thermal_image" in sensors:
            observations["thermal"] = sensors["thermal_image"]
        if "thermal_flame_mask" in sensors:
            observations["thermal_flame_mask"] = sensors["thermal_flame_mask"]


# ---------------------------------------------------------------------------
# 2) Depth-aware dehazing (inverse Beer-Lambert + CLAHE)
# ---------------------------------------------------------------------------
def estimate_smoke_k(
    rgb_smoky: np.ndarray,
    depth_m: np.ndarray,
    *,
    smoke_color: np.ndarray = np.array([180, 180, 180], dtype=np.float32),
    sample_stride: int = 4,
) -> float:
    """Roughly estimate the smoke extinction coefficient ``k``.

    Beer-Lambert: ``I = J*T + A*(1-T)`` with ``T = exp(-k*d)``.
    On dim-radiance regions ``J`` is small, so ``I ≈ A * (1 - exp(-k*d))``.
    We solve for ``k`` on the darkest 25% pixels (least flame contamination)
    using a least-squares fit to ``log((A - I) / A) = -k * d``.
    """
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]

    H, W = depth_m.shape
    if sample_stride > 1:
        rgb_s = rgb_smoky[::sample_stride, ::sample_stride]
        d_s = depth_m[::sample_stride, ::sample_stride]
    else:
        rgb_s, d_s = rgb_smoky, depth_m

    luma = cv2.cvtColor(rgb_s, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    valid = (d_s > 0.3) & (d_s < 50.0) & (luma < np.percentile(luma, 25))
    if valid.sum() < 50:
        return 0.5  # safe default

    A_norm = float(np.mean(smoke_color)) / 255.0
    I_norm = luma[valid]
    d_v = d_s[valid].astype(np.float32)

    # Avoid log(<=0).
    ratio = np.clip((A_norm - I_norm) / max(A_norm, 1e-3), 1e-3, 1.0)
    log_ratio = np.log(ratio)  # = -k * d
    k = -float(np.dot(log_ratio, d_v) / max(np.dot(d_v, d_v), 1e-3))
    return float(np.clip(k, 0.05, 5.0))


def dehaze_with_depth(
    rgb_smoky: np.ndarray,
    depth_m: np.ndarray,
    *,
    k: Optional[float] = None,
    smoke_color: np.ndarray = np.array([180, 180, 180], dtype=np.float32),
    t_min: float = 0.1,
    apply_clahe: bool = True,
) -> np.ndarray:
    """Inverse Beer-Lambert dehazing using a known clean depth.

    ``J = (I - A * (1 - T)) / clip(T, t_min, 1)``

    Why this works in this project:
        - The smoke filter (``SmokeRGBSensor``) literally applies
          ``I = J*T + A*(1-T)``. With clean depth and an estimate of ``k``
          we can invert it analytically.
        - The flame penetration we added on top is monotonic in ``T``, so
          inverting the basic Beer-Lambert just slightly under-corrects
          flame regions, which is desirable (we don't want to amplify the
          self-luminous core).

    Args:
        rgb_smoky: HxWx3 uint8, the smoke-affected RGB.
        depth_m: HxW (or HxWx1) metric depth, must be the *clean* one.
        k: optional override for the extinction coefficient. If None, it is
            estimated from the image / depth pair.
        smoke_color: the fog colour in 0-255, must match the SmokeRGBConfig.
        t_min: lower clamp on the transmittance to avoid noise blow-up at
            far range. 0.1 ≈ 10% transmittance preserved.
        apply_clahe: run a per-channel CLAHE pass after inversion.

    Returns:
        HxWx3 uint8 dehazed RGB.
    """
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]

    if k is None:
        k = estimate_smoke_k(rgb_smoky, depth_m, smoke_color=smoke_color)

    d = np.clip(depth_m.astype(np.float32), 0.0, 50.0)
    T = np.exp(-k * d)
    T = np.clip(T, t_min, 1.0)[..., None]

    A = np.asarray(smoke_color, dtype=np.float32).reshape(1, 1, 3)
    I = rgb_smoky.astype(np.float32)
    J = (I - A * (1.0 - T)) / T
    J = np.clip(J, 0.0, 255.0).astype(np.uint8)

    if apply_clahe:
        # CLAHE on Y channel preserves colour balance better than per-RGB.
        ycrcb = cv2.cvtColor(J, cv2.COLOR_RGB2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
        J = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    return J


# ---------------------------------------------------------------------------
# 3) Thermal -> detection-shaped output
# ---------------------------------------------------------------------------
def thermal_mask_to_detections(
    flame_mask_2d: np.ndarray,
    target_hw: Optional[tuple] = None,
    *,
    min_area: int = 60,
    confidence: float = 0.95,
):
    """Turn a thermal flame mask (float32 in [0,1]) into YOLO-style outputs.

    Returns ``(boxes_xyxy, masks_bool, scores)`` matching the convention
    used by ``utils.detection_segmentation.detect_flames_hsv`` so the
    detection pipeline can ingest it without further plumbing.
    """
    if flame_mask_2d is None or flame_mask_2d.size == 0:
        H, W = (target_hw if target_hw is not None else (0, 0))
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0, H, W), bool),
            np.zeros((0,), np.float32),
        )

    mask_u8 = (flame_mask_2d > 0.5).astype(np.uint8) * 255
    if target_hw is not None and mask_u8.shape != target_hw:
        H, W = target_hw
        mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
    H, W = mask_u8.shape

    n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    boxes, masks, scores = [], [], []
    for lbl in range(1, n_lbl):
        x, y, w, h, area = stats[lbl]
        if area < min_area:
            continue
        boxes.append([float(x), float(y), float(x + w), float(y + h)])
        masks.append(lbl_img == lbl)
        scores.append(confidence)

    if not boxes:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0, H, W), bool),
            np.zeros((0,), np.float32),
        )
    return (
        np.asarray(boxes, dtype=np.float32),
        np.stack(masks, axis=0).astype(bool),
        np.asarray(scores, dtype=np.float32),
    )
