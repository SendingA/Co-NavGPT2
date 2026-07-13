"""Helpers for the smoke-scene perception path.

Two jobs live here:

1. ``apply_clean_depth_and_thermal``:
    Stitch the per-step output of the fire-sensor suite back into the
    Habitat ``observations`` dict so the downstream agent sees:
        - the smoke-attenuated RGB,
        - the *clean* (un-degraded) depth, when ``--depth_use_clean=1``,
        - the thermal grayscale image and flame mask, when
          ``--use_thermal_perception=1``.

2. ``thermal_mask_to_detections``:
    Turn a thermal flame mask into YOLO-style (boxes, masks, scores) so
    the detection pipeline can ingest a smoke-invariant fire detection.
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
        if "thermal_human_mask" in sensors:
            observations["thermal_human_mask"] = sensors["thermal_human_mask"]


# ---------------------------------------------------------------------------
# 2) Thermal -> detection-shaped output
# ---------------------------------------------------------------------------
def thermal_mask_to_detections(
    flame_mask_2d: np.ndarray,
    target_hw: Optional[tuple] = None,
    *,
    min_area: int = 60,
    confidence: float = 0.95,
):
    """Turn a thermal flame mask (float32 in [0,1]) into YOLO-style outputs.

    Returns ``(boxes_xyxy, masks_bool, scores)`` in the same convention as
    the YOLO/SAM detections so the detection pipeline can ingest a
    smoke-invariant fire detection without further plumbing.
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
