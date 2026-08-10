"""Normalisation and spatial projection for fire-risk fields.

The functions in this module are stateless and NumPy-only.  In particular,
sensor projection consumes ordinary arrays and an explicit camera pose; it has
no way to access a :class:`FireWorld` object.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .config import RiskConfig
from .model import GridFrame, RiskEvidence, RiskLayers


def normalize_temperature_c(
    temperature_c: np.ndarray,
    reference_c: float,
    hazard_c: float,
) -> np.ndarray:
    """Map degrees Celsius to a bounded dimensionless temperature risk."""

    if float(hazard_c) <= float(reference_c):
        raise ValueError("hazard_c must exceed reference_c")
    temperature = np.asarray(temperature_c, dtype=np.float32)
    normalised = np.clip(
        (temperature - float(reference_c)) / (float(hazard_c) - float(reference_c)),
        0.0,
        1.0,
    )
    return np.nan_to_num(
        normalised, nan=0.0, posinf=1.0, neginf=0.0
    ).astype(np.float32)


def combine_normalized_risk(
    temperature: np.ndarray,
    smoke: np.ndarray,
    config: RiskConfig,
) -> np.ndarray:
    """Combine heat and smoke into the continuous physical-risk score.

    Flame is handled separately by :func:`hard_unsafe_mask`; omitting it here
    prevents flame intensity and flame-driven temperature from being counted
    twice.
    """

    temp_n, smoke_n = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float32),
        np.asarray(smoke, dtype=np.float32),
    )
    temp_n = np.nan_to_num(temp_n, nan=0.0, posinf=1.0, neginf=0.0)
    smoke_n = np.nan_to_num(smoke_n, nan=0.0, posinf=1.0, neginf=0.0)
    result = (
        float(config.weights.temperature) * np.clip(temp_n, 0.0, 1.0)
        + float(config.weights.smoke) * np.clip(smoke_n, 0.0, 1.0)
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def compute_physical_risk(
    temperature_c: np.ndarray,
    smoke: np.ndarray,
    config: RiskConfig,
) -> np.ndarray:
    """Return ``H=clip(wT*rT+wS*rS, 0, 1)``.

    Smoke is expected to be normalised and temperature is in degrees Celsius.
    Explicit flame belongs to the hard-exclusion layer, not this continuous
    exposure score.
    """

    temperature = normalize_temperature_c(
        temperature_c,
        config.temperature_reference_c,
        config.temperature_hazard_c,
    )
    return combine_normalized_risk(temperature, smoke, config)


def dilate_disk(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary disk dilation without scipy/OpenCV dependencies."""

    source = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius == 0 or not source.any():
        return source.copy()
    height, width = source.shape
    out = source.copy()
    for di in range(-radius, radius + 1):
        span = int(np.floor(np.sqrt(radius * radius - di * di)))
        for dj in range(-span, span + 1):
            src_i0 = max(0, -di)
            src_i1 = min(height, height - di)
            src_j0 = max(0, -dj)
            src_j1 = min(width, width - dj)
            if src_i0 >= src_i1 or src_j0 >= src_j1:
                continue
            out[
                src_i0 + di:src_i1 + di,
                src_j0 + dj:src_j1 + dj,
            ] |= source[src_i0:src_i1, src_j0:src_j1]
    return out


def hard_unsafe_mask(
    flame: np.ndarray,
    temperature_c: np.ndarray,
    frame: GridFrame,
    config: RiskConfig,
) -> np.ndarray:
    """Return the non-negotiable core hazard used by local/global planners.

    By default this is only the high-intensity flame core. Heat and smoke are
    retained in the continuous physical-risk field so their influence fades
    spatially instead of becoming one oversized binary region. Experiments
    that specifically need the legacy temperature veto or a flame clearance
    ring can opt into them through :class:`RiskConfig`.
    """

    flame_source = np.asarray(flame) >= float(config.flame_hard_threshold)
    radius = int(np.ceil(
        float(config.flame_safety_distance_m) / frame.resolution_m
    ))
    flame_unsafe = dilate_disk(flame_source, radius)
    temperature_unsafe = np.zeros(flame_unsafe.shape, dtype=bool)
    if bool(config.temperature_hard_enabled):
        temperature_unsafe = (
            np.asarray(temperature_c) >= float(config.temperature_hard_c)
        )
    return np.asarray(flame_unsafe | temperature_unsafe, dtype=bool)


def _vertical_voxel_indices(
    origin_y: float,
    voxel_m: float,
    ny: int,
    floor_y_m: float,
    min_offset_m: float,
    max_offset_m: float,
) -> np.ndarray:
    """Return y voxels whose physical cells overlap the body-height band."""

    low = float(floor_y_m) + float(min_offset_m)
    high = float(floor_y_m) + float(max_offset_m)
    voxel_lows = float(origin_y) + np.arange(ny, dtype=np.float64) * voxel_m
    voxel_highs = voxel_lows + voxel_m
    return np.flatnonzero((voxel_highs > low) & (voxel_lows < high))


def project_fire_fields(
    flame_field: np.ndarray,
    smoke_field: np.ndarray,
    temperature_field_c: np.ndarray,
    *,
    voxel_origin: np.ndarray,
    voxel_m: float,
    frame: GridFrame,
    floor_y_m: float,
    map_floor_y_m: Optional[float] = None,
    config: RiskConfig,
    timestamp_s: float = 0.0,
) -> RiskLayers:
    """Project a FireWorld frame onto a floor/body-height navigation grid.

    A map cell is inverse-sampled into the FireWorld grid.  This is important
    when the FireWorld voxels (typically 15 cm) are coarser than navigation
    cells (typically 5 cm): forward-splatting voxel centres would otherwise
    leave holes.  Max aggregation over the vertical body band is conservative
    for all three simulated hazard fields.
    """

    flame_3d = np.asarray(flame_field, dtype=np.float32)
    smoke_3d = np.asarray(smoke_field, dtype=np.float32)
    temp_3d = np.asarray(temperature_field_c, dtype=np.float32)
    if flame_3d.ndim != 3:
        raise ValueError("FireWorld fields must have shape (Nx, Ny, Nz)")
    if smoke_3d.shape != flame_3d.shape or temp_3d.shape != flame_3d.shape:
        raise ValueError("FireWorld flame/smoke/temperature shapes must match")
    flame_3d = np.nan_to_num(flame_3d, nan=0.0, posinf=1.0, neginf=0.0)
    smoke_3d = np.nan_to_num(smoke_3d, nan=0.0, posinf=1.0, neginf=0.0)
    temp_3d = np.nan_to_num(
        temp_3d,
        nan=float(config.temperature_ambient_c),
        posinf=float(config.temperature_hazard_c),
        neginf=float(config.temperature_ambient_c),
    )
    if float(voxel_m) <= 0.0:
        raise ValueError("voxel_m must be positive")
    origin = np.asarray(voxel_origin, dtype=np.float64).reshape(3)

    # Navigation cell centres, transformed back to Habitat world axes.
    centres_world = frame.cell_centers_world(map_y_m=float(
        floor_y_m if map_floor_y_m is None else map_floor_y_m
    ))
    ix = np.floor((centres_world[..., 0] - origin[0]) / voxel_m).astype(np.int64)
    iz = np.floor((centres_world[..., 2] - origin[2]) / voxel_m).astype(np.int64)
    valid_xz = (
        (ix >= 0) & (ix < flame_3d.shape[0])
        & (iz >= 0) & (iz < flame_3d.shape[2])
    )
    y_indices = _vertical_voxel_indices(
        origin[1], float(voxel_m), flame_3d.shape[1], floor_y_m,
        config.floor_min_offset_m, config.floor_max_offset_m,
    )

    flame = np.zeros(frame.shape, dtype=np.float32)
    smoke = np.zeros(frame.shape, dtype=np.float32)
    temperature_c = np.full(
        frame.shape, float(config.temperature_ambient_c), dtype=np.float32
    )
    confidence = np.zeros(frame.shape, dtype=np.float32)

    if y_indices.size and valid_xz.any():
        safe_ix = np.clip(ix, 0, flame_3d.shape[0] - 1)
        safe_iz = np.clip(iz, 0, flame_3d.shape[2] - 1)
        temperature_max = np.full(frame.shape, -np.inf, dtype=np.float32)
        for iy in y_indices:
            flame = np.maximum(flame, flame_3d[safe_ix, iy, safe_iz])
            smoke = np.maximum(smoke, smoke_3d[safe_ix, iy, safe_iz])
            temperature_max = np.maximum(
                temperature_max, temp_3d[safe_ix, iy, safe_iz]
            )
        temperature_c[valid_xz] = temperature_max[valid_xz]
        flame[~valid_xz] = 0.0
        smoke[~valid_xz] = 0.0
        temperature_c[~valid_xz] = float(config.temperature_ambient_c)
        confidence[valid_xz] = 1.0

    flame = np.clip(flame, 0.0, 1.0)
    smoke = np.clip(smoke, 0.0, 1.0)
    temperature = normalize_temperature_c(
        temperature_c,
        config.temperature_reference_c,
        config.temperature_hazard_c,
    )
    physical_risk = combine_normalized_risk(temperature, smoke, config)
    hard_unsafe = hard_unsafe_mask(flame, temperature_c, frame, config)
    unknown = confidence < float(config.minimum_known_confidence)
    uncertainty = 1.0 - confidence
    last_update = np.full(frame.shape, -np.inf, dtype=np.float64)
    last_update[~unknown] = float(timestamp_s)
    return RiskLayers(
        flame=flame.astype(np.float32),
        temperature_c=temperature_c.astype(np.float32),
        temperature=temperature.astype(np.float32),
        smoke=smoke.astype(np.float32),
        physical_risk=physical_risk.astype(np.float32),
        hard_unsafe=hard_unsafe,
        confidence=confidence,
        uncertainty=uncertainty.astype(np.float32),
        last_update=last_update,
        unknown=unknown,
    )


def _camera_parameter(camera_k: Any, name: str) -> float:
    if hasattr(camera_k, name):
        return float(getattr(camera_k, name))
    array = np.asarray(camera_k, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError("camera_k must expose fx/fy/cx/cy or be a 3x3 matrix")
    locations = {"fx": (0, 0), "fy": (1, 1), "cx": (0, 2), "cy": (1, 2)}
    return float(array[locations[name]])


def evidence_from_sensor_images(
    *,
    agent_id: int,
    timestamp_s: float,
    depth_m: np.ndarray,
    camera_k: Any,
    camera_position_world: np.ndarray,
    rotation_camera_to_world: np.ndarray,
    thermal_temperature_c: np.ndarray,
    thermal_flame: np.ndarray,
    smoke_estimate: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None,
    privileged_transmittance: Optional[np.ndarray] = None,
    allow_privileged_transmittance: bool = False,
    floor_y_m: Optional[float] = None,
    config: Optional[RiskConfig] = None,
) -> RiskEvidence:
    """Back-project per-pixel sensed hazard evidence into world points.

    Camera convention matches ``utils.fire_sensors.voxel_render``: Habitat
    depth is distance along camera ``-Z`` and image rows point down.  Passing
    renderer transmittance requires the explicit privileged flag so an oracle
    simulator product cannot silently enter the primary sensed benchmark.

    The current thermal product stores the maximum temperature along a ray;
    this first implementation projects it to the observed depth surface, as in
    the paper pipeline.  A future hazard-depth estimator can replace this
    function without changing :class:`RiskEvidence` or the shared map.
    """

    cfg = config or RiskConfig()
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    temperature = np.asarray(thermal_temperature_c, dtype=np.float32)
    flame = np.asarray(thermal_flame, dtype=np.float32)
    if depth.ndim != 2 or temperature.shape != depth.shape or flame.shape != depth.shape:
        raise ValueError("depth, thermal temperature and flame must share HxW")

    if smoke_estimate is None:
        if privileged_transmittance is not None:
            if not allow_privileged_transmittance:
                raise ValueError(
                    "renderer transmittance is privileged; set "
                    "allow_privileged_transmittance=True only for an ablation"
                )
            transmittance = np.asarray(privileged_transmittance, dtype=np.float32)
            if transmittance.shape != depth.shape:
                raise ValueError("privileged transmittance must share depth shape")
            optical_depth = -np.log(np.clip(transmittance, 1e-6, 1.0))
            smoke = np.clip(
                optical_depth
                / np.maximum(
                    float(cfg.smoke_extinction_coefficient) * depth, 1e-6
                ),
                0.0,
                1.0,
            )
            privileged = True
        else:
            smoke = np.zeros_like(depth, dtype=np.float32)
            privileged = False
    else:
        smoke = np.asarray(smoke_estimate, dtype=np.float32)
        if smoke.shape != depth.shape:
            raise ValueError("smoke_estimate must share depth shape")
        smoke = np.clip(smoke, 0.0, 1.0)
        privileged = False

    stride = max(1, int(cfg.sensor_stride))
    rows = np.arange(0, depth.shape[0], stride)
    cols = np.arange(0, depth.shape[1], stride)
    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    d = depth[vv, uu]
    valid = np.isfinite(d) & (d > 0.0)

    fx = _camera_parameter(camera_k, "fx")
    fy = _camera_parameter(camera_k, "fy")
    cx = _camera_parameter(camera_k, "cx")
    cy = _camera_parameter(camera_k, "cy")
    x_cam = (uu.astype(np.float32) - cx) * d / fx
    y_cam = -(vv.astype(np.float32) - cy) * d / fy
    z_cam = -d
    points_camera = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rotation = np.asarray(rotation_camera_to_world, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation_camera_to_world must have shape (3, 3)")
    position = np.asarray(camera_position_world, dtype=np.float64).reshape(3)
    points_world = points_camera @ rotation.T + position
    if floor_y_m is not None:
        valid &= (
            points_world[..., 1]
            >= float(floor_y_m) + float(cfg.floor_min_offset_m)
        ) & (
            points_world[..., 1]
            <= float(floor_y_m) + float(cfg.floor_max_offset_m)
        )

    sampled_smoke = smoke[vv, uu]
    if confidence is None:
        # Dense smoke and missing depth are the two observable confidence
        # degradations available at this layer.  Invalid depth is filtered;
        # smoke makes retained points increasingly uncertain.
        sampled_confidence = np.clip(1.0 - 0.5 * sampled_smoke, 0.05, 1.0)
    else:
        confidence_image = np.asarray(confidence, dtype=np.float32)
        if confidence_image.shape != depth.shape:
            raise ValueError("confidence must share depth shape")
        sampled_confidence = np.clip(confidence_image[vv, uu], 0.0, 1.0)

    return RiskEvidence(
        agent_id=agent_id,
        timestamp_s=timestamp_s,
        points_world=points_world[valid],
        flame=flame[vv, uu][valid],
        temperature_c=temperature[vv, uu][valid],
        smoke=sampled_smoke[valid],
        confidence=sampled_confidence[valid],
        uncertainty=(1.0 - sampled_confidence[valid]),
        privileged_transmittance=privileged,
    )
