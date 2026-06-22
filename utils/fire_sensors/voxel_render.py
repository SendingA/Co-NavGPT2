"""Voxel ray-march: composite RGB / Thermal from a FireWorld field.

This is the math previously hosted in ``utils/fire_world/runtime.py``
(``FireWorldRenderer``). It now lives on the **sensor** side, because
the role of this code is to *observe* the fire world from a camera
pose, not to model the world itself. ``utils.fire_world`` should only
worry about flame/smoke/temperature voxels and how they evolve in time.

The world is passed in via three ``(Nx, Ny, Nz)`` fields plus origin /
voxel size; the camera is described by intrinsics ``K``, world position,
and rotation. The function returns the same dict that
``SmokeRGBSensor.process`` does so it slots straight into the rest of
the sensor suite.

Everything is pure-numpy and stateless — :class:`VoxelSmokeSensor` is
the BaseSensor wrapper that owns the configuration and the camera
intrinsics.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-pixel ray construction: cam-frame depth -> world-space segments
# ---------------------------------------------------------------------------
def build_pixel_rays(
    depth_m: np.ndarray,
    K,
    cam_pos: np.ndarray,
    R_cam2world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(start_xyz, end_xyz)`` world positions for each pixel.

    Habitat depth values are perpendicular distances along the camera
    -Z axis (z-buffer style), so the world end point is::

        x_cam = (u - cx) * d / fx
        y_cam = -(v - cy) * d / fy        # image y down vs +Y_cam up
        z_cam = -d                         # cam looks down -Z
        end_world = R @ (x_cam, y_cam, z_cam) + cam_pos
    """
    H, W = depth_m.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    U, V = np.meshgrid(u, v)
    fx = float(K.fx); fy = float(K.fy); cx = float(K.cx); cy = float(K.cy)
    x_cam = (U - cx) * depth_m / fx
    y_cam = -(V - cy) * depth_m / fy
    z_cam = -depth_m
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    pts_world = pts_cam @ R_cam2world.T + cam_pos[None, None]
    start = np.broadcast_to(cam_pos[None, None, :].astype(np.float32),
                            (H, W, 3)).copy()
    return start, pts_world.astype(np.float32)


# ---------------------------------------------------------------------------
# Trilinear voxel sampler (multi-field gather for speed)
# ---------------------------------------------------------------------------
def sample_voxels_trilinear_multi(
    fields: List[np.ndarray],
    points_world: np.ndarray,
    origin: np.ndarray,
    voxel_m: float,
) -> List[np.ndarray]:
    """Trilinear gather of several ``(Nx, Ny, Nz)`` fields at once.

    All fields are sampled at the same world positions, so we share the
    8 corner indices and 7 lerp weights between them. Out-of-grid
    samples produce 0 (Neumann fade-out).
    """
    if not fields:
        return []
    Nx, Ny, Nz = fields[0].shape
    inv_v = np.float32(1.0 / voxel_m)
    coord = (points_world - origin.astype(np.float32)) * inv_v - np.float32(0.5)
    valid = (
        (coord[..., 0] >= -0.5) & (coord[..., 0] <= Nx - 0.5)
        & (coord[..., 1] >= -0.5) & (coord[..., 1] <= Ny - 0.5)
        & (coord[..., 2] >= -0.5) & (coord[..., 2] <= Nz - 0.5)
    )
    outs = [np.zeros(coord.shape[:-1], dtype=np.float32) for _ in fields]
    if not valid.any():
        return outs

    cv = coord[valid]
    i0 = np.floor(cv).astype(np.int32)
    f = cv - i0.astype(np.float32)
    i1 = i0 + 1
    np.clip(i0[:, 0], 0, Nx - 1, out=i0[:, 0])
    np.clip(i0[:, 1], 0, Ny - 1, out=i0[:, 1])
    np.clip(i0[:, 2], 0, Nz - 1, out=i0[:, 2])
    np.clip(i1[:, 0], 0, Nx - 1, out=i1[:, 0])
    np.clip(i1[:, 1], 0, Ny - 1, out=i1[:, 1])
    np.clip(i1[:, 2], 0, Nz - 1, out=i1[:, 2])

    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    one_mfx = 1.0 - fx
    one_mfy = 1.0 - fy
    one_mfz = 1.0 - fz
    w000 = one_mfx * one_mfy * one_mfz
    w100 = fx       * one_mfy * one_mfz
    w010 = one_mfx * fy       * one_mfz
    w110 = fx       * fy       * one_mfz
    w001 = one_mfx * one_mfy * fz
    w101 = fx       * one_mfy * fz
    w011 = one_mfx * fy       * fz
    w111 = fx       * fy       * fz

    ix0, iy0, iz0 = i0[:, 0], i0[:, 1], i0[:, 2]
    ix1, iy1, iz1 = i1[:, 0], i1[:, 1], i1[:, 2]
    for k, fld in enumerate(fields):
        sampled = (
            fld[ix0, iy0, iz0] * w000
            + fld[ix1, iy0, iz0] * w100
            + fld[ix0, iy1, iz0] * w010
            + fld[ix1, iy1, iz0] * w110
            + fld[ix0, iy0, iz1] * w001
            + fld[ix1, iy0, iz1] * w101
            + fld[ix0, iy1, iz1] * w011
            + fld[ix1, iy1, iz1] * w111
        )
        outs[k][valid] = sampled.astype(np.float32)
    return outs


# ---------------------------------------------------------------------------
# Flame colour LUT
# ---------------------------------------------------------------------------
_FLAME_LUT_X = np.array([0.00, 0.10, 0.30, 0.55, 0.80, 1.00], dtype=np.float32)
_FLAME_LUT_RGB = np.array([
    [0.05, 0.00, 0.00],   # almost black (very faint embers)
    [0.40, 0.05, 0.02],   # dark red
    [0.95, 0.30, 0.05],   # deep orange
    [1.00, 0.60, 0.10],   # orange
    [1.00, 0.85, 0.30],   # yellow
    [1.00, 0.97, 0.78],   # near white core
], dtype=np.float32)


def flame_lut(intensity: np.ndarray) -> np.ndarray:
    x = np.clip(intensity, 0.0, 1.0)
    out = np.empty(x.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(x, _FLAME_LUT_X, _FLAME_LUT_RGB[:, ch])
    return out


# ---------------------------------------------------------------------------
# Thermal compositing (luma + flame contribution + ambient)
# ---------------------------------------------------------------------------
def compose_thermal(
    rgb_clean: np.ndarray,
    temp_max: np.ndarray,
    flame_along: np.ndarray,
    ambient_c: float,
    color_blend: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import cv2
    except Exception:  # pragma: no cover
        cv2 = None  # type: ignore
    if cv2 is not None:
        luma = cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        luma = rgb_clean.mean(axis=-1).astype(np.float32) / 255.0
    scene_field = (luma - 0.5) * 35.0
    temperature = float(ambient_c) + scene_field + np.maximum(temp_max - float(ambient_c), 0.0)

    t_lo = float(np.percentile(temperature, 2))
    t_hi = float(max(np.percentile(temperature, 99.5), 600.0 * 0.6))
    norm = np.clip((temperature - t_lo) / max(t_hi - t_lo, 1e-3), 0.0, 1.0)
    norm = np.power(norm, 0.7)
    gray_u8 = (norm * 255).astype(np.uint8)
    if cv2 is not None:
        image_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        if color_blend > 0.0:
            ir = cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)
            a = float(np.clip(color_blend, 0.0, 1.0))
            image_bgr = cv2.addWeighted(image_bgr, 1 - a, ir, a, 0)
    else:
        image_bgr = np.stack([gray_u8] * 3, axis=-1)
    return image_bgr, temperature.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level: render one frame from a sampled voxel field at a given time
# ---------------------------------------------------------------------------
@dataclass
class VoxelRenderParams:
    max_depth_m: float = 5.0
    n_steps: int = 16
    smoke_k_ext: float = 1.5
    smoke_color_rgb: Tuple[int, int, int] = (180, 180, 180)
    # Cut-off applied to interpolated flame samples. Values above 0.1
    # were too aggressive: the propagation pins source voxels at ~0.6,
    # but trilinear interpolation bleeds the boundary down to 0.05-0.30,
    # so a 0.20 threshold zeroed everything except the source core
    # (one or two pixels). 0.04 lets the flame envelope render as a
    # proper volumetric blob.
    flame_threshold: float = 0.04
    flame_emission_gain: float = 8.0
    flame_k_ext: float = 0.8
    flame_glow_ksize: int = 41
    flame_glow_gain: float = 0.55
    thermal_color_blend: float = 0.0
    render_scale: float = 0.5
    # Fraction of the smoke extinction the flame emission ignores while
    # ray-marching. 0 = flame is attenuated by smoke just like the scene
    # (old behaviour); 1 = smoke is invisible to flame radiation.
    # Realistic value 0.9-0.95 because:
    #   * flame is self-luminous; emission > extinction at the source,
    #   * red/orange wavelengths Mie-scatter less in soot (paper Fig. 7),
    #   * hot soot in the flame envelope itself glows.
    flame_smoke_passthrough: float = 0.95


def volumetric_composite(
    *,
    rgb_clean: np.ndarray,
    depth_m: np.ndarray,
    cam_pos_world: np.ndarray,
    R_cam2world: np.ndarray,
    flame_field: np.ndarray,
    smoke_field: np.ndarray,
    temp_field: np.ndarray,
    origin: np.ndarray,
    voxel_m: float,
    grid_shape: Tuple[int, int, int],
    ambient_c: float,
    camera_K,
    params: VoxelRenderParams,
) -> Dict[str, np.ndarray]:
    """Front-to-back emission/absorption composite.

    For each ray we accumulate::

        sigma_i  = smoke_k_ext * smoke_i  +  flame_k_ext * flame_i
        T_i      = exp(-sigma_i * step_m)
        emission = flame_color_lut(flame_i) * flame_i * gain * step_m
        scatter  = smoke_color * smoke_i * step_m * smoke_k_ext

        color_acc += T_acc * (emission + scatter)
        T_acc     *= T_i

    The remaining ``T_acc`` weights the original scene RGB at the end.

    Returns a dict ``{image, transmittance, flame_mask, thermal_image,
    thermal_temperature}`` matching the legacy FireWorldRenderer.render().
    """
    try:
        import cv2
    except Exception:  # pragma: no cover
        cv2 = None  # type: ignore

    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    depth_m = np.clip(depth_m.astype(np.float32), 0.0, params.max_depth_m)

    # Optional render-scale downsample.
    scale = float(np.clip(params.render_scale, 0.05, 1.0))
    H_full, W_full = depth_m.shape
    if scale < 1.0 and cv2 is not None:
        Wd = max(64, int(round(W_full * scale)))
        Hd = max(64, int(round(H_full * scale)))
        depth_used = cv2.resize(depth_m, (Wd, Hd), interpolation=cv2.INTER_AREA)
        K_eff = SimpleNamespace(
            cx=camera_K.cx * (Wd / float(W_full)),
            cy=camera_K.cy * (Hd / float(H_full)),
            fx=camera_K.fx * (Wd / float(W_full)),
            fy=camera_K.fy * (Hd / float(H_full)),
        )
    else:
        K_eff = camera_K
        depth_used = depth_m
        Hd, Wd = H_full, W_full

    flame_field = flame_field.astype(np.float32, copy=False)
    smoke_field = smoke_field.astype(np.float32, copy=False)
    temp_field = temp_field.astype(np.float32, copy=False)

    start, end = build_pixel_rays(
        depth_used, K_eff, cam_pos_world.astype(np.float32),
        R_cam2world.astype(np.float32),
    )
    H, W = depth_used.shape

    amin = origin.astype(np.float32)
    amax = amin + np.asarray(grid_shape, dtype=np.float32) * np.float32(voxel_m)
    seg_min = np.minimum(start, end)
    seg_max = np.maximum(start, end)
    ray_hits = (
        (seg_max[..., 0] >= amin[0]) & (seg_min[..., 0] <= amax[0])
        & (seg_max[..., 1] >= amin[1]) & (seg_min[..., 1] <= amax[1])
        & (seg_max[..., 2] >= amin[2]) & (seg_min[..., 2] <= amax[2])
    )

    N = max(2, int(params.n_steps))
    ts = np.linspace(0.0, 1.0, N, dtype=np.float32)
    chord = np.linalg.norm(end - start, axis=-1).astype(np.float32)
    step_m = chord / np.float32(N - 1)

    smoke_color = (
        np.array(params.smoke_color_rgb, dtype=np.float32) / 255.0
    ).reshape(1, 1, 3)
    smoke_k = np.float32(params.smoke_k_ext)
    flame_k = np.float32(params.flame_k_ext)
    emission_gain = np.float32(params.flame_emission_gain)
    # How much of the smoke extinction the flame emission ignores. A
    # value of 0.85 means flame radiance is attenuated only by 15% of
    # the smoke optical depth, plus the flame's own self-extinction.
    flame_passthrough = float(np.clip(params.flame_smoke_passthrough, 0.0, 1.0))
    smoke_k_for_flame = smoke_k * np.float32(1.0 - flame_passthrough)
    t_lo = float(params.flame_threshold)
    # The ramp width (t_lo .. t_hi) determines how soft the flame edge
    # is. With t_lo=0.04 we want the ramp to cover roughly one full
    # voxel of trilinear bleed, i.e. an extra 0.05 above t_lo.
    t_hi = t_lo + max(0.05, t_lo)
    thr_inv = np.float32(1.0 / max(t_hi - t_lo, 1e-3))
    thr_lo = np.float32(t_lo)

    color_acc = np.zeros((H, W, 3), dtype=np.float32)
    T_acc = np.ones((H, W), dtype=np.float32)
    flame_seen = np.zeros((H, W), dtype=np.float32)
    temp_max = np.full((H, W), float(ambient_c), dtype=np.float32)

    if ray_hits.any():
        start_h = start[ray_hits]
        end_h = end[ray_hits]
        step_m_h = step_m[ray_hits]
        T_acc_h = np.ones(start_h.shape[0], dtype=np.float32)        # for scene + scatter
        T_acc_flame_h = np.ones(start_h.shape[0], dtype=np.float32)  # for flame emission
        color_acc_h = np.zeros((start_h.shape[0], 3), dtype=np.float32)
        flame_seen_h = np.zeros(start_h.shape[0], dtype=np.float32)
        temp_max_h = np.full(start_h.shape[0], float(ambient_c), dtype=np.float32)

        for i in range(N):
            t = np.float32(ts[i])
            pts = (1.0 - t) * start_h + t * end_h
            sm, fl, te = sample_voxels_trilinear_multi(
                [smoke_field, flame_field, temp_field],
                pts, origin, voxel_m,
            )
            fl_used = np.clip((fl - thr_lo) * thr_inv, 0.0, 1.0) * fl

            # Two separate optical depths: the scene path sees full
            # smoke + flame extinction; the flame path sees only a
            # fraction of the smoke (passthrough) plus flame's own
            # absorption. Both share step_m_h.
            sigma_scene = smoke_k * sm + flame_k * fl_used
            sigma_flame = smoke_k_for_flame * sm + flame_k * fl_used
            T_step_scene = np.exp(-(sigma_scene * step_m_h))
            T_step_flame = np.exp(-(sigma_flame * step_m_h))

            flame_rgb = flame_lut(fl_used)
            emission = (
                flame_rgb * (fl_used * emission_gain)[:, None]
                * step_m_h[:, None]
            )
            scatter = (
                smoke_color.reshape(1, 3)
                * (sm * smoke_k * step_m_h)[:, None]
            )

            color_acc_h = (
                color_acc_h
                + T_acc_flame_h[:, None] * emission   # flame uses its own T
                + T_acc_h[:, None] * scatter           # scatter uses scene T
            )
            T_acc_h = T_acc_h * T_step_scene
            T_acc_flame_h = T_acc_flame_h * T_step_flame
            np.maximum(flame_seen_h, fl_used, out=flame_seen_h)
            np.maximum(temp_max_h, te, out=temp_max_h)

        color_acc[ray_hits] = color_acc_h
        T_acc[ray_hits] = T_acc_h
        flame_seen[ray_hits] = flame_seen_h
        temp_max[ray_hits] = temp_max_h

    if scale < 1.0 and (Hd != H_full or Wd != W_full) and cv2 is not None:
        T_acc = cv2.resize(T_acc, (W_full, H_full), interpolation=cv2.INTER_LINEAR)
        color_acc = cv2.resize(color_acc, (W_full, H_full), interpolation=cv2.INTER_LINEAR)
        flame_seen = cv2.resize(flame_seen, (W_full, H_full), interpolation=cv2.INTER_LINEAR)
        temp_max = cv2.resize(temp_max, (W_full, H_full), interpolation=cv2.INTER_LINEAR)

    scene = rgb_clean.astype(np.float32) / 255.0
    out = scene * T_acc[..., None] + color_acc
    out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    # Optional flame glow halo.
    if params.flame_glow_gain > 0.0 and flame_seen.max() > 1e-3 and cv2 is not None:
        k = max(3, int(params.flame_glow_ksize) | 1)
        glow = cv2.GaussianBlur(flame_seen, (k, k), 0)
        m = float(glow.max())
        if m > 1e-6:
            glow = glow / m
        glow = np.clip(glow * float(params.flame_glow_gain), 0.0, 1.0)
        halo_col = np.array([1.00, 0.55, 0.10], dtype=np.float32) * 255.0
        halo = halo_col.reshape(1, 1, 3) * glow[..., None]
        out_u8 = np.clip(
            out_u8.astype(np.float32) * (1.0 - 0.35 * glow[..., None])
            + 0.35 * halo,
            0, 255,
        ).astype(np.uint8)

    thermal_image, thermal_temp = compose_thermal(
        rgb_clean, temp_max, flame_seen,
        ambient_c=ambient_c,
        color_blend=params.thermal_color_blend,
    )

    return {
        "image": out_u8,
        "transmittance": T_acc,
        "flame_mask": (flame_seen > params.flame_threshold).astype(np.float32),
        "thermal_image": thermal_image,
        "thermal_temperature": thermal_temp,
    }
