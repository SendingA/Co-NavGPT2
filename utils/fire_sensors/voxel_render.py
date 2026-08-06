"""Voxel ray-march: composite RGB / Thermal from a FireWorld field.

This code lives on the **sensor** side, because its role is to
*observe* the fire world from a camera pose, not to model the world
itself. ``utils.fire_world`` only worries about flame/smoke/temperature
voxels and how they evolve in time.

The world is passed in via three ``(Nx, Ny, Nz)`` fields plus origin /
voxel size; the camera is described by intrinsics ``K``, world position,
and rotation. The function returns a dict of smoky-RGB / thermal
arrays that slots straight into the rest of the sensor suite.

The reference implementation in this module is pure NumPy and stateless.
An optional Torch implementation lives in :mod:`voxel_render_torch`; both
backends share the final NumPy/OpenCV display composition below.
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
    [1.00, 0.90, 0.48],   # yellow-white core, still chromatic
], dtype=np.float32)


def flame_lut(intensity: np.ndarray) -> np.ndarray:
    x = np.clip(intensity, 0.0, 1.0)
    out = np.empty(x.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(x, _FLAME_LUT_X, _FLAME_LUT_RGB[:, ch])
    return out


# ---------------------------------------------------------------------------
# Procedural flame noise: sub-voxel texture + time-domain flicker
# ---------------------------------------------------------------------------
# We keep a small 3D random field per renderer instance and tri-linearly
# sample it at the ray-march points. Multiple octaves give a fractal
# look without paying for a full perlin/simplex implementation in
# Python. The world-space coordinates are scaled by ``frequency`` and
# offset by ``time_phase`` so the texture appears to scroll upward
# (looks like a rising plume).
_NOISE_CACHE: Dict[Tuple[int, int, int, int], np.ndarray] = {}


def _hash_noise_field(shape: Tuple[int, int, int], seed: int) -> np.ndarray:
    """A small 3D random scalar field in [0,1] used as a value-noise
    texture. Cached so we don't reallocate every frame.
    """
    key = (shape[0], shape[1], shape[2], int(seed))
    if key not in _NOISE_CACHE:
        rng = np.random.default_rng(int(seed))
        _NOISE_CACHE[key] = rng.random(shape, dtype=np.float32)
    return _NOISE_CACHE[key]


def _sample_noise(points_world: np.ndarray,
                  shape: Tuple[int, int, int],
                  frequency: float,
                  time_phase: float,
                  seed: int) -> np.ndarray:
    """Trilinear sample of a tileable 3D random texture.

    ``points_world`` are positions in metres; we just multiply by
    ``frequency`` and wrap into the noise grid via modulo. Time is
    folded into a y-axis offset so the texture "rises" with t.
    """
    field = _hash_noise_field(shape, seed=seed)
    Nx, Ny, Nz = shape
    f = np.asarray(frequency, dtype=np.float32)
    coord = points_world * f
    coord[..., 1] = coord[..., 1] + np.float32(time_phase)
    # Wrap. ``np.mod`` on float32 can return exactly N at boundaries
    # (e.g. mod(-1e-7, 16) -> 16.0 due to rounding), which then makes
    # floor() yield N and indexes one past the last cell. Wrap the
    # integer indices defensively with another modulo to guarantee
    # i0, i1 stay in [0, N).
    coord_mod = np.mod(coord, np.array([Nx, Ny, Nz], dtype=np.float32))
    i0 = np.floor(coord_mod).astype(np.int32)
    fr = coord_mod - i0.astype(np.float32)
    N_arr = np.array([Nx, Ny, Nz], dtype=np.int32)
    i0 = i0 % N_arr
    i1 = (i0 + 1) % N_arr
    # Trilinear blend.
    c000 = field[i0[..., 0], i0[..., 1], i0[..., 2]]
    c100 = field[i1[..., 0], i0[..., 1], i0[..., 2]]
    c010 = field[i0[..., 0], i1[..., 1], i0[..., 2]]
    c110 = field[i1[..., 0], i1[..., 1], i0[..., 2]]
    c001 = field[i0[..., 0], i0[..., 1], i1[..., 2]]
    c101 = field[i1[..., 0], i0[..., 1], i1[..., 2]]
    c011 = field[i0[..., 0], i1[..., 1], i1[..., 2]]
    c111 = field[i1[..., 0], i1[..., 1], i1[..., 2]]
    fx, fy, fz = fr[..., 0], fr[..., 1], fr[..., 2]
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def fractal_flame_noise(points_world: np.ndarray,
                        time_phase: float,
                        seed: int = 1) -> np.ndarray:
    """Three-octave value noise, centred on 0 with amplitude ~0.5.

    Returns shape ``(..., )`` matching the leading dims of
    ``points_world``. The output is intended to *modulate* a flame
    intensity field, so we centre it on zero: positive values brighten
    the local sample, negative ones dim it. Octaves have geometric
    frequencies (8, 16, 32 per metre roughly) so structure exists at
    multiple scales - this is what gives the flame its fractal "wisp"
    look at any viewing distance.
    """
    n1 = _sample_noise(points_world, (16, 32, 16), frequency=4.0,
                       time_phase=time_phase * 1.6, seed=seed)
    n2 = _sample_noise(points_world, (16, 32, 16), frequency=8.0,
                       time_phase=time_phase * 2.4, seed=seed + 1)
    n3 = _sample_noise(points_world, (16, 32, 16), frequency=16.0,
                       time_phase=time_phase * 3.6, seed=seed + 2)
    # Weighted sum, centred and clipped to [-1, 1].
    raw = 0.55 * n1 + 0.30 * n2 + 0.15 * n3
    return np.clip((raw - 0.5) * 2.0, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Thermal compositing (physical temperature field + FLIR-style display AGC)
# ---------------------------------------------------------------------------
def compose_thermal(
    rgb_clean: np.ndarray,
    temp_max: np.ndarray,
    flame_along: np.ndarray,
    ambient_c: float,
    color_blend: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose a thermal IR image + physical temperature field.

    Returns ``(image_bgr, temperature_c)`` where:

    * ``temperature_c`` is a physical apparent-temperature map in deg C
      built from the ray-marched surface/hot-air estimate plus flame heat.
      RGB is never folded into this array, so downstream temperature
      thresholds remain meaningful.
    * ``image_bgr`` keeps cold scene structure as a dark, low-saturation RGB
      image. Temperature excess progressively replaces that background with
      a grayscale/INFERNO heat palette. A fixed logarithmic response keeps
      ambient, warm objects, and flames comparable between frames instead
      of letting a per-frame maximum wash out the whole image.

    ``temp_max`` retains its historical parameter name for call-site
    compatibility; callers now pass the localized apparent voxel
    temperature, not the maximum sample along the full ray.
    """
    try:
        import cv2
    except Exception:  # pragma: no cover
        cv2 = None  # type: ignore
    if cv2 is not None:
        luma = cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        luma = rgb_clean.mean(axis=-1).astype(np.float32) / 255.0

    # ---- Physical temperature field (returned for downstream detection) --
    # Scene surfaces sit at ambient; only the voxel temperature field and
    # the flame contribute real heat. We deliberately do NOT fold RGB luma
    # into the temperature here — the old code added ``(luma-0.5)*35`` which
    # made bright walls/lamps read as +17 C and dark corners as -17 C, so
    # the "thermal" map was really just a recoloured RGB and the returned
    # ``thermal_temperature`` was unusable for a temperature threshold.
    #
    # ``temp_max`` is the localized apparent voxel temperature produced by
    # the ray-march; ``flame_along`` is the per-ray flame intensity. Take
    # their maximum so a visible flame cannot be dimmed by a cold surface
    # just behind it.
    voxel_excess = np.maximum(temp_max - float(ambient_c), 0.0)
    flame_excess = np.clip(flame_along, 0.0, 1.0) * 600.0
    excess = np.maximum(voxel_excess, flame_excess)
    temperature = float(ambient_c) + excess

    # ---- Display image: dark structure + localized heat -----------------
    # Fixed logarithmic response in excess-Celsius. Unlike percentile AGC,
    # a remote flame cannot redefine the whole frame's black/white anchors.
    # Representative mapping: +5 C -> 0.15, +25 C -> 0.37,
    # +100 C -> 0.64, +600 C -> 1.0.
    excess_display = np.maximum(temperature - float(ambient_c), 0.0)
    norm = np.log1p(excess_display / 5.0) / np.log1p(600.0 / 5.0)
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    gray_u8 = (np.power(norm, 0.82) * 255.0).astype(np.uint8)

    # Cold context is intentionally a dark version of the clean RGB frame,
    # not a fabricated physical temperature. Keeping its weak chroma makes
    # doors, furniture, and people readable without turning ambient walls
    # into hot yellow surfaces.
    rgb_bgr = rgb_clean[..., ::-1].astype(np.float32)
    dark_gain = 0.12 + 0.10 * luma[..., None]
    dark_context = rgb_bgr * dark_gain

    if cv2 is not None:
        heat_gray = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
        ir = cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO).astype(np.float32)
        palette_mix = float(np.clip(color_blend, 0.0, 1.0))
        heat_color = heat_gray * (1.0 - palette_mix) + ir * palette_mix
    else:
        heat_color = np.stack([gray_u8] * 3, axis=-1).astype(np.float32)

    # Temperature controls how much of the palette replaces the structural
    # context. Ambient stays dark; warm bodies and heated objects become
    # increasingly vivid; flames dominate completely.
    heat_alpha = np.clip(norm * 1.55, 0.0, 1.0)[..., None]
    image_bgr = (
        dark_context * (1.0 - heat_alpha) + heat_color * heat_alpha
    )
    image_bgr = np.clip(image_bgr, 0.0, 255.0).astype(np.uint8)
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
    flame_emission_gain: float = 3.2
    flame_k_ext: float = 0.50
    flame_glow_ksize: int = 21
    flame_glow_gain: float = 0.18
    thermal_color_blend: float = 0.85
    thermal_surface_start: float = 0.72
    thermal_air_coupling: float = 0.025
    render_scale: float = 0.5
    # Fraction of the smoke extinction the flame emission ignores while
    # ray-marching. 0 = flame is attenuated by smoke just like the scene
    # (old behaviour); 1 = smoke is invisible to flame radiation.
    # Realistic value 0.9-0.95 because:
    #   * flame is self-luminous; emission > extinction at the source,
    #   * red/orange wavelengths Mie-scatter less in soot (paper Fig. 7),
    #   * hot soot in the flame envelope itself glows.
    flame_smoke_passthrough: float = 0.95
    # Fraction of smoke extinction/scattering removed at a strong flame
    # sample. Hot combustion gases are locally clearer than the surrounding
    # cool soot plume.
    flame_smoke_displacement: float = 0.52
    # Maximum clean-surface texture mixed through pixels containing flame.
    # This models the translucency of an emissive volume and keeps burning
    # furniture visually recognizable.
    flame_surface_reveal: float = 0.13
    # Luminance-preserving compression of integrated flame radiance.
    flame_highlight_compression: float = 1.0

    # ---- Procedural flame texturing --------------------------------------
    # All values below are dimensionless multipliers applied to the
    # fractal value-noise field sampled at ray-march points.
    #
    # ``flame_noise_strength`` is the amplitude of the multiplicative
    # modulation applied to the flame intensity itself: a value of 0.5
    # means the flame can be brightened by up to 50% or dimmed by 50%
    # at sub-voxel scale. 0 disables the effect and the flame renders
    # as a smooth blob (the legacy look).
    flame_noise_strength: float = 0.75
    # ``flame_edge_break`` controls how much noise is applied at the
    # flame edge (where fl_used is between threshold and ~0.4). High
    # values produce ragged "tongues" of flame breaking off the main
    # body; low values keep the silhouette smooth.
    flame_edge_break: float = 1.05
    # ``flame_color_jitter`` shifts the LUT lookup by +/- this fraction
    # of [0,1] using a second noise field, so the same flame intensity
    # produces a range of colors from deep red to a chromatic yellow core.
    flame_color_jitter: float = 0.32
    # ``flame_time_speed`` is how fast (in flicker units per fire-second)
    # the noise pattern advances. Real flames flicker at 5-15 Hz which
    # at speedup=1 maps to ~1.5 phase units per second; we let it
    # scroll faster to look "lively" even at slow simulation rates.
    flame_time_speed: float = 12.0
    # Smoke texture modulation: very mild noise applied to the smoke
    # field so the plume isn't a uniform grey blob. 0.24 means each
    # ray-march point's smoke density can vary up to 24% from its
    # smooth voxel value.
    smoke_noise_strength: float = 0.24


def finalize_volumetric_outputs(
    *,
    rgb_clean: np.ndarray,
    transmittance: np.ndarray,
    smoke_color_acc: np.ndarray,
    flame_color_acc: np.ndarray,
    flame_seen: np.ndarray,
    temp_apparent: np.ndarray,
    output_hw: Tuple[int, int],
    ambient_c: float,
    params: VoxelRenderParams,
) -> Dict[str, np.ndarray]:
    """Finish a ray-marched frame using the shared CPU display pipeline.

    The NumPy and Torch backends both call this function after their
    volumetric integration. Keeping resize, glow and thermal palette handling
    here guarantees that changing the compute backend does not change the
    downstream sensor contract.
    """
    try:
        import cv2
    except Exception:  # pragma: no cover
        cv2 = None  # type: ignore

    h_full, w_full = int(output_hw[0]), int(output_hw[1])
    if (
        transmittance.shape != (h_full, w_full)
        and cv2 is not None
    ):
        transmittance = cv2.resize(
            transmittance, (w_full, h_full), interpolation=cv2.INTER_LINEAR
        )
        smoke_color_acc = cv2.resize(
            smoke_color_acc, (w_full, h_full), interpolation=cv2.INTER_LINEAR
        )
        flame_color_acc = cv2.resize(
            flame_color_acc, (w_full, h_full), interpolation=cv2.INTER_LINEAR
        )
        flame_seen = cv2.resize(
            flame_seen, (w_full, h_full), interpolation=cv2.INTER_LINEAR
        )
        temp_apparent = cv2.resize(
            temp_apparent, (w_full, h_full), interpolation=cv2.INTER_LINEAR
        )

    scene = rgb_clean.astype(np.float32) / 255.0
    # A long ray through a uniformly burning Bounding Box can accumulate
    # radiance well above display white. Clipping that sum independently per
    # channel turns every flame into a textureless white patch. Compress by
    # the RGB peak instead: this bounds the highlight while preserving the
    # red/orange/yellow channel ratios created by the LUT and procedural
    # texture.
    flame_peak = np.max(flame_color_acc, axis=-1)
    compression = max(0.0, float(params.flame_highlight_compression))
    flame_scale = 1.0 / (1.0 + compression * flame_peak)
    flame_display = flame_color_acc * flame_scale[..., None]

    out = (
        scene * transmittance[..., None]
        + smoke_color_acc
        + flame_display
    )

    # A real flame volume is partially transparent. Recover a bounded amount
    # of the clean surface only where a flame is actually visible. This keeps
    # a bedspread, sofa cushion or wood grain legible instead of replacing the
    # whole burning object with emissive fog.
    reveal_strength = float(
        np.clip(params.flame_surface_reveal, 0.0, 0.75)
    )
    if reveal_strength > 0.0:
        reveal_ramp = max(0.20, float(params.flame_threshold) * 4.0)
        reveal_mask = np.clip(
            (flame_seen - float(params.flame_threshold))
            / max(reveal_ramp - float(params.flame_threshold), 1e-4),
            0.0,
            1.0,
        )
        reveal = reveal_strength * reveal_mask[..., None]
        out = out * (1.0 - reveal) + scene * reveal
    out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    # Optional flame glow halo.
    if (
        params.flame_glow_gain > 0.0
        and flame_seen.max() > 1e-3
        and cv2 is not None
    ):
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
        rgb_clean, temp_apparent, flame_seen,
        ambient_c=ambient_c,
        color_blend=params.thermal_color_blend,
    )

    return {
        "image": out_u8,
        "transmittance": transmittance.astype(np.float32, copy=False),
        "flame_mask": (
            flame_seen > params.flame_threshold
        ).astype(np.float32),
        "thermal_image": thermal_image,
        "thermal_temperature": thermal_temp,
    }


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
    t_sim: float = 0.0,
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
    thermal_temperature}``.
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

    # Procedural-noise phase: scrolls with t_sim so the flame
    # "flickers" without us needing to bake animation into the voxel
    # timeline. We use t_sim modulo a large period so the float stays
    # bounded over very long episodes.
    noise_phase = float(np.fmod(float(t_sim) * float(params.flame_time_speed),
                                10_000.0))
    noise_strength = float(np.clip(params.flame_noise_strength, 0.0, 1.5))
    edge_break = float(np.clip(params.flame_edge_break, 0.0, 1.5))
    color_jitter_amp = float(np.clip(params.flame_color_jitter, 0.0, 1.0))
    smoke_noise_amp = float(np.clip(params.smoke_noise_strength, 0.0, 1.0))

    smoke_color_acc = np.zeros((H, W, 3), dtype=np.float32)
    flame_color_acc = np.zeros((H, W, 3), dtype=np.float32)
    T_acc = np.ones((H, W), dtype=np.float32)
    flame_seen = np.zeros((H, W), dtype=np.float32)
    temp_apparent = np.full((H, W), float(ambient_c), dtype=np.float32)

    if ray_hits.any():
        start_h = start[ray_hits]
        end_h = end[ray_hits]
        step_m_h = step_m[ray_hits]
        T_acc_h = np.ones(start_h.shape[0], dtype=np.float32)        # for scene + scatter
        T_acc_flame_h = np.ones(start_h.shape[0], dtype=np.float32)  # for flame emission
        smoke_color_acc_h = np.zeros(
            (start_h.shape[0], 3), dtype=np.float32
        )
        flame_color_acc_h = np.zeros(
            (start_h.shape[0], 3), dtype=np.float32
        )
        flame_seen_h = np.zeros(start_h.shape[0], dtype=np.float32)
        temp_surface_num_h = np.zeros(start_h.shape[0], dtype=np.float32)
        temp_surface_den_h = np.zeros(start_h.shape[0], dtype=np.float32)
        temp_path_excess_h = np.zeros(start_h.shape[0], dtype=np.float32)
        surface_start = float(np.clip(params.thermal_surface_start, 0.0, 0.95))
        air_coupling = float(np.clip(params.thermal_air_coupling, 0.0, 1.0))
        # A surface-anchored noise value is shared by all samples on one ray.
        # Blending it with 3-D turbulence prevents the detail from averaging
        # to one flat colour when a ray traverses a large burning AABB.
        ray_intensity_noise_h = None
        ray_color_noise_h = None
        if noise_strength > 0.0:
            ray_intensity_noise_h = fractal_flame_noise(
                end_h,
                time_phase=noise_phase,
                seed=17,
            )
        if color_jitter_amp > 0.0:
            ray_color_noise_h = fractal_flame_noise(
                end_h,
                time_phase=noise_phase * 0.7,
                seed=23,
            )

        for i in range(N):
            t = np.float32(ts[i])
            pts = (1.0 - t) * start_h + t * end_h
            sm, fl, te = sample_voxels_trilinear_multi(
                [smoke_field, flame_field, temp_field],
                pts, origin, voxel_m,
            )
            # The sampler returns zero outside the voxel AABB. Thermal
            # aggregation treats that as ambient rather than freezing air.
            te = np.where(te > 0.0, te, float(ambient_c)).astype(np.float32)
            # Smoke texture: mild noise so the plume has wisps and
            # bands instead of being a uniform grey wall. Modulation
            # is multiplicative so dense smoke stays dense.
            if smoke_noise_amp > 0.0:
                n_smoke = fractal_flame_noise(pts, time_phase=noise_phase * 0.4,
                                              seed=11)
                sm = np.clip(sm * (1.0 + smoke_noise_amp * n_smoke), 0.0, 1.5)
            # Procedural flame texture: use coherent noise as a density
            # coverage field, not just a small brightness wobble. The core
            # stays stable while edge samples can nearly disappear or stretch
            # into bright tongues. This survives integration along a ray much
            # better than the old symmetric multiply-and-average modulation.
            if noise_strength > 0.0:
                n_intensity = fractal_flame_noise(pts, time_phase=noise_phase,
                                                  seed=1)
                if ray_intensity_noise_h is not None:
                    n_intensity = np.clip(
                        0.35 * n_intensity
                        + 0.65 * ray_intensity_noise_h,
                        -1.0,
                        1.0,
                    )
                fl_norm = np.clip(fl, 0.0, 1.0)
                edge_w = 4.0 * fl_norm * (1.0 - fl_norm)
                detail_weight = np.clip(
                    noise_strength * (0.30 + 0.70 * edge_w),
                    0.0,
                    0.95,
                )
                density_texture = np.clip(
                    0.55 + 0.95 * n_intensity, 0.05, 1.45
                )
                edge_texture = np.clip(
                    0.65 + edge_break * n_intensity, 0.05, 1.40
                )
                core_detail = (
                    (1.0 - detail_weight)
                    + detail_weight * density_texture
                )
                silhouette_detail = (
                    (1.0 - edge_w) + edge_w * edge_texture
                )
                fl = np.clip(
                    fl * core_detail * silhouette_detail,
                    0.0,
                    1.2,
                )

            fl_used = np.clip((fl - thr_lo) * thr_inv, 0.0, 1.0) * fl

            # Color jitter via a second independent noise field. Same
            # phase, different seed: this drifts the LUT lookup point
            # so adjacent voxels with identical intensity show as red
            # vs orange vs yellow, the way real flame colour bands
            # shift unpredictably.
            if color_jitter_amp > 0.0:
                n_color = fractal_flame_noise(pts, time_phase=noise_phase * 0.7,
                                              seed=5)
                if ray_color_noise_h is not None:
                    n_color = np.clip(
                        0.35 * n_color + 0.65 * ray_color_noise_h,
                        -1.0,
                        1.0,
                    )
                lut_input = np.clip(fl_used + color_jitter_amp * n_color, 0.0, 1.0)
            else:
                lut_input = fl_used

            # Hot combustion gas locally displaces soot. Reducing both grey
            # scattering and smoke extinction at a flame sample exposes
            # chromatic flame structure and some of the burning surface.
            smoke_displacement = float(
                np.clip(params.flame_smoke_displacement, 0.0, 1.0)
            )
            flame_presence = np.clip(fl_used / 0.35, 0.0, 1.0)
            sm_visible = sm * (
                1.0 - smoke_displacement * flame_presence
            )

            # Two separate optical depths: the scene path sees displaced
            # smoke + translucent flame extinction; the flame path sees only a
            # fraction of the smoke (passthrough) plus flame's own
            # absorption. Both share step_m_h.
            sigma_scene = smoke_k * sm_visible + flame_k * fl_used
            sigma_flame = (
                smoke_k_for_flame * sm_visible + flame_k * fl_used
            )
            T_step_scene = np.exp(-(sigma_scene * step_m_h))
            T_step_flame = np.exp(-(sigma_flame * step_m_h))

            flame_rgb = flame_lut(lut_input)
            emission = (
                flame_rgb * (fl_used * emission_gain)[:, None]
                * step_m_h[:, None]
            )
            scatter = (
                smoke_color.reshape(1, 3)
                * (sm_visible * smoke_k * step_m_h)[:, None]
            )

            flame_color_acc_h = (
                flame_color_acc_h
                + T_acc_flame_h[:, None] * emission
            )
            smoke_color_acc_h = (
                smoke_color_acc_h
                + T_acc_h[:, None] * scatter
            )
            T_acc_h = T_acc_h * T_step_scene
            T_acc_flame_h = T_acc_flame_h * T_step_flame
            np.maximum(flame_seen_h, fl_used, out=flame_seen_h)

            # Apparent thermal sensing is surface-dominant. The last part of
            # the depth ray estimates the visible object's temperature;
            # mean hot air along the path contributes only weakly. This
            # prevents one hot plume voxel anywhere on a long ray from
            # saturating the background wall.
            te_excess = np.maximum(te - float(ambient_c), 0.0)
            temp_path_excess_h += te_excess / np.float32(N)
            surface_phase = max(
                0.0,
                (float(t) - surface_start) / max(1.0 - surface_start, 1e-6),
            )
            surface_weight = np.float32(surface_phase * surface_phase)
            if surface_weight > 0.0:
                temp_surface_num_h += te_excess * surface_weight
                temp_surface_den_h += surface_weight

        surface_excess_h = temp_surface_num_h / np.maximum(
            temp_surface_den_h, np.float32(1e-6)
        )
        apparent_excess_h = np.maximum(
            surface_excess_h,
            temp_path_excess_h * np.float32(air_coupling),
        )
        temp_apparent_h = float(ambient_c) + apparent_excess_h

        smoke_color_acc[ray_hits] = smoke_color_acc_h
        flame_color_acc[ray_hits] = flame_color_acc_h
        T_acc[ray_hits] = T_acc_h
        flame_seen[ray_hits] = flame_seen_h
        temp_apparent[ray_hits] = temp_apparent_h

    return finalize_volumetric_outputs(
        rgb_clean=rgb_clean,
        transmittance=T_acc,
        smoke_color_acc=smoke_color_acc,
        flame_color_acc=flame_color_acc,
        flame_seen=flame_seen,
        temp_apparent=temp_apparent,
        output_hw=(H_full, W_full),
        ambient_c=ambient_c,
        params=params,
    )
