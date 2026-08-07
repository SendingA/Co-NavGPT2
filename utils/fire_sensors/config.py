"""Single dataclass holding every knob of the fire-sensor simulator.

Splitting per-sensor config into nested dataclasses keeps the public
surface backward-compatible (callers still use :class:`FireSensorConfig`)
while letting each sensor module own its own parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Per-sensor sub-configs
# ---------------------------------------------------------------------------


@dataclass
class SmokeConfig:
    """Shared smoke parameters for the depth / lidar / radar sensors.

    Visibility V ≈ 2.3 / k (Jin's eq.; Starr & Lattimer 2014, eq. 2).
    These knobs drive the smoke-layer clipping and range-dependent noise
    in the depth / lidar / radar modalities. The smoky RGB and thermal
    images themselves are produced exclusively by the voxel renderer
    (:mod:`utils.fire_sensors.voxel_render`).
    """

    smoke_density: float = 0.6  # [0,1] dimensionless control
    smoke_color_rgb: Tuple[int, int, int] = (72, 72, 72)
    smoke_k_max: float = 2.3  # density=1 -> V ≈ 1 m


@dataclass
class DepthDegradeConfig:
    """LIDAR-style smoke-aware noise + dropout + quantization."""

    sigma_base_m: float = 0.01
    sigma_range_m: float = 0.005   # 0.5% of range
    sigma_smoke_m: float = 0.05    # extra error per meter under heavy smoke
    quant_m: float = 0.02          # cm-level quantization
    dropout_max: float = 0.5       # max dropout prob at density=1
    clip_to_smoke: bool = True     # below visibility, see only smoke layer


@dataclass
class RadarConfig:
    """Single-chip mmWave radar (Starr & Lattimer 2014; RadarHD 2023)."""

    range_bins: int = 256
    az_bins: int = 64
    max_range_m: float = 10.0
    az_fov_deg: float = 90.0          # ±90°
    elev_fov_deg: float = 25.0        # ±25° in elevation
    sinc_sigma_bins: float = 1.2      # azimuth sidelobe spread
    noise_std: float = 0.03
    threshold: float = 0.18           # heatmap → point cloud cut-off

    # 3D point-cloud generation mode.
    #   'raw'     - threshold the range-az heatmap (no elevation info)
    #   'learned' - bypass the network, return lidar-like 3D cloud
    mode: str = "learned"

    # 'learned' mode parameters: tuned to roughly match the paper's
    # post-training median Hausdorff error of ~24 cm.
    learned_noise_m: float = 0.10
    learned_stride: int = 4
    learned_target_points: int = 4000

    # BEV preview canvas size.
    bev_size_px: int = 360


@dataclass
class LidarConfig:
    """360°-style spinning LIDAR (Starr & Lattimer 2014, Fig. 5).

    The normal runtime consumes four yaw-rotated Habitat depth sensors and
    stitches their 90° slices into a full surround scan. A forward-depth
    fallback remains for legacy callers that do not install those sensors.
    """

    max_range_m: float = 10.0
    fov_deg_yaw: float = 360.0     # logical (not enforced by the depth source)
    elev_fov_deg: float = 30.0
    stride: int = 2                # subsample depth pixels for speed

    # Range-/density-dependent Gaussian noise (m).
    sigma_base_m: float = 0.01
    sigma_range_m: float = 0.005
    sigma_smoke_m: float = 0.05

    dropout_max: float = 0.5
    clip_to_smoke: bool = True

    # BEV preview.
    bev_size_px: int = 360
    z_color_range_m: Tuple[float, float] = (-0.6, 1.6)


@dataclass
class VoxelSmokeConfig:
    """Voxel-driven smoky-RGB / Thermal camera (FireWorld observer).

    These knobs control the ray-march that turns the
    :class:`utils.fire_world.scene.FireScene` voxels into a per-step
    RGB / Thermal image. They belong on the sensor config because the
    renderer is just *how the camera looks at the world*.
    """

    n_steps: int = 16                  # ray-march samples per pixel
    smoke_k_ext: float = 4.0           # extinction coefficient on smoke voxels (1/m)
    smoke_color_rgb: Tuple[int, int, int] = (72, 72, 72)
    # Trilinear-friendly cutoff: propagation pins flame source voxels at
    # ~0.6, but trilinear interpolation bleeds the boundary down to
    # 0.05-0.30. The previous default 0.20 zeroed everything except the
    # source core (rendered as one or two pixels). 0.04 keeps the flame
    # envelope visible.
    flame_threshold: float = 0.04
    # Keep the self-luminous flame bright without clipping every burning
    # surface to white. Highlight compression in the compositor preserves
    # the orange/yellow colour ratios after the ray integral.
    flame_emission_gain: float = 3.2
    flame_k_ext: float = 0.50
    flame_glow_ksize: int = 21
    flame_glow_gain: float = 0.18
    # Fraction of the smoke extinction the flame radiation ignores.
    # 0 -> flame attenuates exactly like the scene RGB (will be eaten
    # by smoke); 1 -> smoke is invisible to flame radiation. Realistic
    # ~0.95: visible-band flame leaks through medium-thick smoke
    # (Starr & Lattimer 2014, Fig. 7).
    flame_smoke_passthrough: float = 0.95
    # Hot flame locally displaces/consumes soot, so smoke extinction and
    # grey scattering are reduced at flame samples. This creates clear
    # orange tongues instead of embedding one white blob in grey fog.
    flame_smoke_displacement: float = 0.52
    # Flames are emissive volumes, not opaque decals. Blend a controlled
    # fraction of the clean surface texture back through visible flame so
    # burning beds, sofas and tables remain recognizable.
    flame_surface_reveal: float = 0.13
    # Luminance-preserving Reinhard compression applied only to accumulated
    # flame radiance. 1.0 prevents long ray paths from saturating all three
    # RGB channels while retaining their orange/yellow ratios.
    flame_highlight_compression: float = 1.0
    # Thermal display palette. Cold pixels always retain a darkened RGB
    # structure; this value only blends grayscale heat toward INFERNO as
    # apparent temperature rises.
    thermal_color_blend: float = 0.85
    # A thermal camera primarily measures the visible surface at the depth
    # endpoint. Only the final part of each ray contributes to that surface
    # estimate; hot air along the full ray is included separately below.
    thermal_surface_start: float = 0.72
    # Fraction of mean line-of-sight hot-air excess folded into apparent
    # temperature. Keep this small: long rays through a hot room must not
    # inherit the single hottest voxel and become uniformly saturated.
    thermal_air_coupling: float = 0.025
    render_scale: float = 0.5          # fraction of camera resolution

    # --- Compute backend ------------------------------------------------
    # ``auto`` selects Torch when CUDA is visible and otherwise keeps the
    # NumPy reference path. The render device is deliberately independent
    # from Habitat-Sim's --gpu_id so FireWorld can use another GPU.
    render_backend: str = "auto"       # auto | numpy | torch
    render_device: str = "auto"        # auto | cpu | cuda[:N]
    render_dtype: str = "float16"      # FP16 volume on CUDA; CPU uses FP32
    # Bound peak memory by tiling rays when H*W*n_steps exceeds this value.
    max_sample_points: int = 2_000_000

    # --- Procedural flame texturing (pure eye-candy) --------------------
    # These drive the fractal value-noise that makes the flame flicker
    # and wisp. Each non-zero term costs ~3 trilinear noise gathers per
    # ray-march step, so they dominate the render cost (see the
    # micro-benchmark in docs). For navigation / benchmarking set them
    # to 0 (or use --fire_fast) to get a 4-5x speedup; keep them on
    # only for teleop demos where the fire needs to look alive.
    flame_noise_strength: float = 0.75
    flame_edge_break: float = 1.05
    flame_color_jitter: float = 0.32
    flame_time_speed: float = 12.0
    smoke_noise_strength: float = 0.24


# ---------------------------------------------------------------------------
# Top-level config (kept compatible with the original flat dataclass)
# ---------------------------------------------------------------------------


@dataclass
class FireSensorConfig:
    """All knobs of the fire-scene sensor simulator.

    Common fields stay at the top level so existing call sites
    (``FireSensorConfig(max_depth_m=..., smoke_density=...)``) keep
    working. Per-sensor configs can be passed explicitly via the nested
    dataclasses for fine control.
    """

    # --- common (used by multiple sensors) ------------------------------
    max_depth_m: float = 5.0
    hfov_deg: float = 79.0

    # --- per-sensor ----------------------------------------------------
    # ``smoke`` holds the shared smoke params consumed by the depth /
    # lidar / radar modalities. The smoky RGB + thermal images are always
    # produced by the voxel renderer (``voxel``).
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    depth: DepthDegradeConfig = field(default_factory=DepthDegradeConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    voxel: VoxelSmokeConfig = field(default_factory=VoxelSmokeConfig)

    # --- IO -------------------------------------------------------------
    save_npz: bool = False
    # Save the dashboard image alongside per-modality PNGs.
    save_dashboard: bool = True
    dashboard_size: Tuple[int, int] = (2000, 900)  # (W, H) of the big image

    # --- Convenience ----------------------------------------------------
    # ``smoke_density`` is exposed flat so callers can write
    # ``FireSensorConfig(smoke_density=0.7)``; it is synced into the
    # nested SmokeConfig the depth / lidar / radar sensors read from.
    smoke_density: float = 0.6

    def __post_init__(self) -> None:
        self.smoke.smoke_density = float(self.smoke_density)
