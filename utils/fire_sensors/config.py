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
class SmokeRGBConfig:
    """Beer-Lambert smoke filter on RGB.

    Visibility V ≈ 2.3 / k (Jin's eq.; Starr & Lattimer 2014, eq. 2).

    Flames are self-luminous, and their radiation (especially the red/orange
    band) penetrates smoke far better than reflected ambient light. The
    ``flame_*`` knobs below let the smoke filter preserve flame pixels and
    bleed an orange glow into the surrounding smoke, mimicking how human
    observers and visible-light cameras still perceive flames through
    light/medium smoke (see Starr & Lattimer 2014, Fig. 7).
    """

    smoke_density: float = 0.6  # [0,1] dimensionless control
    smoke_color_rgb: Tuple[int, int, int] = (180, 180, 180)
    smoke_k_max: float = 2.3  # density=1 -> V ≈ 1 m

    # --- Flame penetration (visible-band radiation through smoke) -----
    # Per-pixel additional transmittance applied to flame regions: 0 disables
    # the effect, 1 makes flames fully visible regardless of smoke density.
    flame_smoke_passthrough: float = 0.9
    # Gaussian blur kernel used to spread the flame mask into a soft glow
    # halo around the flame.
    flame_glow_ksize: int = 61
    # Strength of the glow halo (controls how much the smoke transmittance
    # is restored *around* flame pixels).
    flame_glow_gain: float = 0.55
    # How strongly the surrounding smoke gets tinted by the flame color.
    flame_color_bleed: float = 0.35
    # HSV ranges for the flame detector (mirror ThermalConfig defaults so the
    # RGB filter and the thermal sensor agree on what constitutes a flame).
    flame_hsv_low1: Tuple[int, int, int] = (0, 100, 200)
    flame_hsv_high1: Tuple[int, int, int] = (35, 255, 255)
    flame_hsv_low2: Tuple[int, int, int] = (160, 100, 200)
    flame_hsv_high2: Tuple[int, int, int] = (180, 255, 255)


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

    NOTE: We currently consume Habitat's forward-facing depth sensor and
    document the limitation. The same back-projection generalises to true
    360° depth once additional yaw-rotated depth sensors (or an
    equirectangular sensor) are wired through.
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
class ThermalConfig:
    """FLIR-style long-wave IR camera (Fig. 10 of Starr & Lattimer 2014)."""

    # Flame detector (HSV ranges; two ranges to cover hue wrap-around).
    flame_hsv_low1: Tuple[int, int, int] = (0, 100, 200)
    flame_hsv_high1: Tuple[int, int, int] = (35, 255, 255)
    flame_hsv_low2: Tuple[int, int, int] = (160, 100, 200)
    flame_hsv_high2: Tuple[int, int, int] = (180, 255, 255)
    # Halo around flames (radiative + conductive heating of nearby surfaces).
    halo_ksize: int = 61
    halo_gain: float = 0.55
    # Temperature scale.
    ambient_c: float = 25.0
    flame_c: float = 600.0
    # Scene structure proxy: emissivity / thermal-mass differences.
    scene_contrast_c: float = 35.0
    edge_gain: float = 0.45
    gamma: float = 0.7
    # 0 = pure FLIR-like grayscale; >0 blends INFERNO false color.
    color_blend: float = 0.0


@dataclass
class VoxelSmokeConfig:
    """Voxel-driven smoky-RGB / Thermal camera (FireWorld observer).

    These knobs control the ray-march that turns the
    :class:`utils.fire_world.scene.FireScene` voxels into a per-step
    RGB / Thermal image. They were previously attributes of
    ``FireWorldRenderer`` but conceptually belong on the sensor
    config: the renderer is just *how the camera looks at the world*.
    """

    n_steps: int = 16                  # ray-march samples per pixel
    smoke_k_ext: float = 4.0           # extinction coefficient on smoke voxels (1/m)
    smoke_color_rgb: Tuple[int, int, int] = (180, 180, 180)
    # Trilinear-friendly cutoff: propagation pins flame source voxels at
    # ~0.6, but trilinear interpolation bleeds the boundary down to
    # 0.05-0.30. The previous default 0.20 zeroed everything except the
    # source core (rendered as one or two pixels). 0.04 keeps the flame
    # envelope visible.
    flame_threshold: float = 0.04
    # Bumped from 4.0 so the flame still survives 1-2 m of dense smoke.
    flame_emission_gain: float = 8.0
    flame_k_ext: float = 0.8
    flame_glow_ksize: int = 41
    flame_glow_gain: float = 0.55
    # Fraction of the smoke extinction the flame radiation ignores.
    # 0 -> flame attenuates exactly like the scene RGB (will be eaten
    # by smoke); 1 -> smoke is invisible to flame radiation. Realistic
    # ~0.95: visible-band flame leaks through medium-thick smoke
    # (Starr & Lattimer 2014, Fig. 7).
    flame_smoke_passthrough: float = 0.95
    thermal_color_blend: float = 0.0   # 0=grayscale, 1=full INFERNO
    render_scale: float = 0.5          # fraction of camera resolution


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
    smoke: SmokeRGBConfig = field(default_factory=SmokeRGBConfig)
    depth: DepthDegradeConfig = field(default_factory=DepthDegradeConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    voxel: VoxelSmokeConfig = field(default_factory=VoxelSmokeConfig)

    # --- Sensor source selection ---------------------------------------
    # Where the smoky RGB / thermal images come from. ``"beer_lambert"``
    # uses the legacy density-driven SmokeRGBSensor / HSV thermal; ``"voxel"``
    # asks the suite to plug in :class:`VoxelSmokeSensor`, which expects a
    # :class:`utils.fire_world.scene.FireScene` to be bound. ``"auto"``
    # picks ``voxel`` whenever a scene is bound and ``beer_lambert``
    # otherwise.
    rgb_source: str = "auto"           # "auto" | "beer_lambert" | "voxel"
    thermal_source: str = "auto"       # same options
    # When ``rgb_source="voxel"`` and the FireSensorSuite is active,
    # stack a global Beer-Lambert pass on top of the voxel RGB to
    # simulate "environment smoke outside the active fire room".
    compound_rgb: bool = False

    # --- IO -------------------------------------------------------------
    save_npz: bool = False
    # Save the dashboard image alongside per-modality PNGs.
    save_dashboard: bool = True
    dashboard_size: Tuple[int, int] = (2000, 900)  # (W, H) of the big image

    # --- Backwards-compat shims -----------------------------------------
    # The first version of this module exposed flat fields like
    # ``smoke_density`` directly on FireSensorConfig. Mirror the most
    # common ones so callers built against the old API keep working.
    smoke_density: float = 0.6

    def __post_init__(self) -> None:
        # Sync the legacy flat field into the nested config so callers
        # can still write ``FireSensorConfig(smoke_density=0.7)``.
        self.smoke.smoke_density = float(self.smoke_density)
