"""Runtime hook: FireWorld + camera ray-march renderer.

This is stage-5 of the fire-world pipeline. Given a precomputed
``timeline.npz`` (stage 3) and the live agent pose, we composite per-step
RGB / Thermal observations that *see* the 3D fire/smoke field instead of
the old global-density approximation in ``SmokeRGBSensor``.

Public surface:

    fw = FireWorld.load(scene_id, plan_id)
    flame, smoke, temp = fw.query(t_sim)              # (Nx, Ny, Nz) f32

    renderer = FireWorldRenderer(fw, camera_K, max_depth_m=5.0)
    out = renderer.render(rgb_clean, depth_m, cam_pos_world, R_cam2world, t_sim)

``out`` is a dict with the same keys as ``SmokeRGBSensor.process`` so it
plugs into ``FireSensorSuite`` and the existing perception path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# FireWorld: time-indexed access to the precomputed voxel timeline
# ---------------------------------------------------------------------------
@dataclass
class FireWorld:
    flame: np.ndarray      # (T, Nx, Ny, Nz) float16
    smoke: np.ndarray      # (T, Nx, Ny, Nz) float16
    temp: np.ndarray       # (T, Nx, Ny, Nz) float16, deg C
    times: np.ndarray      # (T,) float32, seconds
    voxel_m: float
    origin: np.ndarray     # (3,) float64 world coordinates of voxel (0,0,0)
    shape: Tuple[int, int, int]
    ambient_c: float
    scene_id: str
    plan_id: str

    @classmethod
    def load(
        cls,
        scene_id: str,
        plan_id: str,
        out_root: Path = Path("outputs/fire_world"),
    ) -> "FireWorld":
        npz_path = Path(out_root) / scene_id / plan_id / "timeline.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"timeline.npz not found at {npz_path}. "
                f"Run propagation for scene={scene_id} plan={plan_id} first."
            )
        d = np.load(npz_path, allow_pickle=True)
        meta = json.loads(d["meta"][0]) if "meta" in d.files else {}
        return cls(
            flame=d["flame"],
            smoke=d["smoke"],
            temp=d["temp"],
            times=d["times"].astype(np.float32),
            voxel_m=float(meta.get("voxel_m", 0.15)),
            origin=np.asarray(meta.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64),
            shape=tuple(meta.get("shape", d["flame"].shape[1:])),
            ambient_c=float(meta.get("ambient_c", 25.0)),
            scene_id=scene_id,
            plan_id=plan_id,
        )

    # ------------------------------------------------------------------
    def frame_index(self, t_sim: float) -> int:
        """Pick the timeline frame closest to ``t_sim`` (seconds)."""
        t_sim = float(np.clip(t_sim, float(self.times[0]), float(self.times[-1])))
        return int(np.argmin(np.abs(self.times - t_sim)))

    def query(self, t_sim: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fi = self.frame_index(t_sim)
        return (
            self.flame[fi].astype(np.float32),
            self.smoke[fi].astype(np.float32),
            self.temp[fi].astype(np.float32),
        )


# ---------------------------------------------------------------------------
# Renderer: per-pixel ray-march through the voxel field
# ---------------------------------------------------------------------------
def _build_pixel_rays(
    depth_m: np.ndarray,
    K,
    cam_pos: np.ndarray,
    R_cam2world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(start_xyz, end_xyz)`` world positions for each pixel.

    Habitat depth values are perpendicular distances along the camera
    -Z axis (z-buffer style), so the world end point is:

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
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)         # (H, W, 3)
    pts_world = pts_cam @ R_cam2world.T + cam_pos[None, None]  # (H, W, 3)
    start = np.broadcast_to(cam_pos[None, None, :].astype(np.float32),
                            (H, W, 3)).copy()
    return start, pts_world.astype(np.float32)


def _sample_voxels(
    field_xyz: np.ndarray,
    points_world: np.ndarray,
    origin: np.ndarray,
    voxel_m: float,
) -> np.ndarray:
    """Nearest-neighbour sample of ``field_xyz`` at world ``points_world``.

    points_world: (..., 3) float
    Returns: (...) sampled field values, 0 for points outside the grid.
    """
    Nx, Ny, Nz = field_xyz.shape
    ijk = np.floor((points_world - origin) / voxel_m).astype(np.int32)
    valid = (
        (ijk[..., 0] >= 0) & (ijk[..., 0] < Nx)
        & (ijk[..., 1] >= 0) & (ijk[..., 1] < Ny)
        & (ijk[..., 2] >= 0) & (ijk[..., 2] < Nz)
    )
    ijk_safe = np.where(
        valid[..., None],
        ijk,
        np.zeros_like(ijk),
    )
    sampled = field_xyz[ijk_safe[..., 0], ijk_safe[..., 1], ijk_safe[..., 2]]
    return sampled * valid


def _sample_voxels_trilinear(
    field_xyz: np.ndarray,
    points_world: np.ndarray,
    origin: np.ndarray,
    voxel_m: float,
) -> np.ndarray:
    """Trilinear sample of ``field_xyz`` at world ``points_world``.

    Removes the blocky aliasing of nearest-neighbour sampling so flame
    and smoke voxels appear as smooth volumetric blobs in screen space.
    Outside-of-grid samples return 0 (Neumann fade-out).
    """
    Nx, Ny, Nz = field_xyz.shape
    coord = (points_world - origin) / voxel_m - 0.5  # cell-centre alignment
    i0 = np.floor(coord).astype(np.int32)
    f = (coord - i0).astype(np.float32)              # fractional offsets
    i1 = i0 + 1
    # Clip indices and zero-out contributions for out-of-grid samples.
    valid = (
        (coord[..., 0] >= -0.5) & (coord[..., 0] <= Nx - 0.5)
        & (coord[..., 1] >= -0.5) & (coord[..., 1] <= Ny - 0.5)
        & (coord[..., 2] >= -0.5) & (coord[..., 2] <= Nz - 0.5)
    )
    i0[..., 0] = np.clip(i0[..., 0], 0, Nx - 1)
    i0[..., 1] = np.clip(i0[..., 1], 0, Ny - 1)
    i0[..., 2] = np.clip(i0[..., 2], 0, Nz - 1)
    i1[..., 0] = np.clip(i1[..., 0], 0, Nx - 1)
    i1[..., 1] = np.clip(i1[..., 1], 0, Ny - 1)
    i1[..., 2] = np.clip(i1[..., 2], 0, Nz - 1)

    # 8-corner gather.
    c000 = field_xyz[i0[..., 0], i0[..., 1], i0[..., 2]]
    c100 = field_xyz[i1[..., 0], i0[..., 1], i0[..., 2]]
    c010 = field_xyz[i0[..., 0], i1[..., 1], i0[..., 2]]
    c110 = field_xyz[i1[..., 0], i1[..., 1], i0[..., 2]]
    c001 = field_xyz[i0[..., 0], i0[..., 1], i1[..., 2]]
    c101 = field_xyz[i1[..., 0], i0[..., 1], i1[..., 2]]
    c011 = field_xyz[i0[..., 0], i1[..., 1], i1[..., 2]]
    c111 = field_xyz[i1[..., 0], i1[..., 1], i1[..., 2]]

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    out = c0 * (1.0 - fz) + c1 * fz
    return out.astype(np.float32) * valid.astype(np.float32)


# ---------------------------------------------------------------------------
# Flame colour LUT (black -> dark red -> orange -> yellow -> near white)
# ---------------------------------------------------------------------------
# RGB stops in [0,1]; the renderer interpolates linearly on the sampled
# flame intensity to get the per-voxel emission colour. This is what gives
# the flame its volumetric look: hot core voxels glow yellow-white, mid
# intensity voxels are saturated orange, cool tongue tips fade through
# deep red into a smoky brown before vanishing.
_FLAME_LUT_X = np.array([0.00, 0.10, 0.30, 0.55, 0.80, 1.00], dtype=np.float32)
_FLAME_LUT_RGB = np.array([
    [0.05, 0.00, 0.00],   # almost black (very faint embers)
    [0.40, 0.05, 0.02],   # dark red
    [0.95, 0.30, 0.05],   # deep orange
    [1.00, 0.60, 0.10],   # orange
    [1.00, 0.85, 0.30],   # yellow
    [1.00, 0.97, 0.78],   # near white core
], dtype=np.float32)


def _flame_lut(intensity: np.ndarray) -> np.ndarray:
    """Map (..., ) flame intensity in [0, 1] to (..., 3) RGB in [0, 1]."""
    x = np.clip(intensity, 0.0, 1.0)
    out = np.empty(x.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(x, _FLAME_LUT_X, _FLAME_LUT_RGB[:, ch])
    return out


@dataclass
class FireWorldRenderer:
    """Composite RGB / Thermal observations from the live FireWorld."""

    fw: FireWorld
    camera_K: object
    max_depth_m: float = 5.0
    n_steps: int = 16              # ray-march samples per pixel
    smoke_k_ext: float = 1.5       # smoke extinction coefficient (per metre)
    smoke_color_rgb: Tuple[int, int, int] = (180, 180, 180)
    flame_threshold: float = 0.20  # below this voxels emit nothing
    flame_emission_gain: float = 4.0  # multiplier on per-voxel flame contribution
    flame_k_ext: float = 0.8       # extinction added by flames themselves
    flame_glow_ksize: int = 41     # Gaussian halo around the flame core
    flame_glow_gain: float = 0.55
    thermal_color_blend: float = 0.0   # 0=grayscale, 1=full INFERNO

    # ------------------------------------------------------------------
    def render(
        self,
        rgb_clean: np.ndarray,
        depth_m: np.ndarray,
        cam_pos_world: np.ndarray,
        R_cam2world: np.ndarray,
        t_sim: float,
    ) -> Dict[str, np.ndarray]:
        """Volumetric front-to-back composite of flame + smoke onto rgb_clean.

        Per-step emission-absorption integration::

            sigma_i  = smoke_k_ext * smoke_i  +  flame_k_ext * flame_i
            T_i      = exp(-sigma_i * step_m)               # transmittance of step i
            emission_i = flame_color_lut(flame_i) * gain * flame_i * step_m
            scatter_i  = smoke_color * smoke_i * step_m * smoke_k_ext

            color_acc += T_acc * (emission_i + scatter_i)
            T_acc     *= T_i

        At the end the remaining ``T_acc`` weights the original scene
        colour (rgb_clean), so distant geometry is dimmed by the
        accumulated optical depth, exactly like a real foreground plume
        eating the background's contrast.
        """
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        depth_m = np.clip(depth_m.astype(np.float32), 0.0, self.max_depth_m)

        flame_field, smoke_field, temp_field = self.fw.query(t_sim)

        start, end = _build_pixel_rays(
            depth_m, self.camera_K, cam_pos_world.astype(np.float32),
            R_cam2world.astype(np.float32),
        )
        H, W = depth_m.shape

        # Sample positions along each ray. We use n_steps + 1 break-points
        # so each "step" corresponds to one segment with a finite length.
        N = max(2, int(self.n_steps))
        ts = np.linspace(0.0, 1.0, N, dtype=np.float32)
        chord = np.linalg.norm(end - start, axis=-1).astype(np.float32)  # (H, W)
        step_m = chord / float(N - 1)                                    # (H, W)

        smoke_color = (
            np.array(self.smoke_color_rgb, dtype=np.float32) / 255.0
        ).reshape(1, 1, 3)

        # Accumulators.
        color_acc = np.zeros((H, W, 3), dtype=np.float32)
        T_acc = np.ones((H, W), dtype=np.float32)
        flame_seen = np.zeros((H, W), dtype=np.float32)
        temp_max = np.full((H, W), float(self.fw.ambient_c), dtype=np.float32)

        for i, t in enumerate(ts):
            pts = (1.0 - t) * start + t * end                        # (H, W, 3)
            sm = _sample_voxels_trilinear(
                smoke_field, pts, self.fw.origin, self.fw.voxel_m,
            )
            fl = _sample_voxels_trilinear(
                flame_field, pts, self.fw.origin, self.fw.voxel_m,
            )
            te = _sample_voxels_trilinear(
                temp_field, pts, self.fw.origin, self.fw.voxel_m,
            )

            # Threshold flame so faint diffusion noise doesn't pre-light
            # the whole frustum. A soft ramp ([thr, 1.5*thr]) keeps the
            # boundary smooth instead of clipping to a binary mask.
            t_lo = float(self.flame_threshold)
            t_hi = max(t_lo * 1.5, t_lo + 0.05)
            fl_used = np.clip((fl - t_lo) / max(t_hi - t_lo, 1e-3), 0.0, 1.0) * fl

            # Per-voxel optical depth and transmittance for this step.
            sigma = self.smoke_k_ext * sm + self.flame_k_ext * fl_used
            tau_step = sigma * step_m
            T_step = np.exp(-tau_step).astype(np.float32)

            # Emission: flame self-luminance + smoke in-scatter from ambient.
            flame_rgb = _flame_lut(fl_used)                          # (H, W, 3) in [0, 1]
            emission = (
                flame_rgb * (fl_used * self.flame_emission_gain)[..., None]
                * step_m[..., None]
            )
            scatter = (
                smoke_color * (sm * self.smoke_k_ext)[..., None]
                * step_m[..., None]
            )

            color_acc = color_acc + T_acc[..., None] * (emission + scatter)
            T_acc = T_acc * T_step

            flame_seen = np.maximum(flame_seen, fl_used)
            temp_max = np.maximum(temp_max, te)

        # Final composite: the remaining transmittance multiplies the
        # clean scene RGB; the accumulated colour adds on top.
        scene = rgb_clean.astype(np.float32) / 255.0
        out = scene * T_acc[..., None] + color_acc
        out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        # Optional soft glow halo around the flame: it's mostly a
        # cosmetic touch but matches real video where flame edges leak
        # warm light into nearby smoke.
        if self.flame_glow_gain > 0.0 and flame_seen.max() > 1e-3:
            try:
                import cv2
                k = max(3, int(self.flame_glow_ksize) | 1)
                glow = cv2.GaussianBlur(flame_seen, (k, k), 0)
                m = float(glow.max())
                if m > 1e-6:
                    glow = glow / m
                glow = np.clip(glow * float(self.flame_glow_gain), 0.0, 1.0)
                halo_col = np.array([1.00, 0.55, 0.10], dtype=np.float32) * 255.0
                halo = halo_col.reshape(1, 1, 3) * glow[..., None]
                out_u8 = np.clip(
                    out_u8.astype(np.float32) * (1.0 - 0.35 * glow[..., None])
                    + 0.35 * halo,
                    0, 255,
                ).astype(np.uint8)
            except Exception:
                pass

        thermal_image, thermal_temp = self._compose_thermal(
            rgb_clean, temp_max, flame_seen
        )

        return {
            "image": out_u8,
            "transmittance": T_acc,
            "flame_mask": (flame_seen > self.flame_threshold).astype(np.float32),
            "thermal_image": thermal_image,
            "thermal_temperature": thermal_temp,
        }

    # ------------------------------------------------------------------
    def _compose_thermal(
        self,
        rgb_clean: np.ndarray,
        temp_max: np.ndarray,
        flame_along: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            import cv2
        except Exception:
            cv2 = None  # type: ignore
        amb = float(self.fw.ambient_c)
        # Use scene luminance as the thermal "scene structure" baseline.
        if cv2 is not None:
            luma = cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            luma = rgb_clean.mean(axis=-1).astype(np.float32) / 255.0
        scene_field = (luma - 0.5) * 35.0
        temperature = amb + scene_field + np.maximum(temp_max - amb, 0.0)

        # Auto stretch.
        t_lo = float(np.percentile(temperature, 2))
        t_hi = float(max(np.percentile(temperature, 99.5), 600.0 * 0.6))
        norm = np.clip((temperature - t_lo) / max(t_hi - t_lo, 1e-3), 0.0, 1.0)
        norm = np.power(norm, 0.7)
        gray_u8 = (norm * 255).astype(np.uint8)
        if cv2 is not None:
            image_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
            if self.thermal_color_blend > 0.0:
                ir = cv2.applyColorMap(gray_u8, cv2.COLORMAP_INFERNO)
                a = float(np.clip(self.thermal_color_blend, 0.0, 1.0))
                image_bgr = cv2.addWeighted(image_bgr, 1 - a, ir, a, 0)
        else:
            image_bgr = np.stack([gray_u8] * 3, axis=-1)
        return image_bgr, temperature.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: a process()-style adapter so this can drop into the suite.
# ---------------------------------------------------------------------------
def runtime_process(
    fire_world: FireWorld,
    renderer: FireWorldRenderer,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cam_pos_world: np.ndarray,
    R_cam2world: np.ndarray,
    t_sim: float,
) -> Dict[str, np.ndarray]:
    """Same return shape as the smoke / thermal sensor combo so the existing
    ``apply_clean_depth_and_thermal`` helper can consume it unchanged.
    """
    rendered = renderer.render(rgb, depth_m, cam_pos_world, R_cam2world, t_sim)
    return {
        "rgb": rgb,
        "depth_clean": (depth_m if depth_m.ndim == 3 else depth_m[..., None]).astype(np.float32),
        "rgb_smoke": rendered["image"],
        "depth_smoke": (depth_m if depth_m.ndim == 3 else depth_m[..., None]).astype(np.float32),
        "transmittance": rendered["transmittance"],
        "thermal_image": rendered["thermal_image"],
        "thermal_temperature": rendered["thermal_temperature"],
        "thermal_flame_mask": rendered["flame_mask"],
    }
