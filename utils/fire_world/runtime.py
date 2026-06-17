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


@dataclass
class FireWorldRenderer:
    """Composite RGB / Thermal observations from the live FireWorld."""

    fw: FireWorld
    camera_K: object
    max_depth_m: float = 5.0
    n_steps: int = 16              # ray-march samples per pixel
    smoke_k_ext: float = 1.5       # extinction coefficient (per metre, scaled)
    smoke_color_rgb: Tuple[int, int, int] = (180, 180, 180)
    flame_threshold: float = 0.25
    flame_color_lo_rgb: Tuple[int, int, int] = (210, 90, 20)    # deep orange
    flame_color_hi_rgb: Tuple[int, int, int] = (255, 220, 130)  # bright yellow
    flame_glow_ksize: int = 31
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
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]
        depth_m = np.clip(depth_m.astype(np.float32), 0.0, self.max_depth_m)

        flame, smoke, temp = self.fw.query(t_sim)

        start, end = _build_pixel_rays(
            depth_m, self.camera_K, cam_pos_world.astype(np.float32),
            R_cam2world.astype(np.float32),
        )
        H, W = depth_m.shape
        ts = np.linspace(0.0, 1.0, self.n_steps, dtype=np.float32)
        rays = (
            (1.0 - ts[:, None, None, None]) * start[None]
            + ts[:, None, None, None] * end[None]
        )

        smoke_samples = _sample_voxels(smoke, rays, self.fw.origin, self.fw.voxel_m)  # (N, H, W)
        flame_samples = _sample_voxels(flame, rays, self.fw.origin, self.fw.voxel_m)
        temp_along = _sample_voxels(temp, rays, self.fw.origin, self.fw.voxel_m)

        chord = np.linalg.norm(end - start, axis=-1)  # (H, W)
        step_m = chord / max(self.n_steps - 1, 1)     # (H, W)

        # Optical-depth integration along the ray. We accumulate from the
        # camera (t=0) outward; the front-to-back transmittance at sample
        # i is exp(-cumsum(tau_i)).
        tau_per_step = smoke_samples * step_m[None, ...] * float(self.smoke_k_ext)
        tau_cum = np.cumsum(tau_per_step, axis=0)        # (N, H, W)
        T_per_step = np.exp(-tau_cum).astype(np.float32) # transmittance up to step i
        T_final = T_per_step[-1]                         # (H, W) end-to-end T

        # Flame visibility: per-step flame intensity weighted by the
        # transmittance from the camera to that step. The first hot
        # voxel's contribution dominates because everything behind it
        # gets attenuated by smoke in front.
        flame_visible = (flame_samples * T_per_step).max(axis=0).astype(np.float32)

        # Thermal: hottest temperature seen along the ray (smoke is
        # transparent in IR, so we don't apply transmittance here).
        temp_max = temp_along.max(axis=0).astype(np.float32)

        out_rgb = self._composite_rgb(rgb_clean, T_final, flame_visible)
        thermal_image, thermal_temp = self._compose_thermal(
            rgb_clean, temp_max, flame_visible
        )

        return {
            "image": out_rgb,
            "transmittance": T_final,
            "flame_mask": (flame_visible > self.flame_threshold).astype(np.float32),
            "thermal_image": thermal_image,
            "thermal_temperature": thermal_temp,
        }

    # ------------------------------------------------------------------
    def _composite_rgb(
        self,
        rgb_clean: np.ndarray,
        T: np.ndarray,
        flame_along: np.ndarray,
    ) -> np.ndarray:
        smoke_col = np.array(self.smoke_color_rgb, dtype=np.float32).reshape(1, 1, 3)
        T_3 = T[..., None]
        out = rgb_clean.astype(np.float32) * T_3 + smoke_col * (1.0 - T_3)

        if flame_along.max() > 1e-6:
            flame_alpha = np.clip(flame_along, 0.0, 1.0)
            # Soft glow halo via Gaussian blur on the flame mask.
            try:
                import cv2
                k = max(3, int(self.flame_glow_ksize) | 1)
                glow = cv2.GaussianBlur(flame_alpha, (k, k), 0)
                m = float(glow.max())
                if m > 1e-6:
                    glow = glow / m
            except Exception:
                glow = flame_alpha
            lo = np.array(self.flame_color_lo_rgb, dtype=np.float32)
            hi = np.array(self.flame_color_hi_rgb, dtype=np.float32)
            flame_col = (
                (1.0 - flame_alpha[..., None]) * lo
                + flame_alpha[..., None] * hi
            )
            # Direct flame pixels: replace with hot colour. Halo: blend.
            out = (
                (1.0 - flame_alpha[..., None]) * out
                + flame_alpha[..., None] * flame_col
            )
            halo_w = (glow * 0.4)[..., None]
            out = out * (1.0 - halo_w) + lo * halo_w

        return np.clip(out, 0, 255).astype(np.uint8)

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
