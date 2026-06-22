"""``FireWorld``: the world-model data layer of the fire pipeline.

After the architecture refactor, this module owns *only* the
voxel-timeline data structure and the time-indexed accessor. Everything
related to **rendering** (turning the world into RGB / Thermal images)
now lives on the sensor side at
:mod:`utils.fire_sensors.voxel_render` /
:mod:`utils.fire_sensors.sensors.voxel_smoke`.

The pre-refactor public API is preserved as thin shims:

* ``FireWorldRenderer``   -> re-exports
  :class:`utils.fire_sensors.sensors.voxel_smoke.VoxelSmokeSensor` so
  legacy demo / test scripts keep importing from here.
* ``runtime_process``     -> thin wrapper that calls
  :func:`utils.fire_sensors.voxel_render.volumetric_composite` and
  formats the result with the legacy keys.

New code should depend on :class:`utils.fire_world.scene.FireScene`
and the sensor suite instead.
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
def plan_id_to_path(plan_id: str) -> Path:
    """Where to find the plan json relative to ``scenes/<scene>/``."""
    return Path("plans") / f"{plan_id}.json"


def _load_timeline_meta(npz_path: Path) -> dict:
    """Read the ``timeline_meta.json`` sidecar if present.

    propagation.py writes both ``timeline.npz`` and a sibling JSON file
    on every run. Loading the JSON sidecar first lets us skip the
    pickle/object-array path inside numpy entirely, which is the
    cleanest fix for cross-numpy-version load failures.
    """
    sidecar = npz_path.with_name("timeline_meta.json")
    try:
        if sidecar.exists():
            return json.loads(sidecar.read_text())
    except Exception:
        pass
    return {}


def _meta_from_npz(d) -> dict:
    """Best-effort meta extraction from a loaded npz.

    Supports three formats:
      - ``meta_json`` : 0-d numpy unicode array (new format, no pickle).
      - ``meta``      : 1-d object array carrying a single JSON string
                        (legacy pre-2026-06-21 format; relies on pickle).
      - missing       : returns {} so the caller falls back to defaults.
    """
    if "meta_json" in d.files:
        try:
            return json.loads(str(d["meta_json"]))
        except Exception:
            return {}
    if "meta" in d.files:
        try:
            return json.loads(d["meta"][0])
        except Exception:
            return {}
    return {}


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
        # Detect a stale cache: if plan.json was edited after the
        # timeline was baked, the user is almost certainly running with
        # the *old* fire field. Warn loudly so they don't spend an hour
        # wondering why their changes don't show up.
        plan_path = Path("scenes") / scene_id / plan_id_to_path(plan_id)
        try:
            if (
                plan_path.exists()
                and plan_path.stat().st_mtime > npz_path.stat().st_mtime + 1.0
            ):
                import warnings
                warnings.warn(
                    f"[fire_world] timeline.npz at {npz_path} is older than "
                    f"{plan_path}. The on-disk cache will be loaded as-is, so "
                    f"any edits you made to the plan are NOT in the rendered "
                    f"fire. Rerun:\n  python -m utils.fire_world.propagation "
                    f"--scene {scene_id} --plan_id {plan_id} --voxel_m 0.15",
                    stacklevel=2,
                )
        except OSError:
            pass

        # Prefer the JSON sidecar if available: that bypasses any
        # pickle-version skew between the numpy that wrote the npz and
        # the numpy loading it. ``meta_json`` (numpy unicode) is the
        # secondary route; the legacy ``meta`` object array is last
        # resort and may fail across numpy 1.x <-> 2.x boundaries.
        meta = _load_timeline_meta(npz_path)
        try:
            d = np.load(npz_path, allow_pickle=False)
        except ValueError:
            d = np.load(npz_path, allow_pickle=True)
        if not meta:
            meta = _meta_from_npz(d)
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
# Back-compat shims
# ---------------------------------------------------------------------------
class FireWorldRenderer:
    """Backwards-compatible wrapper around the new voxel renderer.

    ``utils/fire_sensors/voxel_render.py`` owns the actual ray-march;
    we just expose the legacy ``render(rgb_clean, depth_m, cam_pos,
    R_cam2world, t_sim)`` signature so the demo and test scripts keep
    working unchanged. New code should drive
    :class:`utils.fire_sensors.sensors.voxel_smoke.VoxelSmokeSensor`
    through the suite instead.
    """

    def __init__(
        self,
        fw: FireWorld,
        camera_K,
        max_depth_m: float = 5.0,
        n_steps: int = 16,
        smoke_k_ext: float = 1.5,
        smoke_color_rgb: Tuple[int, int, int] = (180, 180, 180),
        flame_threshold: float = 0.20,
        flame_emission_gain: float = 4.0,
        flame_k_ext: float = 0.8,
        flame_glow_ksize: int = 41,
        flame_glow_gain: float = 0.55,
        flame_smoke_passthrough: float = 0.85,
        thermal_color_blend: float = 0.0,
        render_scale: float = 0.5,
    ) -> None:
        from utils.fire_sensors.voxel_render import VoxelRenderParams
        self.fw = fw
        self.camera_K = camera_K
        self.params = VoxelRenderParams(
            max_depth_m=float(max_depth_m),
            n_steps=int(n_steps),
            smoke_k_ext=float(smoke_k_ext),
            smoke_color_rgb=smoke_color_rgb,
            flame_threshold=float(flame_threshold),
            flame_emission_gain=float(flame_emission_gain),
            flame_k_ext=float(flame_k_ext),
            flame_glow_ksize=int(flame_glow_ksize),
            flame_glow_gain=float(flame_glow_gain),
            flame_smoke_passthrough=float(flame_smoke_passthrough),
            thermal_color_blend=float(thermal_color_blend),
            render_scale=float(render_scale),
        )

    @property
    def render_scale(self) -> float:
        return self.params.render_scale

    @render_scale.setter
    def render_scale(self, v: float) -> None:
        self.params.render_scale = float(v)

    @property
    def n_steps(self) -> int:
        return self.params.n_steps

    @n_steps.setter
    def n_steps(self, v: int) -> None:
        self.params.n_steps = int(v)

    @property
    def smoke_k_ext(self) -> float:
        return self.params.smoke_k_ext

    @smoke_k_ext.setter
    def smoke_k_ext(self, v: float) -> None:
        self.params.smoke_k_ext = float(v)

    def render(
        self,
        rgb_clean: np.ndarray,
        depth_m: np.ndarray,
        cam_pos_world: np.ndarray,
        R_cam2world: np.ndarray,
        t_sim: float,
    ) -> Dict[str, np.ndarray]:
        from utils.fire_sensors.voxel_render import volumetric_composite
        flame_field, smoke_field, temp_field = self.fw.query(t_sim)
        return volumetric_composite(
            rgb_clean=rgb_clean,
            depth_m=depth_m,
            cam_pos_world=cam_pos_world,
            R_cam2world=R_cam2world,
            flame_field=flame_field,
            smoke_field=smoke_field,
            temp_field=temp_field,
            origin=self.fw.origin,
            voxel_m=self.fw.voxel_m,
            grid_shape=self.fw.shape,
            ambient_c=self.fw.ambient_c,
            camera_K=self.camera_K,
            params=self.params,
        )


def runtime_process(
    fire_world: FireWorld,
    renderer: FireWorldRenderer,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    cam_pos_world: np.ndarray,
    R_cam2world: np.ndarray,
    t_sim: float,
) -> Dict[str, np.ndarray]:
    """Adapter that returns the smoky-RGB / thermal dict shape used by
    ``apply_clean_depth_and_thermal``.
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
