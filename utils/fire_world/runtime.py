"""``FireWorld``: the world-model data layer of the fire pipeline.

After the architecture refactor, this module owns *only* the
voxel-timeline data structure and the time-indexed accessor. Everything
related to **rendering** (turning the world into RGB / Thermal images)
now lives on the sensor side at
:mod:`utils.fire_sensors.voxel_render` /
:mod:`utils.fire_sensors.sensors.voxel_smoke`.

New code depends on :class:`utils.fire_world.scene.FireScene` and the
sensor suite (:class:`utils.fire_sensors.FireSensorSuite`), which drives
:class:`utils.fire_sensors.sensors.voxel_smoke.VoxelSmokeSensor` to
observe this world model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
