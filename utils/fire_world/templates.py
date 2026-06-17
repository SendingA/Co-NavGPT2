"""Fire-scenario templates.

Each template is a *deterministic* function that, given an inventory and a
seeded RNG, picks ignition objects and fills in propagation parameters.
Templates intentionally do not call any LLM yet; that comes in a later PR
(``planner.py`` will gain an ``--use_llm`` switch). For now they encode
common-sense defaults that are good enough to drive the propagation
engine and the top-down validation video.

A template returns a list of ``Ignition`` dicts (small JSON-friendly) plus
a ``propagation_rules`` dict. The planner glues them into a plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


TEMPLATE_VERSION = 1


# ---------------------------------------------------------------------------
# Intensity presets
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IntensityPreset:
    n_ignitions_min: int
    n_ignitions_max: int
    source_temp_c: float
    fuel_kg: float
    duration_s: float
    secondary_delay_s: Tuple[float, float]


INTENSITIES: Dict[str, IntensityPreset] = {
    "light": IntensityPreset(
        n_ignitions_min=1, n_ignitions_max=1,
        source_temp_c=550.0, fuel_kg=2.0,
        duration_s=300.0, secondary_delay_s=(60.0, 90.0),
    ),
    "medium": IntensityPreset(
        n_ignitions_min=1, n_ignitions_max=2,
        source_temp_c=750.0, fuel_kg=5.0,
        duration_s=600.0, secondary_delay_s=(45.0, 90.0),
    ),
    "severe": IntensityPreset(
        n_ignitions_min=2, n_ignitions_max=3,
        source_temp_c=950.0, fuel_kg=10.0,
        duration_s=900.0, secondary_delay_s=(20.0, 60.0),
    ),
}


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
def _object_center(obj: Dict) -> np.ndarray:
    return np.asarray(obj.get("position", obj["aabb_min"]), dtype=np.float64)


def _filter_by_category(objects: List[Dict], cats: List[str]) -> List[Dict]:
    cats = {c.lower() for c in cats}
    return [o for o in objects if o["category"].lower() in cats]


def _pick_primary(
    objects: List[Dict],
    rng: np.random.Generator,
    preferred_cats: List[str],
    fallback_cats: List[str],
) -> Optional[Dict]:
    """Pick a primary ignition object weighted by flammability."""
    pool = _filter_by_category(objects, preferred_cats) or \
           _filter_by_category(objects, fallback_cats) or \
           [o for o in objects if o.get("flammability", 0) > 0.3]
    if not pool:
        return None
    weights = np.array([max(0.05, o.get("flammability", 0.3)) for o in pool])
    weights /= weights.sum()
    idx = int(rng.choice(len(pool), p=weights))
    return pool[idx]


def _pick_secondary(
    objects: List[Dict],
    primary: Dict,
    rng: np.random.Generator,
    radius_m: float,
    n: int,
    avoid_cats: Tuple[str, ...] = ("toilet", "bathtub", "sink"),
    same_floor_y_tol: float = 1.5,
) -> List[Dict]:
    """Pick up to ``n`` secondary ignitions within ``radius_m`` of primary,
    weighted by (flammability * 1/distance). Restricted to the same floor
    via a vertical tolerance (``same_floor_y_tol``) so a kitchen fire
    cannot 'jump' to a bedroom on a different storey."""
    if n <= 0:
        return []
    p_center = _object_center(primary)
    avoid = {c.lower() for c in avoid_cats}
    pool = []
    for o in objects:
        if o["object_id"] == primary["object_id"]:
            continue
        if o["category"].lower() in avoid:
            continue
        oc = _object_center(o)
        if abs(oc[1] - p_center[1]) > same_floor_y_tol:
            continue
        d_xz = float(np.linalg.norm(oc[[0, 2]] - p_center[[0, 2]]))
        if d_xz > radius_m:
            continue
        f = max(0.05, o.get("flammability", 0.3))
        w = f / max(0.5, d_xz)
        pool.append((o, w))
    if not pool:
        return []
    objs, weights = zip(*pool)
    weights = np.array(weights)
    weights /= weights.sum()
    n_pick = min(n, len(objs))
    idx = rng.choice(len(objs), size=n_pick, replace=False, p=weights)
    return [objs[int(i)] for i in idx]


def _make_ignition(obj: Dict, t_s: float, preset: IntensityPreset, rng: np.random.Generator) -> Dict:
    # Source radius scales with object footprint (clamped).
    bb_min = np.array(obj["aabb_min"])
    bb_max = np.array(obj["aabb_max"])
    extent_xz = max(bb_max[0] - bb_min[0], bb_max[2] - bb_min[2])
    src_r = float(np.clip(0.25 + 0.25 * extent_xz, 0.20, 0.80))
    fuel = preset.fuel_kg * (0.6 + 0.8 * obj.get("flammability", 0.3))
    smoke_yield = float(np.clip(obj.get("smoke_yield", 0.4), 0.05, 1.0))
    # Temperature: small object-level jitter for variety, deterministic by RNG.
    temp = float(preset.source_temp_c + rng.uniform(-30.0, 30.0))
    return {
        "object_id": int(obj["object_id"]),
        "category": obj["category"],
        "position": list(map(float, _object_center(obj))),
        "ignite_time_s": float(t_s),
        "source_radius_m": float(round(src_r, 3)),
        "source_temp_c": float(round(temp, 1)),
        "fuel_kg": float(round(fuel, 3)),
        "smoke_yield": float(round(smoke_yield, 3)),
    }


def _default_propagation_rules(intensity: str) -> Dict:
    # These values feed the stage-3 propagation engine.
    base = {
        "flammable_threshold": 0.4,    # min fuel level a voxel needs to ignite
        "ignition_temp_c": 350.0,      # T at which a fuel voxel ignites
        "spread_speed_m_per_s": 0.04,  # surface flame spread along fuel
        "ceiling_jet_speed_m_per_s": 0.30,
        "buoyancy_v_m_per_s": 0.5,     # vertical plume velocity
        "thermal_diffusivity": 0.05,   # alpha used in heat diffusion
        "ambient_temp_c": 25.0,
    }
    if intensity == "light":
        base["spread_speed_m_per_s"] *= 0.7
        base["buoyancy_v_m_per_s"] *= 0.8
    elif intensity == "severe":
        base["spread_speed_m_per_s"] *= 1.5
        base["ceiling_jet_speed_m_per_s"] *= 1.4
        base["buoyancy_v_m_per_s"] *= 1.3
    return base


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
def _template_kitchen_grease_fire(inv: Dict, rng: np.random.Generator,
                                  preset: IntensityPreset) -> List[Dict]:
    objs = inv["objects"]
    # HM3D goals don't include stove; prefer plant/tv_monitor/chair as a
    # rough stand-in for "warm appliance area" then propagate to nearby
    # flammables. We keep the scenario semantically labelled regardless.
    primary = _pick_primary(objs, rng,
                            preferred_cats=["stove", "tv_monitor"],
                            fallback_cats=["chair", "plant", "sofa"])
    if primary is None:
        return []
    n_extra = rng.integers(preset.n_ignitions_min - 1,
                           preset.n_ignitions_max) if preset.n_ignitions_max > 1 else 0
    secondaries = _pick_secondary(objs, primary, rng, radius_m=3.5, n=int(n_extra))
    igns = [_make_ignition(primary, 0.0, preset, rng)]
    for k, s in enumerate(secondaries):
        t = float(rng.uniform(*preset.secondary_delay_s)) + 30.0 * k
        igns.append(_make_ignition(s, t, preset, rng))
    return igns


def _template_bedroom_textile(inv: Dict, rng: np.random.Generator,
                              preset: IntensityPreset) -> List[Dict]:
    objs = inv["objects"]
    primary = _pick_primary(objs, rng,
                            preferred_cats=["bed", "sofa"],
                            fallback_cats=["chair", "plant"])
    if primary is None:
        return []
    n_extra = rng.integers(preset.n_ignitions_min - 1,
                           preset.n_ignitions_max) if preset.n_ignitions_max > 1 else 0
    secondaries = _pick_secondary(objs, primary, rng, radius_m=4.0, n=int(n_extra))
    igns = [_make_ignition(primary, 0.0, preset, rng)]
    for k, s in enumerate(secondaries):
        t = float(rng.uniform(*preset.secondary_delay_s)) + 30.0 * k
        igns.append(_make_ignition(s, t, preset, rng))
    return igns


def _template_living_room_electric(inv: Dict, rng: np.random.Generator,
                                   preset: IntensityPreset) -> List[Dict]:
    objs = inv["objects"]
    primary = _pick_primary(objs, rng,
                            preferred_cats=["tv_monitor", "tv", "monitor"],
                            fallback_cats=["sofa", "chair"])
    if primary is None:
        return []
    n_extra = rng.integers(preset.n_ignitions_min - 1,
                           preset.n_ignitions_max) if preset.n_ignitions_max > 1 else 0
    secondaries = _pick_secondary(objs, primary, rng, radius_m=3.0, n=int(n_extra))
    igns = [_make_ignition(primary, 0.0, preset, rng)]
    for k, s in enumerate(secondaries):
        t = float(rng.uniform(*preset.secondary_delay_s)) + 30.0 * k
        igns.append(_make_ignition(s, t, preset, rng))
    return igns


def _template_multi_origin(inv: Dict, rng: np.random.Generator,
                           preset: IntensityPreset) -> List[Dict]:
    """Multiple ignitions on the *same floor* (stress test)."""
    candidates = [o for o in inv["objects"] if o.get("flammability", 0) >= 0.4]
    if len(candidates) < 2:
        return []
    # Cluster by Y so we keep all picks on a single floor.
    ys = np.array([_object_center(o)[1] for o in candidates])
    bins = np.round(ys / 1.5).astype(int)
    counts = {b: int(np.sum(bins == b)) for b in set(bins.tolist())}
    best_bin = max(counts, key=counts.get)
    objs = [o for o, b in zip(candidates, bins) if b == best_bin]
    if len(objs) < 2:
        objs = candidates
    n = max(2, int(preset.n_ignitions_max))
    n = min(n, len(objs))
    centers = np.array([_object_center(o) for o in objs])
    first = int(rng.integers(0, len(objs)))
    picks = [first]
    for _ in range(n - 1):
        d = np.min(
            np.linalg.norm(
                centers[:, None, [0, 2]] - centers[picks][None, :, [0, 2]],
                axis=-1,
            ),
            axis=1,
        )
        d[picks] = -1
        picks.append(int(np.argmax(d)))
    igns = []
    for k, i in enumerate(picks):
        t = 0.0 if k == 0 else float(rng.uniform(*preset.secondary_delay_s)) + 30.0 * k
        igns.append(_make_ignition(objs[i], t, preset, rng))
    return igns


TEMPLATES: Dict[str, Callable[[Dict, np.random.Generator, IntensityPreset], List[Dict]]] = {
    "kitchen_grease_fire":   _template_kitchen_grease_fire,
    "bedroom_textile":       _template_bedroom_textile,
    "living_room_electric":  _template_living_room_electric,
    "multi_origin":          _template_multi_origin,
}
