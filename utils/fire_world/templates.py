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


TEMPLATE_VERSION = 2


# Exact HM3D semantic.txt category spellings.  Keep these as raw lowercase
# annotation values: no detector aliases (for example ``tv_monitor``) belong
# here.  Tests cross-check every entry against all installed semantic.txt
# files and against scene_scan.MATERIAL_TABLE.
KITCHEN_PRIMARY_CATEGORIES = (
    "stove",
    "stovetop",
    "oven and stove",
    "oven",
    "cooker",
)
KITCHEN_FALLBACK_CATEGORIES = (
    "microwave",
    "toaster",
    "kitchen appliance",
    "ventilation hood",
    "range hood",
    "kitchen extractor",
    "oven vent",
)
KITCHEN_SECONDARY_CATEGORIES = (
    *KITCHEN_PRIMARY_CATEGORIES,
    *KITCHEN_FALLBACK_CATEGORIES,
    "kitchen cabinet",
    "kitchen lower cabinet",
    "cabinet",
    "table",
    "kitchen counter",
    "curtain",
    "towel",
    "paper towel",
    "trashcan",
    "trash can",
)

BEDROOM_PRIMARY_CATEGORIES = (
    "bed",
    "bed small",
    "bedframe",
    "pillow",
    "blanket",
    "bed sheet",
)
BEDROOM_FALLBACK_CATEGORIES = (
    "curtain",
    "window curtain",
    "rug",
    "carpet",
    "clothes",
    "sofa",
    "couch",
    "armchair",
    "chair",
)
BEDROOM_SECONDARY_CATEGORIES = (
    *BEDROOM_PRIMARY_CATEGORIES,
    *BEDROOM_FALLBACK_CATEGORIES,
    "nightstand",
    "wardrobe",
    "cloth",
    "throw blanket",
)

LIVING_ELECTRIC_PRIMARY_CATEGORIES = (
    "tv",
    "led tv",
    "wall tv",
    "monitor",
    "computer",
    "computer tower",
    "pc tower",
    "laptop",
)
LIVING_ELECTRIC_FALLBACK_CATEGORIES = (
    "speaker",
    "stereo",
    "amplifier",
    "dvd player",
    "record player",
    "radio",
)
LIVING_ELECTRIC_SECONDARY_CATEGORIES = (
    *LIVING_ELECTRIC_PRIMARY_CATEGORIES,
    *LIVING_ELECTRIC_FALLBACK_CATEGORIES,
    "media console",
    "sofa",
    "couch",
    "armchair",
    "chair",
    "cabinet",
    "curtain",
    "rug",
    "carpet",
)

TEMPLATE_CATEGORY_GROUPS = {
    "kitchen_grease_fire": {
        "primary": KITCHEN_PRIMARY_CATEGORIES,
        "fallback": KITCHEN_FALLBACK_CATEGORIES,
        "secondary": KITCHEN_SECONDARY_CATEGORIES,
    },
    "bedroom_textile": {
        "primary": BEDROOM_PRIMARY_CATEGORIES,
        "fallback": BEDROOM_FALLBACK_CATEGORIES,
        "secondary": BEDROOM_SECONDARY_CATEGORIES,
    },
    "living_room_electric": {
        "primary": LIVING_ELECTRIC_PRIMARY_CATEGORIES,
        "fallback": LIVING_ELECTRIC_FALLBACK_CATEGORIES,
        "secondary": LIVING_ELECTRIC_SECONDARY_CATEGORIES,
    },
}


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
    position = obj.get("position")
    return np.asarray(
        position if position is not None else obj["aabb_min"],
        dtype=np.float64,
    )


def _filter_by_category(objects: List[Dict], cats: List[str]) -> List[Dict]:
    cats = {c.lower() for c in cats}
    return [o for o in objects if o["category"].lower() in cats]


def _inventory_pool(inv: Dict) -> List[Dict]:
    """Return the per-object pool used by templates.

    Schema v2 carries an ``instances`` list with every recovered HM3D
    instance (628+ entries on a typical scene); each entry has the same
    keys ``aabb_min/aabb_max/category/flammability/structural`` we rely
    on. Schema v1 only has ``objects``. We dedupe by id and drop
    structural items (walls, floors, ceilings) so they can never be
    picked as ignition sources.
    """
    items = inv.get("instances")
    if not items:
        items = inv.get("objects", [])
    out: List[Dict] = []
    seen = set()
    for it in items:
        if bool(it.get("structural", False)):
            continue
        oid = it.get("object_id", it.get("instance_id"))
        if oid is None or oid in seen:
            continue
        seen.add(oid)
        # Normalise to the shape templates expect.
        norm = {
            "object_id": int(oid),
            "category": it["category"],
            "position": it.get("position", it.get("centroid")),
            "aabb_min": it["aabb_min"],
            "aabb_max": it["aabb_max"],
            "flammability": float(it.get("flammability", 0.0)),
            "smoke_yield": float(it.get("smoke_yield", 0.4)),
        }
        out.append(norm)
    return out


def _pick_primary(
    objects: List[Dict],
    rng: np.random.Generator,
    preferred_cats: List[str],
    fallback_cats: List[str],
) -> Optional[Dict]:
    """Pick a primary ignition object weighted by flammability."""
    pool = (
        _filter_by_category(objects, preferred_cats)
        or _filter_by_category(objects, fallback_cats)
    )
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
    allowed_cats: Tuple[str, ...],
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
    allowed = {c.lower() for c in allowed_cats}
    pool = []
    for o in objects:
        if o["object_id"] == primary["object_id"]:
            continue
        if o["category"].lower() not in allowed:
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
        # Lowered from the literature autoignition value (350 C) so the
        # discrete voxel grid actually reaches it. Real fires preheat
        # neighbours to far below autoignition before piloted ignition;
        # at 200 C the radiative_gain heat path can sustainably push
        # adjacent furniture into the reaction loop within 30-60 s.
        "ignition_temp_c": 200.0,
        # Surface flame spread along fuel. Bumped from 0.04 to 0.18 so
        # the fire visibly grows on a 0.15 m grid: at 0.04 m/s it took
        # >5 minutes to cross a single voxel, which is why the original
        # benchmark looked like static fixed sources.
        "spread_speed_m_per_s": 0.18,
        # 'laplacian' (legacy) or 'gaussian' (smoother, NIST-FDS-like).
        "spread_kernel": "laplacian",
        "ceiling_jet_speed_m_per_s": 0.30,
        "buoyancy_v_m_per_s": 0.5,     # vertical plume velocity
        "thermal_diffusivity": 0.05,   # alpha used in heat diffusion
        "ambient_temp_c": 25.0,
        # Radiative pre-heating that lets heat jump ~0.6 m to the next
        # piece of furniture in the same room. Scales linearly with
        # local flame intensity. 250 C/s on a fully-developed flame
        # voxel inside the 4-cell radiative kernel means a fuel voxel
        # 0.6 m away crosses ignition_temp in ~30 s.
        "radiative_gain_c": 250.0,
        "radiative_radius_cells": 4,
        # Floors are *semi-transparent* thermal barriers (default 20 %
        # of normal heat conduction) so a fire on one storey can still
        # raise the assembly's temperature on the storey below over
        # several minutes (NFPA 921 §5.10). Set to 0 for a fully
        # fire-rated assembly; 1.0 makes the floor an air gap.
        "floor_thermal_attenuation": 0.20,
        # Should the flame field actually live inside floor voxels?
        # Default on now that floors get a synthetic fuel deposit when
        # heated (see `floor_fuel_value`): the floor is supposed to
        # carry fire from one piece of furniture to the next.
        "flame_through_floors": 1,
        # Local fuel-abundance modulation. Larger neighbourhood / wider
        # max means a fuel-dense corner burns faster than an isolated
        # chair.
        "fuel_neighborhood_cells": 2,
        "fuel_abundance_min": 0.5,
        "fuel_abundance_max": 2.5,
        # Floor ignition. Above ``floor_ignite_temp_c`` each floor
        # voxel gets a synthetic fuel deposit of ``floor_fuel_value``,
        # which feeds the regular reaction loop. Set the value to 0 to
        # disable floor combustion (e.g. for tile / concrete).
        "floor_ignite_temp_c": 250.0,
        "floor_fuel_value": 0.7,
        # Direct contact ignition: floor voxels within this many cells
        # of a flame voxel ignite immediately, bypassing heat
        # diffusion's smoothing. Without this the per-voxel thermal
        # spike gets averaged out before reaction sees it.
        "floor_ignite_radius_cells": 2,
        "floor_flame_contact_thresh": 0.2,
        # The combustible floor grows outward from each actual ignition
        # source at 1.5 cm/s and stops at a 2 m radius.  This prevents the
        # old recursive floor-cell dilation from carpeting the whole room
        # with flame while keeping a visibly evolving local hazard patch.
        "floor_spread_speed_m_per_s": 0.015,
        "floor_max_spread_radius_m": 2.0,
        # Sustained point sources never expire when 1, ensuring the
        # initial ignitions don't burn themselves out before they have
        # a chance to set the rest of the room alight.
        "inextinguishable_sources": 1,
        # Flame column height: each active flame voxel projects an
        # upward plume up to ``flame_column_cells`` voxels with linear
        # intensity decay (controlled by ``flame_column_decay``). This
        # is what gives flames a visible vertical extent rather than
        # rendering as a flat blob at the source.
        "flame_column_cells": 6,
        "flame_column_decay": 0.75,
    }
    if intensity == "light":
        base["spread_speed_m_per_s"] *= 0.7
        base["buoyancy_v_m_per_s"] *= 0.8
        base["radiative_gain_c"] *= 0.6
    elif intensity == "severe":
        base["spread_speed_m_per_s"] *= 1.5
        base["ceiling_jet_speed_m_per_s"] *= 1.4
        base["buoyancy_v_m_per_s"] *= 1.3
        base["radiative_gain_c"] *= 1.5
    return base


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------
def _template_kitchen_grease_fire(inv: Dict, rng: np.random.Generator,
                                  preset: IntensityPreset) -> List[Dict]:
    objs = _inventory_pool(inv)
    primary = _pick_primary(
        objs,
        rng,
        preferred_cats=list(KITCHEN_PRIMARY_CATEGORIES),
        fallback_cats=list(KITCHEN_FALLBACK_CATEGORIES),
    )
    if primary is None:
        return []
    n_extra = rng.integers(preset.n_ignitions_min - 1,
                           preset.n_ignitions_max) if preset.n_ignitions_max > 1 else 0
    secondaries = _pick_secondary(
        objs,
        primary,
        rng,
        radius_m=3.5,
        n=int(n_extra),
        allowed_cats=KITCHEN_SECONDARY_CATEGORIES,
    )
    igns = [_make_ignition(primary, 0.0, preset, rng)]
    for k, s in enumerate(secondaries):
        t = float(rng.uniform(*preset.secondary_delay_s)) + 30.0 * k
        igns.append(_make_ignition(s, t, preset, rng))
    return igns


def _template_bedroom_textile(inv: Dict, rng: np.random.Generator,
                              preset: IntensityPreset) -> List[Dict]:
    objs = _inventory_pool(inv)
    primary = _pick_primary(
        objs,
        rng,
        preferred_cats=list(BEDROOM_PRIMARY_CATEGORIES),
        fallback_cats=list(BEDROOM_FALLBACK_CATEGORIES),
    )
    if primary is None:
        return []
    n_extra = rng.integers(preset.n_ignitions_min - 1,
                           preset.n_ignitions_max) if preset.n_ignitions_max > 1 else 0
    secondaries = _pick_secondary(
        objs,
        primary,
        rng,
        radius_m=4.0,
        n=int(n_extra),
        allowed_cats=BEDROOM_SECONDARY_CATEGORIES,
    )
    igns = [_make_ignition(primary, 0.0, preset, rng)]
    for k, s in enumerate(secondaries):
        t = float(rng.uniform(*preset.secondary_delay_s)) + 30.0 * k
        igns.append(_make_ignition(s, t, preset, rng))
    return igns


def _template_living_room_electric(inv: Dict, rng: np.random.Generator,
                                   preset: IntensityPreset) -> List[Dict]:
    objs = _inventory_pool(inv)
    primary = _pick_primary(
        objs,
        rng,
        preferred_cats=list(LIVING_ELECTRIC_PRIMARY_CATEGORIES),
        fallback_cats=list(LIVING_ELECTRIC_FALLBACK_CATEGORIES),
    )
    if primary is None:
        return []
    n_extra = rng.integers(preset.n_ignitions_min - 1,
                           preset.n_ignitions_max) if preset.n_ignitions_max > 1 else 0
    secondaries = _pick_secondary(
        objs,
        primary,
        rng,
        radius_m=3.0,
        n=int(n_extra),
        allowed_cats=LIVING_ELECTRIC_SECONDARY_CATEGORIES,
    )
    igns = [_make_ignition(primary, 0.0, preset, rng)]
    for k, s in enumerate(secondaries):
        t = float(rng.uniform(*preset.secondary_delay_s)) + 30.0 * k
        igns.append(_make_ignition(s, t, preset, rng))
    return igns


def _template_multi_origin(inv: Dict, rng: np.random.Generator,
                           preset: IntensityPreset) -> List[Dict]:
    """Multiple ignitions on the *same floor* (stress test)."""
    candidates = [o for o in _inventory_pool(inv) if o.get("flammability", 0) >= 0.4]
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
