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
from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np


TEMPLATE_VERSION = 10
# Version of the contract in which templates select t=0 sources only.
IGNITION_SELECTION_VERSION = 1


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
# multi_origin sources should be stable, low floor-standing furniture.
# Cabinets/wardrobes and small or elevated objects (lamp, pillow, laptop,
# wall TV, curtain) create implausible initial flames near head/ceiling height,
# so they are excluded from the planner's t=0 source pool. They may still
# ignite later when the propagation solver heats their fuel voxels.
MULTI_ORIGIN_INITIAL_CATEGORIES = (
    "bed",
    "bed small",
    "bedframe",
    "sofa",
    "couch",
    "armchair",
    "chair",
    "table",
    "rug",
    "carpet",
    "desk",
    "ottoman",
)

TEMPLATE_CATEGORY_GROUPS = {
    "kitchen_grease_fire": {
        "primary": KITCHEN_PRIMARY_CATEGORIES,
        "fallback": KITCHEN_FALLBACK_CATEGORIES,
    },
    "bedroom_textile": {
        "primary": BEDROOM_PRIMARY_CATEGORIES,
        "fallback": BEDROOM_FALLBACK_CATEGORIES,
    },
    "living_room_electric": {
        "primary": LIVING_ELECTRIC_PRIMARY_CATEGORIES,
        "fallback": LIVING_ELECTRIC_FALLBACK_CATEGORIES,
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


INTENSITIES: Dict[str, IntensityPreset] = {
    "light": IntensityPreset(
        n_ignitions_min=1, n_ignitions_max=1,
        source_temp_c=550.0, fuel_kg=2.0,
        duration_s=300.0,
    ),
    "medium": IntensityPreset(
        n_ignitions_min=1, n_ignitions_max=2,
        source_temp_c=750.0, fuel_kg=5.0,
        duration_s=600.0,
    ),
    "severe": IntensityPreset(
        n_ignitions_min=2, n_ignitions_max=3,
        source_temp_c=950.0, fuel_kg=10.0,
        duration_s=900.0,
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
            "region_id": it.get("region_id"),
            "floor_id": it.get("floor_id"),
        }
        out.append(norm)
    return out


def _make_ignition(obj: Dict, t_s: float, preset: IntensityPreset, rng: np.random.Generator) -> Dict:
    # Source radius scales with object footprint (clamped).
    bb_min = np.array(obj["aabb_min"])
    bb_max = np.array(obj["aabb_max"])
    extent_xz = max(bb_max[0] - bb_min[0], bb_max[2] - bb_min[2])
    # Keep the ignition core local even for a large bed/sofa. Wider visual
    # coverage should come from gradual propagation, not an oversized t=0
    # sphere that immediately fills the camera.
    src_r = float(np.clip(0.25 + 0.25 * extent_xz, 0.20, 0.60))
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


def _weighted_sample_without_replacement(
    objects: List[Dict],
    n: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Sample ``n`` distinct objects using flammability as the weight."""
    if n > len(objects):
        return []
    weights = np.asarray(
        [max(0.05, o.get("flammability", 0.3)) for o in objects],
        dtype=np.float64,
    )
    weights /= weights.sum()
    indices = rng.choice(len(objects), size=n, replace=False, p=weights)
    return [objects[int(i)] for i in np.atleast_1d(indices)]


def _pick_explicit_initials(
    objects: List[Dict],
    fire_type: str,
    n_initial: int,
    rng: np.random.Generator,
) -> List[Dict]:
    """Pick exactly ``n_initial`` objects that the planner lights at t=0.

    Templates constrain only the initial cause of the fire. They never choose
    which objects ignite later; that is decided by the propagation solver from
    the voxel temperature, fuel, contact and radiation fields.
    """
    if fire_type != "multi_origin":
        groups = TEMPLATE_CATEGORY_GROUPS[fire_type]
        preferred = _filter_by_category(objects, list(groups["primary"]))
        preferred_ids = {o["object_id"] for o in preferred}
        fallback = [
            o
            for o in _filter_by_category(objects, list(groups["fallback"]))
            if o["object_id"] not in preferred_ids
        ]
        # Preserve each template's primary-category preference. Fall back only
        # when the preferred pool cannot satisfy the requested initial count.
        pool = preferred if len(preferred) >= n_initial else preferred + fallback
        if len(pool) < n_initial:
            raise RuntimeError(
                f"template {fire_type!r} could not satisfy "
                f"num_ignitions={n_initial}: only {len(pool)} eligible "
                "initial ignition objects are available in its primary "
                "and fallback categories"
            )
        return _weighted_sample_without_replacement(pool, n_initial, rng)

    initial_categories = {
        category.lower() for category in MULTI_ORIGIN_INITIAL_CATEGORIES
    }
    raw_candidates = [
        o
        for o in objects
        if o.get("flammability", 0.0) >= 0.4
        and o["category"].lower() in initial_categories
    ]
    if len(raw_candidates) < n_initial:
        raise RuntimeError(
            f"template {fire_type!r} could not satisfy "
            f"num_ignitions={n_initial}: only {len(raw_candidates)} eligible "
            "flammable initial ignition objects are available"
        )

    # multi_origin remains a same-floor stress test. Choose the floor with the
    # richest eligible pool, then use farthest-point sampling so the t=0
    # sources cover distinct areas rather than collapsing into one corner.
    ys = np.asarray([_object_center(o)[1] for o in raw_candidates])
    bins = np.round(ys / 1.5).astype(int)
    counts = {
        int(bin_id): int(np.sum(bins == bin_id))
        for bin_id in sorted(set(bins.tolist()))
    }
    valid_bins = [bin_id for bin_id, count in counts.items()
                  if count >= n_initial]
    if not valid_bins:
        largest_floor_pool = max(counts.values(), default=0)
        raise RuntimeError(
            f"template {fire_type!r} could not satisfy "
            f"num_ignitions={n_initial}: the largest same-floor group has "
            f"only {largest_floor_pool} eligible initial ignition objects"
        )
    best_bin = max(valid_bins, key=lambda bin_id: (counts[bin_id], -bin_id))
    floor_objects = [
        obj for obj, bin_id in zip(raw_candidates, bins)
        if int(bin_id) == best_bin
    ]
    centres = np.asarray([_object_center(o) for o in floor_objects])
    weights = np.asarray(
        [max(0.05, obj.get("flammability", 0.3)) for obj in floor_objects],
        dtype=np.float64,
    )
    weights /= weights.sum()
    first = int(rng.choice(len(floor_objects), p=weights))
    selected = [first]
    for _ in range(1, n_initial):
        distances = np.min(
            np.linalg.norm(
                centres[:, None, [0, 2]]
                - centres[selected][None, :, [0, 2]],
                axis=-1,
            ),
            axis=1,
        )
        distances[selected] = -1.0
        # Prefer high flammability only when spatial distances tie.
        score = distances + 1e-6 * weights
        selected.append(int(np.argmax(score)))
    return [floor_objects[index] for index in selected]


def _initial_candidate_capacity(objects: List[Dict], fire_type: str) -> int:
    """Maximum initial-source count supported by template categories/floor."""
    if fire_type != "multi_origin":
        groups = TEMPLATE_CATEGORY_GROUPS[fire_type]
        initial_categories = {
            category.lower()
            for category in (*groups["primary"], *groups["fallback"])
        }
        return sum(
            obj["category"].lower() in initial_categories
            for obj in objects
        )

    initial_categories = {
        category.lower() for category in MULTI_ORIGIN_INITIAL_CATEGORIES
    }
    raw_candidates = [
        obj
        for obj in objects
        if obj.get("flammability", 0.0) >= 0.4
        and obj["category"].lower() in initial_categories
    ]
    if not raw_candidates:
        return 0
    ys = np.asarray([_object_center(obj)[1] for obj in raw_candidates])
    bins = np.round(ys / 1.5).astype(int)
    return max(
        (int(np.sum(bins == bin_id)) for bin_id in set(bins.tolist())),
        default=0,
    )


def build_template_ignitions(
    inv: Dict,
    fire_type: str,
    rng: np.random.Generator,
    preset: IntensityPreset,
    n_initial: Optional[int] = None,
) -> List[Dict]:
    """Generate only the initial ignition objects lit at t=0.

    When ``n_initial`` is omitted, the initial count is drawn from the
    intensity preset after capping the range to the template's available
    initial-object capacity. An explicit value is never reduced silently.
    No later object is scheduled here.
    """
    if fire_type == "multi_origin" and n_initial is not None and n_initial < 2:
        raise ValueError(
            "fire_type='multi_origin' requires at least two initial sources"
        )
    if n_initial is None:
        capacity = _initial_candidate_capacity(
            _inventory_pool(inv), fire_type
        )
        minimum = int(preset.n_ignitions_min)
        if fire_type == "multi_origin":
            minimum = max(2, minimum)
        preset_maximum = int(preset.n_ignitions_max)
        if fire_type == "multi_origin":
            preset_maximum = max(2, preset_maximum)
        maximum = min(preset_maximum, capacity)
        if maximum < minimum:
            raise RuntimeError(
                f"template {fire_type!r} needs at least {minimum} initial "
                f"ignition objects for this intensity, but only {capacity} "
                "eligible objects are available"
            )
        if minimum == maximum:
            n_initial = minimum
        else:
            n_initial = int(rng.integers(minimum, maximum + 1))

    objects = _inventory_pool(inv)
    selected = _pick_explicit_initials(
        objects, fire_type, int(n_initial), rng
    )
    ignitions = []
    for obj in selected:
        ignition = _make_ignition(obj, 0.0, preset, rng)
        ignition["ignition_role"] = "initial"
        ignitions.append(ignition)
    return ignitions


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
        # Restore the pasted solver's 0.12 m/s local surface transport. Hard
        # source envelopes still prevent the old room-wide expansion.
        "spread_speed_m_per_s": 0.12,
        # 'laplacian' (legacy) or 'gaussian' (smoother, NIST-FDS-like).
        "spread_kernel": "laplacian",
        "ceiling_jet_speed_m_per_s": 0.30,
        "buoyancy_v_m_per_s": 0.5,     # vertical plume velocity
        "thermal_diffusivity": 0.05,   # alpha used in heat diffusion
        "ambient_temp_c": 25.0,
        # Radiative pre-heating that lets heat jump ~0.6 m to the next
        # piece of furniture in the same room. Scales linearly with
        # local flame intensity. The pasted solver's 200 C/s, 4-cell kernel
        # restores a continuous preheat bridge between nearby furniture; the
        # object envelope still clips actual ignition to a local domain.
        "radiative_gain_c": 200.0,
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
        "floor_fuel_value": 0.52,
        # Direct contact ignition: floor voxels within this many cells
        # of a flame voxel ignite immediately, bypassing heat
        # diffusion's smoothing. Without this the per-voxel thermal
        # spike gets averaged out before reaction sees it.
        "floor_ignite_radius_cells": 2,
        "floor_flame_contact_thresh": 0.12,
        # The combustible floor grows radially around each actual source.
        # The fixed speed is a lower bound; floor_spread_reach_fraction
        # supplies a duration-relative deadline that may increase the
        # per-source constant speed just enough to reach its hard radius.
        "floor_spread_speed_m_per_s": 0.0044,
        "floor_max_spread_radius_m": 2.20,
        "floor_spread_reach_fraction": 0.90,
        # Object ignition is solver-driven but local. This wider, slowly
        # growing envelope lets nearby furniture ignite naturally without
        # allowing overlapping HM3D AABBs to cascade across the whole room.
        "object_spread_speed_m_per_s": 0.0066,
        "object_max_spread_radius_m": 1.35,
        # After a real object ignites, advance a deterministic six-connected
        # flame front through voxels belonging to that exact object ID. At the
        # medium default, one 0.15 m layer takes about 18.75 s.
        "object_bbox_fill_speed_m_per_s": 0.008,
        # Bound the BBox shortcut to the lower part of tall furniture. Heat
        # and smoke still rise above this height through their own fields.
        "object_bbox_max_vertical_spread_m": 0.75,
        "object_bbox_fill_flame_min": 0.16,
        "object_bbox_fill_temp_margin_c": 40.0,
        # The next six-connected voxel shell is preheated and faded in over
        # one layer instead of appearing in a single discrete jump.
        "object_bbox_fill_front_width_layers": 1.0,
        # Do not add another resolution-dependent layer above the metric
        # flame-column limit.
        "object_flame_extra_height_cells": 0,
        # A soft Gaussian front expands radially around each initial source.
        # Later furniture ignition comes from physical heat transfer rather
        # than a planner-authored corridor or delayed source.
        "floor_gaussian_sigma_fraction": 0.58,
        "floor_gaussian_min_influence": 0.08,
        "floor_front_softness_m": 0.18,
        "floor_min_visible_flame": 0.06,
        # Air/floor flame is clipped to source-centred XZ envelopes. Flame on
        # real fuel-bearing object voxels is retained outside the envelope so
        # solver-ignited furniture remains visible.
        "limit_flame_to_source_envelope": 1,
        # Lower seed intensity prevents a newly ignited floor patch from
        # appearing as an opaque, uniformly bright sheet.
        "floor_seed_flame_min": 0.10,
        "floor_seed_flame_max": 0.58,
        # Sustained point sources never expire when 1, ensuring the
        # initial ignitions don't burn themselves out before they have
        # a chance to set the rest of the room alight.
        "inextinguishable_sources": 1,
        # The cell count remains an upper bound for compatibility, while the
        # metric cap makes the final plume height independent of bake
        # resolution. The source envelope also clips physical flame
        # transported higher by buoyancy.
        "flame_column_cells": 3,
        "max_flame_column_height_m": 0.30,
        "flame_column_decay": 0.50,
    }
    if intensity == "light":
        base["spread_speed_m_per_s"] *= 0.7
        base["buoyancy_v_m_per_s"] *= 0.8
        base["radiative_gain_c"] *= 0.6
        base["floor_spread_speed_m_per_s"] *= 0.75
        base["floor_max_spread_radius_m"] *= 0.65
        base["object_spread_speed_m_per_s"] *= 0.75
        base["object_max_spread_radius_m"] *= 0.88
        base["object_bbox_fill_speed_m_per_s"] *= 0.75
        base["object_bbox_max_vertical_spread_m"] = 0.55
        base["flame_column_cells"] = 2
        base["max_flame_column_height_m"] = 0.20
        base["flame_column_decay"] = 0.45
    elif intensity == "severe":
        base["spread_speed_m_per_s"] *= 1.5
        base["ceiling_jet_speed_m_per_s"] *= 1.4
        base["buoyancy_v_m_per_s"] *= 1.3
        base["radiative_gain_c"] *= 1.5
        base["floor_spread_speed_m_per_s"] *= 1.25
        base["floor_max_spread_radius_m"] *= 1.35
        base["object_spread_speed_m_per_s"] *= 1.25
        base["object_max_spread_radius_m"] *= 1.12
        base["object_bbox_fill_speed_m_per_s"] *= 1.25
        base["object_bbox_max_vertical_spread_m"] = 0.90
        base["max_flame_column_height_m"] = 0.35
        base["flame_column_decay"] = 0.55
    return base


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------
def _registered_initial_template(
    inv: Dict,
    rng: np.random.Generator,
    preset: IntensityPreset,
    *,
    fire_type: str,
) -> List[Dict]:
    return build_template_ignitions(inv, fire_type, rng, preset)


TEMPLATES: Dict[
    str,
    Callable[[Dict, np.random.Generator, IntensityPreset], List[Dict]],
] = {
    fire_type: partial(_registered_initial_template, fire_type=fire_type)
    for fire_type in (
        "kitchen_grease_fire",
        "bedroom_textile",
        "living_room_electric",
        "multi_origin",
    )
}
