"""Route-contrast objectives for curated FireWorld plans.

This module turns the qualitative requirement "the blind planner takes the
short unsafe route while the risk-aware planner detours" into a deterministic
grid calculation.  It deliberately has no Habitat dependency, so candidate
ranking and regressions can run on CPU with compact synthetic fixtures.  The
scene-level CLI in :mod:`scripts.tune_fire_route_scenarios` supplies real HM3D
navmesh grids, ObjectNav starts/goals and semantic ignition objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.risk.config import RiskConfig
from utils.risk.model import GridFrame
from utils.risk.projection import project_fire_fields

from .plan_ids import semantic_plan_id


GridCell = Tuple[int, int]


_MOVES: Tuple[Tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class RouteContrastThresholds:
    """Acceptance bounds for one blind-versus-aware route pair."""

    min_blind_max_risk: float = 0.60
    max_aware_max_risk: float = 0.35
    min_exposure_reduction: float = 0.70
    min_detour_ratio: float = 1.15
    max_detour_ratio: float = 1.80
    min_path_divergence: float = 0.20

    def to_dict(self) -> Dict[str, float]:
        return {
            key: float(value)
            for key, value in self.__dict__.items()
        }

    def __post_init__(self) -> None:
        for name in (
            "min_blind_max_risk",
            "max_aware_max_risk",
            "min_exposure_reduction",
            "min_path_divergence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 1.0 <= float(self.min_detour_ratio):
            raise ValueError("min_detour_ratio must be at least one")
        if float(self.max_detour_ratio) < float(self.min_detour_ratio):
            raise ValueError(
                "max_detour_ratio must be >= min_detour_ratio"
            )


@dataclass(frozen=True)
class RoutePath:
    """One path and its geometric/risk measurements."""

    cells: Tuple[GridCell, ...]
    length_cells: float
    objective_cost: float
    cumulative_exposure: float
    mean_risk: float
    max_risk: float
    hard_unsafe_cells: int

    def to_dict(self, *, resolution_m: float) -> Dict[str, object]:
        payload = asdict(self)
        payload["cells"] = [list(cell) for cell in self.cells]
        payload["length_m"] = float(self.length_cells * resolution_m)
        return payload


@dataclass(frozen=True)
class RouteContrast:
    """Counterfactual result for one fixed grid/start/goal/risk map."""

    accepted: bool
    score: float
    blind: RoutePath
    aware: RoutePath
    detour_ratio: float
    exposure_reduction: float
    path_divergence: float
    reasons: Tuple[str, ...]

    def to_dict(self, *, resolution_m: float) -> Dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "score": float(self.score),
            "detour_ratio": float(self.detour_ratio),
            "exposure_reduction": float(self.exposure_reduction),
            "path_divergence": float(self.path_divergence),
            "reasons": list(self.reasons),
            "blind": self.blind.to_dict(resolution_m=resolution_m),
            "aware": self.aware.to_dict(resolution_m=resolution_m),
        }


@dataclass(frozen=True)
class CuratedFireProfile:
    """Plan and geometric-surrogate settings for one scenario family."""

    name: str
    duration_s: float
    source_radius_m: float
    source_temp_c: float
    smoke_yield: float
    synthetic_core_radius_m: float
    synthetic_risk_radius_m: float
    thresholds: RouteContrastThresholds
    propagation_rules: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.name not in {"stable", "dynamic"}:
            raise ValueError("curated profile must be stable or dynamic")
        if float(self.duration_s) <= 0.0:
            raise ValueError("duration_s must be positive")
        if not 0.0 < float(self.synthetic_core_radius_m):
            raise ValueError("synthetic_core_radius_m must be positive")
        if (
            float(self.synthetic_risk_radius_m)
            <= float(self.synthetic_core_radius_m)
        ):
            raise ValueError(
                "synthetic_risk_radius_m must exceed the core radius"
            )


_COMMON_RULES: Dict[str, object] = {
    "flammable_threshold": 0.4,
    "ignition_temp_c": 250.0,
    "spread_kernel": "gaussian",
    "ceiling_jet_speed_m_per_s": 0.30,
    "buoyancy_v_m_per_s": 0.50,
    "thermal_diffusivity": 0.05,
    "ambient_temp_c": 25.0,
    "floor_thermal_attenuation": 0.20,
    "flame_through_floors": 0,
    "fuel_neighborhood_cells": 2,
    "fuel_abundance_min": 0.5,
    "fuel_abundance_max": 2.0,
    "floor_ignite_temp_c": 275.0,
    "floor_ignite_radius_cells": 1,
    "floor_flame_contact_thresh": 0.18,
    "secondary_floor_spread_scale": 0.45,
    "limit_flame_to_source_envelope": 1,
    "floor_seed_flame_min": 0.10,
    "floor_seed_flame_max": 0.45,
    "inextinguishable_sources": 1,
    "flame_column_cells": 3,
    "flame_column_decay": 0.55,
}


CURATED_FIRE_PROFILES: Dict[str, CuratedFireProfile] = {
    "stable": CuratedFireProfile(
        name="stable",
        duration_s=300.0,
        source_radius_m=0.42,
        source_temp_c=820.0,
        smoke_yield=0.48,
        synthetic_core_radius_m=0.55,
        synthetic_risk_radius_m=1.55,
        thresholds=RouteContrastThresholds(),
        propagation_rules={
            **_COMMON_RULES,
            "spread_speed_m_per_s": 0.035,
            "radiative_gain_c": 150.0,
            "radiative_radius_cells": 2,
            "floor_fuel_value": 0.28,
            "floor_spread_speed_m_per_s": 0.0025,
            "floor_max_spread_radius_m": 0.95,
        },
    ),
    "dynamic": CuratedFireProfile(
        name="dynamic",
        duration_s=420.0,
        source_radius_m=0.32,
        source_temp_c=850.0,
        smoke_yield=0.52,
        # The surrogate core includes the runtime 0.50 m flame-safety
        # dilation plus the expected local floor-fire footprint.  The first
        # real TEEsav bake showed that a 0.65 m core was optimistic and could
        # approve a topology whose alternate corridor was later hard-blocked.
        synthetic_core_radius_m=1.20,
        synthetic_risk_radius_m=2.30,
        # A dynamic scene is allowed to show an efficient lateral avoidance:
        # the route must still be distinct and much safer, but need not be
        # 15% longer when a parallel corridor happens to be available.
        thresholds=RouteContrastThresholds(min_detour_ratio=1.05),
        propagation_rules={
            **_COMMON_RULES,
            # Route-contrast scenes need a growing local hazard, not a
            # whole-room flashover that eventually removes every feasible
            # route.  Keep the sustained source and bounded floor front, but
            # prevent ordinary background furniture from becoming a chain
            # of unplanned secondary ignitions.
            "flammable_threshold": 0.95,
            "spread_speed_m_per_s": 0.025,
            "radiative_gain_c": 100.0,
            "radiative_radius_cells": 2,
            "object_spread_speed_m_per_s": 0.0035,
            "object_max_spread_radius_m": 0.85,
            "object_bbox_fill_speed_m_per_s": 0.0,
            "floor_fuel_value": 0.26,
            "floor_spread_speed_m_per_s": 0.0035,
            "floor_max_spread_radius_m": 0.85,
        },
    ),
}


def _as_cell(cell: Sequence[int], shape: Tuple[int, int]) -> GridCell:
    if len(cell) < 2:
        raise ValueError("grid cell must contain row and column")
    result = (int(cell[0]), int(cell[1]))
    if not (0 <= result[0] < shape[0] and 0 <= result[1] < shape[1]):
        raise ValueError(f"grid cell {result} lies outside shape {shape}")
    return result


def _grid_inputs(
    traversible: np.ndarray,
    risk: Optional[np.ndarray],
    hard_unsafe: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    domain = np.asarray(traversible) != 0
    if domain.ndim != 2:
        raise ValueError("traversible must be a 2-D grid")
    if risk is None:
        risk_grid = np.zeros(domain.shape, dtype=np.float32)
    else:
        risk_grid = np.asarray(risk, dtype=np.float32)
        if risk_grid.shape != domain.shape:
            raise ValueError("risk shape must match traversible")
        risk_grid = np.clip(
            np.nan_to_num(risk_grid, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
    if hard_unsafe is None:
        hard = np.zeros(domain.shape, dtype=bool)
    else:
        hard = np.asarray(hard_unsafe, dtype=bool)
        if hard.shape != domain.shape:
            raise ValueError("hard_unsafe shape must match traversible")
    return domain, risk_grid, hard


def shortest_grid_path(
    traversible: np.ndarray,
    start: Sequence[int],
    goals: Iterable[Sequence[int]],
    *,
    risk: Optional[np.ndarray] = None,
    risk_alpha: float = 0.0,
    hard_unsafe: Optional[np.ndarray] = None,
) -> Optional[Tuple[Tuple[GridCell, ...], float, float]]:
    """Return an eight-connected optimal path, objective and length.

    Diagonal moves may not cut an obstacle or hard-risk corner.  The edge
    cost matches the local A* implementation: geometric length multiplied by
    ``1 + risk_alpha * mean_endpoint_risk``.
    """

    domain, risk_grid, hard = _grid_inputs(
        traversible, risk, hard_unsafe
    )
    shape = domain.shape
    source = _as_cell(start, shape)
    targets = {_as_cell(goal, shape) for goal in goals}
    targets = {goal for goal in targets if domain[goal] and not hard[goal]}
    if not targets or not domain[source] or hard[source]:
        return None

    alpha = max(0.0, float(risk_alpha))
    queue: List[Tuple[float, int, GridCell]] = [(0.0, 0, source)]
    costs: Dict[GridCell, float] = {source: 0.0}
    lengths: Dict[GridCell, float] = {source: 0.0}
    parents: Dict[GridCell, Optional[GridCell]] = {source: None}
    ordinal = 1
    reached: Optional[GridCell] = None

    while queue:
        cost, _, current = heapq.heappop(queue)
        if cost > costs[current] + 1e-9:
            continue
        if current in targets:
            reached = current
            break
        for drow, dcol, geometric in _MOVES:
            nxt = (current[0] + drow, current[1] + dcol)
            if not (
                0 <= nxt[0] < shape[0]
                and 0 <= nxt[1] < shape[1]
                and domain[nxt]
                and not hard[nxt]
            ):
                continue
            if drow != 0 and dcol != 0:
                side_a = (current[0] + drow, current[1])
                side_b = (current[0], current[1] + dcol)
                if (
                    not domain[side_a]
                    or hard[side_a]
                    or not domain[side_b]
                    or hard[side_b]
                ):
                    continue
            mean_risk = 0.5 * (
                float(risk_grid[current]) + float(risk_grid[nxt])
            )
            candidate = cost + geometric * (1.0 + alpha * mean_risk)
            if candidate + 1e-9 >= costs.get(nxt, math.inf):
                continue
            costs[nxt] = candidate
            lengths[nxt] = lengths[current] + geometric
            parents[nxt] = current
            heapq.heappush(queue, (candidate, ordinal, nxt))
            ordinal += 1

    if reached is None:
        return None
    cells: List[GridCell] = []
    cursor: Optional[GridCell] = reached
    while cursor is not None:
        cells.append(cursor)
        cursor = parents[cursor]
    cells.reverse()
    return tuple(cells), float(costs[reached]), float(lengths[reached])


def _measure_path(
    path: Tuple[Tuple[GridCell, ...], float, float],
    risk: np.ndarray,
    hard: np.ndarray,
) -> RoutePath:
    cells, objective, length = path
    if len(cells) <= 1:
        cumulative = 0.0
    else:
        cumulative = 0.0
        for current, nxt in zip(cells, cells[1:]):
            geometric = float(np.linalg.norm(np.subtract(nxt, current)))
            cumulative += geometric * 0.5 * (
                float(risk[current]) + float(risk[nxt])
            )
    values = np.asarray([risk[cell] for cell in cells], dtype=np.float64)
    return RoutePath(
        cells=cells,
        length_cells=float(length),
        objective_cost=float(objective),
        cumulative_exposure=float(cumulative),
        mean_risk=float(values.mean()) if values.size else 0.0,
        max_risk=float(values.max()) if values.size else 0.0,
        hard_unsafe_cells=int(sum(bool(hard[cell]) for cell in cells)),
    )


def evaluate_route_contrast(
    traversible: np.ndarray,
    start: Sequence[int],
    goals: Iterable[Sequence[int]],
    risk: np.ndarray,
    hard_unsafe: np.ndarray,
    *,
    risk_alpha: float = 4.0,
    thresholds: RouteContrastThresholds = RouteContrastThresholds(),
) -> Optional[RouteContrast]:
    """Compare obstacle-only shortest routing with risk-aware routing."""

    domain, risk_grid, hard = _grid_inputs(
        traversible, risk, hard_unsafe
    )
    blind_raw = shortest_grid_path(domain, start, goals)
    aware_raw = shortest_grid_path(
        domain,
        start,
        goals,
        risk=risk_grid,
        risk_alpha=risk_alpha,
        hard_unsafe=hard,
    )
    if blind_raw is None or aware_raw is None:
        return None
    blind = _measure_path(blind_raw, risk_grid, hard)
    aware = _measure_path(aware_raw, risk_grid, hard)
    detour = aware.length_cells / max(blind.length_cells, 1e-9)
    exposure_reduction = (
        1.0
        - aware.cumulative_exposure
        / max(blind.cumulative_exposure, 1e-9)
        if blind.cumulative_exposure > 1e-9
        else 0.0
    )
    blind_cells = set(blind.cells)
    aware_cells = set(aware.cells)
    union = blind_cells | aware_cells
    divergence = (
        1.0 - len(blind_cells & aware_cells) / len(union)
        if union
        else 0.0
    )

    reasons: List[str] = []
    if blind.max_risk < thresholds.min_blind_max_risk:
        reasons.append("blind_path_not_dangerous")
    if aware.max_risk > thresholds.max_aware_max_risk:
        reasons.append("aware_path_still_dangerous")
    if aware.hard_unsafe_cells:
        reasons.append("aware_path_crosses_hard_unsafe")
    if exposure_reduction < thresholds.min_exposure_reduction:
        reasons.append("insufficient_exposure_reduction")
    if detour < thresholds.min_detour_ratio:
        reasons.append("detour_too_short")
    if detour > thresholds.max_detour_ratio:
        reasons.append("detour_too_long")
    if divergence < thresholds.min_path_divergence:
        reasons.append("paths_not_topologically_distinct")

    # Higher is better.  The ratio term peaks halfway through the accepted
    # interval so an absurdly long detour cannot win on exposure alone.
    ratio_mid = 0.5 * (
        thresholds.min_detour_ratio + thresholds.max_detour_ratio
    )
    ratio_half_width = max(
        0.5 * (
            thresholds.max_detour_ratio - thresholds.min_detour_ratio
        ),
        1e-6,
    )
    ratio_quality = max(0.0, 1.0 - abs(detour - ratio_mid) / ratio_half_width)
    score = (
        0.40 * float(np.clip(exposure_reduction, 0.0, 1.0))
        + 0.25 * float(np.clip(divergence, 0.0, 1.0))
        + 0.20 * ratio_quality
        + 0.15 * float(np.clip(blind.max_risk - aware.max_risk, 0.0, 1.0))
        - 0.10 * len(reasons)
    )
    return RouteContrast(
        accepted=not reasons,
        score=float(score),
        blind=blind,
        aware=aware,
        detour_ratio=float(detour),
        exposure_reduction=float(exposure_reduction),
        path_divergence=float(divergence),
        reasons=tuple(reasons),
    )


def radial_hazard_map(
    shape: Tuple[int, int],
    centre: Sequence[int],
    *,
    resolution_m: float,
    core_radius_m: float,
    risk_radius_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a deterministic radial surrogate used before expensive baking."""

    if float(resolution_m) <= 0.0:
        raise ValueError("resolution_m must be positive")
    if float(risk_radius_m) <= float(core_radius_m):
        raise ValueError("risk_radius_m must exceed core_radius_m")
    cell = _as_cell(centre, shape)
    rows, cols = np.indices(shape, dtype=np.float64)
    distance_m = np.hypot(rows - cell[0], cols - cell[1]) * float(
        resolution_m
    )
    span = float(risk_radius_m) - float(core_radius_m)
    risk = np.clip((float(risk_radius_m) - distance_m) / span, 0.0, 1.0)
    risk[distance_m <= float(core_radius_m)] = 1.0
    hard = distance_m <= float(core_radius_m)
    return risk.astype(np.float32), hard


def _instance_value(instance: Mapping[str, object], *keys: str):
    for key in keys:
        if key in instance and instance[key] is not None:
            return instance[key]
    return None


def curated_plan_hash(payload: Mapping[str, object]) -> str:
    """Hash the complete tuned payload rather than only template inputs."""

    canonical = {
        key: value
        for key, value in payload.items()
        if key not in {"plan_id", "plan_hash"}
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_curated_plan(
    inventory: Mapping[str, object],
    ignition_instance: Mapping[str, object],
    profile: CuratedFireProfile,
    *,
    seed: int,
    curation: Mapping[str, object],
) -> Dict[str, object]:
    """Create one canonical route-contrast plan from a semantic object."""

    scene_id = str(inventory["scene_id"])
    position = _instance_value(ignition_instance, "centroid", "position")
    if position is None or len(position) < 3:
        raise ValueError("ignition instance must expose a 3-D centroid")
    object_id = _instance_value(
        ignition_instance, "instance_id", "object_id", "id"
    )
    if object_id is None:
        raise ValueError("ignition instance must expose an object id")
    category = str(ignition_instance.get("category") or "curated fuel")
    ignition = {
        "object_id": int(object_id),
        "category": category,
        "position": [float(value) for value in position[:3]],
        "ignite_time_s": 0.0,
        "source_radius_m": float(profile.source_radius_m),
        "source_temp_c": float(profile.source_temp_c),
        "fuel_kg": 1.0,
        "smoke_yield": float(profile.smoke_yield),
        "sustain_s": float(profile.duration_s),
        "floor_spread_scale": 1.0,
        "ignition_role": "initial",
    }
    payload: Dict[str, object] = {
        "schema_version": 4,
        "scene_id": scene_id,
        "scene_glb": inventory.get("scene_glb"),
        "world_aabb": inventory["world_aabb"],
        "fire_type": "route_contrast",
        "intensity": profile.name,
        "seed": int(seed),
        "template_version": 1,
        "duration_s": float(profile.duration_s),
        "num_initial_ignitions": 1,
        "ignition_selection_mode": "curated_route_contrast",
        "ignition_selection_version": 1,
        "ignitions": [ignition],
        "propagation_rules": dict(profile.propagation_rules),
        "curation": dict(curation),
    }
    plan_hash = curated_plan_hash(payload)
    payload["plan_hash"] = plan_hash
    payload["plan_id"] = semantic_plan_id(
        scene_id,
        "route_contrast",
        profile.name,
        plan_hash,
    )
    return payload


def evaluate_fireworld_snapshot(
    fire_world,
    *,
    timestamp_s: float,
    frame: GridFrame,
    floor_y_m: float,
    traversible: np.ndarray,
    start: Sequence[int],
    goals: Iterable[Sequence[int]],
    risk_alpha: float = 4.0,
    risk_config: Optional[RiskConfig] = None,
    thresholds: RouteContrastThresholds = RouteContrastThresholds(),
) -> Tuple[Optional[RouteContrast], object]:
    """Project a baked frame and evaluate its real route contrast."""

    config = risk_config or RiskConfig(enabled=True, source="oracle")
    flame, smoke, temperature = fire_world.query(float(timestamp_s))
    layers = project_fire_fields(
        flame,
        smoke,
        temperature,
        voxel_origin=np.asarray(fire_world.origin, dtype=np.float64),
        voxel_m=float(fire_world.voxel_m),
        frame=frame,
        floor_y_m=float(floor_y_m),
        config=config,
        timestamp_s=float(timestamp_s),
    )
    result = evaluate_route_contrast(
        traversible,
        start,
        goals,
        layers.physical_risk,
        layers.hard_unsafe,
        risk_alpha=risk_alpha,
        thresholds=thresholds,
    )
    return result, layers


def route_overlay(
    traversible: np.ndarray,
    risk: np.ndarray,
    contrast: RouteContrast,
    *,
    start: Sequence[int],
    goal: Sequence[int],
    ignition: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Render a compact RGB diagnostic without plotting dependencies."""

    domain = np.asarray(traversible) != 0
    risk_grid = np.asarray(risk, dtype=np.float32)
    if risk_grid.shape != domain.shape:
        raise ValueError("risk shape must match traversible")
    image = np.zeros(domain.shape + (3,), dtype=np.uint8)
    image[domain] = (225, 225, 225)
    red = (255.0 * np.clip(risk_grid, 0.0, 1.0)).astype(np.uint8)
    image[..., 0] = np.maximum(image[..., 0], red)
    image[..., 1] = np.where(domain, image[..., 1] * (1.0 - 0.65 * risk_grid), 0)
    image[..., 2] = np.where(domain, image[..., 2] * (1.0 - 0.65 * risk_grid), 0)
    for cell in contrast.blind.cells:
        image[cell] = (30, 100, 255)
    for cell in contrast.aware.cells:
        image[cell] = (30, 220, 80)
    image[_as_cell(start, domain.shape)] = (255, 255, 255)
    image[_as_cell(goal, domain.shape)] = (170, 50, 220)
    if ignition is not None:
        image[_as_cell(ignition, domain.shape)] = (255, 220, 0)
    return image


def write_curated_plan(plan: Mapping[str, object], scenes_root: Path) -> Path:
    """Persist a curated plan without overwriting a different payload."""

    scene_id = str(plan["scene_id"])
    plan_id = str(plan["plan_id"])
    path = Path(scenes_root) / scene_id / "plans" / f"{plan_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(plan), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return path
        raise FileExistsError(
            f"refusing to overwrite different curated plan at {path}"
        )
    path.write_text(serialized, encoding="utf-8")
    return path


__all__ = [
    "CURATED_FIRE_PROFILES",
    "CuratedFireProfile",
    "RouteContrast",
    "RouteContrastThresholds",
    "RoutePath",
    "build_curated_plan",
    "curated_plan_hash",
    "evaluate_fireworld_snapshot",
    "evaluate_route_contrast",
    "radial_hazard_map",
    "route_overlay",
    "shortest_grid_path",
    "write_curated_plan",
]
