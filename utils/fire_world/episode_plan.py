"""Episode-level source search for route-contrast FireWorld plans.

The functions in this module deliberately avoid Habitat dependencies.  The
CLI adapter builds the navigation grid, while this module owns the reusable
ObjectNav episode contract and the bounded multi-source combination search.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fine_tuning import (
    RouteContrast,
    RouteContrastThresholds,
    evaluate_route_contrast,
    radial_hazard_map,
)


GridCell = Tuple[int, int]


@dataclass(frozen=True)
class EpisodeGoalGeometry:
    """Native ObjectNav goal centres and success view points on one floor."""

    centres: np.ndarray
    viewpoints: np.ndarray


@dataclass(frozen=True)
class EpisodeSourceCandidate:
    """One semantic fuel object projected onto the episode navigation grid."""

    instance: Mapping[str, object]
    cell: GridCell
    position: Tuple[float, float, float]
    distance_to_blind_path_m: float
    distance_to_start_m: float
    nearest_goal_centre_clearance_m: float
    nearest_goal_viewpoint_clearance_m: float
    blind_path_progress_m: float

    @property
    def object_id(self) -> int:
        return int(self.instance["instance_id"])

    def to_dict(self) -> Dict[str, object]:
        return {
            "object_id": self.object_id,
            "category": str(self.instance.get("category") or "curated fuel"),
            "position": [float(value) for value in self.position],
            "cell": [int(value) for value in self.cell],
            "flammability": float(self.instance.get("flammability", 0.0)),
            "distance_to_blind_path_m": float(self.distance_to_blind_path_m),
            "distance_to_start_m": float(self.distance_to_start_m),
            "nearest_goal_centre_clearance_m": float(
                self.nearest_goal_centre_clearance_m
            ),
            "nearest_goal_viewpoint_clearance_m": float(
                self.nearest_goal_viewpoint_clearance_m
            ),
            "blind_path_progress_m": float(self.blind_path_progress_m),
        }


@dataclass(frozen=True)
class SourceCombinationResult:
    """Best accepted combined-source counterfactual and search diagnostics."""

    sources: Tuple[EpisodeSourceCandidate, ...]
    contrast: RouteContrast
    risk: np.ndarray
    hard_unsafe: np.ndarray
    max_goal_risk: float
    source_progress_span_m: float
    evaluated_combinations: int
    rejection_counts: Mapping[str, int]


def resolve_episode(
    dataset: Mapping[str, object],
    episode_id: object,
    *,
    object_category: Optional[str] = None,
) -> Mapping[str, object]:
    """Resolve an exact episode and reject ambiguous reused episode ids."""

    matches = [
        episode for episode in dataset.get("episodes", [])
        if str(episode.get("episode_id")) == str(episode_id)
        and (
            object_category is None
            or str(episode.get("object_category")) == str(object_category)
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        suffix = (
            "" if object_category is None
            else f" and object_category={object_category!r}"
        )
        raise ValueError(f"no episode_id={str(episode_id)!r}{suffix}")
    categories = sorted({
        str(episode.get("object_category")) for episode in matches
    })
    raise ValueError(
        f"episode_id={str(episode_id)!r} is ambiguous across categories "
        f"{categories}; pass --object-category"
    )


def episode_scene_id(episode: Mapping[str, object]) -> str:
    """Extract the short HM3D scene id from an episode scene path."""

    name = Path(str(episode.get("scene_id") or "")).name
    for suffix in (".basis.glb", ".glb"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        raise ValueError("episode scene_id does not contain a scene name")
    return name


def native_goal_geometry(
    dataset: Mapping[str, object],
    *,
    scene_id: str,
    object_category: str,
    floor_y_m: float,
    floor_tolerance_m: float = 0.60,
) -> EpisodeGoalGeometry:
    """Load the category's native goal centres and same-floor VIEW_POINTS."""

    exact_key = f"{scene_id}.basis.glb_{object_category}"
    goals_by_category = dataset.get("goals_by_category") or {}
    goals = list(goals_by_category.get(exact_key, []))
    if not goals:
        suffix = f"_{object_category}"
        matching_keys = [
            key for key in goals_by_category
            if Path(str(key).split("_", 1)[0]).name.startswith(scene_id)
            and str(key).endswith(suffix)
        ]
        if len(matching_keys) == 1:
            goals = list(goals_by_category[matching_keys[0]])
    centres: List[List[float]] = []
    viewpoints: List[List[float]] = []
    for goal in goals:
        centre = goal.get("position")
        if centre is not None and len(centre) >= 3:
            centres.append([float(value) for value in centre[:3]])
        for view in goal.get("view_points", []):
            position = (view.get("agent_state") or {}).get("position")
            if (
                position is not None
                and len(position) >= 3
                and abs(float(position[1]) - float(floor_y_m))
                <= float(floor_tolerance_m)
            ):
                viewpoints.append([float(value) for value in position[:3]])
    if not centres or not viewpoints:
        raise ValueError(
            f"native goals for {scene_id}/{object_category} must contain "
            "centres and same-floor VIEW_POINTS"
        )
    return EpisodeGoalGeometry(
        centres=np.asarray(centres, dtype=np.float64),
        viewpoints=np.asarray(viewpoints, dtype=np.float64),
    )


def combined_radial_hazard_map(
    shape: Tuple[int, int],
    centres: Iterable[Sequence[int]],
    *,
    resolution_m: float,
    core_radius_m: float,
    risk_radius_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Union hard cores and take pointwise maximum risk for every source."""

    source_cells = [tuple(int(value) for value in cell[:2]) for cell in centres]
    if not source_cells:
        raise ValueError("at least one ignition centre is required")
    risk = np.zeros(shape, dtype=np.float32)
    hard = np.zeros(shape, dtype=bool)
    for cell in source_cells:
        source_risk, source_hard = radial_hazard_map(
            shape,
            cell,
            resolution_m=resolution_m,
            core_radius_m=core_radius_m,
            risk_radius_m=risk_radius_m,
        )
        risk = np.maximum(risk, source_risk)
        hard |= source_hard
    return risk, hard


def source_clearance_metrics(
    position: Sequence[float],
    *,
    start_position: Sequence[float],
    goal_geometry: EpisodeGoalGeometry,
) -> Dict[str, float]:
    """Measure one source in the horizontal plane used by route planning."""

    source_xz = np.asarray(position, dtype=np.float64)[[0, 2]]
    start_xz = np.asarray(start_position, dtype=np.float64)[[0, 2]]
    centre_xz = np.asarray(goal_geometry.centres, dtype=np.float64)[:, [0, 2]]
    view_xz = np.asarray(goal_geometry.viewpoints, dtype=np.float64)[:, [0, 2]]
    return {
        "distance_to_start_m": float(np.linalg.norm(source_xz - start_xz)),
        "nearest_goal_centre_clearance_m": float(np.min(np.linalg.norm(
            centre_xz - source_xz[None, :], axis=1
        ))),
        "nearest_goal_viewpoint_clearance_m": float(np.min(np.linalg.norm(
            view_xz - source_xz[None, :], axis=1
        ))),
    }


def select_best_source_combination(
    traversible: np.ndarray,
    *,
    start: Sequence[int],
    goals: Iterable[Sequence[int]],
    candidates: Sequence[EpisodeSourceCandidate],
    source_count: int,
    resolution_m: float,
    core_radius_m: float,
    risk_radius_m: float,
    risk_alpha: float,
    thresholds: RouteContrastThresholds,
    max_goal_risk: float,
    min_source_spacing_m: float,
    max_combinations: int,
    diagnostics: Optional[Dict[str, object]] = None,
) -> Optional[SourceCombinationResult]:
    """Search bounded semantic-source combinations and replan every union."""

    requested = int(source_count)
    if requested < 1:
        raise ValueError("source_count must be positive")
    if len(candidates) < requested:
        if diagnostics is not None:
            diagnostics.update({
                "candidate_count": int(len(candidates)),
                "evaluated_combinations": 0,
                "rejection_counts": {"insufficient_source_candidates": 1},
            })
        return None
    if int(max_combinations) < 1:
        raise ValueError("max_combinations must be positive")
    domain = np.asarray(traversible) != 0
    goal_cells = sorted({(int(cell[0]), int(cell[1])) for cell in goals})
    if not goal_cells:
        raise ValueError("at least one goal cell is required")

    ordered = sorted(
        candidates,
        key=lambda item: (
            item.distance_to_blind_path_m,
            -item.nearest_goal_viewpoint_clearance_m,
            item.object_id,
        ),
    )
    rejection_counts: Counter[str] = Counter()
    best = None
    evaluated = 0
    for combo in combinations(ordered, requested):
        if evaluated >= int(max_combinations):
            break
        evaluated += 1
        positions = np.asarray([item.position for item in combo], dtype=np.float64)
        if len(combo) > 1:
            distances = np.linalg.norm(
                positions[:, None, [0, 2]] - positions[None, :, [0, 2]],
                axis=2,
            )
            distances += np.eye(len(combo), dtype=np.float64) * 1e9
            if float(np.min(distances)) < float(min_source_spacing_m):
                rejection_counts["sources_too_close"] += 1
                continue
        risk, hard = combined_radial_hazard_map(
            domain.shape,
            [item.cell for item in combo],
            resolution_m=float(resolution_m),
            core_radius_m=float(core_radius_m),
            risk_radius_m=float(risk_radius_m),
        )
        goal_risks = np.asarray([risk[cell] for cell in goal_cells])
        goal_hard = np.asarray([hard[cell] for cell in goal_cells])
        goal_max = float(np.max(goal_risks, initial=0.0))
        if bool(np.any(goal_hard)):
            rejection_counts["goal_enters_hard_unsafe"] += 1
            continue
        if goal_max > float(max_goal_risk):
            rejection_counts["goal_risk_too_high"] += 1
            continue
        contrast = evaluate_route_contrast(
            domain,
            start,
            goal_cells,
            risk,
            hard,
            risk_alpha=float(risk_alpha),
            thresholds=thresholds,
        )
        if contrast is None:
            rejection_counts["no_route"] += 1
            continue
        if not contrast.accepted:
            rejection_counts.update(contrast.reasons)
            continue
        progress = [item.blind_path_progress_m for item in combo]
        span = float(max(progress) - min(progress)) if len(progress) > 1 else 0.0
        key = (float(contrast.score), span, -goal_max)
        if best is None or key > best[0]:
            best = (key, combo, contrast, risk, hard, goal_max, span)
    if diagnostics is not None:
        diagnostics.update({
            "candidate_count": int(len(candidates)),
            "evaluated_combinations": int(evaluated),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        })
    if best is None:
        return None
    _, sources, contrast, risk, hard, goal_max, span = best
    return SourceCombinationResult(
        sources=tuple(sources),
        contrast=contrast,
        risk=risk,
        hard_unsafe=hard,
        max_goal_risk=float(goal_max),
        source_progress_span_m=float(span),
        evaluated_combinations=int(evaluated),
        rejection_counts=dict(sorted(rejection_counts.items())),
    )


__all__ = [
    "EpisodeGoalGeometry",
    "EpisodeSourceCandidate",
    "SourceCombinationResult",
    "combined_radial_hazard_map",
    "episode_scene_id",
    "native_goal_geometry",
    "resolve_episode",
    "select_best_source_combination",
    "source_clearance_metrics",
]
