"""Frontier-level risk reports and deterministic team assignment.

This module deliberately contains no Habitat or model dependencies.  It sits
between a 2-D risk/confidence map and both global-planning backends:

* classical policies can rank or assign frontiers with :func:`assign_frontiers`;
* a VLM can receive the serialisable :class:`FrontierRiskReport` objects; and
* :func:`guard_frontier_assignments` remains the final, deterministic safety
  gate after a VLM response.

All public frontier ids are zero-based.  A labelled frontier image therefore
uses the project convention ``label 1 -> frontier 0`` by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


GridPoint = Tuple[int, int]
FrontierMasks = Union[
    np.ndarray,
    Sequence[np.ndarray],
    Mapping[int, np.ndarray],
]


@dataclass(frozen=True)
class SeverityThresholds:
    """Thresholds for qualitative risk and hard route rejection."""

    safe_max: float = 0.25
    moderate_max: float = 0.55
    hard_max: float = 0.85

    def __post_init__(self) -> None:
        if not 0.0 <= self.safe_max <= self.moderate_max <= self.hard_max <= 1.0:
            raise ValueError(
                "risk thresholds must satisfy "
                "0 <= safe_max <= moderate_max <= hard_max <= 1"
            )


@dataclass(frozen=True)
class UtilityWeights:
    """Weights for the deterministic, maximised frontier utility."""

    information_gain: float = 1.0
    distance: float = 0.35
    risk: float = 1.50
    uncertainty: float = 0.50
    redundancy: float = 0.75


@dataclass(frozen=True)
class FrontierRiskReport:
    """Risk summary for one frontier and, optionally, its approach route.

    ``route_risk`` is the mean risk over the supplied route cells.  The route
    maximum is retained separately because a short lethal segment must not be
    hidden by averaging a long safe route.
    """

    frontier_id: int
    point: GridPoint
    cell_count: int
    mean_risk: float
    p95_risk: float
    max_risk: float
    route_risk: Optional[float]
    route_max_risk: Optional[float]
    severity: str
    confidence: float
    hard_blocked: bool
    route_is_proxy: bool = False

    @property
    def uncertainty(self) -> float:
        return float(np.clip(1.0 - self.confidence, 0.0, 1.0))

    @property
    def planning_risk(self) -> float:
        """Conservative continuous cost used by the assignment helpers."""
        route = 0.0 if self.route_risk is None else self.route_risk
        return float(max(self.p95_risk, route))

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["point"] = [int(self.point[0]), int(self.point[1])]
        payload["uncertainty"] = self.uncertainty
        return payload


@dataclass(frozen=True)
class FrontierScore:
    """Per-robot score for one frontier before team redundancy penalties."""

    robot_id: int
    frontier_id: int
    distance: float
    normalized_distance: float
    information_gain: float
    normalized_information_gain: float
    risk: float
    uncertainty: float
    feasible: bool
    utility: float


@dataclass(frozen=True)
class AssignmentGuardResult:
    """Safety-filtered VLM assignments plus rejection reasons."""

    assignments: Dict[int, Optional[int]]
    rejected: Dict[int, str]


def _as_unit_map(array: np.ndarray, *, nan_value: float) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D map, got shape {values.shape}")
    values = np.nan_to_num(values, nan=nan_value, posinf=1.0, neginf=0.0)
    return np.clip(values, 0.0, 1.0)


def _validate_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    if result.shape != shape:
        raise ValueError(
            f"frontier mask shape {result.shape} does not match risk map {shape}"
        )
    return result


def _valid_route_values(
    values: np.ndarray,
    route_cells: Optional[Iterable[Sequence[int]]],
) -> np.ndarray:
    if route_cells is None:
        return np.empty((0,), dtype=np.float32)
    height, width = values.shape
    samples: List[float] = []
    for cell in route_cells:
        if len(cell) < 2:
            continue
        row, col = int(cell[0]), int(cell[1])
        if 0 <= row < height and 0 <= col < width:
            samples.append(float(values[row, col]))
    return np.asarray(samples, dtype=np.float32)


def _centroid(mask: np.ndarray) -> GridPoint:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return (0, 0)
    centre = np.rint(cells.mean(axis=0)).astype(int)
    return (int(centre[0]), int(centre[1]))


def _severity(risk: float, thresholds: SeverityThresholds) -> str:
    if risk <= thresholds.safe_max:
        return "safe"
    if risk <= thresholds.moderate_max:
        return "moderate"
    return "dangerous"


def frontier_risk_report(
    frontier_id: int,
    frontier_mask: np.ndarray,
    risk_map: np.ndarray,
    confidence_map: Optional[np.ndarray] = None,
    *,
    hard_unsafe_map: Optional[np.ndarray] = None,
    frontier_point: Optional[Sequence[int]] = None,
    route_cells: Optional[Iterable[Sequence[int]]] = None,
    route_is_proxy: bool = False,
    thresholds: SeverityThresholds = SeverityThresholds(),
) -> FrontierRiskReport:
    """Summarise one frontier and its optional route.

    NaN risk is represented as zero risk with zero confidence when callers
    also provide a confidence map.  Unknown space is consequently penalised
    by the uncertainty term, rather than silently becoming a hard obstacle.
    Positive infinity is treated conservatively as maximum risk.
    """

    risk = _as_unit_map(risk_map, nan_value=0.0)
    mask = _validate_mask(frontier_mask, risk.shape)
    if confidence_map is None:
        confidence = np.ones(risk.shape, dtype=np.float32)
    else:
        confidence = _as_unit_map(confidence_map, nan_value=0.0)
        if confidence.shape != risk.shape:
            raise ValueError(
                f"confidence map shape {confidence.shape} does not match "
                f"risk map {risk.shape}"
            )
    if hard_unsafe_map is None:
        hard_unsafe = np.zeros(risk.shape, dtype=bool)
    else:
        hard_unsafe = _validate_mask(hard_unsafe_map, risk.shape)

    point = (
        (int(frontier_point[0]), int(frontier_point[1]))
        if frontier_point is not None
        else _centroid(mask)
    )
    frontier_values = risk[mask]
    confidence_values = confidence[mask]

    # A point-only frontier is useful for lightweight/global planners.  Sample
    # its cell when a mask was not available instead of producing a misleading
    # fully-confident zero-risk report.
    if frontier_values.size == 0:
        row, col = point
        if 0 <= row < risk.shape[0] and 0 <= col < risk.shape[1]:
            frontier_values = risk[row : row + 1, col : col + 1]
            confidence_values = confidence[row : row + 1, col : col + 1]

    if frontier_values.size:
        mean_risk = float(frontier_values.mean())
        p95_risk = float(np.percentile(frontier_values, 95))
        max_risk = float(frontier_values.max())
        mean_confidence = float(confidence_values.mean())
    else:
        mean_risk = p95_risk = max_risk = 0.0
        mean_confidence = 0.0

    route_values = _valid_route_values(risk, route_cells)
    route_hard_values = _valid_route_values(
        hard_unsafe.astype(np.float32), route_cells
    )
    route_risk = float(route_values.mean()) if route_values.size else None
    route_max = float(route_values.max()) if route_values.size else None
    classification_risk = max(p95_risk, route_risk or 0.0)
    hard_blocked = bool(
        max_risk >= thresholds.hard_max
        or (
            not route_is_proxy
            and route_max is not None
            and route_max >= thresholds.hard_max
        )
        or bool(hard_unsafe[mask].any())
        or bool(
            not route_is_proxy
            and route_hard_values.size
            and route_hard_values.max() > 0.5
        )
    )
    if hard_blocked:
        classification_risk = max(classification_risk, thresholds.moderate_max + 1e-6)

    return FrontierRiskReport(
        frontier_id=int(frontier_id),
        point=point,
        cell_count=max(int(mask.sum()), int(frontier_values.size)),
        mean_risk=mean_risk,
        p95_risk=p95_risk,
        max_risk=max_risk,
        route_risk=route_risk,
        route_max_risk=route_max,
        severity=_severity(classification_risk, thresholds),
        confidence=float(np.clip(mean_confidence, 0.0, 1.0)),
        hard_blocked=hard_blocked,
        route_is_proxy=bool(route_is_proxy),
    )


def _normalise_frontiers(
    frontiers: FrontierMasks,
    risk_shape: Tuple[int, int],
    *,
    label_offset: int,
) -> List[Tuple[int, np.ndarray]]:
    if isinstance(frontiers, Mapping):
        return [
            (int(frontier_id), _validate_mask(mask, risk_shape))
            for frontier_id, mask in sorted(
                frontiers.items(), key=lambda item: int(item[0])
            )
        ]

    if isinstance(frontiers, np.ndarray):
        array = np.asarray(frontiers)
        if array.ndim == 3:
            if array.shape[1:] != risk_shape:
                raise ValueError(
                    f"frontier mask stack shape {array.shape} does not match "
                    f"risk map {risk_shape}"
                )
            return [
                (index, _validate_mask(mask, risk_shape))
                for index, mask in enumerate(array)
            ]
        if array.shape != risk_shape:
            raise ValueError(
                f"frontier map shape {array.shape} does not match risk map {risk_shape}"
            )
        if array.dtype == np.bool_:
            return [(0, array.astype(bool))]
        labels = [int(value) for value in np.unique(array) if int(value) > 0]
        return [
            (int(label - label_offset), array == label)
            for label in labels
        ]

    return [
        (index, _validate_mask(mask, risk_shape))
        for index, mask in enumerate(frontiers)
    ]


def _optional_item(
    collection,
    frontier_id: int,
    ordinal: int,
):
    if collection is None:
        return None
    if isinstance(collection, Mapping):
        return collection.get(frontier_id)
    if ordinal < len(collection):
        return collection[ordinal]
    return None


def _is_grid_point(value) -> bool:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return False
    return np.isscalar(value[0]) and np.isscalar(value[1])


def _is_route(value) -> bool:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) == 0:
        return False
    return _is_grid_point(value[0])


def build_frontier_risk_reports(
    frontiers: FrontierMasks,
    risk_map: np.ndarray,
    confidence_map: Optional[np.ndarray] = None,
    *,
    hard_unsafe_map: Optional[np.ndarray] = None,
    frontier_points: Optional[
        Union[Sequence[Sequence[int]], Mapping[int, Sequence[int]]]
    ] = None,
    route_cells: Optional[
        Union[
            Sequence[Iterable[Sequence[int]]],
            Mapping[int, Iterable[Sequence[int]]],
        ]
    ] = None,
    thresholds: SeverityThresholds = SeverityThresholds(),
    label_offset: int = 1,
    route_is_proxy: Union[bool, Sequence[bool], Mapping[int, bool]] = False,
) -> List[FrontierRiskReport]:
    """Build zero-based reports from masks or a positive-integer label map.

    ``route_is_proxy`` may be one shared flag or one flag per frontier.  The
    latter lets a planner distinguish exact traversable routes from isolated
    frontiers that had to retain a straight-line diagnostic proxy.
    """

    risk = _as_unit_map(risk_map, nan_value=0.0)
    masks = _normalise_frontiers(frontiers, risk.shape, label_offset=label_offset)
    reports = []
    for ordinal, (frontier_id, mask) in enumerate(masks):
        if (
            len(masks) == 1
            and not isinstance(frontier_points, Mapping)
            and _is_grid_point(frontier_points)
        ):
            point = frontier_points
        else:
            point = _optional_item(frontier_points, frontier_id, ordinal)
        if (
            len(masks) == 1
            and not isinstance(route_cells, Mapping)
            and _is_route(route_cells)
        ):
            route = route_cells
        else:
            route = _optional_item(route_cells, frontier_id, ordinal)
        if isinstance(route_is_proxy, (bool, np.bool_)):
            proxy = bool(route_is_proxy)
        else:
            proxy = bool(
                _optional_item(route_is_proxy, frontier_id, ordinal)
            )
        reports.append(
            frontier_risk_report(
                frontier_id,
                mask,
                risk,
                confidence_map,
                hard_unsafe_map=hard_unsafe_map,
                frontier_point=point,
                route_cells=route,
                route_is_proxy=proxy,
                thresholds=thresholds,
            )
        )
    return reports


def risk_context_payload(
    reports: Sequence[FrontierRiskReport],
) -> Dict[str, List[Dict[str, object]]]:
    """Return the stable JSON shape consumed by ``chat_utils.message_prepare``."""
    return {"hazard_report": [report.to_dict() for report in reports]}


def _normalise_agent_positions(
    positions: Union[Sequence[Sequence[float]], Mapping[int, Sequence[float]]]
) -> List[Tuple[int, Tuple[float, float]]]:
    if isinstance(positions, Mapping):
        items = sorted(positions.items(), key=lambda item: int(item[0]))
    else:
        items = list(enumerate(positions))
    return [
        (int(robot_id), (float(point[0]), float(point[1])))
        for robot_id, point in items
    ]


def _information_value(
    information_gain: Optional[Union[Sequence[float], Mapping[int, float]]],
    report: FrontierRiskReport,
    ordinal: int,
) -> float:
    if information_gain is None:
        return float(report.cell_count)
    if isinstance(information_gain, Mapping):
        return float(information_gain.get(report.frontier_id, 0.0))
    return float(information_gain[ordinal]) if ordinal < len(information_gain) else 0.0


def score_frontiers(
    agent_positions: Union[
        Sequence[Sequence[float]], Mapping[int, Sequence[float]]
    ],
    reports: Sequence[FrontierRiskReport],
    information_gain: Optional[Union[Sequence[float], Mapping[int, float]]] = None,
    *,
    weights: UtilityWeights = UtilityWeights(),
    hard_risk_threshold: Optional[float] = None,
) -> Dict[int, List[FrontierScore]]:
    """Score every robot/frontier pair with deterministic normalisation.

    Distance is normalised independently for each robot; information gain is
    normalised across frontiers.  Risk and uncertainty are already in [0, 1].
    Hard filtering happens before assignment and is independent of any VLM.
    """

    ordered_reports = sorted(reports, key=lambda report: report.frontier_id)
    agents = _normalise_agent_positions(agent_positions)
    if not ordered_reports:
        return {robot_id: [] for robot_id, _ in agents}

    info = np.asarray(
        [
            max(0.0, _information_value(information_gain, report, ordinal))
            for ordinal, report in enumerate(ordered_reports)
        ],
        dtype=np.float64,
    )
    info_scale = float(info.max()) if info.size and info.max() > 0 else 1.0
    info_norm = info / info_scale

    threshold = None
    if hard_risk_threshold is not None:
        threshold = float(np.clip(hard_risk_threshold, 0.0, 1.0))

    all_scores: Dict[int, List[FrontierScore]] = {}
    for robot_id, position in agents:
        distances = np.asarray(
            [math.dist(position, report.point) for report in ordered_reports],
            dtype=np.float64,
        )
        distance_scale = (
            float(distances.max()) if distances.size and distances.max() > 0 else 1.0
        )
        distance_norm = distances / distance_scale
        robot_scores: List[FrontierScore] = []
        for ordinal, report in enumerate(ordered_reports):
            threshold_blocked = bool(
                threshold is not None
                and (
                    report.max_risk >= threshold
                    or (
                        not report.route_is_proxy
                        and report.route_max_risk is not None
                        and report.route_max_risk >= threshold
                    )
                )
            )
            feasible = not report.hard_blocked and not threshold_blocked
            utility = (
                weights.information_gain * float(info_norm[ordinal])
                - weights.distance * float(distance_norm[ordinal])
                - weights.risk * report.planning_risk
                - weights.uncertainty * report.uncertainty
            )
            if not feasible:
                utility = float("-inf")
            robot_scores.append(
                FrontierScore(
                    robot_id=robot_id,
                    frontier_id=report.frontier_id,
                    distance=float(distances[ordinal]),
                    normalized_distance=float(distance_norm[ordinal]),
                    information_gain=float(info[ordinal]),
                    normalized_information_gain=float(info_norm[ordinal]),
                    risk=report.planning_risk,
                    uncertainty=report.uncertainty,
                    feasible=feasible,
                    utility=float(utility),
                )
            )
        all_scores[robot_id] = robot_scores
    return all_scores


def _redundancy_cost(
    frontier_ids: Sequence[int],
    report_by_id: Mapping[int, FrontierRiskReport],
    radius_cells: float,
) -> float:
    cost = 0.0
    for left in range(len(frontier_ids)):
        for right in range(left + 1, len(frontier_ids)):
            first, second = frontier_ids[left], frontier_ids[right]
            if first == second:
                cost += 1.0
                continue
            if radius_cells > 0.0:
                distance = math.dist(
                    report_by_id[first].point,
                    report_by_id[second].point,
                )
                cost += max(0.0, 1.0 - distance / radius_cells)
    return cost


def _greedy_assignment(
    scores: Mapping[int, Sequence[FrontierScore]],
    report_by_id: Mapping[int, FrontierRiskReport],
    weights: UtilityWeights,
    redundancy_radius_cells: float,
    allow_shared: bool,
) -> Dict[int, Optional[int]]:
    result: Dict[int, Optional[int]] = {}
    chosen: List[int] = []
    for robot_id in sorted(scores):
        best = None
        best_value = float("-inf")
        for score in scores[robot_id]:
            if not score.feasible or (not allow_shared and score.frontier_id in chosen):
                continue
            penalty = _redundancy_cost(
                chosen + [score.frontier_id], report_by_id, redundancy_radius_cells
            ) - _redundancy_cost(chosen, report_by_id, redundancy_radius_cells)
            value = score.utility - weights.redundancy * penalty
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12
                and (best is None or score.frontier_id < best)
            ):
                best = score.frontier_id
                best_value = value
        result[robot_id] = best
        if best is not None:
            chosen.append(best)
    return result


def assign_frontiers(
    agent_positions: Union[
        Sequence[Sequence[float]], Mapping[int, Sequence[float]]
    ],
    reports: Sequence[FrontierRiskReport],
    information_gain: Optional[Union[Sequence[float], Mapping[int, float]]] = None,
    *,
    weights: UtilityWeights = UtilityWeights(),
    hard_risk_threshold: Optional[float] = None,
    allow_shared: bool = True,
    redundancy_radius_cells: float = 0.0,
    max_exact_combinations: int = 200_000,
) -> Dict[int, Optional[int]]:
    """Assign frontiers by maximising team utility with hard safety filtering.

    Exhaustive search is deterministic and practical for the project's usual
    two robots and at most six frontiers.  Larger products use a deterministic
    greedy fallback.  If a robot has no safe frontier, its value is ``None``;
    the helper never silently converts an unsafe VLM choice into permission to
    traverse it.
    """

    scores = score_frontiers(
        agent_positions,
        reports,
        information_gain,
        weights=weights,
        hard_risk_threshold=hard_risk_threshold,
    )
    report_by_id = {report.frontier_id: report for report in reports}
    robot_ids = sorted(scores)
    candidates: Dict[int, List[FrontierScore]] = {
        robot_id: [score for score in scores[robot_id] if score.feasible]
        for robot_id in robot_ids
    }
    result: Dict[int, Optional[int]] = {
        robot_id: None for robot_id in robot_ids if not candidates[robot_id]
    }
    active_ids = [robot_id for robot_id in robot_ids if candidates[robot_id]]
    if not active_ids:
        return result

    combination_count = math.prod(len(candidates[robot_id]) for robot_id in active_ids)
    if combination_count > max(1, int(max_exact_combinations)):
        result.update(
            _greedy_assignment(
                {robot_id: candidates[robot_id] for robot_id in active_ids},
                report_by_id,
                weights,
                float(redundancy_radius_cells),
                allow_shared,
            )
        )
        return {robot_id: result.get(robot_id) for robot_id in robot_ids}

    best_assignment: Optional[Tuple[int, ...]] = None
    best_value = float("-inf")
    score_lists = [candidates[robot_id] for robot_id in active_ids]
    for choice in product(*score_lists):
        frontier_ids = tuple(score.frontier_id for score in choice)
        if not allow_shared and len(set(frontier_ids)) != len(frontier_ids):
            continue
        value = sum(score.utility for score in choice)
        value -= weights.redundancy * _redundancy_cost(
            frontier_ids, report_by_id, float(redundancy_radius_cells)
        )
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12
            and (best_assignment is None or frontier_ids < best_assignment)
        ):
            best_value = value
            best_assignment = frontier_ids

    if best_assignment is not None:
        result.update(dict(zip(active_ids, best_assignment)))
    return {robot_id: result.get(robot_id) for robot_id in robot_ids}


def _parse_assignment_id(value, prefix: str) -> Optional[int]:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw.startswith(prefix + "_"):
            raw = raw[len(prefix) + 1 :]
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def guard_frontier_assignments(
    assignments: Optional[Mapping[object, object]],
    reports: Sequence[FrontierRiskReport],
    *,
    fallback_assignments: Optional[Mapping[int, Optional[int]]] = None,
    expected_robot_ids: Optional[Iterable[int]] = None,
    hard_risk_threshold: Optional[float] = None,
) -> AssignmentGuardResult:
    """Reject unsafe/invalid VLM choices and optionally apply safe fallbacks.

    Keys may be ``0`` or ``"robot_0"`` and values may be ``1`` or
    ``"frontier_1"``.  A fallback is accepted only when it is itself present
    and not hard-blocked, so the VLM can never be the sole safety guard.
    """

    report_by_id = {report.frontier_id: report for report in reports}
    threshold = (
        None
        if hard_risk_threshold is None
        else float(np.clip(hard_risk_threshold, 0.0, 1.0))
    )

    def is_blocked(report: Optional[FrontierRiskReport]) -> bool:
        if report is None:
            return True
        return bool(
            report.hard_blocked
            or (
                threshold is not None
                and (
                    report.max_risk >= threshold
                    or (
                        not report.route_is_proxy
                        and report.route_max_risk is not None
                        and report.route_max_risk >= threshold
                    )
                )
            )
        )

    if not isinstance(assignments, Mapping):
        assignments = {}

    parsed: Dict[int, Optional[int]] = {}
    for raw_robot, raw_frontier in assignments.items():
        robot_id = _parse_assignment_id(raw_robot, "robot")
        if robot_id is None:
            continue
        parsed[robot_id] = _parse_assignment_id(raw_frontier, "frontier")

    if expected_robot_ids is None:
        robot_ids = sorted(parsed)
    else:
        robot_ids = sorted(int(robot_id) for robot_id in expected_robot_ids)

    guarded: Dict[int, Optional[int]] = {}
    rejected: Dict[int, str] = {}
    for robot_id in robot_ids:
        frontier_id = parsed.get(robot_id)
        report = report_by_id.get(frontier_id) if frontier_id is not None else None
        if frontier_id is None:
            reason = "missing_or_invalid"
        elif report is None:
            reason = "unknown_frontier"
        elif is_blocked(report):
            reason = "hard_blocked"
        else:
            guarded[robot_id] = frontier_id
            continue

        rejected[robot_id] = reason
        fallback = (
            fallback_assignments.get(robot_id)
            if fallback_assignments is not None
            else None
        )
        fallback_report = report_by_id.get(fallback) if fallback is not None else None
        guarded[robot_id] = (
            int(fallback)
            if fallback_report is not None and not is_blocked(fallback_report)
            else None
        )

    return AssignmentGuardResult(assignments=guarded, rejected=rejected)


__all__ = [
    "AssignmentGuardResult",
    "FrontierRiskReport",
    "FrontierScore",
    "SeverityThresholds",
    "UtilityWeights",
    "assign_frontiers",
    "build_frontier_risk_reports",
    "frontier_risk_report",
    "guard_frontier_assignments",
    "risk_context_payload",
    "score_frontiers",
]
