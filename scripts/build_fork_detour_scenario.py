#!/usr/bin/env python3
"""Build a two-start ObjectNav package around a proven fork/fire plan.

The primary start and its blind/aware paths come from a route-tuning candidate
report.  A second start is selected on the same floor whose shortest route to
the goal is longer and has little cell overlap with the primary safe route.
The resulting per-agent starts are stored in ``episode.info`` and are applied
by :mod:`utils.multi_agent_start` at runtime.
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.tune_fire_route_scenarios import (  # noqa: E402
    _read_dataset,
    _scene_grid,
)
from utils.fire_world.fine_tuning import (  # noqa: E402
    CURATED_FIRE_PROFILES,
    radial_hazard_map,
    route_overlay,
)
from utils.multi_agent_start import (  # noqa: E402
    GOAL_POSITIONS_KEY,
    START_STATES_KEY,
    TARGET_AGENT_IDS_KEY,
)


GridCell = Tuple[int, int]
_MOVES = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--plan-path", required=True)
    parser.add_argument(
        "--source-shard",
        default=None,
        help="defaults to objectnav_hm3d_v2/val/content/<scene>.json.gz",
    )
    parser.add_argument("--dataset-output-dir", required=True)
    parser.add_argument("--report-output-dir", required=True)
    parser.add_argument(
        "--scene-dataset-config",
        default=(
            "data/scene_datasets/hm3d_v0.2/"
            "hm3d_annotated_basis.scene_dataset_config.json"
        ),
    )
    parser.add_argument("--resolution-m", type=float, default=0.10)
    parser.add_argument("--min-start-separation-m", type=float, default=5.0)
    parser.add_argument("--min-secondary-length-ratio", type=float, default=1.0)
    parser.add_argument("--max-route-overlap", type=float, default=0.20)
    parser.add_argument("--min-ignition-clearance-m", type=float, default=2.0)
    parser.add_argument(
        "--secondary-avoid-target-visibility",
        action="store_true",
        help="require the secondary start to be occluded from the target center",
    )
    parser.add_argument(
        "--primary-target-agent-only",
        action="store_true",
        help="only agent 0 may accept target detections in this controlled run",
    )
    parser.add_argument(
        "--secondary-min-occlusion-margin-m",
        type=float,
        default=2.0,
        help="minimum target-ray distance hidden behind scene geometry",
    )
    return parser


def _distance_tree(
    traversible: np.ndarray,
    goals: Iterable[Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return obstacle-only distance and a next-cell tree toward the goals."""

    domain = np.asarray(traversible, dtype=bool)
    distance = np.full(domain.shape, np.inf, dtype=np.float64)
    parent = np.full(domain.shape + (2,), -1, dtype=np.int32)
    queue: List[Tuple[float, int, GridCell]] = []
    ordinal = 0
    for raw_goal in goals:
        goal = (int(raw_goal[0]), int(raw_goal[1]))
        if not domain[goal] or distance[goal] == 0.0:
            continue
        distance[goal] = 0.0
        heapq.heappush(queue, (0.0, ordinal, goal))
        ordinal += 1

    while queue:
        cost, _, current = heapq.heappop(queue)
        if cost > distance[current] + 1e-9:
            continue
        for drow, dcol, geometric in _MOVES:
            nxt = (current[0] + drow, current[1] + dcol)
            if not (
                0 <= nxt[0] < domain.shape[0]
                and 0 <= nxt[1] < domain.shape[1]
                and domain[nxt]
            ):
                continue
            if drow != 0 and dcol != 0:
                if (
                    not domain[current[0] + drow, current[1]]
                    or not domain[current[0], current[1] + dcol]
                ):
                    continue
            candidate = cost + geometric
            if candidate + 1e-9 >= distance[nxt]:
                continue
            distance[nxt] = candidate
            parent[nxt] = current
            heapq.heappush(queue, (candidate, ordinal, nxt))
            ordinal += 1
    return distance, parent


def _tree_path(
    start: GridCell,
    distance: np.ndarray,
    parent: np.ndarray,
) -> Tuple[GridCell, ...]:
    if not np.isfinite(distance[start]):
        return ()
    path = [start]
    current = start
    limit = int(np.prod(distance.shape)) + 1
    while distance[current] > 1e-9 and len(path) < limit:
        raw = parent[current]
        nxt = (int(raw[0]), int(raw[1]))
        if nxt[0] < 0:
            return ()
        path.append(nxt)
        current = nxt
    return tuple(path)


def select_secondary_route(
    traversible: np.ndarray,
    *,
    primary_start: Sequence[int],
    primary_aware_cells: Iterable[Sequence[int]],
    goals: Iterable[Sequence[int]],
    ignition_cells: Iterable[Sequence[int]],
    resolution_m: float,
    min_start_separation_m: float,
    min_secondary_length_ratio: float,
    max_route_overlap: float,
    min_ignition_clearance_m: float,
    candidate_validator: Optional[Callable[[GridCell], bool]] = None,
) -> Dict[str, object]:
    """Pick a distant, low-overlap second start on the same navmesh island."""

    resolution = float(resolution_m)
    primary = (int(primary_start[0]), int(primary_start[1]))
    aware = tuple((int(cell[0]), int(cell[1])) for cell in primary_aware_cells)
    goal_cells = tuple((int(cell[0]), int(cell[1])) for cell in goals)
    ignitions = tuple((int(cell[0]), int(cell[1])) for cell in ignition_cells)
    if not aware or not goal_cells:
        raise ValueError("primary aware path and goals must be non-empty")
    distance, parent = _distance_tree(traversible, goal_cells)
    primary_length_cells = sum(
        float(np.linalg.norm(np.subtract(nxt, current)))
        for current, nxt in zip(aware, aware[1:])
    )
    aware_set = set(aware)

    accepted = []
    for raw in np.argwhere(np.isfinite(distance)):
        cell = (int(raw[0]), int(raw[1]))
        start_separation_m = float(
            np.linalg.norm(np.subtract(cell, primary)) * resolution
        )
        if start_separation_m < float(min_start_separation_m):
            continue
        if ignitions:
            ignition_clearance_m = min(
                float(np.linalg.norm(np.subtract(cell, source)) * resolution)
                for source in ignitions
            )
            if ignition_clearance_m < float(min_ignition_clearance_m):
                continue
        else:
            ignition_clearance_m = math.inf
        secondary_length_cells = float(distance[cell])
        length_ratio = secondary_length_cells / max(primary_length_cells, 1e-9)
        if length_ratio < float(min_secondary_length_ratio):
            continue
        path = _tree_path(cell, distance, parent)
        if not path:
            continue
        overlap_cells = len(set(path) & aware_set)
        overlap_fraction = overlap_cells / max(1, len(set(path)))
        if overlap_fraction > float(max_route_overlap):
            continue
        score = (
            1.5 * (1.0 - overlap_fraction)
            + min(length_ratio, 2.0)
            + 0.05 * start_separation_m
            + 0.02 * min(ignition_clearance_m, 10.0)
        )
        accepted.append((score, cell, path, {
            "start_separation_m": start_separation_m,
            "ignition_clearance_m": ignition_clearance_m,
            "length_m": secondary_length_cells * resolution,
            "length_ratio_to_primary_safe": length_ratio,
            "overlap_cells_with_primary_safe": overlap_cells,
            "route_overlap_fraction": overlap_fraction,
        }))
    if not accepted:
        raise RuntimeError(
            "no secondary start satisfies separation, route-length and "
            "overlap thresholds"
        )
    ranked = sorted(accepted, key=lambda item: item[0], reverse=True)
    if candidate_validator is not None:
        ranked = [
            item for item in ranked
            if bool(candidate_validator(item[1]))
        ]
        if not ranked:
            raise RuntimeError(
                "no secondary start passes the runtime visibility validator"
            )
    _, cell, path, metrics = ranked[0]
    return {
        "cell": list(cell),
        "route_cells": [list(point) for point in path],
        **metrics,
    }


def _line_of_sight(
    simulator,
    source: Sequence[float],
    target: Sequence[float],
) -> bool:
    """Return whether scene geometry leaves camera-to-target torso visible."""
    return _target_occlusion_margin_m(simulator, source, target) <= 0.08


def _target_occlusion_margin_m(
    simulator,
    source: Sequence[float],
    target: Sequence[float],
) -> float:
    """Return how far before the target the first blocking surface lies."""

    import habitat_sim

    origin = np.asarray(source, dtype=np.float32).copy()
    destination = np.asarray(target, dtype=np.float32).copy()
    origin[1] += 0.88
    destination[1] += 1.0
    delta = destination - origin
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return 0.0
    result = simulator.cast_ray(habitat_sim.geo.Ray(origin, delta / distance))
    if not result.has_hits():
        return 0.0
    first_hit = min(float(hit.ray_distance) for hit in result.hits)
    return max(0.0, distance - first_hit)


def _yaw_offset(rotation: Sequence[float], radians: float) -> List[float]:
    x, y, z, w = (float(value) for value in rotation)
    sy = math.sin(0.5 * float(radians))
    cy = math.cos(0.5 * float(radians))
    result = [cy * x + sy * z, cy * y + sy * w, cy * z - sy * x, cy * w - sy * y]
    norm = float(np.linalg.norm(result))
    return [value / norm for value in result]


def _write_dataset(
    source: Mapping[str, object],
    episode: Mapping[str, object],
    *,
    scene_id: str,
    plan_id: str,
    profile: str,
    output_dir: Path,
    secondary_position: Sequence[float],
    primary_goal_position: Sequence[float],
    primary_target_agent_only: bool = False,
) -> Path:
    selected = dict(episode)
    info = dict(selected.get("info") or {})
    primary_rotation = [float(value) for value in selected["start_rotation"]]
    info[START_STATES_KEY] = [
        {
            "position": [float(value) for value in selected["start_position"]],
            "rotation": primary_rotation,
        },
        {
            "position": [float(value) for value in secondary_position],
            "rotation": _yaw_offset(primary_rotation, math.pi),
        },
    ]
    info[GOAL_POSITIONS_KEY] = [
        [float(value) for value in primary_goal_position],
        None,
    ]
    if primary_target_agent_only:
        info[TARGET_AGENT_IDS_KEY] = [0]
    selected["info"] = info
    scenario = {
        "schema_version": 2,
        "scene_id": scene_id,
        "plan_id": plan_id,
        "profile": profile,
        "episode_id": str(selected["episode_id"]),
        "object_category": str(selected["object_category"]),
        "separate_agent_starts": True,
        "controlled_primary_goal_hint": True,
    }
    shard = dict(source)
    shard["episodes"] = [selected]
    shard["fire_route_scenario"] = scenario
    root = {
        key: value
        for key, value in source.items()
        if key not in {"episodes", "goals_by_category"}
    }
    root["episodes"] = []
    root["content_scenes_path"] = "{data_path}/content/{scene}.json.gz"
    root["fire_route_scenario"] = scenario

    content_dir = output_dir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    root_path = output_dir / "val.json.gz"
    for path, payload in (
        (root_path, root),
        (content_dir / f"{scene_id}.json.gz", shard),
    ):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
    return root_path


def _write_overlay(
    path: Path,
    traversible: np.ndarray,
    candidate: Mapping[str, object],
    secondary: Mapping[str, object],
    ignition_cells: Iterable[Sequence[int]],
    resolution_m: float,
) -> None:
    profile = CURATED_FIRE_PROFILES[str(candidate["profile"])]
    ignition_cells = tuple(
        (int(cell[0]), int(cell[1])) for cell in ignition_cells
    )
    risk = np.zeros(traversible.shape, dtype=np.float32)
    for ignition_cell in ignition_cells:
        source_risk, _ = radial_hazard_map(
            traversible.shape,
            ignition_cell,
            resolution_m=float(resolution_m),
            core_radius_m=profile.synthetic_core_radius_m,
            risk_radius_m=profile.synthetic_risk_radius_m,
        )
        risk = np.maximum(risk, source_risk)
    contrast = SimpleNamespace(
        blind=SimpleNamespace(cells=tuple(
            (int(cell[0]), int(cell[1]))
            for cell in candidate["contrast"]["blind"]["cells"]
        )),
        aware=SimpleNamespace(cells=tuple(
            (int(cell[0]), int(cell[1]))
            for cell in candidate["contrast"]["aware"]["cells"]
        )),
    )
    image = route_overlay(
        traversible,
        risk,
        contrast,
        start=candidate["start_cell"],
        goal=candidate["chosen_goal_cell"],
        ignitions=ignition_cells,
    )
    for row, col in secondary["route_cells"]:
        image[int(row), int(col)] = (190, 70, 220)
    row, col = secondary["cell"]
    cv2.circle(image, (int(col), int(row)), 2, (0, 255, 255), -1)
    scale = max(1, int(round(0.40 / float(resolution_m))))
    image = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(path), image)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    candidate_path = (ROOT / args.candidate_report).resolve()
    plan_path = (ROOT / args.plan_path).resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    scene_id = str(candidate["scene_id"])
    if str(plan["scene_id"]) != scene_id:
        raise ValueError("candidate and plan scene ids differ")
    source_path = (
        (ROOT / args.source_shard).resolve()
        if args.source_shard
        else ROOT / "data/datasets/objectnav_hm3d_v2/val/content" / f"{scene_id}.json.gz"
    )
    source = _read_dataset(source_path)
    matches = [
        episode for episode in source.get("episodes", [])
        if str(episode.get("episode_id")) == str(candidate["episode_id"])
        and str(episode.get("object_category")) == str(candidate["object_category"])
        and np.allclose(
            episode.get("start_position", ()), candidate["start_position"],
            rtol=0.0, atol=1e-5,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one source episode, found {len(matches)}")
    episode = matches[0]
    inventory = json.loads(
        (ROOT / "scenes" / scene_id / "inventory.json").read_text(encoding="utf-8")
    )
    grid = _scene_grid(
        inventory,
        (ROOT / args.scene_dataset_config).resolve(),
        float(args.resolution_m),
        float(episode["start_position"][1]),
        enable_physics=bool(args.secondary_avoid_target_visibility),
    )
    try:
        ignition_cells = [
            grid.frame.world_to_grid(np.asarray(item["position"], dtype=np.float64))
            for item in plan["ignitions"]
        ]
        target_position = None
        secondary_validator = None
        if args.secondary_avoid_target_visibility:
            key = f"{scene_id}.basis.glb_{candidate['object_category']}"
            goals = list((source.get("goals_by_category") or {}).get(key, []))
            if len(goals) != 1 or goals[0].get("position") is None:
                raise RuntimeError(
                    "visibility-constrained secondary start requires one target center"
                )
            target_position = [float(value) for value in goals[0]["position"]]

            def secondary_validator(cell):
                world = grid.frame.grid_to_world(
                    np.asarray(cell),
                    map_y_m=float(episode["start_position"][1]),
                )
                return _target_occlusion_margin_m(
                    grid.simulator, world, target_position
                ) >= float(args.secondary_min_occlusion_margin_m)

        secondary = select_secondary_route(
            grid.traversible,
            primary_start=candidate["start_cell"],
            primary_aware_cells=candidate["contrast"]["aware"]["cells"],
            # ObjectNav Success uses the closest viewpoint of every instance
            # in the category.  Selecting against only the route-tuning goal
            # could put agent 1 beside another bed and end the episode before
            # agent 0 demonstrates the fork detour.
            goals=candidate["goal_cells"],
            ignition_cells=ignition_cells,
            resolution_m=float(args.resolution_m),
            min_start_separation_m=float(args.min_start_separation_m),
            min_secondary_length_ratio=float(args.min_secondary_length_ratio),
            max_route_overlap=float(args.max_route_overlap),
            min_ignition_clearance_m=float(args.min_ignition_clearance_m),
            candidate_validator=secondary_validator,
        )
        world = grid.frame.grid_to_world(
            np.asarray(secondary["cell"]),
            map_y_m=float(episode["start_position"][1]),
        )
        snapped = np.asarray(grid.simulator.pathfinder.snap_point(world), dtype=np.float64)
        if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
            raise RuntimeError("selected secondary start could not be snapped")
        secondary["position"] = [float(value) for value in snapped]
        primary_goal_world = grid.frame.grid_to_world(
            np.asarray(candidate["chosen_goal_cell"]),
            map_y_m=float(episode["start_position"][1]),
        )
        primary_goal_snapped = np.asarray(
            grid.simulator.pathfinder.snap_point(primary_goal_world),
            dtype=np.float64,
        )
        if primary_goal_snapped.shape != (3,) or not np.all(
            np.isfinite(primary_goal_snapped)
        ):
            raise RuntimeError("primary controlled goal could not be snapped")

        dataset_dir = (ROOT / args.dataset_output_dir).resolve()
        report_dir = (ROOT / args.report_output_dir).resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = _write_dataset(
            source,
            episode,
            scene_id=scene_id,
            plan_id=str(plan["plan_id"]),
            profile=str(candidate["profile"]),
            output_dir=dataset_dir,
            secondary_position=snapped,
            primary_goal_position=primary_goal_snapped,
            primary_target_agent_only=bool(args.primary_target_agent_only),
        )
        report = {
            "schema_version": 1,
            "scene_id": scene_id,
            "plan_id": str(plan["plan_id"]),
            "dataset_path": str(dataset_path),
            "episode_id": str(candidate["episode_id"]),
            "object_category": str(candidate["object_category"]),
            "ignition_sources": [
                {
                    "object_id": int(item["object_id"]),
                    "category": str(item["category"]),
                    "position": [float(value) for value in item["position"]],
                    "grid_cell": [int(value) for value in cell],
                }
                for item, cell in zip(plan["ignitions"], ignition_cells)
            ],
            "primary": {
                "start_position": candidate["start_position"],
                "blind_length_m": candidate["contrast"]["blind"]["length_m"],
                "blind_max_risk": candidate["contrast"]["blind"]["max_risk"],
                "safe_length_m": candidate["contrast"]["aware"]["length_m"],
                "safe_max_risk": candidate["contrast"]["aware"]["max_risk"],
                "detour_ratio": candidate["contrast"]["detour_ratio"],
                "path_divergence": candidate["contrast"]["path_divergence"],
                "objectnav_goal_cell_count": len(candidate["goal_cells"]),
                "controlled_goal_position": [
                    float(value) for value in primary_goal_snapped
                ],
            },
            "secondary": secondary,
            "selection_thresholds": {
                "min_start_separation_m": float(args.min_start_separation_m),
                "min_secondary_length_ratio": float(args.min_secondary_length_ratio),
                "max_route_overlap": float(args.max_route_overlap),
                "min_ignition_clearance_m": float(args.min_ignition_clearance_m),
                "secondary_avoid_target_visibility": bool(
                    args.secondary_avoid_target_visibility
                ),
                "secondary_min_occlusion_margin_m": float(
                    args.secondary_min_occlusion_margin_m
                ),
            },
            "secondary_target_visible_from_start": (
                None
                if target_position is None
                else bool(_line_of_sight(grid.simulator, snapped, target_position))
            ),
            "secondary_target_occlusion_margin_m": (
                None
                if target_position is None
                else float(_target_occlusion_margin_m(
                    grid.simulator, snapped, target_position
                ))
            ),
            "controlled_target_agent_ids": (
                [0] if args.primary_target_agent_only else None
            ),
            "overlay_legend": {
                "blue": "primary shortest unsafe route",
                "green": "primary longer safe route",
                "purple": "secondary obstacle-only reference route",
                "cyan_square": "secondary start",
                "yellow_square": "primary start",
                "yellow_source_pixels": "all ignition sources",
            },
        }
        report_path = report_dir / "scenario_geometry.json"
        overlay_path = report_dir / "scenario_geometry.png"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_overlay(
            overlay_path,
            grid.traversible,
            candidate,
            secondary,
            ignition_cells,
            float(args.resolution_m),
        )
        print(f"[fork-detour] dataset={dataset_path}")
        print(f"[fork-detour] report={report_path}")
        print(f"[fork-detour] overlay={overlay_path}")
        print(
            "[fork-detour] secondary "
            f"start_separation={secondary['start_separation_m']:.2f}m "
            f"route_overlap={secondary['route_overlap_fraction']:.3f}"
        )
    finally:
        grid.simulator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
