#!/usr/bin/env python3
"""Build and validate a three-source route-contrast plan for native person.

The input candidate supplies a proven short-versus-safe fork.  This script
combines three real, low semantic fuel objects, verifies that the person and
all native ObjectNav view points stay outside a configurable protection zone,
and writes a content-addressed FireWorld plan plus visual evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.tune_fire_route_scenarios import (  # noqa: E402
    _nearest_free,
    _scene_grid,
)
from utils.fire_world.fine_tuning import (  # noqa: E402
    CURATED_FIRE_PROFILES,
    build_curated_plan,
    evaluate_fireworld_snapshot,
    evaluate_route_contrast,
    radial_hazard_map,
    route_overlay,
    write_curated_plan,
)
from utils.fire_world.runtime import FireWorld  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--source-shard", required=True)
    parser.add_argument("--source-object-ids", nargs=3, type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--timeline-root", default="outputs/fire_world")
    parser.add_argument(
        "--scene-dataset-config",
        default=(
            "data/scene_datasets/hm3d_v0.2/"
            "hm3d_annotated_basis.scene_dataset_config.json"
        ),
    )
    parser.add_argument("--resolution-m", type=float, default=0.10)
    parser.add_argument("--risk-alpha", type=float, default=4.0)
    parser.add_argument("--source-radius-m", type=float, default=0.78)
    parser.add_argument("--min-person-clearance-m", type=float, default=3.0)
    parser.add_argument("--min-viewpoint-clearance-m", type=float, default=2.5)
    parser.add_argument("--max-source-height-m", type=float, default=1.0)
    parser.add_argument(
        "--validation-times-s",
        nargs="+",
        type=float,
        default=(40.0, 60.0, 100.0),
        help="timeline times covering the configured <=250-step runtime horizon",
    )
    parser.add_argument("--validate-timeline", action="store_true")
    return parser


def _read_shard(path: Path) -> Mapping[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _selected_episode(
    shard: Mapping[str, object], candidate: Mapping[str, object]
) -> Mapping[str, object]:
    matches = [
        episode for episode in shard.get("episodes", [])
        if str(episode.get("episode_id")) == str(candidate["episode_id"])
        and str(episode.get("object_category")) == "person"
        and np.allclose(
            episode.get("start_position", ()), candidate["start_position"],
            rtol=0.0, atol=1e-5,
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one native person episode, found {len(matches)}")
    return matches[0]


def _person_goal(
    shard: Mapping[str, object], scene_id: str
) -> tuple[np.ndarray, np.ndarray]:
    key = f"{scene_id}.basis.glb_person"
    goals = list((shard.get("goals_by_category") or {}).get(key, []))
    if len(goals) != 1:
        raise RuntimeError(f"expected one static person goal for {scene_id}")
    centre = np.asarray(goals[0]["position"], dtype=np.float64)
    viewpoints = np.asarray([
        view["agent_state"]["position"]
        for view in goals[0].get("view_points", [])
    ], dtype=np.float64)
    if centre.shape != (3,) or viewpoints.ndim != 2 or viewpoints.shape[1] != 3:
        raise ValueError("person goal must provide a center and 3-D VIEW_POINTS")
    return centre, viewpoints


def source_protection_metrics(
    sources: Iterable[Mapping[str, object]],
    *,
    person_centre: Sequence[float],
    person_viewpoints: np.ndarray,
    floor_y_m: float,
) -> list[dict]:
    """Measure each semantic source against person and floor protections."""

    centre = np.asarray(person_centre, dtype=np.float64)
    views = np.asarray(person_viewpoints, dtype=np.float64)
    records = []
    for source in sources:
        position = np.asarray(source["centroid"], dtype=np.float64)
        records.append({
            "object_id": int(source["instance_id"]),
            "category": str(source.get("category") or "curated fuel"),
            "position": [float(value) for value in position],
            "flammability": float(source.get("flammability", 0.0)),
            "height_above_floor_m": float(position[1] - float(floor_y_m)),
            "person_centre_clearance_m": float(np.linalg.norm(
                position[[0, 2]] - centre[[0, 2]]
            )),
            "nearest_viewpoint_clearance_m": float(np.min(np.linalg.norm(
                views[:, [0, 2]] - position[None, [0, 2]], axis=1
            ))),
        })
    return records


def combined_surrogate(
    shape: tuple[int, int],
    ignition_cells: Iterable[Sequence[int]],
    *,
    resolution_m: float,
    core_radius_m: float,
    risk_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    risk = np.zeros(shape, dtype=np.float32)
    hard = np.zeros(shape, dtype=bool)
    for cell in ignition_cells:
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


def _mark_person(
    image: np.ndarray,
    raw_cell: Sequence[int],
    *,
    radius_cells: int,
) -> None:
    row, col = (int(value) for value in raw_cell[:2])
    cv2.circle(image, (col, row), max(1, int(radius_cells)), (255, 0, 255), 1)
    cv2.circle(image, (col, row), 2, (255, 255, 255), -1)


def _write_overlay(
    path: Path,
    traversible: np.ndarray,
    risk: np.ndarray,
    contrast,
    *,
    start: Sequence[int],
    goal: Sequence[int],
    ignition_cells: Iterable[Sequence[int]],
    person_cell: Sequence[int],
    protection_radius_cells: int,
    resolution_m: float,
) -> None:
    image = route_overlay(
        traversible,
        risk,
        contrast,
        start=start,
        goal=goal,
        ignitions=ignition_cells,
    )
    _mark_person(image, person_cell, radius_cells=protection_radius_cells)
    scale = max(1, int(round(0.40 / float(resolution_m))))
    image = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write {path}")


def _validate_timeline(
    *,
    plan: Mapping[str, object],
    candidate: Mapping[str, object],
    grid,
    person_centre: np.ndarray,
    person_viewpoints: np.ndarray,
    output_dir: Path,
    timeline_root: Path,
    risk_alpha: float,
    protection_radius_m: float,
    timestamps_s: Iterable[float],
) -> Mapping[str, object]:
    fire_world = FireWorld.load(
        str(plan["scene_id"]), str(plan["plan_id"]), out_root=timeline_root
    )
    profile = CURATED_FIRE_PROFILES[str(candidate["profile"])]
    person_cell = grid.frame.world_to_grid(person_centre)
    view_cells = np.asarray(
        grid.frame.world_to_grid(person_viewpoints), dtype=np.int64
    )
    valid_views = view_cells[
        (view_cells[:, 0] >= 0)
        & (view_cells[:, 0] < grid.traversible.shape[0])
        & (view_cells[:, 1] >= 0)
        & (view_cells[:, 1] < grid.traversible.shape[1])
    ]
    ignition_cells = [
        grid.frame.world_to_grid(item["position"])
        for item in plan["ignitions"]
    ]
    records = []
    for timestamp_s in timestamps_s:
        timestamp_s = float(timestamp_s)
        contrast, layers = evaluate_fireworld_snapshot(
            fire_world,
            timestamp_s=timestamp_s,
            frame=grid.frame,
            floor_y_m=grid.floor_y_m,
            traversible=grid.traversible,
            start=candidate["start_cell"],
            goals=[candidate["chosen_goal_cell"]],
            risk_alpha=float(risk_alpha),
            thresholds=profile.thresholds,
        )
        if contrast is None:
            raise RuntimeError(f"no traversable route at t={timestamp_s:.0f}s")
        if not contrast.accepted:
            raise RuntimeError(
                f"route contrast rejected inside runtime horizon at "
                f"t={timestamp_s:.0f}s: {list(contrast.reasons)}"
            )
        person_risk = float(layers.physical_risk[tuple(person_cell)])
        person_hard = bool(layers.hard_unsafe[tuple(person_cell)])
        view_risk = layers.physical_risk[valid_views[:, 0], valid_views[:, 1]]
        view_hard = layers.hard_unsafe[valid_views[:, 0], valid_views[:, 1]]
        if person_hard or bool(np.any(view_hard)):
            raise RuntimeError(f"person protection enters hard unsafe at {timestamp_s}s")
        if person_risk > 0.10 or float(np.max(view_risk, initial=0.0)) > 0.35:
            raise RuntimeError(f"person protection risk is too high at {timestamp_s}s")
        overlay_path = output_dir / f"actual_t{int(timestamp_s):04d}.png"
        _write_overlay(
            overlay_path,
            grid.traversible,
            layers.physical_risk,
            contrast,
            start=candidate["start_cell"],
            goal=candidate["chosen_goal_cell"],
            ignition_cells=ignition_cells,
            person_cell=person_cell,
            protection_radius_cells=int(round(protection_radius_m / grid.frame.resolution_m)),
            resolution_m=grid.frame.resolution_m,
        )
        records.append({
            "timestamp_s": timestamp_s,
            "contrast": contrast.to_dict(resolution_m=grid.frame.resolution_m),
            "person_centre_risk": person_risk,
            "person_centre_hard_unsafe": person_hard,
            "max_viewpoint_risk": float(np.max(view_risk, initial=0.0)),
            "hard_unsafe_viewpoint_count": int(np.count_nonzero(view_hard)),
            "overlay_path": str(overlay_path.relative_to(ROOT)),
        })
    return {
        "plan_id": str(plan["plan_id"]),
        "person_viewpoint_count": int(len(person_viewpoints)),
        "frames": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    candidate_path = (ROOT / args.candidate_report).resolve()
    shard_path = (ROOT / args.source_shard).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if str(candidate.get("object_category")) != "person":
        raise ValueError("candidate must use native object_category=person")
    scene_id = str(candidate["scene_id"])
    shard = _read_shard(shard_path)
    episode = _selected_episode(shard, candidate)
    person_centre, person_viewpoints = _person_goal(shard, scene_id)
    inventory = json.loads(
        (ROOT / args.scenes_root / scene_id / "inventory.json").read_text(
            encoding="utf-8"
        )
    )
    by_id = {int(item["instance_id"]): item for item in inventory["instances"]}
    if len(set(args.source_object_ids)) != 3:
        raise ValueError("source object ids must be distinct")
    sources = [by_id[object_id] for object_id in args.source_object_ids]
    grid = _scene_grid(
        inventory,
        (ROOT / args.scene_dataset_config).resolve(),
        float(args.resolution_m),
        float(episode["start_position"][1]),
    )
    try:
        protections = source_protection_metrics(
            sources,
            person_centre=person_centre,
            person_viewpoints=person_viewpoints,
            floor_y_m=grid.floor_y_m,
        )
        for source, metrics in zip(sources, protections):
            if bool(source.get("structural")):
                raise ValueError(f"source {source['instance_id']} is structural")
            if float(source.get("flammability", 0.0)) < 0.4:
                raise ValueError(f"source {source['instance_id']} is not flammable")
            if abs(metrics["height_above_floor_m"]) > float(args.max_source_height_m):
                raise ValueError(f"source {source['instance_id']} is not a low floor source")
            if metrics["person_centre_clearance_m"] < float(args.min_person_clearance_m):
                raise ValueError(f"source {source['instance_id']} is too close to person")
            if metrics["nearest_viewpoint_clearance_m"] < float(args.min_viewpoint_clearance_m):
                raise ValueError(f"source {source['instance_id']} is too close to a viewpoint")

        ignition_cells = []
        for source in sources:
            cell = _nearest_free(
                grid.traversible,
                grid.frame.world_to_grid(source["centroid"]),
                max_radius_cells=max(8, int(round(1.0 / grid.frame.resolution_m))),
            )
            if cell is None:
                raise RuntimeError(f"source {source['instance_id']} has no nearby nav cell")
            ignition_cells.append(cell)
        profile = replace(
            CURATED_FIRE_PROFILES[str(candidate["profile"])],
            num_initial_ignitions=3,
            source_radius_m=float(args.source_radius_m),
        )
        risk, hard = combined_surrogate(
            grid.traversible.shape,
            ignition_cells,
            resolution_m=grid.frame.resolution_m,
            core_radius_m=profile.synthetic_core_radius_m,
            risk_radius_m=profile.synthetic_risk_radius_m,
        )
        contrast = evaluate_route_contrast(
            grid.traversible,
            candidate["start_cell"],
            candidate["goal_cells"],
            risk,
            hard,
            risk_alpha=float(args.risk_alpha),
            thresholds=profile.thresholds,
        )
        if contrast is None or not contrast.accepted:
            reasons = None if contrast is None else list(contrast.reasons)
            raise RuntimeError(f"combined three-source contrast rejected: {reasons}")
        chosen_goal = list(contrast.blind.cells[-1])
        curation = {
            "scene_id": scene_id,
            "profile": str(candidate["profile"]),
            "episode_id": str(candidate["episode_id"]),
            "object_category": "person",
            "start_position": [float(value) for value in candidate["start_position"]],
            "start_cell": [int(value) for value in candidate["start_cell"]],
            "goal_cells": [[int(value) for value in cell] for cell in candidate["goal_cells"]],
            "chosen_goal_cell": chosen_goal,
            "contrast": contrast.to_dict(resolution_m=grid.frame.resolution_m),
            "ignitions": [
                {
                    "instance_id": int(source["instance_id"]),
                    "category": str(source["category"]),
                    "centroid": [float(value) for value in source["centroid"]],
                    "flammability": float(source["flammability"]),
                    "grid_cell": [int(value) for value in cell],
                }
                for source, cell in zip(sources, ignition_cells)
            ],
            "person_protection": {
                "centre": [float(value) for value in person_centre],
                "viewpoint_count": int(len(person_viewpoints)),
                "min_person_clearance_m": float(args.min_person_clearance_m),
                "min_viewpoint_clearance_m": float(args.min_viewpoint_clearance_m),
                "synthetic_risk_radius_m": float(profile.synthetic_risk_radius_m),
                "source_radius_m": float(profile.source_radius_m),
                "sources": protections,
            },
        }
        plan = build_curated_plan(
            inventory,
            sources[0],
            profile,
            seed=42,
            curation=curation,
            additional_ignition_instances=sources[1:],
        )
        plan_path = write_curated_plan(plan, (ROOT / args.scenes_root).resolve())
        output_dir.mkdir(parents=True, exist_ok=True)
        person_cell = grid.frame.world_to_grid(person_centre)
        surrogate_path = output_dir / "person_three_source_surrogate.png"
        _write_overlay(
            surrogate_path,
            grid.traversible,
            risk,
            contrast,
            start=candidate["start_cell"],
            goal=chosen_goal,
            ignition_cells=ignition_cells,
            person_cell=person_cell,
            protection_radius_cells=int(round(
                float(args.min_person_clearance_m) / grid.frame.resolution_m
            )),
            resolution_m=grid.frame.resolution_m,
        )
        report = {
            "schema_version": 1,
            "plan_id": str(plan["plan_id"]),
            "plan_path": str(plan_path.relative_to(ROOT)),
            "candidate_report": str(candidate_path.relative_to(ROOT)),
            "source_shard": str(shard_path.relative_to(ROOT)),
            "episode_id": str(candidate["episode_id"]),
            "object_category": "person",
            "person_centre": [float(value) for value in person_centre],
            "person_viewpoint_count": int(len(person_viewpoints)),
            "sources": protections,
            "ignition_cells": [[int(value) for value in cell] for cell in ignition_cells],
            "surrogate_contrast": contrast.to_dict(
                resolution_m=grid.frame.resolution_m
            ),
            "surrogate_overlay": str(surrogate_path.relative_to(ROOT)),
        }
        if args.validate_timeline:
            report["actual_timeline"] = _validate_timeline(
                plan=plan,
                candidate={**candidate, "chosen_goal_cell": chosen_goal},
                grid=grid,
                person_centre=person_centre,
                person_viewpoints=person_viewpoints,
                output_dir=output_dir,
                timeline_root=(ROOT / args.timeline_root).resolve(),
                risk_alpha=float(args.risk_alpha),
                protection_radius_m=float(args.min_person_clearance_m),
                timestamps_s=args.validation_times_s,
            )
        report_path = output_dir / "person_fire_plan.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[person-fire-plan] plan_id={plan['plan_id']}")
        print(f"[person-fire-plan] plan={plan_path}")
        print(f"[person-fire-plan] report={report_path}")
        print(f"[person-fire-plan] overlay={surrogate_path}")
    finally:
        grid.simulator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
