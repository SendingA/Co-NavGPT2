#!/usr/bin/env python3
"""Generate a short-dangerous/long-safe FireWorld plan for one ObjectNav episode.

Unlike the scene-wide ranker, this entry point resolves one exact episode and
scores the *combined* hazard from every selected semantic source.  It preserves
the native category goal centres and VIEW_POINTS, protects those success
regions, writes a content-addressed plan, and can optionally emit a one-episode
dataset and validate a baked FireWorld timeline.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.tune_fire_route_scenarios import (  # noqa: E402
    Candidate,
    _floor_id,
    _ignition_instances,
    _nearest_free,
    _scene_grid,
    _write_episode_dataset,
)
from utils.fire_world.episode_plan import (  # noqa: E402
    EpisodeSourceCandidate,
    episode_scene_id,
    native_goal_geometry,
    resolve_episode,
    select_best_source_combination,
    source_clearance_metrics,
)
from utils.fire_world.fine_tuning import (  # noqa: E402
    CURATED_FIRE_PROFILES,
    build_curated_plan,
    evaluate_fireworld_snapshot,
    route_overlay,
    shortest_grid_path,
    write_curated_plan,
)
from utils.fire_world.propagation import run_propagation  # noqa: E402
from utils.fire_world.runtime import FireWorld  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-shard",
        required=True,
        help="ObjectNav content/<scene>.json.gz containing the episode",
    )
    parser.add_argument("--episode-id", required=True)
    parser.add_argument(
        "--object-category",
        help="required when the shard reuses the episode id across categories",
    )
    parser.add_argument(
        "--profile", choices=sorted(CURATED_FIRE_PROFILES), default="stable"
    )
    parser.add_argument("--source-count", type=int, default=3)
    parser.add_argument(
        "--source-radius-m",
        type=float,
        help="override the profile's real FireWorld source radius",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--timeline-root", default="outputs/fire_world")
    parser.add_argument(
        "--scene-dataset-config",
        default=(
            "data/scene_datasets/hm3d_v0.2/"
            "hm3d_annotated_basis.scene_dataset_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "diagnostic directory; default outputs/fire_episode_plans/"
            "<scene>_ep_<id>_<category>_<profile>"
        ),
    )
    parser.add_argument(
        "--dataset-output-root",
        default="data/processed/fire_route_scenarios",
    )
    parser.add_argument("--write-dataset", action="store_true")
    parser.add_argument("--bake", action="store_true")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate an existing canonical timeline without baking it",
    )
    parser.add_argument(
        "--validation-times-s", nargs="+", type=float, default=(40.0, 60.0, 100.0)
    )
    parser.add_argument("--voxel-m", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--save-dt", type=float, default=1.0)
    parser.add_argument("--resolution-m", type=float, default=0.10)
    parser.add_argument("--risk-alpha", type=float, default=4.0)
    parser.add_argument("--min-route-length-m", type=float, default=4.0)
    parser.add_argument("--max-source-height-m", type=float, default=1.0)
    parser.add_argument("--max-source-to-blind-route-m", type=float, default=0.90)
    parser.add_argument("--min-source-start-clearance-m", type=float, default=1.50)
    parser.add_argument(
        "--min-goal-centre-clearance-m",
        type=float,
        help="default 3.0 for person and 1.5 for other categories",
    )
    parser.add_argument(
        "--min-goal-viewpoint-clearance-m",
        type=float,
        help="default 2.5 for person and 1.2 for other categories",
    )
    parser.add_argument("--max-goal-risk", type=float, default=0.35)
    parser.add_argument("--min-source-spacing-m", type=float, default=0.45)
    parser.add_argument("--max-source-candidates", type=int, default=20)
    parser.add_argument("--max-combinations", type=int, default=1000)
    parser.add_argument(
        "--allow-target-category-sources",
        action="store_true",
        help="allow semantic objects with the same category as the target to burn",
    )
    return parser


def _read_shard(path: Path) -> Mapping[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _goal_cells(grid, viewpoints: np.ndarray) -> list[tuple[int, int]]:
    cells = []
    for position in viewpoints:
        cell = _nearest_free(
            grid.traversible,
            grid.frame.world_to_grid(position),
            max_radius_cells=8,
        )
        if cell is not None:
            cells.append(tuple(int(value) for value in cell))
    result = sorted(set(cells))
    if not result:
        raise RuntimeError("no native goal VIEW_POINT projects onto the episode floor")
    return result


def _source_candidates(
    *,
    inventory: Mapping[str, object],
    grid,
    start_position: Sequence[float],
    object_category: str,
    goal_geometry,
    blind_cells: Sequence[Sequence[int]],
    max_source_height_m: float,
    max_source_to_blind_route_m: float,
    min_source_start_clearance_m: float,
    min_goal_centre_clearance_m: float,
    min_goal_viewpoint_clearance_m: float,
    allow_target_category_sources: bool,
) -> list[EpisodeSourceCandidate]:
    blind = np.asarray(blind_cells, dtype=np.float64)
    deltas = np.linalg.norm(np.diff(blind, axis=0), axis=1)
    progress = np.concatenate(([0.0], np.cumsum(deltas))) * float(
        grid.frame.resolution_m
    )
    floor_id = _floor_id(inventory, grid.floor_y_m)
    candidates = []
    for instance in _ignition_instances(inventory, floor_id=floor_id):
        if (
            not allow_target_category_sources
            and str(instance.get("category")) == str(object_category)
        ):
            continue
        position = np.asarray(instance["centroid"], dtype=np.float64)
        if abs(float(position[1]) - float(grid.floor_y_m)) > float(
            max_source_height_m
        ):
            continue
        cell = _nearest_free(
            grid.traversible,
            grid.frame.world_to_grid(position),
            max_radius_cells=max(8, int(round(1.0 / grid.frame.resolution_m))),
        )
        if cell is None:
            continue
        distances = np.linalg.norm(
            blind - np.asarray(cell, dtype=np.float64)[None, :], axis=1
        )
        path_index = int(np.argmin(distances))
        path_distance_m = float(distances[path_index] * grid.frame.resolution_m)
        if path_distance_m > float(max_source_to_blind_route_m):
            continue
        clearances = source_clearance_metrics(
            position,
            start_position=start_position,
            goal_geometry=goal_geometry,
        )
        if clearances["distance_to_start_m"] < float(
            min_source_start_clearance_m
        ):
            continue
        if clearances["nearest_goal_centre_clearance_m"] < float(
            min_goal_centre_clearance_m
        ):
            continue
        if clearances["nearest_goal_viewpoint_clearance_m"] < float(
            min_goal_viewpoint_clearance_m
        ):
            continue
        candidates.append(EpisodeSourceCandidate(
            instance=instance,
            cell=(int(cell[0]), int(cell[1])),
            position=tuple(float(value) for value in position[:3]),
            distance_to_blind_path_m=path_distance_m,
            distance_to_start_m=clearances["distance_to_start_m"],
            nearest_goal_centre_clearance_m=clearances[
                "nearest_goal_centre_clearance_m"
            ],
            nearest_goal_viewpoint_clearance_m=clearances[
                "nearest_goal_viewpoint_clearance_m"
            ],
            blind_path_progress_m=float(progress[path_index]),
        ))
    return sorted(candidates, key=lambda item: (
        item.distance_to_blind_path_m,
        -item.nearest_goal_viewpoint_clearance_m,
        item.object_id,
    ))


def _write_overlay(
    path: Path,
    *,
    traversible: np.ndarray,
    risk: np.ndarray,
    contrast,
    start: Sequence[int],
    ignition_cells: Iterable[Sequence[int]],
    resolution_m: float,
) -> None:
    image = route_overlay(
        traversible,
        risk,
        contrast,
        start=start,
        goal=contrast.blind.cells[-1],
        ignitions=ignition_cells,
    )
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
        raise RuntimeError(f"could not write overlay {path}")


def _validate_timeline(
    *,
    plan: Mapping[str, object],
    grid,
    start_cell: Sequence[int],
    goal_cells: Sequence[Sequence[int]],
    profile,
    risk_alpha: float,
    max_goal_risk: float,
    timestamps_s: Iterable[float],
    timeline_root: Path,
    output_dir: Path,
) -> Mapping[str, object]:
    fire_world = FireWorld.load(
        str(plan["scene_id"]), str(plan["plan_id"]), out_root=timeline_root
    )
    records = []
    ignition_cells = [
        grid.frame.world_to_grid(item["position"]) for item in plan["ignitions"]
    ]
    goal_cells = [tuple(int(value) for value in cell) for cell in goal_cells]
    for timestamp_s in timestamps_s:
        timestamp_s = float(timestamp_s)
        contrast, layers = evaluate_fireworld_snapshot(
            fire_world,
            timestamp_s=timestamp_s,
            frame=grid.frame,
            floor_y_m=grid.floor_y_m,
            traversible=grid.traversible,
            start=start_cell,
            goals=goal_cells,
            risk_alpha=float(risk_alpha),
            thresholds=profile.thresholds,
        )
        if contrast is None:
            raise RuntimeError(f"actual timeline has no route at t={timestamp_s:g}s")
        if not contrast.accepted:
            raise RuntimeError(
                f"actual route contrast rejected at t={timestamp_s:g}s: "
                f"{list(contrast.reasons)}"
            )
        goal_risk = np.asarray([
            layers.physical_risk[cell] for cell in goal_cells
        ])
        goal_hard = np.asarray([layers.hard_unsafe[cell] for cell in goal_cells])
        max_risk = float(np.max(goal_risk, initial=0.0))
        if bool(np.any(goal_hard)) or max_risk > float(max_goal_risk):
            raise RuntimeError(
                f"native goal protection rejected at t={timestamp_s:g}s: "
                f"hard={int(np.count_nonzero(goal_hard))}, max_risk={max_risk:.3f}"
            )
        overlay_path = output_dir / f"actual_t{int(round(timestamp_s)):04d}.png"
        _write_overlay(
            overlay_path,
            traversible=grid.traversible,
            risk=layers.physical_risk,
            contrast=contrast,
            start=start_cell,
            ignition_cells=ignition_cells,
            resolution_m=grid.frame.resolution_m,
        )
        records.append({
            "timestamp_s": timestamp_s,
            "contrast": contrast.to_dict(resolution_m=grid.frame.resolution_m),
            "max_goal_viewpoint_risk": max_risk,
            "hard_unsafe_goal_viewpoint_count": int(np.count_nonzero(goal_hard)),
            "overlay_path": str(overlay_path.relative_to(ROOT)),
        })
    return {"plan_id": str(plan["plan_id"]), "frames": records}


def _validate_args(args) -> None:
    positive = (
        "source_count", "resolution_m", "risk_alpha", "min_route_length_m",
        "max_source_height_m", "max_source_to_blind_route_m",
        "min_source_start_clearance_m", "min_source_spacing_m",
        "max_source_candidates", "max_combinations", "voxel_m", "dt", "save_dt",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= float(args.max_goal_risk) <= 1.0:
        raise ValueError("--max-goal-risk must be in [0, 1]")
    if int(args.max_source_candidates) < int(args.source_count):
        raise ValueError("--max-source-candidates must be >= --source-count")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    shard_path = (ROOT / args.source_shard).resolve()
    shard = _read_shard(shard_path)
    episode = resolve_episode(
        shard,
        args.episode_id,
        object_category=args.object_category,
    )
    scene_id = episode_scene_id(episode)
    category = str(episode["object_category"])
    start_position = [float(value) for value in episode["start_position"][:3]]
    profile = CURATED_FIRE_PROFILES[str(args.profile)]
    profile = replace(
        profile,
        num_initial_ignitions=int(args.source_count),
        source_radius_m=(
            float(profile.source_radius_m)
            if args.source_radius_m is None
            else float(args.source_radius_m)
        ),
    )
    min_goal_centre_clearance_m = (
        3.0 if category == "person" else 1.5
    ) if args.min_goal_centre_clearance_m is None else float(
        args.min_goal_centre_clearance_m
    )
    min_goal_viewpoint_clearance_m = (
        2.5 if category == "person" else 1.2
    ) if args.min_goal_viewpoint_clearance_m is None else float(
        args.min_goal_viewpoint_clearance_m
    )
    output_dir = (
        ROOT / args.output_dir
        if args.output_dir
        else ROOT / "outputs/fire_episode_plans" / (
            f"{scene_id}_ep_{args.episode_id}_{category}_{args.profile}"
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(
        (ROOT / args.scenes_root / scene_id / "inventory.json").read_text(
            encoding="utf-8"
        )
    )
    goal_geometry = native_goal_geometry(
        shard,
        scene_id=scene_id,
        object_category=category,
        floor_y_m=float(start_position[1]),
    )
    grid = _scene_grid(
        inventory,
        (ROOT / args.scene_dataset_config).resolve(),
        float(args.resolution_m),
        float(start_position[1]),
    )
    try:
        start_cell = _nearest_free(
            grid.traversible, grid.frame.world_to_grid(start_position)
        )
        if start_cell is None:
            raise RuntimeError("episode start has no nearby traversible cell")
        goal_cells = _goal_cells(grid, goal_geometry.viewpoints)
        blind = shortest_grid_path(grid.traversible, start_cell, goal_cells)
        if blind is None:
            raise RuntimeError("episode has no obstacle-only route to a native goal")
        if float(blind[2] * grid.frame.resolution_m) < float(args.min_route_length_m):
            raise RuntimeError(
                f"episode shortest route is only "
                f"{blind[2] * grid.frame.resolution_m:.2f}m; choose a longer episode"
            )
        source_candidates = _source_candidates(
            inventory=inventory,
            grid=grid,
            start_position=start_position,
            object_category=category,
            goal_geometry=goal_geometry,
            blind_cells=blind[0],
            max_source_height_m=float(args.max_source_height_m),
            max_source_to_blind_route_m=float(args.max_source_to_blind_route_m),
            min_source_start_clearance_m=float(args.min_source_start_clearance_m),
            min_goal_centre_clearance_m=min_goal_centre_clearance_m,
            min_goal_viewpoint_clearance_m=min_goal_viewpoint_clearance_m,
            allow_target_category_sources=bool(args.allow_target_category_sources),
        )[: int(args.max_source_candidates)]
        diagnostics: Dict[str, object] = {}
        result = select_best_source_combination(
            grid.traversible,
            start=start_cell,
            goals=goal_cells,
            candidates=source_candidates,
            source_count=int(args.source_count),
            resolution_m=grid.frame.resolution_m,
            core_radius_m=profile.synthetic_core_radius_m,
            risk_radius_m=profile.synthetic_risk_radius_m,
            risk_alpha=float(args.risk_alpha),
            thresholds=profile.thresholds,
            max_goal_risk=float(args.max_goal_risk),
            min_source_spacing_m=float(args.min_source_spacing_m),
            max_combinations=int(args.max_combinations),
            diagnostics=diagnostics,
        )
        report: Dict[str, object] = {
            "schema_version": 1,
            "scene_id": scene_id,
            "episode_id": str(episode["episode_id"]),
            "object_category": category,
            "profile": str(args.profile),
            "source_count": int(args.source_count),
            "start_position": start_position,
            "start_cell": [int(value) for value in start_cell],
            "native_goal_centre_count": int(len(goal_geometry.centres)),
            "native_goal_viewpoint_count": int(len(goal_geometry.viewpoints)),
            "goal_cell_count": int(len(goal_cells)),
            "obstacle_shortest_length_m": float(
                blind[2] * grid.frame.resolution_m
            ),
            "protection": {
                "min_goal_centre_clearance_m": min_goal_centre_clearance_m,
                "min_goal_viewpoint_clearance_m": min_goal_viewpoint_clearance_m,
                "max_goal_risk": float(args.max_goal_risk),
                "allow_target_category_sources": bool(
                    args.allow_target_category_sources
                ),
            },
            "search": {
                **diagnostics,
                "eligible_sources": [item.to_dict() for item in source_candidates],
            },
        }
        report_path = output_dir / "episode_fire_plan.json"
        if result is None:
            report["accepted"] = False
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "no accepted source combination; inspect "
                f"{report_path} and relax only the reported failed constraint"
            )
        selected_sources = [item.instance for item in result.sources]
        curation = {
            "schema_version": 1,
            "generator": "scripts/generate_episode_fire_plan.py",
            "scene_id": scene_id,
            "episode_id": str(episode["episode_id"]),
            "object_category": category,
            "start_position": start_position,
            "start_cell": [int(value) for value in start_cell],
            "chosen_goal_cell": [int(value) for value in result.contrast.blind.cells[-1]],
            "goal_cell_count": int(len(goal_cells)),
            "native_goal_viewpoint_count": int(len(goal_geometry.viewpoints)),
            "combined_contrast": result.contrast.to_dict(
                resolution_m=grid.frame.resolution_m
            ),
            "combined_max_goal_risk": float(result.max_goal_risk),
            "source_progress_span_m": float(result.source_progress_span_m),
            "sources": [item.to_dict() for item in result.sources],
            "protection": report["protection"],
        }
        plan = build_curated_plan(
            inventory,
            selected_sources[0],
            profile,
            seed=int(args.seed),
            curation=curation,
            additional_ignition_instances=selected_sources[1:],
        )
        plan_path = write_curated_plan(
            plan, (ROOT / args.scenes_root).resolve()
        )
        surrogate_path = output_dir / "surrogate_route_contrast.png"
        _write_overlay(
            surrogate_path,
            traversible=grid.traversible,
            risk=result.risk,
            contrast=result.contrast,
            start=start_cell,
            ignition_cells=[item.cell for item in result.sources],
            resolution_m=grid.frame.resolution_m,
        )
        report.update({
            "accepted": True,
            "plan_id": str(plan["plan_id"]),
            "plan_path": str(plan_path.relative_to(ROOT)),
            "source_radius_m": float(profile.source_radius_m),
            "selected_sources": [item.to_dict() for item in result.sources],
            "combined_surrogate": result.contrast.to_dict(
                resolution_m=grid.frame.resolution_m
            ),
            "combined_max_goal_risk": float(result.max_goal_risk),
            "source_progress_span_m": float(result.source_progress_span_m),
            "surrogate_overlay": str(surrogate_path.relative_to(ROOT)),
        })
        selected = Candidate(
            scene_id=scene_id,
            profile_name=str(args.profile),
            episode_id=str(episode["episode_id"]),
            object_category=category,
            start_position=start_position,
            start_cell=tuple(start_cell),
            goal_cells=goal_cells,
            chosen_goal_cell=tuple(result.contrast.blind.cells[-1]),
            ignition_cell=result.sources[0].cell,
            ignition_instance=result.sources[0].instance,
            contrast=result.contrast,
            risk=result.risk,
            hard_unsafe=result.hard_unsafe,
        )
        if args.write_dataset:
            dataset_path = _write_episode_dataset(
                shard_path.parent,
                (ROOT / args.dataset_output_root).resolve(),
                selected,
                plan,
            )
            report["dataset_path"] = str(dataset_path.relative_to(ROOT))
        report["validation_status"] = "surrogate_only"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        timeline_root = (ROOT / args.timeline_root).resolve()
        if args.bake:
            try:
                run_propagation(
                    inventory=dict(inventory),
                    plan=dict(plan),
                    voxel_m=float(args.voxel_m),
                    dt=float(args.dt),
                    save_dt=float(args.save_dt),
                    out_dir=timeline_root / scene_id / str(plan["plan_id"]),
                    verbose=True,
                )
            except Exception as exc:
                report["validation_status"] = "bake_failed"
                report["validation_error"] = f"{type(exc).__name__}: {exc}"
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise
        if args.bake or args.validate_existing:
            try:
                report["actual_timeline"] = _validate_timeline(
                    plan=plan,
                    grid=grid,
                    start_cell=start_cell,
                    goal_cells=goal_cells,
                    profile=profile,
                    risk_alpha=float(args.risk_alpha),
                    max_goal_risk=float(args.max_goal_risk),
                    timestamps_s=args.validation_times_s,
                    timeline_root=timeline_root,
                    output_dir=output_dir,
                )
                report["validation_status"] = "actual_timeline_passed"
            except Exception as exc:
                report["validation_status"] = "actual_timeline_rejected"
                report["validation_error"] = f"{type(exc).__name__}: {exc}"
                report_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[episode-fire-plan] plan_id={plan['plan_id']}")
        print(f"[episode-fire-plan] plan={plan_path}")
        print(f"[episode-fire-plan] report={report_path}")
        print(f"[episode-fire-plan] overlay={surrogate_path}")
        if report.get("dataset_path"):
            print(f"[episode-fire-plan] dataset={ROOT / str(report['dataset_path'])}")
    finally:
        grid.simulator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
