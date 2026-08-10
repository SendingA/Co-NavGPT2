#!/usr/bin/env python3
"""Discover, build and validate curated risk-awareness fire scenarios.

The expensive voxel solver is the final stage, not the search primitive.  A
cheap radial fire surrogate first screens real HM3D ObjectNav starts, goal
viewpoints and semantic objects for the required route topology.  Selected
plans can then be baked and rescored with their actual FireWorld fields.

Example::

    python scripts/tune_fire_route_scenarios.py \
      --scenarios Nfvxx8J5NCo:stable TEEsavR23oF:dynamic \
      --write-plans --bake
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.fine_tuning import (  # noqa: E402
    CURATED_FIRE_PROFILES,
    RouteContrast,
    build_curated_plan,
    evaluate_fireworld_snapshot,
    evaluate_route_contrast,
    radial_hazard_map,
    route_overlay,
    shortest_grid_path,
    write_curated_plan,
)
from utils.fire_world.propagation import run_propagation  # noqa: E402
from utils.fire_world.runtime import FireWorld  # noqa: E402
from utils.risk.model import GridFrame  # noqa: E402


@dataclass
class SceneGrid:
    simulator: object
    traversible: np.ndarray
    frame: GridFrame
    floor_y_m: float


@dataclass
class Candidate:
    scene_id: str
    profile_name: str
    episode_id: str
    object_category: str
    start_position: List[float]
    start_cell: Tuple[int, int]
    goal_cells: List[Tuple[int, int]]
    chosen_goal_cell: Tuple[int, int]
    ignition_cell: Tuple[int, int]
    ignition_instance: Mapping[str, object]
    contrast: RouteContrast
    risk: np.ndarray
    hard_unsafe: np.ndarray

    def to_dict(self, *, resolution_m: float) -> Dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "profile": self.profile_name,
            "episode_id": self.episode_id,
            "object_category": self.object_category,
            "start_position": self.start_position,
            "start_cell": list(self.start_cell),
            "goal_cells": [list(cell) for cell in self.goal_cells],
            "chosen_goal_cell": list(self.chosen_goal_cell),
            "ignition_cell": list(self.ignition_cell),
            "ignition": {
                "instance_id": int(self.ignition_instance["instance_id"]),
                "category": str(self.ignition_instance.get("category")),
                "centroid": [
                    float(value)
                    for value in self.ignition_instance["centroid"]
                ],
                "flammability": float(
                    self.ignition_instance.get("flammability", 0.0)
                ),
            },
            "contrast": self.contrast.to_dict(resolution_m=resolution_m),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        required=True,
        metavar="SCENE:PROFILE",
        help="one or more HM3D scene/profile pairs; profile=stable|dynamic",
    )
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument(
        "--dataset-root",
        default="data/datasets/objectnav_hm3d_v2/val/content",
    )
    parser.add_argument(
        "--scene-dataset-config",
        default=(
            "data/scene_datasets/hm3d_v0.2/"
            "hm3d_annotated_basis.scene_dataset_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/fire_route_tuning"
    )
    parser.add_argument("--timeline-root", default="outputs/fire_world")
    parser.add_argument("--resolution-m", type=float, default=0.10)
    parser.add_argument("--risk-alpha", type=float, default=4.0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--write-plans",
        action="store_true",
        help="persist the best accepted plan for each scene/profile",
    )
    parser.add_argument(
        "--write-datasets",
        action="store_true",
        help=(
            "write a one-episode ObjectNav dataset for each selected plan "
            "under data/processed/fire_route_scenarios"
        ),
    )
    parser.add_argument(
        "--bake",
        action="store_true",
        help="bake each selected plan to its canonical FireWorld directory",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help=(
            "rescore an already-baked selected timeline without running the "
            "voxel solver again"
        ),
    )
    parser.add_argument("--voxel-m", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--save-dt", type=float, default=1.0)
    return parser


def _scenario(value: str) -> Tuple[str, str]:
    try:
        scene_id, profile_name = value.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"scenario must be SCENE:PROFILE, got {value!r}"
        ) from exc
    scene_id = scene_id.strip()
    profile_name = profile_name.strip().lower()
    if not scene_id or profile_name not in CURATED_FIRE_PROFILES:
        raise argparse.ArgumentTypeError(
            f"invalid scenario {value!r}; profile must be stable or dynamic"
        )
    return scene_id, profile_name


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_dataset(path: Path) -> Dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _make_simulator(
    scene_glb: Path,
    scene_dataset_config: Path,
    *,
    agent_radius_m: float = 0.18,
    enable_physics: bool = False,
):
    try:
        import habitat_sim
    except ImportError as exc:
        raise RuntimeError(
            "habitat_sim is required; activate the Co-Nav Habitat environment"
        ) from exc

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_glb)
    backend.scene_dataset_config_file = str(scene_dataset_config)
    backend.enable_physics = bool(enable_physics)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.radius = float(agent_radius_m)
    agent.height = 0.88
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def _scene_grid(
    inventory: Mapping[str, object],
    scene_dataset_config: Path,
    resolution_m: float,
    floor_y_m: float,
    *,
    enable_physics: bool = False,
) -> SceneGrid:
    simulator = _make_simulator(
        ROOT / str(inventory["scene_glb"]),
        scene_dataset_config,
        enable_physics=enable_physics,
    )
    nav_zx = np.asarray(
        simulator.pathfinder.get_topdown_view(
            float(resolution_m), float(floor_y_m)
        ),
        dtype=bool,
    )
    # Habitat top-down maps index (z, x); GridFrame and the FireWorld fields
    # index (x, z), so transpose once at the boundary.
    traversible = nav_zx.T.copy()
    lower, _ = simulator.pathfinder.get_bounds()
    frame = GridFrame(
        shape=traversible.shape,
        resolution_m=float(resolution_m),
        origin_xz=(float(lower[0]), float(lower[2])),
    )
    return SceneGrid(
        simulator=simulator,
        traversible=traversible,
        frame=frame,
        floor_y_m=float(floor_y_m),
    )


def _nearest_free(
    traversible: np.ndarray,
    cell: Sequence[int],
    *,
    max_radius_cells: int = 8,
) -> Optional[Tuple[int, int]]:
    row = int(np.clip(cell[0], 0, traversible.shape[0] - 1))
    col = int(np.clip(cell[1], 0, traversible.shape[1] - 1))
    if traversible[row, col]:
        return row, col
    for radius in range(1, max_radius_cells + 1):
        r0, r1 = max(0, row - radius), min(traversible.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(traversible.shape[1], col + radius + 1)
        cells = np.argwhere(traversible[r0:r1, c0:c1])
        if cells.size:
            cells[:, 0] += r0
            cells[:, 1] += c0
            index = int(
                np.argmin(np.linalg.norm(cells - np.asarray([row, col]), axis=1))
            )
            return int(cells[index, 0]), int(cells[index, 1])
    return None


def _goal_positions(
    dataset: Mapping[str, object],
    scene_id: str,
    category: str,
    floor_y_m: float,
) -> List[List[float]]:
    key = f"{scene_id}.basis.glb_{category}"
    goals = dataset.get("goals_by_category", {}).get(key, [])
    positions: List[List[float]] = []
    for goal in goals:
        for view in goal.get("view_points", []):
            position = view.get("agent_state", {}).get("position")
            if (
                position is not None
                and len(position) >= 3
                and abs(float(position[1]) - float(floor_y_m)) <= 0.60
            ):
                positions.append([float(value) for value in position[:3]])
    return positions


def _goal_cells(
    positions: Iterable[Sequence[float]],
    grid: SceneGrid,
) -> List[Tuple[int, int]]:
    cells = []
    for position in positions:
        raw = grid.frame.world_to_grid(np.asarray(position, dtype=np.float64))
        cell = _nearest_free(grid.traversible, raw)
        if cell is not None:
            cells.append(cell)
    return sorted(set(cells))


def _ignition_instances(
    inventory: Mapping[str, object],
    *,
    floor_id: Optional[int],
) -> List[Mapping[str, object]]:
    candidates = []
    for instance in inventory.get("instances", []):
        if bool(instance.get("structural", False)):
            continue
        if float(instance.get("flammability", 0.0)) < 0.40:
            continue
        if floor_id is not None and instance.get("floor_id") not in {
            None,
            floor_id,
        }:
            continue
        centroid = instance.get("centroid")
        if centroid is None or len(centroid) < 3:
            continue
        candidates.append(instance)
    return candidates


def _select_plan_ignitions(
    candidate: Candidate,
    inventory: Mapping[str, object],
    grid: SceneGrid,
    count: int,
) -> List[Mapping[str, object]]:
    """Select sources near the blind route and clear of the safe detour.

    The primary source remains the object that won candidate screening.
    Additional sources must be ordinary same-floor semantic objects, sit near
    the blind route, remain at least 1.10 m from the surrogate aware route,
    and be spatially distinct from already selected sources.
    """

    requested = int(count)
    if requested < 1:
        raise ValueError("ignition count must be positive")
    selected = [candidate.ignition_instance]
    if requested == 1:
        return selected

    blind = np.asarray(candidate.contrast.blind.cells, dtype=np.float64)
    aware = np.asarray(candidate.contrast.aware.cells, dtype=np.float64)
    floor_id = _floor_id(inventory, grid.floor_y_m)
    primary_id = int(candidate.ignition_instance["instance_id"])
    ranked = []
    for instance in _ignition_instances(inventory, floor_id=floor_id):
        object_id = int(instance["instance_id"])
        if object_id == primary_id:
            continue
        centroid = np.asarray(instance["centroid"], dtype=np.float64)
        # Avoid ceiling fixtures: their centroid can project near the route
        # while the physical flame sphere never reaches the floor corridor.
        if abs(float(centroid[1]) - float(grid.floor_y_m)) > 1.0:
            continue
        raw_cell = grid.frame.world_to_grid(centroid)
        cell = _nearest_free(grid.traversible, raw_cell, max_radius_cells=10)
        if cell is None:
            continue
        point = np.asarray(cell, dtype=np.float64)
        blind_distance_m = float(
            np.min(np.linalg.norm(blind - point[None, :], axis=1))
            * grid.frame.resolution_m
        )
        aware_distance_m = float(
            np.min(np.linalg.norm(aware - point[None, :], axis=1))
            * grid.frame.resolution_m
        )
        if blind_distance_m > 0.90 or aware_distance_m < 1.10:
            continue
        ranked.append((
            blind_distance_m,
            -aware_distance_m,
            object_id,
            instance,
        ))

    for _, _, _, instance in sorted(ranked):
        position = np.asarray(instance["centroid"], dtype=np.float64)
        if any(
            np.linalg.norm(
                position[[0, 2]]
                - np.asarray(other["centroid"], dtype=np.float64)[[0, 2]]
            ) < 0.45
            for other in selected
        ):
            continue
        selected.append(instance)
        if len(selected) == requested:
            break
    if len(selected) != requested:
        raise RuntimeError(
            f"could only select {len(selected)} of {requested} route-safe "
            "ignition objects"
        )
    return selected


def _ignition_summary(
    instances: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    return [
        {
            "instance_id": int(instance["instance_id"]),
            "category": str(instance.get("category")),
            "centroid": [
                float(value) for value in instance["centroid"][:3]
            ],
            "flammability": float(instance.get("flammability", 0.0)),
        }
        for instance in instances
    ]


def _floor_id(inventory: Mapping[str, object], floor_y_m: float) -> Optional[int]:
    floors = inventory.get("floors") or []
    if not floors:
        return None
    best = min(floors, key=lambda floor: abs(float(floor["y"]) - floor_y_m))
    return int(best["id"])


def _screen_scene(
    scene_id: str,
    profile_name: str,
    *,
    scenes_root: Path,
    dataset_root: Path,
    scene_dataset_config: Path,
    resolution_m: float,
    risk_alpha: float,
    max_episodes: int,
) -> Tuple[SceneGrid, Mapping[str, object], List[Candidate]]:
    inventory = _read_json(scenes_root / scene_id / "inventory.json")
    dataset = _read_dataset(dataset_root / f"{scene_id}.json.gz")
    episodes = list(dataset.get("episodes", []))
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise RuntimeError(f"scene {scene_id} contains no ObjectNav episodes")

    # HM3D ObjectNav shards are single-floor per episode.  Rebuild the grid if
    # a later episode belongs to another floor; candidate objects stay tied to
    # that exact floor.
    profile = CURATED_FIRE_PROFILES[profile_name]
    thresholds = profile.thresholds
    candidates: List[Candidate] = []
    active_grid: Optional[SceneGrid] = None
    active_floor: Optional[float] = None

    try:
        for episode in episodes:
            start_position = [
                float(value) for value in episode["start_position"][:3]
            ]
            floor_y = float(start_position[1])
            if active_grid is None or abs(floor_y - float(active_floor)) > 0.60:
                if active_grid is not None:
                    active_grid.simulator.close()
                active_grid = _scene_grid(
                    inventory,
                    scene_dataset_config,
                    resolution_m,
                    floor_y,
                )
                active_floor = floor_y
            grid = active_grid
            start_raw = grid.frame.world_to_grid(start_position)
            start = _nearest_free(grid.traversible, start_raw)
            if start is None:
                continue
            category = str(episode["object_category"])
            goals = _goal_cells(
                _goal_positions(dataset, scene_id, category, floor_y),
                grid,
            )
            if not goals:
                continue
            blind = shortest_grid_path(grid.traversible, start, goals)
            if blind is None or blind[2] * resolution_m < 4.0:
                continue
            blind_cells = np.asarray(blind[0], dtype=np.float64)
            for instance in _ignition_instances(
                inventory,
                floor_id=_floor_id(inventory, floor_y),
            ):
                ignition_raw = grid.frame.world_to_grid(instance["centroid"])
                ignition = _nearest_free(
                    grid.traversible,
                    ignition_raw,
                    max_radius_cells=max(
                        8, int(math.ceil(1.0 / resolution_m))
                    ),
                )
                if ignition is None:
                    continue
                distance_to_path_m = float(
                    np.min(
                        np.linalg.norm(
                            blind_cells - np.asarray(ignition)[None, :],
                            axis=1,
                        )
                    )
                    * resolution_m
                )
                if distance_to_path_m > 0.85:
                    continue
                distance_to_start_m = float(
                    np.linalg.norm(np.subtract(ignition, start)) * resolution_m
                )
                distance_to_goal_m = min(
                    float(np.linalg.norm(np.subtract(ignition, goal)) * resolution_m)
                    for goal in goals
                )
                if distance_to_start_m < 1.5 or distance_to_goal_m < 1.2:
                    continue
                risk, hard = radial_hazard_map(
                    grid.traversible.shape,
                    ignition,
                    resolution_m=resolution_m,
                    core_radius_m=profile.synthetic_core_radius_m,
                    risk_radius_m=profile.synthetic_risk_radius_m,
                )
                contrast = evaluate_route_contrast(
                    grid.traversible,
                    start,
                    goals,
                    risk,
                    hard,
                    risk_alpha=risk_alpha,
                    thresholds=thresholds,
                )
                if contrast is None:
                    continue
                candidates.append(Candidate(
                    scene_id=scene_id,
                    profile_name=profile_name,
                    episode_id=str(episode["episode_id"]),
                    object_category=category,
                    start_position=start_position,
                    start_cell=start,
                    goal_cells=goals,
                    chosen_goal_cell=contrast.blind.cells[-1],
                    ignition_cell=ignition,
                    ignition_instance=instance,
                    contrast=contrast,
                    risk=risk,
                    hard_unsafe=hard,
                ))
    except Exception:
        if active_grid is not None:
            active_grid.simulator.close()
        raise

    if active_grid is None:
        raise RuntimeError(f"could not build a navigation grid for {scene_id}")
    candidates.sort(
        key=lambda candidate: (
            not candidate.contrast.accepted,
            -candidate.contrast.score,
            candidate.episode_id,
            int(candidate.ignition_instance["instance_id"]),
        )
    )
    return active_grid, inventory, candidates


def _summary_without_cells(candidate: Candidate, resolution_m: float) -> Dict:
    payload = candidate.to_dict(resolution_m=resolution_m)
    for planner_name in ("blind", "aware"):
        payload["contrast"][planner_name].pop("cells", None)
    payload["goal_cells"] = [list(candidate.chosen_goal_cell)]
    payload["acceptance_thresholds"] = (
        CURATED_FIRE_PROFILES[candidate.profile_name].thresholds.to_dict()
    )
    return payload


def _write_candidate_outputs(
    output_dir: Path,
    candidate: Candidate,
    *,
    traversible: np.ndarray,
    resolution_m: float,
    rank: int,
) -> None:
    stem = f"rank_{rank:02d}_ep_{candidate.episode_id}_obj_{candidate.ignition_instance['instance_id']}"
    payload = candidate.to_dict(resolution_m=resolution_m)
    (output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    overlay = route_overlay(
        traversible,
        candidate.risk,
        candidate.contrast,
        start=candidate.start_cell,
        goal=candidate.chosen_goal_cell,
        ignition=candidate.ignition_cell,
    )
    # Enlarge nearest-neighbour so one navigation cell remains legible.
    scale = max(1, int(round(0.40 / resolution_m)))
    overlay = cv2.resize(
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imwrite(str(output_dir / f"{stem}.png"), overlay)


def _write_episode_dataset(
    dataset_root: Path,
    output_root: Path,
    candidate: Candidate,
    plan: Mapping[str, object],
) -> Path:
    """Write one scenario using Habitat's native root-plus-content layout."""

    source = _read_dataset(dataset_root / f"{candidate.scene_id}.json.gz")
    matches = []
    for episode in source.get("episodes", []):
        if str(episode.get("episode_id")) != candidate.episode_id:
            continue
        if str(episode.get("object_category")) != candidate.object_category:
            continue
        start = episode.get("start_position")
        if (
            start is None
            or len(start) < 3
            or not np.allclose(
                np.asarray(start[:3], dtype=np.float64),
                np.asarray(candidate.start_position[:3], dtype=np.float64),
                rtol=0.0,
                atol=1e-5,
            )
        ):
            continue
        matches.append(episode)
    if len(matches) != 1:
        raise RuntimeError(
            "expected one episode matching "
            f"(id={candidate.episode_id!r}, "
            f"category={candidate.object_category!r}, "
            f"start={candidate.start_position!r}) in "
            f"{candidate.scene_id}, found {len(matches)}"
        )
    scenario = {
        "schema_version": 1,
        "scene_id": candidate.scene_id,
        "plan_id": str(plan["plan_id"]),
        "profile": candidate.profile_name,
        "episode_id": candidate.episode_id,
        "object_category": candidate.object_category,
    }
    shard_payload = dict(source)
    shard_payload["episodes"] = matches
    shard_payload["fire_route_scenario"] = scenario

    # A direct content shard is not a valid data_path when main.py also sets
    # content_scenes: Habitat falls back to filtering the embedded scene path
    # token (e.g. Nfvxx8J5NCo.basis) against the FireWorld short ID
    # (Nfvxx8J5NCo), producing zero episodes.  Preserve the normal ObjectNav
    # package boundary so content_scenes selects a shard filename instead.
    root_payload = {
        key: value
        for key, value in source.items()
        if key not in {"episodes", "goals_by_category"}
    }
    root_payload["episodes"] = []
    root_payload["content_scenes_path"] = (
        "{data_path}/content/{scene}.json.gz"
    )
    root_payload["fire_route_scenario"] = scenario

    package_dir = output_root / str(plan["plan_id"])
    content_dir = package_dir / "content"
    content_dir.mkdir(parents=True, exist_ok=True)
    root_path = package_dir / "val.json.gz"
    shard_path = content_dir / f"{candidate.scene_id}.json.gz"
    for path, payload in (
        (root_path, root_payload),
        (shard_path, shard_payload),
    ):
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
    return root_path


def _actual_timeline_report(
    candidate: Candidate,
    grid: SceneGrid,
    *,
    plan: Mapping[str, object],
    timeline_root: Path,
    risk_alpha: float,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    fire_world = FireWorld.load(
        candidate.scene_id,
        str(plan["plan_id"]),
        out_root=timeline_root,
    )
    profile = CURATED_FIRE_PROFILES[candidate.profile_name]
    ignition_cells = [
        tuple(
            int(value)
            for value in grid.frame.world_to_grid(ignition["position"])
        )
        for ignition in plan["ignitions"]
    ]
    fractions = (0.20, 0.50, 0.80)
    frames = []
    for fraction in fractions:
        timestamp = float(profile.duration_s * fraction)
        contrast, layers = evaluate_fireworld_snapshot(
            fire_world,
            timestamp_s=timestamp,
            frame=grid.frame,
            floor_y_m=grid.floor_y_m,
            traversible=grid.traversible,
            start=candidate.start_cell,
            goals=[candidate.chosen_goal_cell],
            risk_alpha=risk_alpha,
            thresholds=profile.thresholds,
        )
        frame_record = {
            "timestamp_s": timestamp,
            "contrast": (
                None
                if contrast is None
                else contrast.to_dict(resolution_m=grid.frame.resolution_m)
            ),
            "risk": {
                "mean": float(np.mean(layers.physical_risk)),
                "max": float(np.max(layers.physical_risk)),
                "hard_fraction": float(np.mean(layers.hard_unsafe)),
            },
        }
        if contrast is not None and output_dir is not None:
            overlay = route_overlay(
                grid.traversible,
                layers.physical_risk,
                contrast,
                start=candidate.start_cell,
                goal=candidate.chosen_goal_cell,
                ignitions=ignition_cells,
            )
            scale = max(
                1, int(round(0.40 / grid.frame.resolution_m))
            )
            overlay = cv2.resize(
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            overlay_path = output_dir / (
                f"actual_t{int(round(timestamp)):04d}.png"
            )
            cv2.imwrite(str(overlay_path), overlay)
            frame_record["overlay_path"] = str(overlay_path)
        frames.append(frame_record)
    return {
        "scene_id": candidate.scene_id,
        "plan_id": str(plan["plan_id"]),
        "profile": candidate.profile_name,
        "frames": frames,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.resolution_m <= 0.0:
        raise ValueError("--resolution-m must be positive")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    scenarios = [_scenario(value) for value in args.scenarios]
    if len({scene for scene, _ in scenarios}) != len(scenarios):
        raise ValueError("each curated scenario must use a distinct scene")

    scenes_root = (ROOT / args.scenes_root).resolve()
    dataset_root = (ROOT / args.dataset_root).resolve()
    scene_dataset_config = (ROOT / args.scene_dataset_config).resolve()
    output_root = (ROOT / args.output_dir).resolve()
    timeline_root = (ROOT / args.timeline_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    master_records = []
    failures = []
    for scene_id, profile_name in scenarios:
        print(f"[route-tuning] screening {scene_id}:{profile_name}")
        grid = None
        try:
            grid, inventory, candidates = _screen_scene(
                scene_id,
                profile_name,
                scenes_root=scenes_root,
                dataset_root=dataset_root,
                scene_dataset_config=scene_dataset_config,
                resolution_m=float(args.resolution_m),
                risk_alpha=float(args.risk_alpha),
                max_episodes=int(args.max_episodes),
            )
            scenario_dir = output_root / f"{scene_id}_{profile_name}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            for rank, candidate in enumerate(candidates[: args.top_k], 1):
                _write_candidate_outputs(
                    scenario_dir,
                    candidate,
                    traversible=grid.traversible,
                    resolution_m=float(args.resolution_m),
                    rank=rank,
                )
            accepted = [
                candidate for candidate in candidates
                if candidate.contrast.accepted
            ]
            summary = {
                "scene_id": scene_id,
                "profile": profile_name,
                "resolution_m": float(args.resolution_m),
                "risk_alpha": float(args.risk_alpha),
                "candidate_count": len(candidates),
                "accepted_count": len(accepted),
                "top_candidates": [
                    candidate.to_dict(resolution_m=float(args.resolution_m))
                    for candidate in candidates[: args.top_k]
                ],
            }
            plan = None
            if accepted:
                best = accepted[0]
                if (
                    args.write_plans
                    or args.write_datasets
                    or args.bake
                    or args.validate_existing
                ):
                    profile = CURATED_FIRE_PROFILES[profile_name]
                    selected_ignitions = _select_plan_ignitions(
                        best,
                        inventory,
                        grid,
                        profile.num_initial_ignitions,
                    )
                    curation = _summary_without_cells(
                        best, float(args.resolution_m)
                    )
                    curation["ignitions"] = _ignition_summary(
                        selected_ignitions
                    )
                    plan = build_curated_plan(
                        inventory,
                        best.ignition_instance,
                        profile,
                        seed=int(args.seed),
                        curation=curation,
                        additional_ignition_instances=selected_ignitions[1:],
                    )
                    plan_path = write_curated_plan(plan, scenes_root)
                    summary["selected_plan_id"] = str(plan["plan_id"])
                    summary["selected_plan_path"] = str(plan_path)
                    print(
                        f"[route-tuning] selected {plan['plan_id']} "
                        f"score={best.contrast.score:.3f}"
                    )
                if args.write_datasets:
                    dataset_path = _write_episode_dataset(
                        dataset_root,
                        ROOT / "data/processed/fire_route_scenarios",
                        best,
                        plan,
                    )
                    summary["selected_dataset_path"] = str(dataset_path)
                if args.bake:
                    out_dir = timeline_root / scene_id / str(plan["plan_id"])
                    run_propagation(
                        inventory=dict(inventory),
                        plan=dict(plan),
                        voxel_m=float(args.voxel_m),
                        dt=float(args.dt),
                        save_dt=float(args.save_dt),
                        out_dir=out_dir,
                        verbose=True,
                    )
                if args.bake or args.validate_existing:
                    actual = _actual_timeline_report(
                        best,
                        grid,
                        plan=plan,
                        timeline_root=timeline_root,
                        risk_alpha=float(args.risk_alpha),
                        output_dir=scenario_dir,
                    )
                    actual_path = scenario_dir / "actual_timeline_report.json"
                    actual_path.write_text(
                        json.dumps(actual, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    summary["actual_timeline_report"] = str(actual_path)
            else:
                failures.append(
                    f"{scene_id}:{profile_name}: no accepted route contrast"
                )
            summary_path = scenario_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            master_records.append(summary)
        except Exception as exc:
            message = f"{scene_id}:{profile_name}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"[route-tuning] FAILED {message}")
        finally:
            if grid is not None:
                grid.simulator.close()

    master = {
        "schema_version": 1,
        "scenarios": master_records,
        "failures": failures,
    }
    master_path = output_root / "summary.json"
    master_path.write_text(
        json.dumps(master, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[route-tuning] wrote {master_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
