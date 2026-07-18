"""Generate a native HM3D ObjectNav benchmark with a fixed person target.

The source ObjectNav dataset is never modified. For every selected scene this
script chooses a deterministic navigable person position, samples a dense set
of navigable view points with unobstructed line of sight to the person's
torso, verifies close-stop geodesic coverage below Habitat's Success threshold,
and writes person-only episodes into a separate dataset directory.

Example:

    python scripts/build_person_objectnav_dataset.py --split val_mini
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.person_objectnav import (  # noqa: E402
    PERSON_CATEGORY,
    add_person_category_mappings,
    person_goals_key,
    unique_positions,
    validate_person_dataset_dict,
    yaw_quaternion_facing,
)


@dataclass(frozen=True)
class ViewPointSamplingConfig:
    """Reproducible sampling and coverage contract for person view points."""

    view_min_radius: float = 0.20
    view_max_radius: float = 1.10
    view_grid_spacing: float = 0.05
    max_snap_error: float = 0.08
    max_height_delta: float = 0.25
    dedup_tolerance: float = 0.04
    min_view_points: int = 24
    coverage_min_radius: float = 0.20
    coverage_max_radius: float = 1.00
    coverage_grid_spacing: float = 0.05
    coverage_max_snap_error: float = 0.04
    coverage_distance: float = 0.15
    polar_radii: Optional[Tuple[float, ...]] = None
    polar_angle_count: int = 72

    def __post_init__(self) -> None:
        finite_values = {
            "view_min_radius": self.view_min_radius,
            "view_max_radius": self.view_max_radius,
            "view_grid_spacing": self.view_grid_spacing,
            "max_snap_error": self.max_snap_error,
            "max_height_delta": self.max_height_delta,
            "dedup_tolerance": self.dedup_tolerance,
            "coverage_min_radius": self.coverage_min_radius,
            "coverage_max_radius": self.coverage_max_radius,
            "coverage_grid_spacing": self.coverage_grid_spacing,
            "coverage_max_snap_error": self.coverage_max_snap_error,
            "coverage_distance": self.coverage_distance,
        }
        if any(not math.isfinite(float(value))
               for value in finite_values.values()):
            raise ValueError("view-point sampling values must be finite")
        if not (0.0 <= self.view_min_radius < self.view_max_radius):
            raise ValueError("view radii must satisfy 0 <= min < max")
        if not (
            self.view_min_radius <= self.coverage_min_radius
            < self.coverage_max_radius <= self.view_max_radius
        ):
            raise ValueError(
                "coverage radii must lie inside the sampled view-point band"
            )
        if self.view_grid_spacing <= 0.0 or self.coverage_grid_spacing <= 0.0:
            raise ValueError("view and coverage grid spacing must be positive")
        if self.max_snap_error < 0.0 or self.coverage_max_snap_error < 0.0:
            raise ValueError("navmesh snap tolerances must be non-negative")
        if self.max_height_delta < 0.0:
            raise ValueError("max height delta must be non-negative")
        if self.dedup_tolerance <= 0.0:
            raise ValueError("view-point dedup tolerance must be positive")
        if not (0.0 < self.coverage_distance < 0.20):
            raise ValueError(
                "coverage distance must leave margin below Habitat's 0.20m "
                "success distance"
            )
        if self.min_view_points <= 0:
            raise ValueError("min view-point count must be positive")
        if self.polar_angle_count < 4:
            raise ValueError("polar angle count must be at least four")
        if self.polar_radii is not None:
            if not self.polar_radii:
                raise ValueError("explicit view radii cannot be empty")
            if any(
                not math.isfinite(float(radius)) or float(radius) < 0.0
                for radius in self.polar_radii
            ):
                raise ValueError("explicit view radii must be finite and non-negative")


def _read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_gz(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"))


def _scene_path(scene_id: str, scenes_dir: Path) -> Path:
    path = Path(scene_id)
    if path.is_absolute():
        return path
    direct = scenes_dir / path
    if direct.exists():
        return direct

    # HM3D v0.2 val_mini shards use a historical "minival" path while
    # installations commonly store those same scenes under "val".
    scene_name = path.name
    scene_short_id = scene_name.replace(".basis.glb", "")
    matches = sorted(
        scenes_dir.glob(f"hm3d_v0.2/*/*-{scene_short_id}/{scene_name}")
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"scene asset not found for {scene_id}")


def _dataset_scene_id(scene_path: Path, scenes_dir: Path) -> str:
    """Return a scene id that Habitat can resolve below dataset.scenes_dir."""
    try:
        return scene_path.relative_to(scenes_dir).as_posix()
    except ValueError:
        return str(scene_path)


def _make_simulator(scene_path: Path, scene_dataset_config: Path):
    try:
        import habitat_sim
    except ImportError as exc:
        raise RuntimeError(
            "habitat_sim is required to generate view points. Activate the "
            "same Habitat environment used to run main.py."
        ) from exc

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_path)
    backend.scene_dataset_config_file = str(scene_dataset_config)
    # Bullet ray casts are used to reject view points occluded by walls.
    backend.enable_physics = True

    agent = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))


def _geodesic(sim, start: Sequence[float],
              ends: Sequence[Sequence[float]]) -> float:
    import habitat_sim

    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_ends = np.asarray(ends, dtype=np.float32)
    if sim.pathfinder.find_path(path):
        return float(path.geodesic_distance)
    return math.inf


def _line_of_sight(sim, source: Sequence[float],
                   target: Sequence[float]) -> bool:
    """Check environment occlusion between camera height and human torso."""
    import habitat_sim

    origin = np.asarray(source, dtype=np.float32).copy()
    destination = np.asarray(target, dtype=np.float32).copy()
    origin[1] += 0.88
    destination[1] += 1.0
    delta = destination - origin
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        return False
    ray = habitat_sim.geo.Ray(origin, delta / distance)
    result = sim.cast_ray(ray)
    if not result.has_hits():
        return True
    first_hit = min(float(hit.ray_distance) for hit in result.hits)
    return first_hit >= distance - 0.08


def _cartesian_offsets(
    min_radius: float,
    max_radius: float,
    spacing: float,
    *,
    half_cell_offset: bool = False,
) -> List[np.ndarray]:
    """Return a deterministic XZ lattice clipped to a radial band."""
    extent = int(math.ceil(max_radius / spacing))
    phase = 0.5 if half_cell_offset else 0.0
    coordinates = (np.arange(-extent, extent + 1, dtype=np.float64) + phase) \
        * spacing
    offsets: List[np.ndarray] = []
    epsilon = spacing * 1e-6
    for dx in coordinates:
        for dz in coordinates:
            radius = math.hypot(float(dx), float(dz))
            if min_radius - epsilon <= radius <= max_radius + epsilon:
                offsets.append(np.asarray([dx, 0.0, dz], dtype=np.float64))
    return offsets


def _polar_offsets(radii: Sequence[float], angle_count: int) -> List[np.ndarray]:
    """Compatibility path for callers that explicitly request polar rings."""
    return [
        np.asarray(
            [radius * math.sin(angle), 0.0, radius * math.cos(angle)],
            dtype=np.float64,
        )
        for radius in radii
        for angle in (
            2.0 * math.pi * index / angle_count
            for index in range(angle_count)
        )
    ]


def _visible_navigable_positions(
    sim,
    person_position: np.ndarray,
    offsets: Sequence[np.ndarray],
    *,
    min_radius: float,
    max_radius: float,
    max_snap_error: float,
    max_height_delta: float,
    dedup_tolerance: float,
) -> List[np.ndarray]:
    """Snap candidates to the person's island and keep visible near points."""
    person_position = np.asarray(person_position, dtype=np.float64)
    island = sim.pathfinder.get_island(person_position)
    points: List[np.ndarray] = []
    epsilon = 1e-6
    for offset in offsets:
        candidate = person_position + np.asarray(offset, dtype=np.float64)
        snapped = np.asarray(
            sim.pathfinder.snap_point(candidate, island_index=island),
            dtype=np.float64,
        )
        if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
            continue
        if sim.pathfinder.get_island(snapped) != island:
            continue
        horizontal_error = float(
            np.linalg.norm((snapped - candidate)[[0, 2]])
        )
        height_delta = abs(float(snapped[1] - person_position[1]))
        radius = float(
            np.linalg.norm((snapped - person_position)[[0, 2]])
        )
        if horizontal_error > max_snap_error + epsilon:
            continue
        if height_delta > max_height_delta + epsilon:
            continue
        if not (min_radius - epsilon <= radius <= max_radius + epsilon):
            continue
        if not _line_of_sight(sim, snapped, person_position):
            continue
        points.append(snapped)
    return unique_positions(points, tolerance=dedup_tolerance)


def _sample_view_positions(
    sim,
    person_position: np.ndarray,
    config: ViewPointSamplingConfig,
) -> List[np.ndarray]:
    if config.polar_radii is None:
        offsets = _cartesian_offsets(
            config.view_min_radius,
            config.view_max_radius,
            config.view_grid_spacing,
        )
        min_radius = config.view_min_radius
        max_radius = config.view_max_radius
    else:
        offsets = _polar_offsets(
            config.polar_radii, config.polar_angle_count
        )
        min_radius = min(config.polar_radii)
        max_radius = max(config.polar_radii)
    return _visible_navigable_positions(
        sim,
        person_position,
        offsets,
        min_radius=min_radius,
        max_radius=max_radius,
        max_snap_error=config.max_snap_error,
        max_height_delta=config.max_height_delta,
        dedup_tolerance=config.dedup_tolerance,
    )


def _sample_view_points(
    sim,
    person_position: np.ndarray,
    config: ViewPointSamplingConfig,
) -> List[dict]:
    """Generate native ObjectNav view-point records around one person."""
    return [
        {
            "agent_state": {
                "position": point.tolist(),
                "rotation": yaw_quaternion_facing(point, person_position),
            },
            "iou": 1.0,
        }
        for point in _sample_view_positions(sim, person_position, config)
    ]


def _coverage_probe_positions(
    sim,
    person_position: np.ndarray,
    config: ViewPointSamplingConfig,
) -> List[np.ndarray]:
    offsets = _cartesian_offsets(
        config.coverage_min_radius,
        config.coverage_max_radius,
        config.coverage_grid_spacing,
        half_cell_offset=True,
    )
    return _visible_navigable_positions(
        sim,
        person_position,
        offsets,
        min_radius=config.coverage_min_radius,
        max_radius=config.coverage_max_radius,
        max_snap_error=config.coverage_max_snap_error,
        max_height_delta=config.max_height_delta,
        dedup_tolerance=0.5 * config.coverage_grid_spacing,
    )


def _validate_view_point_coverage(
    sim,
    person_position: np.ndarray,
    view_points: Sequence[dict],
    config: ViewPointSamplingConfig,
) -> dict:
    """Require every independent close-stop probe to satisfy native Success.

    Habitat Success uses a strict distance below 0.20m. This generator uses a
    stricter 0.15m geodesic bound so discretisation and float noise retain a
    five-centimetre margin at evaluation time.
    """
    view_positions = [
        view["agent_state"]["position"] for view in view_points
    ]
    if not view_positions:
        raise ValueError("person goal has no view points")
    probes = _coverage_probe_positions(sim, person_position, config)
    if not probes:
        raise ValueError("person position has no visible navigable close probes")

    distances = np.asarray(
        [_geodesic(sim, probe, view_positions) for probe in probes],
        dtype=np.float64,
    )
    finite = np.isfinite(distances)
    max_distance = float(np.max(distances)) if len(distances) else math.inf
    if not np.all(finite) or max_distance >= config.coverage_distance:
        worst_index = int(np.argmax(distances))
        raise ValueError(
            "close-stop coverage failed: "
            f"max_geodesic={max_distance:.3f}m, "
            f"required<{config.coverage_distance:.3f}m, "
            f"worst_probe={probes[worst_index].tolist()}"
        )
    return {
        "probe_count": len(probes),
        "max_geodesic": max_distance,
        "p95_geodesic": float(np.percentile(distances, 95)),
    }


def _select_goal_and_views(
    sim,
    source_episodes: Sequence[dict],
    *,
    config: ViewPointSamplingConfig,
) -> Tuple[np.ndarray, List[dict]]:
    candidates = unique_positions(
        episode.get("start_position", ()) for episode in source_episodes
    )
    last_error: Optional[Exception] = None
    for candidate in candidates:
        snapped = np.asarray(sim.pathfinder.snap_point(candidate), dtype=np.float64)
        if not np.all(np.isfinite(snapped)):
            continue
        views = _sample_view_points(sim, snapped, config)
        if len(views) < config.min_view_points:
            last_error = ValueError(
                f"only {len(views)} visible view points; "
                f"need {config.min_view_points}"
            )
            continue
        try:
            coverage = _validate_view_point_coverage(
                sim, snapped, views, config
            )
        except ValueError as exc:
            last_error = exc
            continue
        print(
            "[person-dataset] coverage: "
            f"{coverage['probe_count']} probes, "
            f"max={coverage['max_geodesic']:.3f}m, "
            f"p95={coverage['p95_geodesic']:.3f}m"
        )
        return snapped, views
    detail = f": {last_error}" if last_error is not None else ""
    raise RuntimeError(
        "could not find a person position with dense close-stop coverage"
        f"{detail}"
    )


def _person_goal(position: np.ndarray, view_points: List[dict]) -> dict:
    return {
        "position": position.tolist(),
        "radius": None,
        "object_id": "person_0",
        "object_name": "person_0",
        "object_name_id": 0,
        "object_category": PERSON_CATEGORY,
        "room_id": None,
        "room_name": None,
        "view_points": view_points,
    }


def _person_episodes(
    sim,
    source_episodes: Sequence[dict],
    person_position: np.ndarray,
    view_points: Sequence[dict],
    *,
    episodes_per_scene: int,
    min_start_distance: float,
) -> List[dict]:
    view_positions = [
        view["agent_state"]["position"] for view in view_points
    ]
    generated: List[dict] = []
    for source in source_episodes:
        start = source.get("start_position")
        if not isinstance(start, list) or len(start) != 3:
            continue
        geo = _geodesic(sim, start, view_positions)
        if not np.isfinite(geo) or geo < min_start_distance:
            continue
        episode = {
            "episode_id": str(len(generated)),
            "scene_id": source["scene_id"],
            "scene_dataset_config": source.get("scene_dataset_config"),
            "additional_obj_config_paths": source.get(
                "additional_obj_config_paths", []
            ),
            "start_position": start,
            "start_rotation": source["start_rotation"],
            "info": {
                "geodesic_distance": geo,
                "euclidean_distance": float(
                    np.linalg.norm(
                        (np.asarray(start) - person_position)[[0, 2]]
                    )
                ),
                "closest_goal_object_id": "person_0",
            },
            "goals": [],
            "start_room": source.get("start_room"),
            "shortest_paths": None,
            "object_category": PERSON_CATEGORY,
        }
        generated.append(episode)
        if len(generated) >= episodes_per_scene:
            break
    if not generated:
        raise RuntimeError("no source starts can reach the person view points")
    return generated


def build_scene_shard(
    source: dict,
    *,
    scene_path: Path,
    scene_dataset_config: Path,
    episodes_per_scene: int,
    min_start_distance: float,
    view_config: ViewPointSamplingConfig,
) -> dict:
    source_episodes = source.get("episodes", [])
    if not source_episodes:
        raise ValueError("source shard contains no episodes")

    sim = _make_simulator(scene_path, scene_dataset_config)
    try:
        person_position, views = _select_goal_and_views(
            sim,
            source_episodes,
            config=view_config,
        )
        episodes = _person_episodes(
            sim,
            source_episodes,
            person_position,
            views,
            episodes_per_scene=episodes_per_scene,
            min_start_distance=min_start_distance,
        )
    finally:
        sim.close()

    shard = {
        "goals_by_category": {
            person_goals_key(episodes[0]["scene_id"]): [
                _person_goal(person_position, views)
            ]
        },
        "episodes": episodes,
        "category_to_task_category_id": dict(
            source.get("category_to_task_category_id", {})
        ),
        "category_to_scene_annotation_category_id": dict(
            source.get("category_to_scene_annotation_category_id", {})
        ),
    }
    add_person_category_mappings(shard)
    validate_person_dataset_dict(shard)
    return shard


def validate_dataset_root(
    root: Path,
    split: str,
    *,
    scenes_dir: Optional[Path] = None,
    scene_dataset_config: Optional[Path] = None,
    view_config: Optional[ViewPointSamplingConfig] = None,
) -> int:
    """Validate structure and, when configured, live navmesh coverage."""
    if (scenes_dir is None) != (scene_dataset_config is None):
        raise ValueError(
            "scenes_dir and scene_dataset_config must be provided together"
        )
    if view_config is not None and scenes_dir is None:
        raise ValueError("live coverage validation requires scene assets")

    index_path = root / split / f"{split}.json.gz"
    index = _read_json_gz(index_path)
    add_person_category_mappings(index)
    scene_ids = index.get("content_scenes", [])
    shard_paths = sorted((root / split / "content").glob("*.json.gz"))
    if scene_ids and len(scene_ids) != len(shard_paths):
        raise ValueError("index content_scenes count does not match shard count")
    episode_count = 0
    coverage_goal_count = 0
    for path in shard_paths:
        shard = _read_json_gz(path)
        validate_person_dataset_dict(shard)
        episode_count += len(shard["episodes"])
        if view_config is None:
            continue
        scene_id = shard["episodes"][0]["scene_id"]
        sim = _make_simulator(
            _scene_path(scene_id, scenes_dir), scene_dataset_config
        )
        try:
            for goals in shard["goals_by_category"].values():
                for goal in goals:
                    coverage = _validate_view_point_coverage(
                        sim,
                        np.asarray(goal["position"], dtype=np.float64),
                        goal["view_points"],
                        view_config,
                    )
                    coverage_goal_count += 1
                    print(
                        f"[person-dataset] {path.stem}: coverage valid, "
                        f"{coverage['probe_count']} probes, "
                        f"max={coverage['max_geodesic']:.3f}m"
                    )
        finally:
            sim.close()
    if not shard_paths:
        raise ValueError("person dataset contains no scene shards")
    print(
        f"[person-dataset] valid: {len(shard_paths)} scenes, "
        f"{episode_count} episodes"
        + (
            f", {coverage_goal_count} goals with live coverage"
            if view_config is not None else ""
        )
    )
    return episode_count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",
                        default="data/datasets/objectnav_hm3d_v2")
    parser.add_argument("--output-root",
                        default="data/datasets/objectnav_hm3d_person_v1")
    parser.add_argument("--scenes-dir", default="data/scene_datasets")
    parser.add_argument(
        "--scene-dataset-config",
        default="data/scene_datasets/hm3d_v0.2/"
                "hm3d_annotated_basis.scene_dataset_config.json",
    )
    parser.add_argument("--split", default="val_mini")
    parser.add_argument("--episodes-per-scene", type=int, default=20)
    parser.add_argument("--min-start-distance", type=float, default=1.5)
    parser.add_argument("--view-min-radius", type=float, default=0.20)
    parser.add_argument("--view-max-radius", type=float, default=1.10)
    parser.add_argument("--view-grid-spacing", type=float, default=0.05)
    parser.add_argument("--view-max-snap-error", type=float, default=0.08)
    parser.add_argument("--view-max-height-delta", type=float, default=0.25)
    parser.add_argument("--view-dedup-tolerance", type=float, default=0.04)
    parser.add_argument("--min-view-points", type=int, default=24)
    parser.add_argument("--coverage-min-radius", type=float, default=0.20)
    parser.add_argument("--coverage-max-radius", type=float, default=1.00)
    parser.add_argument("--coverage-grid-spacing", type=float, default=0.05)
    parser.add_argument("--coverage-max-snap-error", type=float, default=0.04)
    parser.add_argument("--coverage-distance", type=float, default=0.15)
    parser.add_argument(
        "--view-radii",
        type=float,
        nargs="+",
        default=None,
        help="legacy explicit polar rings; omit to use the dense Cartesian grid",
    )
    parser.add_argument("--view-angle-count", type=int, default=72)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="with --validate-only, skip scene loading and geodesic coverage",
    )
    args = parser.parse_args(argv)

    source_root = (ROOT / args.source_root).resolve()
    output_root = (ROOT / args.output_root).resolve()
    scenes_dir = (ROOT / args.scenes_dir).resolve()
    scene_dataset_config = (ROOT / args.scene_dataset_config).resolve()

    view_config = ViewPointSamplingConfig(
        view_min_radius=args.view_min_radius,
        view_max_radius=args.view_max_radius,
        view_grid_spacing=args.view_grid_spacing,
        max_snap_error=args.view_max_snap_error,
        max_height_delta=args.view_max_height_delta,
        dedup_tolerance=args.view_dedup_tolerance,
        min_view_points=args.min_view_points,
        coverage_min_radius=args.coverage_min_radius,
        coverage_max_radius=args.coverage_max_radius,
        coverage_grid_spacing=args.coverage_grid_spacing,
        coverage_max_snap_error=args.coverage_max_snap_error,
        coverage_distance=args.coverage_distance,
        polar_radii=(
            tuple(float(radius) for radius in args.view_radii)
            if args.view_radii is not None else None
        ),
        polar_angle_count=args.view_angle_count,
    )

    if args.validate_only:
        validate_dataset_root(
            output_root,
            args.split,
            scenes_dir=None if args.structural_only else scenes_dir,
            scene_dataset_config=(
                None if args.structural_only else scene_dataset_config
            ),
            view_config=None if args.structural_only else view_config,
        )
        return 0

    source_content = source_root / args.split / "content"
    shard_paths = sorted(source_content.glob("*.json.gz"))
    if args.scene:
        selected = set(args.scene)
        shard_paths = [path for path in shard_paths if path.stem.split(".")[0] in selected]
    if not shard_paths:
        raise SystemExit(f"no source shards found under {source_content}")

    content_scenes: List[str] = []
    total_episodes = 0
    for source_path in shard_paths:
        source = _read_json_gz(source_path)
        first_episode = source.get("episodes", [None])[0]
        if first_episode is None:
            print(f"[person-dataset] skip empty shard {source_path.name}")
            continue
        scene_id = first_episode["scene_id"]
        shard = build_scene_shard(
            source,
            scene_path=_scene_path(scene_id, scenes_dir),
            scene_dataset_config=scene_dataset_config,
            episodes_per_scene=args.episodes_per_scene,
            min_start_distance=args.min_start_distance,
            view_config=view_config,
        )
        canonical_scene_id = _dataset_scene_id(
            _scene_path(scene_id, scenes_dir), scenes_dir
        )
        for episode in shard["episodes"]:
            episode["scene_id"] = canonical_scene_id
        output_path = output_root / args.split / "content" / source_path.name
        _write_json_gz(output_path, shard)
        content_scenes.append(source_path.stem.split(".")[0])
        total_episodes += len(shard["episodes"])
        print(
            f"[person-dataset] {content_scenes[-1]}: "
            f"{len(shard['episodes'])} episodes, "
            f"{len(next(iter(shard['goals_by_category'].values()))[0]['view_points'])} "
            f"view points"
        )

    source_index_path = source_root / args.split / f"{args.split}.json.gz"
    source_index = _read_json_gz(source_index_path)
    index = {
        "episodes": [],
        "content_scenes_path": "{data_path}/content/{scene}.json.gz",
        "content_scenes": content_scenes,
        "category_to_task_category_id": dict(
            source_index.get("category_to_task_category_id", {})
        ),
        "category_to_scene_annotation_category_id": dict(
            source_index.get("category_to_scene_annotation_category_id", {})
        ),
    }
    add_person_category_mappings(index)
    _write_json_gz(output_root / args.split / f"{args.split}.json.gz", index)
    # Each selected goal already passed live coverage before it was written.
    validate_dataset_root(output_root, args.split)
    print(
        f"[person-dataset] wrote {total_episodes} episodes to "
        f"{output_root / args.split}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
