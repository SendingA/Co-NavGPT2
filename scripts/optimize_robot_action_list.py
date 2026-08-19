#!/usr/bin/env python3
"""Convert a dense Habitat action trace into real-robot waypoint commands.

The source ``action_list.json`` remains an exact simulator record: it contains
one primitive action per agent and navigation step, including stationary turn
and camera actions.  A physical robot should not replay those poses one by
one.  This tool removes stationary samples, simplifies only inside a bounded
corridor around the executed x/z path, and caps the distance between adjacent
waypoints.  It deliberately uses straight segments instead of a smoothing
spline so a corner is never rounded beyond the configured corridor.

Output coordinates are unmodified Habitat world coordinates ``[x, y, z]``.
For image overlays, world ``x`` maps to image column and world ``z`` maps to
image row; no Open3D map coordinate is stored in the output.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class TraceSample:
    step: int
    t_sim_s: float
    position_xyz_m: Tuple[float, float, float]
    risk: float
    hard_unsafe: bool
    action_name: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source action_list.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scenario-json",
        default=None,
        help=(
            "optional scenario metadata; defaults to scenario.json beside "
            "the output directory"
        ),
    )
    parser.add_argument(
        "--path-deviation-m",
        type=float,
        default=0.15,
        help="maximum x/z deviation allowed by line simplification",
    )
    parser.add_argument(
        "--max-waypoint-spacing-m",
        type=float,
        default=0.90,
        help="maximum straight-line distance between output waypoints",
    )
    parser.add_argument(
        "--stationary-epsilon-m",
        type=float,
        default=0.02,
        help="positions closer than this are one stationary pose",
    )
    parser.add_argument("--position-tolerance-m", type=float, default=0.20)
    parser.add_argument("--heading-tolerance-deg", type=float, default=20.0)
    parser.add_argument("--max-linear-speed-m-s", type=float, default=0.30)
    parser.add_argument("--max-angular-speed-rad-s", type=float, default=0.60)
    parser.add_argument(
        "--scene-dataset-config",
        default=None,
        help="optional Habitat scene dataset config for 2-D navmesh validation",
    )
    parser.add_argument("--navmesh-resolution-m", type=float, default=0.05)
    return parser


def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _distance_xz(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def _path_length_xz(points: Sequence[Sequence[float]]) -> float:
    return float(sum(
        _distance_xz(points[index - 1], points[index])
        for index in range(1, len(points))
    ))


def _point_segment_distance_xz(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    dx = float(end[0]) - float(start[0])
    dz = float(end[2]) - float(start[2])
    denominator = dx * dx + dz * dz
    if denominator <= 1e-16:
        return _distance_xz(point, start)
    fraction = (
        (float(point[0]) - float(start[0])) * dx
        + (float(point[2]) - float(start[2])) * dz
    ) / denominator
    fraction = min(1.0, max(0.0, fraction))
    projected = (
        float(start[0]) + fraction * dx,
        float(point[1]),
        float(start[2]) + fraction * dz,
    )
    return _distance_xz(point, projected)


def _rdp_indices(points: Sequence[Sequence[float]], tolerance_m: float) -> List[int]:
    """Return ordered Ramer-Douglas-Peucker anchors for an x/z path."""

    tolerance_m = _validate_positive("path_deviation_m", tolerance_m)
    if len(points) <= 2:
        return list(range(len(points)))
    keep = {0, len(points) - 1}
    pending = [(0, len(points) - 1)]
    while pending:
        start, end = pending.pop()
        if end - start <= 1:
            continue
        distances = [
            _point_segment_distance_xz(points[index], points[start], points[end])
            for index in range(start + 1, end)
        ]
        relative = int(np.argmax(distances))
        maximum = float(distances[relative])
        if maximum > tolerance_m:
            split = start + 1 + relative
            keep.add(split)
            pending.extend(((start, split), (split, end)))
    return sorted(keep)


def _extract_agent_samples(
    payload: Mapping[str, object], agent_id: int
) -> List[TraceSample]:
    samples: List[TraceSample] = []
    for record in payload.get("steps", []):
        action = next(
            (
                item
                for item in record.get("actions", [])
                if int(item.get("agent_id", -1)) == int(agent_id)
            ),
            None,
        )
        if action is None:
            raise ValueError(
                f"step {record.get('step')} has no action for agent {agent_id}"
            )
        position = tuple(float(value) for value in action["position_after"])
        if len(position) != 3 or not all(math.isfinite(value) for value in position):
            raise ValueError("position_after must contain three finite xyz values")
        samples.append(TraceSample(
            step=int(record["step"]),
            t_sim_s=float(record.get("t_sim_s", 0.0)),
            position_xyz_m=position,
            risk=float(action.get("risk_after", 0.0)),
            hard_unsafe=bool(action.get("hard_unsafe_after", False)),
            action_name=str(action.get("action_name", "unknown")),
        ))
    if not samples:
        raise ValueError(f"action list has no samples for agent {agent_id}")
    return samples


def _moving_samples(
    raw_samples: Sequence[TraceSample], stationary_epsilon_m: float
) -> List[TraceSample]:
    stationary_epsilon_m = _validate_positive(
        "stationary_epsilon_m", stationary_epsilon_m
    )
    moving = [raw_samples[0]]
    for sample in raw_samples[1:]:
        if _distance_xz(
            sample.position_xyz_m, moving[-1].position_xyz_m
        ) > stationary_epsilon_m:
            moving.append(sample)
    # The terminal step often contains STOP/HOLD at the last moving position.
    # Keep the geometry once, but associate its final point with terminal time.
    final = raw_samples[-1]
    if _distance_xz(final.position_xyz_m, moving[-1].position_xyz_m) <= 1e-9:
        moving[-1] = final
    else:
        moving.append(final)
    return moving


def _source_index_for_fraction(
    samples: Sequence[TraceSample], start: int, end: int, fraction: float
) -> int:
    if end <= start:
        return start
    cumulative = [0.0]
    for index in range(start + 1, end + 1):
        cumulative.append(
            cumulative[-1]
            + _distance_xz(
                samples[index - 1].position_xyz_m,
                samples[index].position_xyz_m,
            )
        )
    target = min(1.0, max(0.0, float(fraction))) * cumulative[-1]
    relative = min(
        range(len(cumulative)), key=lambda index: abs(cumulative[index] - target)
    )
    return start + relative


def _interpolate_position(
    start: Sequence[float], end: Sequence[float], fraction: float
) -> Tuple[float, float, float]:
    return tuple(
        float(start[axis])
        + float(fraction) * (float(end[axis]) - float(start[axis]))
        for axis in range(3)
    )


def _build_route_points(
    moving: Sequence[TraceSample],
    *,
    path_deviation_m: float,
    max_waypoint_spacing_m: float,
) -> List[Dict[str, object]]:
    max_waypoint_spacing_m = _validate_positive(
        "max_waypoint_spacing_m", max_waypoint_spacing_m
    )
    positions = [sample.position_xyz_m for sample in moving]
    anchors = _rdp_indices(positions, path_deviation_m)
    route: List[Dict[str, object]] = [{
        "position_xyz_m": list(positions[0]),
        "source_step": int(moving[0].step),
    }]
    for anchor_index in range(1, len(anchors)):
        source_start = anchors[anchor_index - 1]
        source_end = anchors[anchor_index]
        start = positions[source_start]
        end = positions[source_end]
        segment_length = _distance_xz(start, end)
        subdivisions = max(1, int(math.ceil(
            segment_length / max_waypoint_spacing_m
        )))
        for subdivision in range(1, subdivisions + 1):
            fraction = subdivision / subdivisions
            source_index = _source_index_for_fraction(
                moving, source_start, source_end, fraction
            )
            route.append({
                "position_xyz_m": list(_interpolate_position(
                    start, end, fraction
                )),
                "source_step": int(moving[source_index].step),
            })
    route[-1]["position_xyz_m"] = list(moving[-1].position_xyz_m)
    route[-1]["source_step"] = int(moving[-1].step)
    return route


def _maximum_deviation_m(
    source: Sequence[Sequence[float]], route: Sequence[Sequence[float]]
) -> float:
    if len(route) <= 1:
        return max((_distance_xz(point, route[0]) for point in source), default=0.0)
    return max(
        min(
            _point_segment_distance_xz(point, route[index - 1], route[index])
            for index in range(1, len(route))
        )
        for point in source
    )


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _heading_hint(start: Sequence[float], end: Sequence[float]) -> float:
    # Habitat world convention documented explicitly in the output: yaw zero
    # is +z and positive yaw turns toward +x.
    return float(math.atan2(
        float(end[0]) - float(start[0]),
        float(end[2]) - float(start[2]),
    ))


def _annotate_route(
    route: List[Dict[str, object]], raw: Sequence[TraceSample]
) -> None:
    step_to_sample = {sample.step: sample for sample in raw}
    previous_step = raw[0].step
    positions = [item["position_xyz_m"] for item in route]
    headings = []
    for index in range(len(route)):
        if len(route) == 1:
            headings.append(0.0)
        elif index + 1 < len(route):
            headings.append(_heading_hint(positions[index], positions[index + 1]))
        else:
            headings.append(headings[-1])
    for index, item in enumerate(route):
        source_step = int(item["source_step"])
        interval = [
            sample
            for sample in raw
            if previous_step <= sample.step <= source_step
        ]
        if not interval:
            interval = [step_to_sample.get(source_step, raw[-1])]
        source_sample = step_to_sample.get(source_step, interval[-1])
        item.update({
            "waypoint_id": int(index),
            "heading_hint_rad": float(headings[index]),
            "risk_at_source_step": float(source_sample.risk),
            "max_source_risk_since_previous_waypoint": float(max(
                sample.risk for sample in interval
            )),
            "hard_unsafe_observed_since_previous_waypoint": bool(any(
                sample.hard_unsafe for sample in interval
            )),
        })
        previous_step = source_step


def _spacing_stats(route_positions: Sequence[Sequence[float]]) -> Dict[str, float]:
    spacing = [
        _distance_xz(route_positions[index - 1], route_positions[index])
        for index in range(1, len(route_positions))
    ]
    if not spacing:
        return {"minimum_m": 0.0, "mean_m": 0.0, "maximum_m": 0.0}
    return {
        "minimum_m": float(min(spacing)),
        "mean_m": float(sum(spacing) / len(spacing)),
        "maximum_m": float(max(spacing)),
    }


def optimize_agent(
    payload: Mapping[str, object],
    agent_id: int,
    *,
    path_deviation_m: float,
    max_waypoint_spacing_m: float,
    stationary_epsilon_m: float,
    position_tolerance_m: float,
    heading_tolerance_deg: float,
    max_linear_speed_m_s: float,
    max_angular_speed_rad_s: float,
) -> Dict[str, object]:
    raw = _extract_agent_samples(payload, agent_id)
    moving = _moving_samples(raw, stationary_epsilon_m)
    route = _build_route_points(
        moving,
        path_deviation_m=path_deviation_m,
        max_waypoint_spacing_m=max_waypoint_spacing_m,
    )
    _annotate_route(route, raw)
    route_positions = [item["position_xyz_m"] for item in route]
    source_positions = [sample.position_xyz_m for sample in moving]
    headings = [float(item["heading_hint_rad"]) for item in route]
    corner_rotation = sum(
        abs(_wrap_angle(headings[index] - headings[index - 1]))
        for index in range(1, len(headings))
    )
    optimized_length = _path_length_xz(route_positions)
    terminal_command = (
        "STOP" if raw[-1].action_name == "stop" else "HOLD"
    )
    commands = []
    for item in route[1:]:
        commands.append({
            "command": "GOTO",
            "waypoint_id": int(item["waypoint_id"]),
            "position_xyz_m": list(item["position_xyz_m"]),
            "heading_hint_rad": float(item["heading_hint_rad"]),
            "position_tolerance_m": float(position_tolerance_m),
            "heading_tolerance_deg": float(heading_tolerance_deg),
            "max_linear_speed_m_s": float(max_linear_speed_m_s),
            "max_angular_speed_rad_s": float(max_angular_speed_rad_s),
        })
    commands.append({
        "command": terminal_command,
        "reason": (
            "source_agent_issued_stop"
            if terminal_command == "STOP"
            else "team_episode_finished_before_agent_stop"
        ),
    })
    return {
        "agent_id": int(agent_id),
        "terminal_command": terminal_command,
        "source_action_count": int(len(raw)),
        "source_moving_pose_count": int(len(moving)),
        "source_path_length_m": float(_path_length_xz(source_positions)),
        "source_max_risk": float(max(sample.risk for sample in raw)),
        "source_hard_unsafe_steps": int(sum(
            sample.hard_unsafe for sample in raw
        )),
        "waypoint_count_including_start": int(len(route)),
        "execution_command_count": int(len(commands)),
        "optimized_path_length_m": float(optimized_length),
        "optimized_to_source_path_length_ratio": float(
            optimized_length / max(_path_length_xz(source_positions), 1e-9)
        ),
        "maximum_source_path_deviation_m": float(_maximum_deviation_m(
            source_positions, route_positions
        )),
        "waypoint_spacing": _spacing_stats(route_positions),
        "estimated_translation_time_s": float(
            optimized_length / max_linear_speed_m_s
        ),
        "estimated_corner_rotation_time_s": float(
            corner_rotation / max_angular_speed_rad_s
        ),
        "route_waypoints": route,
        "commands": commands,
    }


def optimize_payload(
    payload: Mapping[str, object],
    *,
    source_action_list: str,
    path_deviation_m: float = 0.15,
    max_waypoint_spacing_m: float = 0.90,
    stationary_epsilon_m: float = 0.02,
    position_tolerance_m: float = 0.20,
    heading_tolerance_deg: float = 20.0,
    max_linear_speed_m_s: float = 0.30,
    max_angular_speed_rad_s: float = 0.60,
) -> Dict[str, object]:
    for name, value in (
        ("path_deviation_m", path_deviation_m),
        ("max_waypoint_spacing_m", max_waypoint_spacing_m),
        ("stationary_epsilon_m", stationary_epsilon_m),
        ("position_tolerance_m", position_tolerance_m),
        ("heading_tolerance_deg", heading_tolerance_deg),
        ("max_linear_speed_m_s", max_linear_speed_m_s),
        ("max_angular_speed_rad_s", max_angular_speed_rad_s),
    ):
        _validate_positive(name, value)
    num_agents = int(payload.get("num_agents", 0))
    if num_agents <= 0:
        raise ValueError("source action list must contain at least one agent")
    agents = {
        str(agent_id): optimize_agent(
            payload,
            agent_id,
            path_deviation_m=path_deviation_m,
            max_waypoint_spacing_m=max_waypoint_spacing_m,
            stationary_epsilon_m=stationary_epsilon_m,
            position_tolerance_m=position_tolerance_m,
            heading_tolerance_deg=heading_tolerance_deg,
            max_linear_speed_m_s=max_linear_speed_m_s,
            max_angular_speed_rad_s=max_angular_speed_rad_s,
        )
        for agent_id in range(num_agents)
    }
    return {
        "schema_version": 1,
        "artifact_type": "real_robot_waypoint_action_list",
        "source_action_list": str(source_action_list),
        "episode_id": payload.get("episode_id"),
        "scene_id": payload.get("scene_id"),
        "fire_plan_id": payload.get("fire_plan_id"),
        "planner_source": payload.get("planner_source"),
        "num_agents": num_agents,
        "coordinate_frame": {
            "name": "habitat_world_xyz_m",
            "position_order": ["x", "y", "z"],
            "heading_convention": (
                "yaw_rad=atan2(delta_x,delta_z); zero is +z; positive toward +x"
            ),
            "image_overlay": "column=x, row=z; do not swap x/z",
            "open3d_coordinates_used": False,
        },
        "execution_semantics": {
            "streams": "independent_per_agent",
            "start_pose": "route_waypoints[0], not an execution command",
            "goto_behavior": "continuous path following; do not stop at every waypoint",
            "team_finish": "target STOP ends the trial; unfinished agents HOLD",
        },
        "optimizer": {
            "method": "stationary_filter_then_bounded_rdp_then_spacing_cap",
            "path_deviation_m": float(path_deviation_m),
            "max_waypoint_spacing_m": float(max_waypoint_spacing_m),
            "stationary_epsilon_m": float(stationary_epsilon_m),
            "position_tolerance_m": float(position_tolerance_m),
            "heading_tolerance_deg": float(heading_tolerance_deg),
            "max_linear_speed_m_s": float(max_linear_speed_m_s),
            "max_angular_speed_rad_s": float(max_angular_speed_rad_s),
        },
        "agents": agents,
    }


def _sample_route_segments(
    positions: Sequence[Sequence[float]], sample_spacing_m: float
) -> np.ndarray:
    sampled: List[Tuple[float, float, float]] = []
    for index in range(1, len(positions)):
        start, end = positions[index - 1], positions[index]
        count = max(1, int(math.ceil(
            _distance_xz(start, end) / sample_spacing_m
        )))
        if not sampled:
            sampled.append(tuple(map(float, start)))
        sampled.extend(
            _interpolate_position(start, end, subdivision / count)
            for subdivision in range(1, count + 1)
        )
    if not sampled and positions:
        sampled.append(tuple(map(float, positions[0])))
    return np.asarray(sampled, dtype=np.float64)


def validate_navmesh(
    optimized: Mapping[str, object],
    *,
    scene_dataset_config: Path,
    resolution_m: float,
) -> Dict[str, object]:
    """Validate every interpolated route sample on Habitat's 2-D navmesh."""

    import cv2

    from scripts.tune_fire_route_scenarios import _scene_grid

    resolution_m = _validate_positive("navmesh_resolution_m", resolution_m)
    scene_id = str(optimized["scene_id"])
    inventory_path = ROOT / "scenes" / scene_id / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    all_y = [
        float(item["position_xyz_m"][1])
        for agent in optimized["agents"].values()
        for item in agent["route_waypoints"]
    ]
    floor_y_m = float(np.median(all_y))
    grid = _scene_grid(
        inventory,
        scene_dataset_config.resolve(),
        resolution_m,
        floor_y_m,
    )
    try:
        clearance = cv2.distanceTransform(
            grid.traversible.astype(np.uint8), cv2.DIST_L2, 5
        ) * resolution_m
        agents: Dict[str, object] = {}
        total_invalid = 0
        for agent_id, agent in optimized["agents"].items():
            positions = [
                item["position_xyz_m"] for item in agent["route_waypoints"]
            ]
            samples = _sample_route_segments(
                positions, sample_spacing_m=0.5 * resolution_m
            )
            cells = grid.frame.world_to_grid(samples)
            in_bounds = grid.frame.in_bounds(cells)
            valid = np.zeros(len(cells), dtype=bool)
            valid[in_bounds] = grid.traversible[
                cells[in_bounds, 0], cells[in_bounds, 1]
            ]
            valid_clearances = clearance[
                cells[in_bounds, 0], cells[in_bounds, 1]
            ]
            invalid = int(np.count_nonzero(~valid))
            total_invalid += invalid
            agents[str(agent_id)] = {
                "sample_count": int(len(samples)),
                "invalid_sample_count": invalid,
                "minimum_grid_clearance_m": float(
                    np.min(valid_clearances) if len(valid_clearances) else 0.0
                ),
            }
        return {
            "status": "passed" if total_invalid == 0 else "failed",
            "method": "Habitat topdown navmesh sampled along every segment",
            "scene_dataset_config": str(scene_dataset_config),
            "resolution_m": float(resolution_m),
            "floor_y_m": floor_y_m,
            "invalid_sample_count": int(total_invalid),
            "agents": agents,
        }
    finally:
        grid.simulator.close()


def _load_ignitions(payload: Mapping[str, object]) -> List[Sequence[float]]:
    scene_id = payload.get("scene_id")
    plan_id = payload.get("fire_plan_id")
    if not scene_id or not plan_id:
        return []
    plan_path = ROOT / "scenes" / str(scene_id) / "plans" / f"{plan_id}.json"
    if not plan_path.exists():
        return []
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return [item["position"] for item in plan.get("ignitions", [])]


def _attach_scenario_metadata(
    optimized: Dict[str, object], scenario_path: Path
) -> None:
    if not scenario_path.exists():
        return
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    optimized["scenario_metadata"] = {
        key: scenario.get(key)
        for key in (
            "experiment_type",
            "scene_id",
            "episode_id",
            "object_category",
            "objectgoal",
            "dataset_path",
            "plan_id",
        )
        if key in scenario
    }
    optimized["scenario_metadata"]["metadata_path"] = str(scenario_path)


def save_preview(
    source_payload: Mapping[str, object],
    optimized: Mapping[str, object],
    output_path: Path,
) -> None:
    colors = ("#ef6548", "#168ac2", "#4daf4a", "#984ea3")
    num_agents = int(optimized["num_agents"])
    figure, axes = plt.subplots(
        1, num_agents, figsize=(7.2 * num_agents, 6.2), squeeze=False
    )
    ignitions = _load_ignitions(optimized)
    for agent_id in range(num_agents):
        axis = axes[0, agent_id]
        raw = _extract_agent_samples(source_payload, agent_id)
        raw_positions = np.asarray(
            [sample.position_xyz_m for sample in raw], dtype=np.float64
        )
        agent = optimized["agents"][str(agent_id)]
        route = np.asarray(
            [item["position_xyz_m"] for item in agent["route_waypoints"]],
            dtype=np.float64,
        )
        color = colors[agent_id % len(colors)]
        axis.plot(
            raw_positions[:, 0], raw_positions[:, 2],
            color="#a8a8a8", linewidth=1.2, alpha=0.65,
            label=f"raw {len(raw)} step poses",
        )
        axis.plot(
            route[:, 0], route[:, 2], color=color, linewidth=2.8,
            label=f"optimized {len(route)} waypoints",
        )
        axis.scatter(
            route[:, 0], route[:, 2], s=38, color="white",
            edgecolors=color, linewidths=1.8, zorder=4,
        )
        axis.scatter(
            route[0, 0], route[0, 2], marker="o", s=90,
            color="#2ca25f", edgecolors="black", zorder=5, label="start",
        )
        axis.scatter(
            route[-1, 0], route[-1, 2], marker="*", s=170,
            color="#ffd92f", edgecolors="black", zorder=5, label="end",
        )
        if ignitions:
            fire = np.asarray(ignitions, dtype=np.float64)
            axis.scatter(
                fire[:, 0], fire[:, 2], marker="X", s=85,
                color="#d7191c", edgecolors="#7f0000", zorder=3,
                label="ignition",
            )
        axis.set_title(
            f"Agent {agent_id}: {len(raw)} actions -> "
            f"{len(agent['commands'])} robot commands"
        )
        axis.set_xlabel("Habitat world x (m) / image column")
        axis.set_ylabel("Habitat world z (m) / image row")
        axis.set_aspect("equal", adjustable="datalim")
        axis.invert_yaxis()
        axis.grid(True, alpha=0.22)
        axis.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"{optimized.get('scene_id')} episode {optimized.get('episode_id')} "
        "real-robot waypoint optimization",
        fontsize=14,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    source_path = (ROOT / args.input).resolve()
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    optimized = optimize_payload(
        source_payload,
        source_action_list=str(args.input),
        path_deviation_m=args.path_deviation_m,
        max_waypoint_spacing_m=args.max_waypoint_spacing_m,
        stationary_epsilon_m=args.stationary_epsilon_m,
        position_tolerance_m=args.position_tolerance_m,
        heading_tolerance_deg=args.heading_tolerance_deg,
        max_linear_speed_m_s=args.max_linear_speed_m_s,
        max_angular_speed_rad_s=args.max_angular_speed_rad_s,
    )
    output_dir = (ROOT / args.output_dir).resolve()
    scenario_path = (
        (ROOT / args.scenario_json).resolve()
        if args.scenario_json
        else output_dir.parent / "scenario.json"
    )
    _attach_scenario_metadata(optimized, scenario_path)
    if args.scene_dataset_config:
        optimized["navmesh_validation"] = validate_navmesh(
            optimized,
            scene_dataset_config=(ROOT / args.scene_dataset_config).resolve(),
            resolution_m=args.navmesh_resolution_m,
        )
        if optimized["navmesh_validation"]["status"] != "passed":
            raise RuntimeError(
                "optimized path left the sampled Habitat navmesh; "
                "reduce --path-deviation-m"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "optimized_action_list.json"
    output_png = output_dir / "optimized_trajectory_preview.png"
    output_json.write_text(
        json.dumps(optimized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_preview(source_payload, optimized, output_png)
    print(f"[robot-actions] json={output_json}")
    print(f"[robot-actions] preview={output_png}")
    for agent_id, agent in optimized["agents"].items():
        print(
            f"[robot-actions] agent={agent_id} "
            f"source_actions={agent['source_action_count']} "
            f"moving_poses={agent['source_moving_pose_count']} "
            f"waypoints={agent['waypoint_count_including_start']} "
            f"commands={agent['execution_command_count']} "
            f"max_deviation_m={agent['maximum_source_path_deviation_m']:.3f} "
            f"max_spacing_m={agent['waypoint_spacing']['maximum_m']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
