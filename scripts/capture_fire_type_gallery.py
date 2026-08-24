#!/usr/bin/env python3
"""Capture publication-quality FireSensor observations for four fire types.

The script selects a navigable camera pose near a semantic ignition source,
uses a lightweight preview render to find a frame with both visible flame and
smoke, and then performs one full-resolution CUDA ray-march.  It writes every
FireSensor modality, compressed raw arrays, a labelled dashboard, and two 2x2
paper montages.

The default scenarios intentionally use four different HM3D scenes so that the
gallery illustrates fire *types*, rather than four variants of one apartment.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


FIRE_TYPE_LABELS = {
    "kitchen_grease_fire": "Kitchen Grease Fire",
    "bedroom_textile": "Bedroom Textile Fire",
    "living_room_electric": "Living-room Electrical Fire",
    "multi_origin": "Multi-origin Fire",
}


@dataclass(frozen=True)
class Scenario:
    fire_type: str
    scene_id: str
    plan_id: str

    @property
    def display_name(self) -> str:
        return FIRE_TYPE_LABELS.get(
            self.fire_type, self.fire_type.replace("_", " ").title()
        )


DEFAULT_SCENARIOS = (
    Scenario(
        "kitchen_grease_fire",
        "4ok3usBNeis",
        "4ok3usBNeis_kitchen_grease_fire_paper_spread_aabb550436d2",
    ),
    Scenario(
        "bedroom_textile",
        "mL8ThkuaVTM",
        "mL8ThkuaVTM_bedroom_textile_paper_spread_3891d6192739",
    ),
    Scenario(
        "living_room_electric",
        "QaLdnwvtxbs",
        "QaLdnwvtxbs_living_room_electric_paper_spread_47767285ec66",
    ),
    Scenario(
        "multi_origin",
        "Dd4bFSTQ8gi",
        "Dd4bFSTQ8gi_multi_origin_paper_spread_0a15e5bd69f9",
    ),
)


def parse_scenario(value: str) -> Scenario:
    """Parse ``FIRE_TYPE:SCENE_ID:PLAN_ID`` from the CLI."""

    parts = value.split(":", 2)
    if len(parts) != 3 or any(not item.strip() for item in parts):
        raise argparse.ArgumentTypeError(
            "scenario must be FIRE_TYPE:SCENE_ID:PLAN_ID"
        )
    scenario = Scenario(*(item.strip() for item in parts))
    if scenario.fire_type not in FIRE_TYPE_LABELS:
        raise argparse.ArgumentTypeError(
            f"unknown fire type {scenario.fire_type!r}; expected one of "
            f"{', '.join(FIRE_TYPE_LABELS)}"
        )
    return scenario


def balanced_visibility_score(outputs: Mapping[str, np.ndarray]) -> Dict[str, float]:
    """Score a preview that shows flame and smoke without hiding the room."""

    flame = np.asarray(outputs["thermal_flame_mask"], dtype=np.float32)
    trans = np.asarray(outputs["transmittance"], dtype=np.float32)
    rgb = np.asarray(outputs["rgb_smoke"], dtype=np.float32)
    visible = flame > 0.03
    flame_fraction = float(np.mean(visible))
    smoke_fraction = float(np.mean(trans < 0.94))
    attenuation = float(np.mean(1.0 - np.clip(trans, 0.0, 1.0)))
    luminance = float(np.mean(rgb) / 255.0)
    depth_clean = outputs.get("depth_clean")
    if depth_clean is None:
        near_obstacle_fraction = 0.0
    else:
        depth_array = np.asarray(depth_clean, dtype=np.float32)
        if depth_array.ndim == 3:
            depth_array = depth_array[..., 0]
        near_obstacle_fraction = float(np.mean(depth_array < 0.8))

    # The thermal mask alone can rate a fire hidden behind a wall or visible
    # only as a sliver very highly.  Require actual orange/yellow emissive
    # pixels in the RGB observation as the publication view's primary signal.
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    fire_colored = (
        visible
        & (red > green + 5.0)
        & (green > blue + 8.0)
        & (red > 90.0)
    )
    rgb_fire_fraction = float(np.mean(fire_colored))
    rgb_fire_ratio = float(
        np.count_nonzero(fire_colored) / max(1, np.count_nonzero(visible))
    )

    central = np.zeros_like(visible)
    height, width = visible.shape
    central[
        int(round(height * 0.08)) : int(round(height * 0.96)),
        int(round(width * 0.12)) : int(round(width * 0.88)),
    ] = True
    central_flame_ratio = float(
        np.count_nonzero(visible & central) / max(1, np.count_nonzero(visible))
    )

    if np.any(visible):
        yy, xx = np.nonzero(visible)
        cy = float(np.mean(yy) / max(1, visible.shape[0] - 1))
        cx = float(np.mean(xx) / max(1, visible.shape[1] - 1))
        center_distance = math.hypot(cx - 0.5, cy - 0.55)
    else:
        center_distance = 1.0

    # Best paper frames usually devote 0.5--10% of the pixels to flame and
    # retain enough clean surface texture to identify the burning object.
    flame_reward = min(flame_fraction / 0.025, 1.0)
    smoke_reward = min(attenuation / 0.12, 1.0)
    score = (
        5.0 * flame_reward
        + 2.2 * smoke_reward
        + 0.8 * min(smoke_fraction / 0.25, 1.0)
        + 0.8 * min(luminance / 0.32, 1.0)
        + 3.5 * min(rgb_fire_fraction / 0.025, 1.0)
        + 6.0 * min(rgb_fire_fraction / 0.12, 1.0)
        + 2.0 * min(rgb_fire_ratio / 0.55, 1.0)
        + 1.5 * central_flame_ratio
        - 4.5 * center_distance
        - 3.0 * max(0.0, 0.78 - central_flame_ratio)
        - 9.0 * near_obstacle_fraction
        - 25.0 * max(0.0, flame_fraction - 0.16)
        - 12.0 * max(0.0, attenuation - 0.58)
    )
    if flame_fraction < 0.0002:
        score -= 8.0
    return {
        "score": float(score),
        "flame_fraction": flame_fraction,
        "smoke_fraction": smoke_fraction,
        "mean_smoke_attenuation": attenuation,
        "mean_luminance": luminance,
        "flame_center_distance": center_distance,
        "rgb_fire_fraction": rgb_fire_fraction,
        "rgb_fire_ratio": rgb_fire_ratio,
        "central_flame_ratio": central_flame_ratio,
        "near_obstacle_fraction": near_obstacle_fraction,
    }


def _save_pose_search_preview(
    path: Path,
    candidates: Sequence[Tuple[float, np.ndarray, Mapping[str, Any], float, Mapping[str, float]]],
) -> None:
    """Save a labelled contact sheet of the strongest preview viewpoints."""

    if not candidates:
        return
    tiles = []
    for score, image_rgb, ignition, time_s, metrics in candidates[:12]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, (320, 240), interpolation=cv2.INTER_AREA)
        tile = np.full((282, 320, 3), 245, dtype=np.uint8)
        tile[42:] = image_bgr
        cv2.putText(
            tile,
            f"{ignition.get('category')}  score={score:.2f}",
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"t={time_s:.0f}s rgb-fire={metrics['rgb_fire_fraction']:.3f}",
            (8, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (55, 55, 55),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    while len(tiles) < 12:
        tiles.append(np.full_like(tiles[0], 245))
    sheet = np.vstack(
        [np.hstack(tiles[row : row + 4]) for row in range(0, 12, 4)]
    )
    _write_image(path, sheet)


def _cluster_ignitions(
    ignitions: Sequence[Mapping[str, Any]],
    *,
    horizontal_m: float = 1.35,
    vertical_m: float = 1.2,
) -> List[Mapping[str, Any]]:
    """Keep one representative from each spatial ignition cluster."""

    selected: List[Mapping[str, Any]] = []
    for ignition in sorted(
        ignitions,
        key=lambda item: (
            -float(item.get("source_radius_m", 0.0)),
            -float(item.get("fuel_kg", 0.0)),
        ),
    ):
        point = np.asarray(ignition["position"], dtype=np.float64)
        is_new = True
        for prior in selected:
            other = np.asarray(prior["position"], dtype=np.float64)
            if (
                np.linalg.norm(point[[0, 2]] - other[[0, 2]]) <= horizontal_m
                and abs(float(point[1] - other[1])) <= vertical_m
            ):
                is_new = False
                break
        if is_new:
            selected.append(ignition)
    return selected


def active_ignitions(
    plan: Mapping[str, Any], time_s: float
) -> List[Mapping[str, Any]]:
    """Return sources that have activated by one paper-sequence time."""

    return [
        ignition
        for ignition in plan.get("ignitions", ())
        if float(ignition.get("ignite_time_s", 0.0)) <= float(time_s)
    ]


def _dominant_spread_chain(
    plan: Mapping[str, Any],
) -> Tuple[List[Mapping[str, Any]], np.ndarray]:
    """Return the largest authored parent chain and its visual centroid."""

    ignitions = list(plan.get("ignitions", ()))
    by_id = {int(item["object_id"]): item for item in ignitions}
    children: Dict[int, List[int]] = {object_id: [] for object_id in by_id}
    roots = []
    for ignition in ignitions:
        object_id = int(ignition["object_id"])
        parent = ignition.get("parent_object_id")
        if parent is None:
            roots.append(object_id)
        elif int(parent) in children:
            children[int(parent)].append(object_id)

    def descendants(root_id: int) -> List[int]:
        result = []
        stack = [root_id]
        while stack:
            current = stack.pop()
            if current in result:
                continue
            result.append(current)
            stack.extend(children.get(current, ()))
        return result

    groups = [descendants(root) for root in roots]
    if not groups:
        groups = [[int(item["object_id"])] for item in ignitions]
    groups.sort(
        key=lambda group: (
            len(group),
            max(float(by_id[item]["ignite_time_s"]) for item in group),
        ),
        reverse=True,
    )
    chain = [by_id[object_id] for object_id in groups[0]]
    positions = np.asarray(
        [item["position"] for item in chain], dtype=np.float64
    )
    # X/Z centroid frames the whole chain. Median Y avoids a tall curtain or
    # lamp pulling a level camera above the burning furniture.
    target = np.array(
        [
            float(np.mean(positions[:, 0])),
            float(np.median(positions[:, 1])),
            float(np.mean(positions[:, 2])),
        ],
        dtype=np.float64,
    )
    return chain, target


def _candidate_positions(pathfinder, source: Sequence[float]) -> List[np.ndarray]:
    """Generate unique navigable ring poses around an ignition source."""

    source = np.asarray(source, dtype=np.float64)
    snapped_source = np.asarray(pathfinder.snap_point(source), dtype=np.float64)
    if snapped_source.shape != (3,) or not np.all(np.isfinite(snapped_source)):
        return []

    candidates: List[np.ndarray] = []
    seen = set()
    for radius in (1.25, 1.7, 2.2, 2.8, 3.5):
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            probe = np.array(
                [
                    source[0] + radius * math.cos(float(angle)),
                    snapped_source[1],
                    source[2] + radius * math.sin(float(angle)),
                ],
                dtype=np.float64,
            )
            snapped = np.asarray(pathfinder.snap_point(probe), dtype=np.float64)
            if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
                continue
            distance = float(np.linalg.norm(snapped[[0, 2]] - source[[0, 2]]))
            if distance < 0.85 or distance > 4.2:
                continue
            if abs(float(snapped[1] - snapped_source[1])) > 0.55:
                continue
            key = tuple(np.round(snapped, 2))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(snapped)

    # Prefer the 1.5--2.8 m band, but retain farther views for large fires.
    candidates.sort(
        key=lambda point: abs(
            float(np.linalg.norm(point[[0, 2]] - source[[0, 2]])) - 2.1
        )
    )
    return candidates[:18]


def _yaw_quaternion_facing(
    source: Sequence[float], target: Sequence[float]
) -> List[float]:
    source_arr = np.asarray(source, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    dx = float(target_arr[0] - source_arr[0])
    dz = float(target_arr[2] - source_arr[2])
    yaw = math.atan2(-dx, -dz)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _extract_visual_observation(sim, agent_id: int = 0) -> Dict[str, np.ndarray]:
    observations = sim.step(None)
    if isinstance(observations, list):
        return dict(observations[agent_id])
    return dict(observations)


def _metric_depth(observation: Mapping[str, np.ndarray], max_depth_m: float) -> np.ndarray:
    depth = np.asarray(observation["depth"])
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype(np.float32, copy=False)
    finite = depth[np.isfinite(depth)]
    if finite.size and float(finite.max()) <= 1.0 + 1e-3:
        depth = depth * float(max_depth_m)
    return depth


def _make_suite(
    *,
    scene,
    camera_k,
    output_dir: Path,
    seed: int,
    max_depth_m: float,
    hfov_deg: float,
    n_steps: int,
    render_scale: float,
    device: str,
    procedural: bool,
):
    from utils.fire_sensors import FireSensorConfig, FireSensorSuite
    from utils.fire_sensors.config import VoxelSmokeConfig

    voxel = VoxelSmokeConfig(
        n_steps=int(n_steps),
        render_scale=float(render_scale),
        render_backend="torch",
        render_device=str(device),
        render_dtype="float16",
        max_sample_points=2_000_000,
        flame_noise_strength=0.75 if procedural else 0.0,
        flame_edge_break=1.05 if procedural else 0.0,
        flame_color_jitter=0.32 if procedural else 0.0,
        flame_time_speed=12.0,
        smoke_noise_strength=0.24 if procedural else 0.0,
    )
    config = FireSensorConfig(
        max_depth_m=float(max_depth_m),
        hfov_deg=float(hfov_deg),
        smoke_density=0.6,
        save_npz=False,
        save_dashboard=True,
        dashboard_size=(2000, 900),
        voxel=voxel,
    )
    return FireSensorSuite(
        cfg=config,
        dump_dir=str(output_dir),
        save_every=0,
        seed=int(seed),
        scene=scene,
        camera_K=camera_k,
    )


def _process_pose(
    *,
    sim,
    suite,
    position: Sequence[float],
    source: Sequence[float],
    time_s: float,
    max_depth_m: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rotation = _yaw_quaternion_facing(position, source)
    success = sim.set_agent_state(position, rotation, agent_id=0)
    if success is False:
        raise RuntimeError(f"simulator rejected camera position {position}")
    observation = _extract_visual_observation(sim)
    rgb = np.asarray(observation["rgb"])[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    depth_m = _metric_depth(observation, max_depth_m)
    outputs = suite.process(
        rgb,
        depth_m,
        obs=observation,
        agent_state=sim.get_agent_state(0),
        robot_step=0,
        t_sim_s=float(time_s),
    )
    return outputs, observation


def _search_pose_and_time(
    *,
    sim,
    scene,
    plan: Mapping[str, Any],
    camera_k,
    output_dir: Path,
    seed: int,
    max_depth_m: float,
    hfov_deg: float,
    device: str,
    preview_steps: int,
    preview_scale: float,
) -> Tuple[np.ndarray, Mapping[str, Any], float, Dict[str, float]]:
    preview = _make_suite(
        scene=scene,
        camera_k=camera_k,
        output_dir=output_dir,
        seed=seed,
        max_depth_m=max_depth_m,
        hfov_deg=hfov_deg,
        n_steps=preview_steps,
        render_scale=preview_scale,
        device=device,
        procedural=False,
    )
    timeline_end = float(scene.fw.times[-1])
    preview_time = min(90.0, timeline_end * 0.4)
    best: Tuple[float, np.ndarray, Mapping[str, Any], Dict[str, float]] | None = None
    representatives = _cluster_ignitions(plan["ignitions"])
    print(
        f"[gallery] previewing {len(representatives)} ignition cluster(s) "
        f"at t={preview_time:.1f}s"
    )
    ranked_previews: List[
        Tuple[float, np.ndarray, Mapping[str, Any], float, Mapping[str, float]]
    ] = []
    for source_index, ignition in enumerate(representatives):
        candidates = _candidate_positions(sim.pathfinder, ignition["position"])
        for candidate in candidates:
            outputs, _ = _process_pose(
                sim=sim,
                suite=preview,
                position=candidate,
                source=ignition["position"],
                time_s=preview_time,
                max_depth_m=max_depth_m,
            )
            metrics = balanced_visibility_score(outputs)
            ranked_previews.append(
                (
                    metrics["score"],
                    np.asarray(outputs["rgb_smoke"]).copy(),
                    ignition,
                    preview_time,
                    metrics,
                )
            )
            ranked_previews.sort(key=lambda item: item[0], reverse=True)
            del ranked_previews[12:]
            if best is None or metrics["score"] > best[0]:
                best = (metrics["score"], candidate.copy(), ignition, metrics)
        print(
            f"[gallery] source {source_index + 1}/{len(representatives)} "
            f"category={ignition.get('category')} candidates={len(candidates)}"
        )
    if best is None:
        raise RuntimeError("no navigable camera pose found near any ignition")

    _, best_position, best_ignition, _ = best
    time_candidates = sorted(
        {
            min(timeline_end, max(0.0, value))
            for value in (30.0, 60.0, 90.0, 150.0, 240.0)
        }
    )
    best_time = preview_time
    best_metrics: Dict[str, float] | None = None
    for time_s in time_candidates:
        outputs, _ = _process_pose(
            sim=sim,
            suite=preview,
            position=best_position,
            source=best_ignition["position"],
            time_s=time_s,
            max_depth_m=max_depth_m,
        )
        metrics = balanced_visibility_score(outputs)
        ranked_previews.append(
            (
                metrics["score"],
                np.asarray(outputs["rgb_smoke"]).copy(),
                best_ignition,
                float(time_s),
                metrics,
            )
        )
        ranked_previews.sort(key=lambda item: item[0], reverse=True)
        del ranked_previews[12:]
        if best_metrics is None or metrics["score"] > best_metrics["score"]:
            best_time = float(time_s)
            best_metrics = metrics
    assert best_metrics is not None
    _save_pose_search_preview(
        output_dir / "pose_search_preview.jpg", ranked_previews
    )
    return best_position, best_ignition, best_time, best_metrics


def _search_sequence_pose(
    *,
    sim,
    scene,
    plan: Mapping[str, Any],
    camera_k,
    output_dir: Path,
    seed: int,
    max_depth_m: float,
    hfov_deg: float,
    device: str,
    preview_steps: int,
    preview_scale: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Mapping[str, Any],
    List[float],
    Dict[str, Any],
]:
    """Select one fixed camera that visibly preserves early-to-late growth."""

    raw_times = plan.get("paper_spread_sequence_times_s")
    if not isinstance(raw_times, list) or len(raw_times) != 3:
        raise ValueError(
            "staged spread plans require three paper_spread_sequence_times_s"
        )
    sequence_times = [float(value) for value in raw_times]
    chain, target = _dominant_spread_chain(plan)
    preview = _make_suite(
        scene=scene,
        camera_k=camera_k,
        output_dir=output_dir,
        seed=seed,
        max_depth_m=max_depth_m,
        hfov_deg=hfov_deg,
        n_steps=preview_steps,
        render_scale=preview_scale,
        device=device,
        procedural=False,
    )
    candidates = _candidate_positions(sim.pathfinder, target)
    if not candidates:
        candidates = _candidate_positions(
            sim.pathfinder, chain[0]["position"]
        )
    if not candidates:
        raise RuntimeError("no navigable camera pose found around spread chain")

    print(
        f"[gallery] sequence chain={len(chain)} source(s) "
        f"target={np.round(target, 2).tolist()} times={sequence_times}"
    )
    late_ranked = []
    late_time = sequence_times[-1]
    for position in candidates:
        outputs, _ = _process_pose(
            sim=sim,
            suite=preview,
            position=position,
            source=target,
            time_s=late_time,
            max_depth_m=max_depth_m,
        )
        metrics = balanced_visibility_score(outputs)
        late_ranked.append((metrics["score"], position.copy(), metrics))
    late_ranked.sort(key=lambda item: item[0], reverse=True)

    best = None
    preview_sheet_entries = []
    for _, position, _ in late_ranked[: min(7, len(late_ranked))]:
        stage_metrics = []
        stage_images = []
        for time_s in sequence_times:
            outputs, _ = _process_pose(
                sim=sim,
                suite=preview,
                position=position,
                source=target,
                time_s=time_s,
                max_depth_m=max_depth_m,
            )
            stage_metrics.append(balanced_visibility_score(outputs))
            stage_images.append(np.asarray(outputs["rgb_smoke"]).copy())
        early, middle, late = stage_metrics
        rgb_growth = late["rgb_fire_fraction"] - early["rgb_fire_fraction"]
        flame_growth = late["flame_fraction"] - early["flame_fraction"]
        combined = (
            0.15 * early["score"]
            + 0.35 * middle["score"]
            + 0.50 * late["score"]
            + 12.0 * max(0.0, rgb_growth)
            + 6.0 * max(0.0, flame_growth)
        )
        if early["flame_fraction"] < 0.0002:
            combined -= 4.0
        if late["rgb_fire_fraction"] <= early["rgb_fire_fraction"]:
            combined -= 1.5
        record = {
            "score": float(combined),
            "rgb_fire_growth": float(rgb_growth),
            "flame_growth": float(flame_growth),
            "stage_metrics": stage_metrics,
            "dominant_chain_object_ids": [
                int(item["object_id"]) for item in chain
            ],
            "dominant_chain_categories": [
                str(item["category"]) for item in chain
            ],
        }
        preview_sheet_entries.append(
            (
                float(combined),
                stage_images[-1],
                chain[-1],
                late_time,
                late,
            )
        )
        if best is None or combined > best[0]:
            best = (float(combined), position.copy(), record)

    assert best is not None
    preview_sheet_entries.sort(key=lambda item: item[0], reverse=True)
    _save_pose_search_preview(
        output_dir / "pose_search_preview.jpg", preview_sheet_entries
    )
    return (
        best[1],
        target,
        chain[0],
        sequence_times,
        best[2],
    )


def _write_image(path: Path, image: np.ndarray, *, rgb: bool = False) -> None:
    array = np.asarray(image)
    if rgb:
        array = cv2.cvtColor(np.ascontiguousarray(array[..., :3]), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), array):
        raise RuntimeError(f"failed to write {path}")


def _compose_labelled_dashboard(
    outputs: Mapping[str, np.ndarray],
    *,
    scenario: Scenario,
    plan: Mapping[str, Any],
    source: Mapping[str, Any],
    time_s: float,
    n_steps: int,
    render_scale: float,
    max_depth_m: float,
    stage_label: str = "",
) -> np.ndarray:
    from scripts.keyboard_teleop_full import compose_view

    active = active_ignitions(plan, time_s)
    active_generations = sorted(
        {int(item.get("spread_generation", 0)) for item in active}
    )
    stage_prefix = f"Stage: {stage_label}    " if stage_label else ""
    status = [
        f"Scene: {scenario.scene_id}    Type: {scenario.display_name}",
        f"Intensity: {plan.get('intensity')}    Active sources: "
        f"{len(active)}/{len(plan.get('ignitions', []))}    "
        f"Spread generations: {active_generations}",
        f"{stage_prefix}Fire time: {time_s:.1f}s    Renderer: Torch CUDA, "
        f"scale={render_scale:.2f}, samples={n_steps}, procedural texture=ON",
    ]
    return compose_view(
        rgb_clean=outputs["rgb"],
        rgb_smoke=outputs["rgb_smoke"],
        depth_clean=outputs["depth_clean"],
        depth_smoke=outputs["depth_smoke"],
        thermal=outputs["thermal_image"],
        lidar=outputs["lidar_image"],
        radar_bev=outputs["radar_image_bev"],
        radar_az=outputs["radar_image_az"],
        radar_el=outputs["radar_image_el"],
        status_lines=status,
        max_d=max_depth_m,
        dashboard_size=(2000, 900),
        title=f"FireSensor Observation | {scenario.display_name}",
    )


def _save_observation(
    *,
    output_dir: Path,
    outputs: Mapping[str, np.ndarray],
    dashboard: np.ndarray,
) -> List[str]:
    from utils.fire_sensors.dashboard import colorize_depth

    output_dir.mkdir(parents=True, exist_ok=True)
    max_depth = float(outputs["sensor_max_depth_m"])
    products = {
        "rgb_clean.png": (outputs["rgb"], True),
        "rgb_fire_smoke.png": (outputs["rgb_smoke"], True),
        "depth_clean.png": (colorize_depth(outputs["depth_clean"], max_depth), False),
        "depth_smoke.png": (colorize_depth(outputs["depth_smoke"], max_depth), False),
        "thermal.png": (outputs["thermal_image"], False),
        "lidar_bev.png": (outputs["lidar_image"], False),
        "radar_bev.png": (outputs["radar_image_bev"], False),
        "radar_range_azimuth.png": (outputs["radar_image_az"], False),
        "radar_range_elevation.png": (outputs["radar_image_el"], False),
        "sensor_dashboard.png": (dashboard, False),
    }
    for filename, (image, rgb) in products.items():
        _write_image(output_dir / filename, image, rgb=rgb)

    np.savez_compressed(
        output_dir / "fire_sensor_arrays.npz",
        rgb=np.asarray(outputs["rgb"]),
        rgb_fire_smoke=np.asarray(outputs["rgb_smoke"]),
        depth_clean=np.asarray(outputs["depth_clean"], dtype=np.float32),
        depth_smoke=np.asarray(outputs["depth_smoke"], dtype=np.float32),
        transmittance=np.asarray(outputs["transmittance"], dtype=np.float32),
        thermal_temperature=np.asarray(
            outputs["thermal_temperature"], dtype=np.float32
        ),
        thermal_flame_mask=np.asarray(
            outputs["thermal_flame_mask"], dtype=np.float32
        ),
        lidar_points=np.asarray(outputs["lidar_points"], dtype=np.float32),
        radar_heatmap=np.asarray(outputs["radar_heatmap"], dtype=np.float32),
        radar_points=np.asarray(outputs["radar_points"], dtype=np.float32),
        radar_points_3d=np.asarray(outputs["radar_points_3d"], dtype=np.float32),
    )
    return [*products, "fire_sensor_arrays.npz"]


def _label_tile(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    image = np.asarray(image)
    bar_h = 72
    canvas = np.full((image.shape[0] + bar_h, image.shape[1], 3), 248, np.uint8)
    canvas[bar_h:] = image[..., :3]
    cv2.putText(
        canvas,
        title,
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (22, 22, 22),
        2,
        cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            canvas,
            subtitle,
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
    return canvas


def make_montage(
    items: Sequence[Tuple[np.ndarray, str, str]],
    *,
    tile_size: Tuple[int, int],
) -> np.ndarray:
    """Return a 2x2 BGR montage with fixed-size labelled tiles."""

    if len(items) != 4:
        raise ValueError("a fire-type montage requires exactly four items")
    width, height = tile_size
    tiles = []
    for image, title, subtitle in items:
        resized = cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_AREA)
        tiles.append(_label_tile(resized, title, subtitle))
    return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))


def make_spread_sequence_montage(
    rows: Sequence[Sequence[Tuple[np.ndarray, str, str]]],
    *,
    tile_size: Tuple[int, int] = (480, 360),
) -> np.ndarray:
    """Compose one three-stage row per fire type."""

    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("spread montage needs one or more three-frame rows")
    width, height = tile_size
    rendered_rows = []
    for row in rows:
        tiles = []
        for image, title, subtitle in row:
            resized = cv2.resize(
                np.asarray(image), (width, height), interpolation=cv2.INTER_AREA
            )
            tiles.append(_label_tile(resized, title, subtitle))
        rendered_rows.append(np.hstack(tiles))
    return np.vstack(rendered_rows)


def _load_config(args: argparse.Namespace):
    from arguments import load_config

    namespace = argparse.Namespace(
        task_config=args.task_config.replace("configs/", ""),
        config=None,
        num_agents=1,
        num_humans=0,
        gpu_id=args.gpu_id,
        seed=args.seed,
        robot_models_enabled=0,
        robot_profiles=None,
        robot_urdfs=None,
        dataset_path=None,
        scenes_dir=None,
        scene_dataset=None,
        frame_width=640,
        frame_height=480,
        hfov=79.0,
        turn_angle=30,
    )
    return load_config(namespace)


def _load_plan(scenario: Scenario, scenes_root: Path) -> Dict[str, Any]:
    path = scenes_root / scenario.scene_id / "plans" / f"{scenario.plan_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("scene_id") != scenario.scene_id:
        raise ValueError(f"scene mismatch in {path}")
    if plan.get("plan_id") != scenario.plan_id:
        raise ValueError(f"plan ID mismatch in {path}")
    if plan.get("fire_type") != scenario.fire_type:
        raise ValueError(f"fire type mismatch in {path}")
    if not plan.get("ignitions"):
        raise ValueError(f"no ignitions in {path}")
    return plan


def run(args: argparse.Namespace) -> None:
    import habitat
    from habitat import Env

    from utils.fire_sensors.lidar_360 import install_lidar_depth_sensors
    from utils.fire_world.runtime import FireWorld
    from utils.fire_world.scene import FireClock, FireScene
    from utils.general_utils import get_camera_K

    scenarios = tuple(args.scenario or DEFAULT_SCENARIOS)
    if len(scenarios) != 4 or {item.fire_type for item in scenarios} != set(
        FIRE_TYPE_LABELS
    ):
        raise ValueError("select exactly one scenario for each of the four fire types")

    scenes_root = Path(args.scenes_root).resolve()
    fire_root = Path(args.fire_world_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(args)
    with habitat.config.read_write(config):
        install_lidar_depth_sensors(
            config,
            resolution=int(args.lidar_resolution),
            num_agents=1,
        )
    env = Env(config=config)

    main_agent = config.habitat.simulator.agents_order[0]
    depth_cfg = config.habitat.simulator.agents[main_agent].sim_sensors.depth_sensor
    rgb_cfg = config.habitat.simulator.agents[main_agent].sim_sensors.rgb_sensor
    max_depth_m = float(depth_cfg.max_depth)
    hfov_deg = float(rgb_cfg.hfov)
    camera_k = get_camera_K(int(rgb_cfg.width), int(rgb_cfg.height), hfov_deg)

    gallery_manifest: Dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "purpose": (
            "publication-quality early-middle-late observations of four "
            "staged FireWorld source-to-source spread scenarios"
        ),
        "spread_disclosure": (
            "Secondary source activation times and parent links are authored "
            "for a controlled publication sequence; they are not claimed as "
            "solver-inferred causal attribution."
        ),
        "renderer": {
            "backend": "torch",
            "device_requested": args.device,
            "resolution": [int(rgb_cfg.width), int(rgb_cfg.height)],
            "render_scale": float(args.render_scale),
            "ray_march_samples": int(args.n_steps),
            "render_dtype": "float16",
            "procedural_flame_texture": True,
            "flame_noise_strength": 0.75,
            "flame_edge_break": 1.05,
            "flame_color_jitter": 0.32,
            "smoke_noise_strength": 0.24,
        },
        "scenarios": [],
    }
    rgb_tiles = []
    dashboard_tiles = []
    spread_rows = []

    try:
        for order, scenario in enumerate(scenarios, start=1):
            print(
                f"[gallery] {order}/4 {scenario.display_name}: "
                f"scene={scenario.scene_id} plan={scenario.plan_id}"
            )
            plan = _load_plan(scenario, scenes_root)
            timeline = fire_root / scenario.scene_id / scenario.plan_id / "timeline.npz"
            if not timeline.exists():
                raise FileNotFoundError(
                    f"missing {timeline}; bake it with utils.fire_world.propagation"
                )

            matching = [
                episode
                for episode in env.episodes
                if scenario.scene_id in str(episode.scene_id)
            ]
            if not matching:
                raise RuntimeError(
                    f"the configured ObjectNav dataset has no episode for {scenario.scene_id}"
                )
            env.current_episode = matching[0]
            env.reset()

            fw = FireWorld.load(
                scenario.scene_id,
                scenario.plan_id,
                out_root=fire_root,
            )
            fire_scene = FireScene(fw=fw, clock=FireClock(mode="step"))
            scenario_dir = output_root / f"{order:02d}_{scenario.fire_type}_{scenario.scene_id}"
            scenario_dir.mkdir(parents=True, exist_ok=True)

            is_spread_sequence = bool(
                plan.get("paper_spread_sequence_times_s")
            )
            if is_spread_sequence:
                (
                    position,
                    view_target,
                    source,
                    sequence_times,
                    preview_metrics,
                ) = _search_sequence_pose(
                    sim=env.sim,
                    scene=fire_scene,
                    plan=plan,
                    camera_k=camera_k,
                    output_dir=scenario_dir,
                    seed=args.seed + order,
                    max_depth_m=max_depth_m,
                    hfov_deg=hfov_deg,
                    device=args.device,
                    preview_steps=args.preview_steps,
                    preview_scale=args.preview_scale,
                )
                time_s = float(sequence_times[-1])
            else:
                position, source, time_s, preview_metrics = _search_pose_and_time(
                    sim=env.sim,
                    scene=fire_scene,
                    plan=plan,
                    camera_k=camera_k,
                    output_dir=scenario_dir,
                    seed=args.seed + order,
                    max_depth_m=max_depth_m,
                    hfov_deg=hfov_deg,
                    device=args.device,
                    preview_steps=args.preview_steps,
                    preview_scale=args.preview_scale,
                )
                view_target = np.asarray(source["position"], dtype=np.float64)
                sequence_times = [float(time_s)]
            print(
                f"[gallery] selected category={source.get('category')} "
                f"pose={np.round(position, 3).tolist()} "
                f"target={np.round(view_target, 3).tolist()} "
                f"times={sequence_times} preview={preview_metrics}"
            )

            final_suite = _make_suite(
                scene=fire_scene,
                camera_k=camera_k,
                output_dir=scenario_dir,
                seed=args.seed + 100 + order,
                max_depth_m=max_depth_m,
                hfov_deg=hfov_deg,
                n_steps=args.n_steps,
                render_scale=args.render_scale,
                device=args.device,
                procedural=True,
            )
            stage_names = (
                ["early", "middle", "late"]
                if is_spread_sequence
                else ["representative"]
            )
            sequence_records = []
            sequence_row = []
            outputs = None
            dashboard = None
            for stage_index, (stage_name, stage_time) in enumerate(
                zip(stage_names, sequence_times), start=1
            ):
                stage_outputs, _ = _process_pose(
                    sim=env.sim,
                    suite=final_suite,
                    position=position,
                    source=view_target,
                    time_s=stage_time,
                    max_depth_m=max_depth_m,
                )
                stage_metrics = balanced_visibility_score(stage_outputs)
                stage_dashboard = _compose_labelled_dashboard(
                    stage_outputs,
                    scenario=scenario,
                    plan=plan,
                    source=source,
                    time_s=stage_time,
                    n_steps=args.n_steps,
                    render_scale=args.render_scale,
                    max_depth_m=max_depth_m,
                    stage_label=stage_name.title(),
                )
                if is_spread_sequence:
                    stage_dir = (
                        scenario_dir
                        / "sequence"
                        / f"{stage_index:02d}_{stage_name}_{stage_time:05.1f}s"
                    )
                    stage_files = _save_observation(
                        output_dir=stage_dir,
                        outputs=stage_outputs,
                        dashboard=stage_dashboard,
                    )
                else:
                    stage_dir = scenario_dir
                    stage_files = []

                actual_backend = str(
                    stage_outputs.get("fire_render_backend", "unknown")
                )
                actual_device = str(
                    stage_outputs.get("fire_render_device", "unknown")
                )
                if actual_backend != "torch" or not actual_device.startswith(
                    "cuda"
                ):
                    raise RuntimeError(
                        "highest-quality render fell back to "
                        f"{actual_backend}/{actual_device}"
                    )
                active = active_ignitions(plan, stage_time)
                stage_record = {
                    "stage": stage_name,
                    "time_s": float(stage_time),
                    "active_source_count": len(active),
                    "total_source_count": len(plan["ignitions"]),
                    "active_object_ids": [
                        int(item["object_id"]) for item in active
                    ],
                    "active_generations": sorted(
                        {
                            int(item.get("spread_generation", 0))
                            for item in active
                        }
                    ),
                    "metrics": stage_metrics,
                    "output_dir": str(stage_dir.relative_to(output_root)),
                    "files": stage_files,
                    "actual_renderer": {
                        "backend": actual_backend,
                        "device": actual_device,
                        "frame_index": int(
                            stage_outputs.get("fire_render_frame_index", -1)
                        ),
                    },
                }
                sequence_records.append(stage_record)
                subtitle = (
                    f"t={stage_time:.0f}s | active "
                    f"{len(active)}/{len(plan['ignitions'])} | "
                    f"gen {stage_record['active_generations']}"
                )
                sequence_row.append(
                    (
                        cv2.cvtColor(
                            stage_outputs["rgb_smoke"], cv2.COLOR_RGB2BGR
                        ),
                        f"{scenario.display_name} | {stage_name.title()}",
                        subtitle,
                    )
                )
                outputs = stage_outputs
                dashboard = stage_dashboard

            assert outputs is not None and dashboard is not None
            final_metrics = balanced_visibility_score(outputs)
            # Keep the late frame at the scenario root for compatibility with
            # the original one-frame gallery, while sequence/ retains all
            # three complete FireSensor observations.
            saved_files = _save_observation(
                output_dir=scenario_dir,
                outputs=outputs,
                dashboard=dashboard,
            )

            if is_spread_sequence:
                spread_rows.append(sequence_row)
                scenario_sequence = make_spread_sequence_montage(
                    [sequence_row], tile_size=(640, 480)
                )
                _write_image(
                    scenario_dir / "spread_sequence.png",
                    scenario_sequence,
                )
                saved_files.append("spread_sequence.png")

            actual_backend = str(outputs.get("fire_render_backend", "unknown"))
            actual_device = str(outputs.get("fire_render_device", "unknown"))
            rotation = _yaw_quaternion_facing(position, view_target)
            scenario_record = {
                "order": order,
                **asdict(scenario),
                "display_name": scenario.display_name,
                "intensity": plan.get("intensity"),
                "template_version": plan.get("template_version"),
                "episode_id_used_for_scene_loading": str(matching[0].episode_id),
                "ignition_count": len(plan["ignitions"]),
                "num_initial_ignitions": int(
                    plan.get("num_initial_ignitions", len(plan["ignitions"]))
                ),
                "ignitions": plan["ignitions"],
                "selected_visible_source": source,
                "view_target_position": [
                    float(value) for value in view_target
                ],
                "camera_position": [float(value) for value in position],
                "camera_rotation_xyzw": [float(value) for value in rotation],
                "fire_time_s": float(time_s),
                "preview_metrics": preview_metrics,
                "final_metrics": final_metrics,
                "actual_renderer": {
                    "backend": actual_backend,
                    "device": actual_device,
                    "frame_index": int(outputs.get("fire_render_frame_index", -1)),
                },
                "lidar_is_360": bool(outputs.get("lidar_is_360", False)),
                "files": saved_files,
                "sequence": sequence_records,
                "spread_disclosure": plan.get("paper_spread_disclosure"),
            }
            (scenario_dir / "manifest.json").write_text(
                json.dumps(scenario_record, indent=2) + "\n",
                encoding="utf-8",
            )
            gallery_manifest["scenarios"].append(scenario_record)

            subtitle = (
                f"{scenario.scene_id} | {plan.get('intensity')} | "
                f"{len(plan['ignitions'])} source(s) | t={time_s:.0f}s"
            )
            rgb_tiles.append(
                (
                    cv2.cvtColor(outputs["rgb_smoke"], cv2.COLOR_RGB2BGR),
                    scenario.display_name,
                    subtitle,
                )
            )
            dashboard_tiles.append((dashboard, scenario.display_name, subtitle))

            # Release the large mmap/GPU volume before loading the next scene.
            cache = getattr(fire_scene, "_fire_torch_volume_cache", None)
            if cache is not None:
                cache.clear()
            del final_suite, fire_scene, fw, outputs
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except (ImportError, RuntimeError):
                pass

        rgb_montage = make_montage(rgb_tiles, tile_size=(640, 480))
        dashboard_montage = make_montage(dashboard_tiles, tile_size=(1000, 450))
        _write_image(output_root / "fire_types_rgb_montage.png", rgb_montage)
        _write_image(
            output_root / "fire_types_sensor_dashboard_montage.png",
            dashboard_montage,
        )
        montage_files = [
            "fire_types_rgb_montage.png",
            "fire_types_sensor_dashboard_montage.png",
        ]
        if spread_rows:
            spread_montage = make_spread_sequence_montage(
                spread_rows, tile_size=(480, 360)
            )
            _write_image(
                output_root / "fire_source_spread_sequence_montage.png",
                spread_montage,
            )
            montage_files.insert(0, "fire_source_spread_sequence_montage.png")
        gallery_manifest["montages"] = montage_files
        (output_root / "manifest.json").write_text(
            json.dumps(gallery_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        command = (
            "/home/liushe10/miniconda3/envs/co-nav3/bin/python "
            "scripts/capture_fire_type_gallery.py --device cuda:0 "
            f"--n-steps {args.n_steps} --render-scale {args.render_scale}"
        )
        (output_root / "README.md").write_text(
            "# FireWorld staged-spread paper gallery\n\n"
            "Each numbered directory contains fixed-camera early, middle and "
            "late FireSensor observations. Every stage saves clean and "
            "fire/smoke RGB, clean and degraded depth, thermal, 360-degree "
            "LiDAR, three radar views, a dashboard and compressed raw arrays.\n\n"
            "Important disclosure: delayed secondary-source times and parent "
            "links are authored for this controlled visual sequence; they are "
            "not solver-inferred causal attribution.\n\n"
            "Reproduce from the repository root:\n\n"
            f"```bash\n{command}\n```\n",
            encoding="utf-8",
        )
        print(f"[gallery] complete: {output_root}")
    finally:
        env.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", default="multi_objectnav_hm3d.yaml")
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--fire-world-root", default="outputs/fire_world")
    parser.add_argument("--output-dir", default="outputs/paper_fire_spread_gallery")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--preview-steps", type=int, default=10)
    parser.add_argument("--preview-scale", type=float, default=0.25)
    parser.add_argument("--lidar-resolution", type=int, default=320)
    parser.add_argument(
        "--scenario",
        action="append",
        type=parse_scenario,
        help="override with FIRE_TYPE:SCENE_ID:PLAN_ID; pass exactly four times",
    )
    args = parser.parse_args(argv)
    if args.n_steps < 32:
        parser.error("--n-steps must be at least 32 for publication capture")
    if not 0.05 <= args.render_scale <= 1.0:
        parser.error("--render-scale must be in [0.05, 1.0]")
    if args.preview_steps < 2:
        parser.error("--preview-steps must be at least 2")
    return args


if __name__ == "__main__":
    run(parse_args())
