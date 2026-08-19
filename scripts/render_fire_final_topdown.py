#!/usr/bin/env python3
"""Render a whole-scene FireWorld final state over a textured HM3D top view.

The base image is captured by a Habitat-Sim orthographic RGB camera placed
inside the ceiling and looking straight down.  Fire fields are sampled from
the baked ``timeline.npz`` in the same world x/z frame, so the overlay is a
measurement of the selected plan rather than an illustrative heatmap.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_DATASET = (
    PROJECT_ROOT
    / "data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
)


def _short_scene_id(value: str) -> str:
    name = Path(value).name
    if "-" in name and name.split("-", 1)[0].isdigit():
        return name.split("-", 1)[1]
    return name


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_dataset_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else path.open
    if path.suffix == ".gz":
        with opener(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with opener("r", encoding="utf-8") as stream:
        return json.load(stream)


def _person_goal_from_dataset(
    dataset: dict,
    *,
    scene_id: str,
    episode_id: Optional[str],
) -> dict:
    """Resolve the native static-person goal used by an ObjectNav episode."""

    episodes = list(dataset.get("episodes", ()))
    if episode_id is not None:
        episodes = [
            episode
            for episode in episodes
            if str(episode.get("episode_id")) == str(episode_id)
        ]
        if not episodes:
            raise ValueError(f"person dataset has no episode {episode_id}")
    person_episodes = [
        episode
        for episode in episodes
        if episode.get("object_category") == "person"
        and scene_id in str(episode.get("scene_id", ""))
    ]
    if not person_episodes:
        raise ValueError(f"dataset has no person episode for scene {scene_id}")

    goal_keys = [
        key
        for key in dataset.get("goals_by_category", {})
        if key.endswith("_person") and scene_id in key
    ]
    if len(goal_keys) != 1:
        raise ValueError(
            f"expected one person goal key for {scene_id}, got {goal_keys}"
        )
    goals = dataset["goals_by_category"][goal_keys[0]]
    if len(goals) != 1:
        raise ValueError(
            f"expected one fixed person goal for {scene_id}, got {len(goals)}"
        )
    goal = dict(goals[0])
    position = np.asarray(goal.get("position", ()), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("person goal must contain one finite xyz position")
    return goal


def _world_to_pixel(
    x_m: float,
    z_m: float,
    bounds_xz: Sequence[float],
    image_hw: Tuple[int, int],
) -> Tuple[int, int]:
    """Map world x/z to image column/row for the downward camera."""

    x_min, z_min, x_max, z_max = map(float, bounds_xz)
    height, width = map(int, image_hw)
    col = int(round((float(x_m) - x_min) / (x_max - x_min) * (width - 1)))
    row = int(round((float(z_m) - z_min) / (z_max - z_min) * (height - 1)))
    return col, row


def _sample_xz_field(
    field_xz: np.ndarray,
    *,
    origin_xyz: Sequence[float],
    voxel_m: float,
    bounds_xz: Sequence[float],
    image_hw: Tuple[int, int],
) -> np.ndarray:
    """Bilinearly sample an x/z FireWorld field into top-view pixels."""

    field = np.asarray(field_xz, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"expected a 2-D x/z field, got shape {field.shape}")
    x_min, z_min, x_max, z_max = map(float, bounds_xz)
    height, width = map(int, image_hw)
    x_world = np.linspace(x_min, x_max, width, dtype=np.float32)
    z_world = np.linspace(z_min, z_max, height, dtype=np.float32)
    map_x = np.broadcast_to(
        ((x_world - float(origin_xyz[0])) / float(voxel_m))[None, :],
        (height, width),
    ).astype(np.float32, copy=False)
    map_y = np.broadcast_to(
        ((z_world - float(origin_xyz[2])) / float(voxel_m))[:, None],
        (height, width),
    ).astype(np.float32, copy=False)
    # FireWorld arrays are x,y,z.  OpenCV images are row=z, column=x.
    return cv2.remap(
        field.T,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _render_textured_topdown(
    *,
    scene_glb: Path,
    scene_dataset_config: Path,
    bounds_xz: Sequence[float],
    camera_y_m: float,
    image_hw: Tuple[int, int],
    gpu_device_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Capture RGB and geometry mask using Habitat's orthographic camera."""

    import habitat_sim
    import magnum as mn

    x_min, z_min, x_max, z_max = map(float, bounds_xz)
    height, width = map(int, image_hw)
    x_span = x_max - x_min

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(scene_glb)
    backend.scene_dataset_config_file = str(scene_dataset_config)
    backend.enable_physics = False
    backend.gpu_device_id = int(gpu_device_id)

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "topdown_rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    sensor.resolution = mn.Vector2i(height, width)
    sensor.position = mn.Vector3(0.0, 0.0, 0.0)
    sensor.orientation = mn.Vector3(-math.pi / 2.0, 0.0, 0.0)
    sensor.near = 0.03
    sensor.far = 50.0
    # Habitat's horizontal orthographic span is 1 / ortho_scale.
    sensor.ortho_scale = 1.0 / float(x_span)
    sensor.clear_color = mn.Color4(0.0, 0.0, 0.0, 0.0)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [agent_cfg])
    )
    try:
        state = habitat_sim.AgentState()
        state.position = np.asarray(
            [
                0.5 * (x_min + x_max),
                float(camera_y_m),
                0.5 * (z_min + z_max),
            ],
            dtype=np.float32,
        )
        simulator.initialize_agent(0, state)
        rgba = np.asarray(simulator.get_sensor_observations()[sensor.uuid])
    finally:
        simulator.close()

    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise RuntimeError(f"unexpected Habitat RGB observation {rgba.shape}")
    rgb = np.asarray(rgba[..., :3], dtype=np.uint8)
    if rgba.shape[2] == 4:
        geometry_mask = np.asarray(rgba[..., 3] > 0, dtype=bool)
    else:
        geometry_mask = np.any(rgb > 2, axis=2)
    return rgb, geometry_mask


def _fractal_noise(shape: Tuple[int, int], seed: int) -> np.ndarray:
    """Deterministic smooth multi-scale noise for smoke/flame appearance."""

    height, width = map(int, shape)
    rng = np.random.default_rng(int(seed))
    result = np.zeros((height, width), dtype=np.float32)
    total_weight = 0.0
    for cell_px, weight in (
        (180, 0.32),
        (78, 0.28),
        (30, 0.24),
        (12, 0.16),
    ):
        grid_h = max(2, int(math.ceil(height / cell_px)) + 2)
        grid_w = max(2, int(math.ceil(width / cell_px)) + 2)
        coarse = rng.random((grid_h, grid_w), dtype=np.float32)
        layer = cv2.resize(
            coarse,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        result += np.clip(layer, 0.0, 1.0) * float(weight)
        total_weight += float(weight)
    result /= total_weight
    low, high = float(np.min(result)), float(np.max(result))
    if high > low:
        result = (result - low) / (high - low)
    return result.astype(np.float32, copy=False)


def _flame_color(intensity: np.ndarray) -> np.ndarray:
    """Return a physical-looking RGB red-edge -> orange -> yellow-core map."""

    value = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
    low = np.asarray([182.0, 28.0, 4.0], dtype=np.float32)
    middle = np.asarray([255.0, 112.0, 8.0], dtype=np.float32)
    high = np.asarray([255.0, 178.0, 22.0], dtype=np.float32)
    lower_mix = np.clip(value / 0.55, 0.0, 1.0)[..., None]
    upper_mix = np.clip((value - 0.55) / 0.45, 0.0, 1.0)[..., None]
    color = low * (1.0 - lower_mix) + middle * lower_mix
    return color * (1.0 - upper_mix) + high * upper_mix


def _compose_overlay(
    rgb: np.ndarray,
    geometry_mask: np.ndarray,
    flame: np.ndarray,
    smoke: np.ndarray,
) -> np.ndarray:
    """Composite textured smoke, heat glow and translucent flame cores.

    Even at maximum smoke and flame, at least roughly one third of the
    textured scene remains in the final pixel.  The overlay therefore conveys
    hazard extent without turning furniture into an opaque segmentation mask.
    """

    result = np.asarray(rgb, dtype=np.float32).copy()
    mask = np.asarray(geometry_mask, dtype=bool)
    smoke_strength = np.clip((smoke - 0.012) / 0.40, 0.0, 1.0)
    smoke_texture = _fractal_noise(smoke.shape, seed=1701)
    smoke_noise = 0.68 + 0.62 * smoke_texture
    smoke_alpha = np.clip(
        0.52 * np.power(smoke_strength, 0.32) * smoke_noise,
        0.0,
        0.52,
    )
    smoke_alpha *= mask
    smoke_low = np.asarray([92.0, 96.0, 102.0], dtype=np.float32)
    smoke_high = np.asarray([12.0, 15.0, 19.0], dtype=np.float32)
    smoke_tone = np.clip(smoke_strength, 0.0, 1.0)[..., None]
    smoke_rgb = (
        smoke_low * (1.0 - smoke_tone) + smoke_high * smoke_tone
    )
    # Alternating lighter and darker wisps make smoke distinguishable from
    # ordinary room shadow while remaining tied to the measured density.
    smoke_rgb += ((smoke_texture - 0.5) * 54.0)[..., None]
    smoke_rgb = np.clip(smoke_rgb, 8.0, 125.0)
    result = (
        result * (1.0 - smoke_alpha[..., None])
        + smoke_rgb * smoke_alpha[..., None]
    )
    # Pale-gray intermittent wisps on top of the darker smoke body make the
    # cloud structure visible over both light floors and dark furniture.
    wisp = (
        np.power(smoke_strength, 0.55)
        * np.clip((smoke_texture - 0.46) / 0.54, 0.0, 1.0)
    )
    wisp_alpha = np.clip(0.16 * wisp, 0.0, 0.16) * mask
    wisp_rgb = np.full_like(result, (142.0, 146.0, 151.0))
    result = (
        result * (1.0 - wisp_alpha[..., None])
        + wisp_rgb * wisp_alpha[..., None]
    )

    # A broad, very light amber glow communicates low flame/heat without
    # painting every affected voxel as a solid flame patch.
    heat = np.clip((flame - 0.018) / 0.28, 0.0, 1.0)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=7.0, sigmaY=7.0)
    heat_alpha = np.clip(0.035 * heat, 0.0, 0.035) * mask
    heat_rgb = np.full_like(result, (255.0, 147.0, 28.0))
    result = (
        result * (1.0 - heat_alpha[..., None])
        + heat_rgb * heat_alpha[..., None]
    )

    # Flame cores are spatially modulated to avoid the flat heatmap look.
    flame_strength = np.clip((flame - 0.060) / 0.68, 0.0, 1.0)
    flame_noise = 0.48 + 0.82 * _fractal_noise(flame.shape, seed=8429)
    textured_flame = np.clip(
        np.power(flame_strength, 0.72) * flame_noise,
        0.0,
        1.0,
    )
    flame_alpha = np.clip(0.48 * textured_flame, 0.0, 0.48)
    flame_alpha *= mask
    flame_rgb = _flame_color(textured_flame)
    result = (
        result * (1.0 - flame_alpha[..., None])
        + flame_rgb * flame_alpha[..., None]
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def _draw_source_markers(
    image_rgb: np.ndarray,
    ignitions: Sequence[dict],
    *,
    bounds_xz: Sequence[float],
) -> Tuple[np.ndarray, list]:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    pixels = []
    for index, source in enumerate(ignitions, start=1):
        position = source["position"]
        col, row = _world_to_pixel(
            position[0], position[2], bounds_xz, image_rgb.shape[:2]
        )
        pixels.append([int(col), int(row)])
        cv2.circle(canvas, (col, row), 18, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.circle(canvas, (col, row), 13, (20, 35, 245), -1, cv2.LINE_AA)
        label = f"S{index}"
        cv2.putText(
            canvas,
            label,
            (col + 15, row - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (col + 15, row - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (25, 25, 235),
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), pixels


def _draw_person_marker(
    image_rgb: np.ndarray,
    person_goal: dict,
    *,
    bounds_xz: Sequence[float],
) -> Tuple[np.ndarray, list]:
    """Draw a coordinate-anchored top-down person target symbol."""

    position = person_goal["position"]
    col, row = _world_to_pixel(
        position[0], position[2], bounds_xz, image_rgb.shape[:2]
    )
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cyan = (215, 170, 15)
    cv2.circle(canvas, (col, row), 22, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.circle(canvas, (col, row), 18, cyan, 3, cv2.LINE_AA)
    cv2.circle(canvas, (col, row - 6), 6, cyan, -1, cv2.LINE_AA)
    cv2.ellipse(
        canvas,
        (col, row + 7),
        (9, 12),
        0,
        180,
        360,
        cyan,
        -1,
        cv2.LINE_AA,
    )
    label = "PERSON"
    cv2.putText(
        canvas,
        label,
        (col + 25, row + 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (col + 25, row + 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        cyan,
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), [int(col), int(row)]


def _source_caption(index: int, source: dict) -> str:
    """Build a source label for both current and legacy fire plans."""

    caption = f"S{int(index)}  {source.get('category', 'source')}"
    region_id = source.get("region_id")
    if region_id is not None:
        caption += f"  region {region_id}"
    return caption


def _decorate(
    map_rgb: np.ndarray,
    *,
    scene_label: str,
    plan: dict,
    final_time_s: float,
    person_goal: Optional[dict] = None,
) -> np.ndarray:
    height, width = map_rgb.shape[:2]
    header_h = 124
    footer_h = 132
    canvas = np.full((header_h + height + footer_h, width, 3), 248, np.uint8)
    canvas[header_h : header_h + height] = map_rgb
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    cv2.putText(
        bgr,
        f"00880 FireWorld final state - {scene_label}",
        (28, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f"{plan['fire_type']} / {plan['intensity']}   t = {final_time_s:.0f} s   "
        f"sources = {len(plan['ignitions'])}"
        + ("   person target = 1" if person_goal is not None else ""),
        (30, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.73,
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )

    legend_y = 104
    flame_x = width - 650
    cv2.rectangle(bgr, (flame_x, legend_y - 23), (flame_x + 75, legend_y - 7), (7, 36, 188), -1)
    cv2.rectangle(bgr, (flame_x + 75, legend_y - 23), (flame_x + 150, legend_y - 7), (8, 112, 255), -1)
    cv2.rectangle(bgr, (flame_x + 150, legend_y - 23), (flame_x + 225, legend_y - 7), (72, 232, 255), -1)
    cv2.putText(bgr, "flame edge", (flame_x, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(bgr, "core", (flame_x + 184, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)

    smoke_x = width - 390
    cv2.rectangle(bgr, (smoke_x, legend_y - 23), (smoke_x + 65, legend_y - 7), (120, 115, 112), -1)
    cv2.rectangle(bgr, (smoke_x + 65, legend_y - 23), (smoke_x + 130, legend_y - 7), (70, 67, 64), -1)
    cv2.rectangle(bgr, (smoke_x + 130, legend_y - 23), (smoke_x + 195, legend_y - 7), (35, 30, 27), -1)
    cv2.putText(bgr, "smoke light", (smoke_x, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(bgr, "dense", (smoke_x + 151, legend_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (60, 60, 60), 1, cv2.LINE_AA)

    cv2.circle(bgr, (width - 142, legend_y - 15), 9, (20, 35, 245), -1, cv2.LINE_AA)
    cv2.putText(bgr, "source", (width - 124, legend_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (60, 60, 60), 1, cv2.LINE_AA)

    # Two compact source rows keep the map itself readable.
    base_y = header_h + height + 38
    column_w = width // 4
    for index, source in enumerate(plan["ignitions"], start=1):
        row_id = (index - 1) // 4
        col_id = (index - 1) % 4
        x = col_id * column_w + 28
        y = base_y + row_id * 44
        cv2.circle(bgr, (x, y - 6), 8, (20, 35, 245), -1, cv2.LINE_AA)
        text = _source_caption(index, source)
        cv2.putText(bgr, text, (x + 17, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (45, 45, 45), 1, cv2.LINE_AA)

    cv2.putText(
        bgr,
        "Textured gray-black = body-height smoke   |   Translucent red-orange-yellow = max vertical flame",
        (30, header_h + height + footer_h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.49,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a textured whole-scene FireWorld final top view"
    )
    parser.add_argument("--scene", required=True, help="HM3D scene id or folder")
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--fire-out-root", default="outputs/fire_world")
    parser.add_argument("--scene-dataset", default=str(DEFAULT_SCENE_DATASET))
    parser.add_argument("--output", default=None)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--padding-fraction", type=float, default=0.035)
    parser.add_argument("--camera-height-above-floor-m", type=float, default=2.60)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--person-dataset-path",
        default=None,
        help="optional person ObjectNav scene shard (.json or .json.gz)",
    )
    parser.add_argument(
        "--person-episode-id",
        default=None,
        help="episode id used to validate --person-dataset-path",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.width < 320:
        raise ValueError("--width must be at least 320 pixels")
    if not 0.0 <= args.padding_fraction <= 0.25:
        raise ValueError("--padding-fraction must be in [0, 0.25]")

    scene_id = _short_scene_id(args.scene)
    scenes_root = (PROJECT_ROOT / args.scenes_root).resolve()
    fire_out_root = (PROJECT_ROOT / args.fire_out_root).resolve()
    inventory_path = scenes_root / scene_id / "inventory.json"
    plan_path = scenes_root / scene_id / "plans" / f"{args.plan_id}.json"
    timeline_dir = fire_out_root / scene_id / args.plan_id
    timeline_path = timeline_dir / "timeline.npz"
    timeline_meta_path = timeline_dir / "timeline_meta.json"
    for required in (
        inventory_path,
        plan_path,
        timeline_path,
        timeline_meta_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    inventory = _load_json(inventory_path)
    plan = _load_json(plan_path)
    timeline_meta = _load_json(timeline_meta_path)
    person_goal = None
    person_dataset_path = None
    if args.person_dataset_path:
        person_dataset_path = Path(args.person_dataset_path).expanduser().resolve()
        if not person_dataset_path.is_file():
            raise FileNotFoundError(person_dataset_path)
        person_goal = _person_goal_from_dataset(
            _load_dataset_json(person_dataset_path),
            scene_id=scene_id,
            episode_id=args.person_episode_id,
        )
    if plan.get("scene_id") != scene_id:
        raise ValueError("plan scene_id does not match --scene")
    if timeline_meta.get("plan_id") != args.plan_id:
        raise ValueError("timeline metadata does not match --plan-id")

    world_aabb = list(map(float, timeline_meta["world_aabb"]))
    x_min, _, z_min, x_max, _, z_max = world_aabb
    x_padding = (x_max - x_min) * float(args.padding_fraction)
    z_padding = (z_max - z_min) * float(args.padding_fraction)
    bounds_xz = (
        x_min - x_padding,
        z_min - z_padding,
        x_max + x_padding,
        z_max + z_padding,
    )
    x_span = bounds_xz[2] - bounds_xz[0]
    z_span = bounds_xz[3] - bounds_xz[1]
    width = int(args.width)
    height = max(320, int(round(width * z_span / x_span)))
    image_hw = (height, width)

    floor_y_m = float(inventory["floors"][0]["y"])
    camera_y_m = floor_y_m + float(args.camera_height_above_floor_m)
    scene_glb = (PROJECT_ROOT / inventory["scene_glb"]).resolve()
    scene_dataset = Path(args.scene_dataset).expanduser().resolve()
    rgb, geometry_mask = _render_textured_topdown(
        scene_glb=scene_glb,
        scene_dataset_config=scene_dataset,
        bounds_xz=bounds_xz,
        camera_y_m=camera_y_m,
        image_hw=image_hw,
        gpu_device_id=args.gpu_device_id,
    )

    with np.load(timeline_path, allow_pickle=False) as timeline:
        final_flame_xyz = np.asarray(timeline["flame"][-1], dtype=np.float32)
        final_smoke_xyz = np.asarray(timeline["smoke"][-1], dtype=np.float32)
        final_time_s = float(timeline["times"][-1])
    flame_xz = np.max(final_flame_xyz, axis=1)
    origin_xyz = timeline_meta["origin"]
    voxel_m = float(timeline_meta["voxel_m"])
    floor_index = int(round((floor_y_m - float(origin_xyz[1])) / voxel_m))
    body_lo = int(np.clip(floor_index, 0, final_smoke_xyz.shape[1] - 1))
    body_hi = int(np.clip(
        floor_index + math.ceil(1.5 / voxel_m),
        body_lo + 1,
        final_smoke_xyz.shape[1],
    ))
    smoke_xz = np.mean(final_smoke_xyz[:, body_lo:body_hi, :], axis=1)
    flame_image = _sample_xz_field(
        flame_xz,
        origin_xyz=origin_xyz,
        voxel_m=voxel_m,
        bounds_xz=bounds_xz,
        image_hw=image_hw,
    )
    smoke_image = _sample_xz_field(
        smoke_xz,
        origin_xyz=origin_xyz,
        voxel_m=voxel_m,
        bounds_xz=bounds_xz,
        image_hw=image_hw,
    )
    composed = _compose_overlay(
        rgb, geometry_mask, flame_image, smoke_image
    )
    marked, source_pixels = _draw_source_markers(
        composed,
        plan["ignitions"],
        bounds_xz=bounds_xz,
    )
    person_pixel = None
    if person_goal is not None:
        marked, person_pixel = _draw_person_marker(
            marked,
            person_goal,
            bounds_xz=bounds_xz,
        )
    decorated = _decorate(
        marked,
        scene_label=scene_id,
        plan=plan,
        final_time_s=final_time_s,
        person_goal=person_goal,
    )

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else timeline_dir / "final_topdown.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path), cv2.cvtColor(decorated, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"failed to write {output_path}")

    scene_pixels = max(1, int(np.count_nonzero(geometry_mask)))
    metadata: Dict[str, object] = {
        "schema_version": 1,
        "scene_id": scene_id,
        "scene_folder": str(args.scene),
        "plan_id": args.plan_id,
        "fire_type": plan["fire_type"],
        "intensity": plan["intensity"],
        "final_frame_index": int(timeline_meta["n_frames"]) - 1,
        "final_time_s": final_time_s,
        "num_ignitions": len(plan["ignitions"]),
        "world_bounds_xz": list(map(float, bounds_xz)),
        "camera_y_m": camera_y_m,
        "map_image_hw": [height, width],
        "output_image_hw": list(map(int, decorated.shape[:2])),
        "flame_coverage_scene_fraction": float(
            np.count_nonzero((flame_image > 0.03) & geometry_mask)
            / scene_pixels
        ),
        "smoke_coverage_scene_fraction": float(
            np.count_nonzero((smoke_image > 0.03) & geometry_mask)
            / scene_pixels
        ),
        "source_pixels_xy": source_pixels,
        "person_dataset_path": (
            str(person_dataset_path) if person_dataset_path is not None else None
        ),
        "person_episode_id": (
            str(args.person_episode_id) if person_goal is not None else None
        ),
        "person_goal_world_xyz": (
            list(map(float, person_goal["position"]))
            if person_goal is not None
            else None
        ),
        "person_pixel_xy": person_pixel,
        "output_png": str(output_path),
        "visualization_style": {
            "version": 2,
            "max_smoke_alpha": 0.52,
            "max_smoke_wisp_alpha": 0.16,
            "max_heat_glow_alpha": 0.035,
            "max_flame_alpha": 0.48,
            "smoke_noise_seed": 1701,
            "flame_noise_seed": 8429,
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[fire-topdown] image={output_path}")
    print(f"[fire-topdown] metadata={metadata_path}")
    print(
        "[fire-topdown] "
        f"scene_pixels={scene_pixels} "
        f"flame_coverage={metadata['flame_coverage_scene_fraction']:.3f} "
        f"smoke_coverage={metadata['smoke_coverage_scene_fraction']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
