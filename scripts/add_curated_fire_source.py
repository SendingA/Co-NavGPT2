#!/usr/bin/env python3
"""Add one validated semantic ignition to an existing curated FireWorld plan.

The source is selected by semantic instance id.  For screenshot-driven edits,
the script also projects every ignition into the mapper frame and writes a
marked reference image, so the requested location is auditable before the new
timeline is baked.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.fine_tuning import (  # noqa: E402
    curated_plan_hash,
    write_curated_plan,
)
from utils.fire_world.plan_ids import semantic_plan_id  # noqa: E402
from utils.local_planners.pointnav import world_to_frontier_grid  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--dataset-shard", required=True)
    parser.add_argument("--object-id", required=True, type=int)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--output-evidence-dir", required=True)
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--map-resolution-cm", type=float, default=5.0)
    parser.add_argument("--map-size-pixels", type=int, default=480)
    parser.add_argument("--map-panel-x", type=int, default=15)
    parser.add_argument("--map-panel-y", type=int, default=50)
    parser.add_argument(
        "--red-box-map-cells",
        nargs=4,
        type=float,
        metavar=("ROW_MIN", "COL_MIN", "ROW_MAX", "COL_MAX"),
        default=(220.0, 125.0, 282.0, 165.0),
        help="approximate user-marked bounds in the 480x480 mapper grid",
    )
    return parser


def quaternion_coeffs_to_matrix(coeffs: Sequence[float]) -> np.ndarray:
    """Convert Habitat's ``[x, y, z, w]`` quaternion to a rotation matrix."""

    values = np.asarray(coeffs, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("start_rotation must contain [x, y, z, w]")
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("start_rotation quaternion has zero norm")
    x, y, z, w = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def build_augmented_plan(
    base_plan: Mapping[str, object],
    instance: Mapping[str, object],
    *,
    map_cell: Sequence[float],
    red_box_map_cells: Sequence[float],
    reference_image: str,
) -> dict:
    """Return a content-addressed clone with one extra semantic ignition."""

    payload = copy.deepcopy(dict(base_plan))
    object_id = int(instance["instance_id"])
    existing_ids = {int(item["object_id"]) for item in payload["ignitions"]}
    if object_id in existing_ids:
        raise ValueError(f"object {object_id} is already an ignition source")
    if bool(instance.get("structural")):
        raise ValueError(f"object {object_id} is structural")
    threshold = float(payload["propagation_rules"].get("flammable_threshold", 0.4))
    flammability = float(instance.get("flammability", 0.0))
    if flammability < threshold:
        raise ValueError(
            f"object {object_id} flammability {flammability:.3f} is below "
            f"the plan threshold {threshold:.3f}"
        )
    centroid = [float(value) for value in instance["centroid"][:3]]
    template = dict(payload["ignitions"][0])
    template.update(
        {
            "object_id": object_id,
            "category": str(instance.get("category") or "curated fuel"),
            "position": centroid,
            "ignite_time_s": 0.0,
            "ignition_role": "initial",
        }
    )
    payload["ignitions"].append(template)
    payload["num_initial_ignitions"] = len(payload["ignitions"])
    payload["ignition_selection_mode"] = "curated_route_contrast_user_marked"
    payload["ignition_selection_version"] = 2

    curation = dict(payload.get("curation") or {})
    curated_ignitions = list(curation.get("ignitions") or [])
    curated_ignitions.append(
        {
            "instance_id": object_id,
            "category": template["category"],
            "centroid": centroid,
            "flammability": flammability,
        }
    )
    curation["ignitions"] = curated_ignitions
    curation["user_marked_additional_ignition"] = {
        "instance_id": object_id,
        "category": template["category"],
        "centroid": centroid,
        "flammability": flammability,
        "floor_id": int(instance.get("floor_id", 0)),
        "map_cell": [float(value) for value in map_cell[:2]],
        "red_box_map_cells": [float(value) for value in red_box_map_cells],
        "reference_image": str(reference_image),
    }
    payload["curation"] = curation
    payload.pop("plan_hash", None)
    payload.pop("plan_id", None)
    plan_hash = curated_plan_hash(payload)
    payload["plan_hash"] = plan_hash
    payload["plan_id"] = semantic_plan_id(
        str(payload["scene_id"]),
        str(payload["fire_type"]),
        str(payload["intensity"]),
        plan_hash,
    )
    return payload


def _read_first_episode(path: Path) -> Mapping[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes") or []
    if len(episodes) != 1:
        raise ValueError(f"expected one episode in {path}, found {len(episodes)}")
    return episodes[0]


def _map_cell(
    position: Sequence[float],
    episode: Mapping[str, object],
    *,
    map_size_pixels: int,
    map_resolution_cm: float,
) -> np.ndarray:
    origin = 0.5 * float(map_size_pixels)
    return world_to_frontier_grid(
        position,
        origins_grid=(origin, origin),
        map_resolution_cm=float(map_resolution_cm),
        initial_agent_position=episode["start_position"],
        initial_sensor_rotation=quaternion_coeffs_to_matrix(
            episode["start_rotation"]
        ),
    )


def _display_pixel(
    map_cell: Sequence[float],
    *,
    panel_x: int,
    panel_y: int,
    map_size_pixels: int,
) -> tuple[int, int]:
    row, col = (float(value) for value in map_cell[:2])
    return (
        int(round(float(panel_x) + col)),
        int(round(float(panel_y) + (float(map_size_pixels) - 1.0 - row))),
    )


def _write_selection_image(
    reference_path: Path,
    output_path: Path,
    plan: Mapping[str, object],
    episode: Mapping[str, object],
    *,
    selected_object_id: int,
    red_box: Sequence[float],
    panel_x: int,
    panel_y: int,
    map_size_pixels: int,
    map_resolution_cm: float,
) -> None:
    image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read reference image {reference_path}")
    row_min, col_min, row_max, col_max = (float(value) for value in red_box)
    box_tl = _display_pixel(
        (row_max, col_min), panel_x=panel_x, panel_y=panel_y,
        map_size_pixels=map_size_pixels,
    )
    box_br = _display_pixel(
        (row_min, col_max), panel_x=panel_x, panel_y=panel_y,
        map_size_pixels=map_size_pixels,
    )
    cv2.rectangle(image, box_tl, box_br, (0, 0, 255), 2)
    for ignition in plan["ignitions"]:
        cell = _map_cell(
            ignition["position"], episode,
            map_size_pixels=map_size_pixels,
            map_resolution_cm=map_resolution_cm,
        )
        pixel = _display_pixel(
            cell, panel_x=panel_x, panel_y=panel_y,
            map_size_pixels=map_size_pixels,
        )
        selected = int(ignition["object_id"]) == int(selected_object_id)
        color = (0, 0, 255) if selected else (0, 220, 255)
        radius = 8 if selected else 5
        cv2.circle(image, pixel, radius, color, 2)
        label = f"new {selected_object_id}" if selected else str(ignition["object_id"])
        cv2.putText(
            image, label, (pixel[0] + 7, pixel[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"could not write selection image {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    base_path = (ROOT / args.base_plan).resolve()
    inventory_path = (ROOT / args.inventory).resolve()
    shard_path = (ROOT / args.dataset_shard).resolve()
    reference_path = (ROOT / args.reference_image).resolve()
    output_dir = (ROOT / args.output_evidence_dir).resolve()
    base_plan = json.loads(base_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    episode = _read_first_episode(shard_path)
    matches = [
        item for item in inventory["instances"]
        if int(item["instance_id"]) == int(args.object_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one inventory object {args.object_id}")
    selected = matches[0]
    cell = _map_cell(
        selected["centroid"], episode,
        map_size_pixels=int(args.map_size_pixels),
        map_resolution_cm=float(args.map_resolution_cm),
    )
    row_min, col_min, row_max, col_max = args.red_box_map_cells
    if not (row_min <= cell[0] <= row_max and col_min <= cell[1] <= col_max):
        raise ValueError(
            f"projected source cell {cell.tolist()} lies outside the marked "
            f"box {args.red_box_map_cells}"
        )
    plan = build_augmented_plan(
        base_plan,
        selected,
        map_cell=cell,
        red_box_map_cells=args.red_box_map_cells,
        reference_image=str(reference_path.relative_to(ROOT)),
    )
    plan_path = write_curated_plan(plan, (ROOT / args.scenes_root).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "source_selection.png"
    _write_selection_image(
        reference_path,
        image_path,
        plan,
        episode,
        selected_object_id=int(args.object_id),
        red_box=args.red_box_map_cells,
        panel_x=int(args.map_panel_x),
        panel_y=int(args.map_panel_y),
        map_size_pixels=int(args.map_size_pixels),
        map_resolution_cm=float(args.map_resolution_cm),
    )
    display_pixel = _display_pixel(
        cell,
        panel_x=int(args.map_panel_x),
        panel_y=int(args.map_panel_y),
        map_size_pixels=int(args.map_size_pixels),
    )
    report = {
        "schema_version": 1,
        "base_plan_id": str(base_plan["plan_id"]),
        "plan_id": str(plan["plan_id"]),
        "plan_path": str(plan_path.relative_to(ROOT)),
        "selected_instance": {
            "instance_id": int(selected["instance_id"]),
            "category": str(selected["category"]),
            "centroid": [float(value) for value in selected["centroid"]],
            "flammability": float(selected["flammability"]),
            "structural": bool(selected["structural"]),
            "floor_id": int(selected["floor_id"]),
            "map_cell": [float(value) for value in cell],
            "display_pixel": list(display_pixel),
        },
        "red_box_map_cells": [float(value) for value in args.red_box_map_cells],
        "source_count": len(plan["ignitions"]),
        "reference_image": str(reference_path.relative_to(ROOT)),
        "selection_image": str(image_path.relative_to(ROOT)),
    }
    report_path = output_dir / "source_selection.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[additional-source] plan_id={plan['plan_id']}")
    print(f"[additional-source] plan={plan_path}")
    print(f"[additional-source] report={report_path}")
    print(f"[additional-source] image={image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
