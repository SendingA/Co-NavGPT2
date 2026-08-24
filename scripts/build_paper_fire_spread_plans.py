#!/usr/bin/env python3
"""Build short, publication-only FireWorld source-to-source spread plans.

These plans are intentionally separate from benchmark plans.  The normal
planner emits t=0 initial sources only and lets the propagation solver ignite
unplanned fuel.  For a controlled paper figure, however, this script authors a
small number of *staged* secondary sources with explicit parent, generation,
distance and activation time metadata.  That makes the visual sequence
reproducible and keeps its causal assumptions honest in the exported manifest.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.templates import INTENSITIES, _make_ignition  # noqa: E402


@dataclass(frozen=True)
class Stage:
    object_id: int
    ignite_time_s: float
    parent_object_id: Optional[int]
    generation: int


@dataclass(frozen=True)
class SpreadPlanSpec:
    scene_id: str
    fire_type: str
    intensity: str
    base_plan_id: str
    stages: Tuple[Stage, ...]
    sequence_times_s: Tuple[float, float, float]
    duration_s: float = 210.0


SPECS: Tuple[SpreadPlanSpec, ...] = (
    SpreadPlanSpec(
        scene_id="4ok3usBNeis",
        fire_type="kitchen_grease_fire",
        intensity="severe",
        base_plan_id="4ok3usBNeis_kitchen_grease_fire_severe_5709cbe790aa",
        stages=(
            Stage(328, 0.0, None, 0),          # oven and stove
            Stage(329, 35.0, 328, 1),         # cloth
            Stage(322, 70.0, 329, 2),         # toaster
            Stage(324, 105.0, 322, 3),        # microwave
        ),
        sequence_times_s=(10.0, 80.0, 130.0),
    ),
    SpreadPlanSpec(
        scene_id="mL8ThkuaVTM",
        fire_type="bedroom_textile",
        intensity="severe",
        base_plan_id="mL8ThkuaVTM_bedroom_textile_severe_c60a8c96d8a7",
        stages=(
            Stage(78, 0.0, None, 0),           # bed, bedroom A
            Stage(26, 0.0, None, 0),           # bed, bedroom B
            Stage(79, 25.0, 78, 1),
            Stage(80, 25.0, 78, 1),
            Stage(27, 25.0, 26, 1),
            Stage(81, 50.0, 79, 2),
            Stage(82, 50.0, 80, 2),
            Stage(83, 75.0, 82, 3),
            Stage(84, 75.0, 81, 3),
            Stage(93, 105.0, 83, 4),           # curtain
            Stage(92, 105.0, 80, 4),           # curtain
        ),
        sequence_times_s=(10.0, 65.0, 130.0),
    ),
    SpreadPlanSpec(
        scene_id="QaLdnwvtxbs",
        fire_type="living_room_electric",
        intensity="severe",
        base_plan_id="QaLdnwvtxbs_living_room_electric_severe_44eb1448cf78",
        stages=(
            Stage(196, 0.0, None, 0),          # television: electrical cause
            Stage(197, 30.0, 196, 1),          # co-located TV stand
            Stage(166, 65.0, 197, 2),          # nearby curtain
            Stage(185, 105.0, 166, 3),         # nearby sofa
        ),
        sequence_times_s=(10.0, 80.0, 130.0),
    ),
    SpreadPlanSpec(
        scene_id="Dd4bFSTQ8gi",
        fire_type="multi_origin",
        intensity="medium",
        base_plan_id="Dd4bFSTQ8gi_multi_origin_medium_936f5508fca4",
        stages=(
            Stage(8, 0.0, None, 0),            # origin A: couch
            Stage(9, 0.0, None, 0),            # origin B: couch
            Stage(57, 30.0, 8, 1),             # A -> pillow
            Stage(53, 30.0, 9, 1),             # B -> pillow
            Stage(58, 55.0, 57, 2),            # A -> second pillow
            Stage(54, 55.0, 53, 2),            # B -> second pillow
            Stage(62, 85.0, 58, 3),            # A -> blanket
            Stage(7, 115.0, 8, 3),             # A -> rug/floor fuel
        ),
        sequence_times_s=(10.0, 65.0, 135.0),
    ),
)


def _inventory_objects(inventory: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Return the full inventory using the shape expected by _make_ignition."""

    objects: Dict[int, Dict[str, Any]] = {}
    for raw in inventory.get("instances") or inventory.get("objects", ()):
        object_id = raw.get("object_id", raw.get("instance_id"))
        if object_id is None or bool(raw.get("structural", False)):
            continue
        position = raw.get("position", raw.get("centroid"))
        if position is None:
            continue
        objects[int(object_id)] = {
            "object_id": int(object_id),
            "category": str(raw["category"]),
            "position": [float(value) for value in position],
            "aabb_min": [float(value) for value in raw["aabb_min"]],
            "aabb_max": [float(value) for value in raw["aabb_max"]],
            "flammability": float(raw.get("flammability", 0.0)),
            "smoke_yield": float(raw.get("smoke_yield", 0.4)),
            "region_id": raw.get("region_id"),
            "floor_id": raw.get("floor_id"),
        }
    return objects


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def validate_spread_plan(plan: Mapping[str, Any]) -> None:
    """Validate staged-source ordering and parent causality."""

    ignitions = list(plan.get("ignitions", ()))
    if not ignitions:
        raise ValueError("spread plan has no ignitions")
    ids = [int(ignition["object_id"]) for ignition in ignitions]
    if len(ids) != len(set(ids)):
        raise ValueError("spread plan contains duplicate object IDs")
    by_id = {int(ignition["object_id"]): ignition for ignition in ignitions}
    initial_count = 0
    for ignition in ignitions:
        object_id = int(ignition["object_id"])
        time_s = float(ignition["ignite_time_s"])
        generation = int(ignition["spread_generation"])
        parent_id = ignition.get("parent_object_id")
        if parent_id is None:
            initial_count += 1
            if time_s != 0.0 or generation != 0:
                raise ValueError(f"initial source {object_id} must start at t=0")
            if ignition.get("ignition_role") != "initial":
                raise ValueError(f"initial source {object_id} has wrong role")
            continue
        parent_id = int(parent_id)
        if parent_id not in by_id:
            raise ValueError(f"source {object_id} has missing parent {parent_id}")
        parent = by_id[parent_id]
        if float(parent["ignite_time_s"]) >= time_s:
            raise ValueError(f"source {object_id} does not ignite after its parent")
        if int(parent["spread_generation"]) >= generation:
            raise ValueError(f"source {object_id} has invalid spread generation")
        if ignition.get("ignition_role") != "secondary_spread":
            raise ValueError(f"secondary source {object_id} has wrong role")
        if float(ignition["spread_distance_m"]) <= 0.0:
            raise ValueError(f"source {object_id} has invalid parent distance")
    if initial_count != int(plan["num_initial_ignitions"]):
        raise ValueError("num_initial_ignitions does not match staged entries")
    if len(ignitions) != int(plan["num_fire_sources"]):
        raise ValueError("num_fire_sources does not match staged entries")
    if max(float(item["ignite_time_s"]) for item in ignitions) >= float(
        plan["duration_s"]
    ):
        raise ValueError("last source activates outside the plan duration")


def build_plan(
    spec: SpreadPlanSpec,
    *,
    scenes_root: Path,
) -> Dict[str, Any]:
    inventory_path = scenes_root / spec.scene_id / "inventory.json"
    base_path = (
        scenes_root / spec.scene_id / "plans" / f"{spec.base_plan_id}.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    objects = _inventory_objects(inventory)
    missing = [stage.object_id for stage in spec.stages if stage.object_id not in objects]
    if missing:
        raise ValueError(f"{spec.scene_id} is missing inventory objects {missing}")

    rng = np.random.default_rng(int(base.get("seed", 0)) + 20260819)
    preset = INTENSITIES[spec.intensity]
    ignitions: List[Dict[str, Any]] = []
    for stage in spec.stages:
        obj = objects[stage.object_id]
        ignition = _make_ignition(obj, stage.ignite_time_s, preset, rng)
        ignition["spread_generation"] = int(stage.generation)
        if stage.parent_object_id is None:
            ignition["ignition_role"] = "initial"
        else:
            parent = objects[stage.parent_object_id]
            ignition["ignition_role"] = "secondary_spread"
            ignition["parent_object_id"] = int(stage.parent_object_id)
            ignition["spread_distance_m"] = round(
                float(
                    np.linalg.norm(
                        np.asarray(obj["position"], dtype=np.float64)
                        - np.asarray(parent["position"], dtype=np.float64)
                    )
                ),
                3,
            )
            # Secondary fronts should remain local enough for adjacent flames
            # to stay visually distinct in the late paper frame.
            ignition["floor_spread_scale"] = 0.72
        ignitions.append(ignition)

    initial_count = sum(item["ignition_role"] == "initial" for item in ignitions)
    plan: Dict[str, Any] = {
        key: value
        for key, value in base.items()
        if key
        not in {
            "plan_id",
            "duration_s",
            "ignitions",
            "num_initial_ignitions",
            "num_initial_ignitions_requested",
            "default_initial_ignition_range",
            "ignition_selection_mode",
            "ignition_selection_version",
        }
    }
    plan.update(
        {
            "template_version": max(12, int(base.get("template_version", 0))),
            "duration_s": float(spec.duration_s),
            "num_initial_ignitions": int(initial_count),
            "num_fire_sources": len(ignitions),
            "ignition_selection_mode": "curated_publication_spread_chain",
            "ignition_selection_version": 1,
            "ignitions": ignitions,
            "paper_spread_sequence_times_s": [
                float(value) for value in spec.sequence_times_s
            ],
            "paper_spread_disclosure": (
                "Secondary sources are explicitly staged for a controlled "
                "publication sequence; parent links are authored scenario "
                "metadata, not solver-inferred causal attribution."
            ),
            "base_plan_id": spec.base_plan_id,
        }
    )
    hash_payload = {
        "scene_id": spec.scene_id,
        "fire_type": spec.fire_type,
        "intensity": spec.intensity,
        "duration_s": spec.duration_s,
        "stages": [
            {
                "object_id": stage.object_id,
                "ignite_time_s": stage.ignite_time_s,
                "parent_object_id": stage.parent_object_id,
                "generation": stage.generation,
            }
            for stage in spec.stages
        ],
        "base_plan_id": spec.base_plan_id,
    }
    plan["plan_id"] = (
        f"{spec.scene_id}_{spec.fire_type}_paper_spread_"
        f"{_canonical_hash(hash_payload)}"
    )
    validate_spread_plan(plan)
    return plan


def build_all(
    *,
    scenes_root: Path,
    manifest_path: Path,
) -> List[Dict[str, Any]]:
    records = []
    for spec in SPECS:
        plan = build_plan(spec, scenes_root=scenes_root)
        path = scenes_root / spec.scene_id / "plans" / f"{plan['plan_id']}.json"
        path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        records.append(
            {
                "scene_id": spec.scene_id,
                "fire_type": spec.fire_type,
                "intensity": spec.intensity,
                "plan_id": plan["plan_id"],
                "plan_path": str(path),
                "base_plan_id": spec.base_plan_id,
                "num_initial_ignitions": plan["num_initial_ignitions"],
                "num_fire_sources": plan["num_fire_sources"],
                "sequence_times_s": plan["paper_spread_sequence_times_s"],
            }
        )
        print(
            f"[paper-spread] {spec.fire_type}: {plan['plan_id']} "
            f"initial={plan['num_initial_ignitions']} "
            f"total={plan['num_fire_sources']}"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"plans": records}, indent=2) + "\n", encoding="utf-8"
    )
    return records


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument(
        "--manifest",
        default="outputs/paper_fire_spread_plans/manifest.json",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    build_all(
        scenes_root=Path(args.scenes_root).resolve(),
        manifest_path=Path(args.manifest).resolve(),
    )


if __name__ == "__main__":
    main()
