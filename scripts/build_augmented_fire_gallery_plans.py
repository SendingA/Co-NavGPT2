#!/usr/bin/env python3
"""Build augmented medium FireWorld plans for the four-type RGB gallery.

The canonical template plans remain untouched.  Each derived plan keeps the
base plan's medium propagation physics, adds a small curated set of semantic
objects using the normal medium ignition preset, and records the preferred
furniture category that the gallery camera should frame.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.templates import INTENSITIES, _make_ignition  # noqa: E402


@dataclass(frozen=True)
class GalleryPlanSpec:
    scene_id: str
    fire_type: str
    base_plan_id: str
    added_object_ids: Tuple[int, ...]
    preferred_source_category: str
    replace_base_ignitions: bool = False


SPECS = (
    GalleryPlanSpec(
        "Dd4bFSTQ8gi",
        "kitchen_grease_fire",
        "Dd4bFSTQ8gi_kitchen_grease_fire_medium_ec574a477063",
        (234, 187, 177),  # towel, kitchen counter and stool in one kitchen
        "oven",
    ),
    GalleryPlanSpec(
        "DYehNKdT76V",
        "bedroom_textile",
        "DYehNKdT76V_bedroom_textile_medium_5db5708eb6f1",
        (302, 64, 65),  # pillows grouped around the two burning beds
        "bed",
    ),
    GalleryPlanSpec(
        "DYehNKdT76V",
        "living_room_electric",
        "DYehNKdT76V_living_room_electric_medium_8a2ef66c295f",
        # TV origin plus nearby couch, coffee table and pillows in region 7.
        (191, 180, 171, 177, 179, 183, 185),
        "led tv",
        True,
    ),
    GalleryPlanSpec(
        "p53SfW6mjZe",
        "multi_origin",
        "p53SfW6mjZe_multi_origin_medium_5ceb066712df",
        (791, 797),  # an additional bed/pillow origin area
        "couch",
    ),
)


def _inventory_objects(inventory: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    objects: Dict[int, Dict[str, Any]] = {}
    for raw in inventory.get("instances") or inventory.get("objects", ()):
        object_id = raw.get("object_id", raw.get("instance_id"))
        position = raw.get("position", raw.get("centroid"))
        if object_id is None or position is None or bool(raw.get("structural", False)):
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


def _plan_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _write_json_if_changed(path: Path, payload: Mapping[str, Any]) -> bool:
    """Write generated JSON only when its bytes changed.

    FireWorld uses plan/timeline modification times to flag stale baked data.
    Rewriting an identical deterministic plan would therefore create a false
    stale-cache warning even though the plan hash and physics are unchanged.
    """

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == encoded:
        return False
    path.write_text(encoded, encoding="utf-8")
    return True


def validate_gallery_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("intensity") != "medium":
        raise ValueError("gallery augmentation must preserve medium intensity")
    ignitions = list(plan.get("ignitions", ()))
    if not ignitions:
        raise ValueError("gallery plan has no ignitions")
    object_ids = [int(item["object_id"]) for item in ignitions]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("gallery plan contains duplicate ignition object IDs")
    for ignition in ignitions:
        if float(ignition["ignite_time_s"]) != 0.0:
            raise ValueError("gallery plan sources must be initial t=0 ignitions")
        if ignition.get("ignition_role") != "initial":
            raise ValueError("gallery plan sources must use the initial role")
    if int(plan["num_initial_ignitions"]) != len(ignitions):
        raise ValueError("num_initial_ignitions does not match ignitions")
    added = [int(value) for value in plan.get("gallery_added_object_ids", ())]
    if not added or not set(added).issubset(object_ids):
        raise ValueError("gallery_added_object_ids are missing from ignitions")


def build_plan(spec: GalleryPlanSpec, *, scenes_root: Path) -> Dict[str, Any]:
    scene_root = scenes_root / spec.scene_id
    base_path = scene_root / "plans" / f"{spec.base_plan_id}.json"
    inventory_path = scene_root / "inventory.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if base.get("intensity") != "medium" or base.get("fire_type") != spec.fire_type:
        raise ValueError(f"unexpected base plan semantics: {base_path}")

    objects = _inventory_objects(inventory)
    missing = [object_id for object_id in spec.added_object_ids if object_id not in objects]
    if missing:
        raise ValueError(f"{spec.scene_id} inventory is missing objects {missing}")

    existing = (
        set()
        if spec.replace_base_ignitions
        else {int(item["object_id"]) for item in base["ignitions"]}
    )
    duplicates = sorted(existing.intersection(spec.added_object_ids))
    if duplicates:
        raise ValueError(f"added objects already exist in base plan: {duplicates}")

    rng = np.random.default_rng(int(base.get("seed", 0)) + 20260828)
    preset = INTENSITIES["medium"]
    added = []
    for object_id in spec.added_object_ids:
        ignition = _make_ignition(objects[object_id], 0.0, preset, rng)
        ignition["ignition_role"] = "initial"
        added.append(ignition)

    plan = {
        key: value
        for key, value in base.items()
        if key not in {"plan_id", "plan_hash"}
    }
    retained = [] if spec.replace_base_ignitions else list(base["ignitions"])
    plan["ignitions"] = retained + added
    plan["num_initial_ignitions"] = len(plan["ignitions"])
    plan["num_fire_sources"] = len(plan["ignitions"])
    base_range = base.get("default_initial_ignition_range")
    min_sources = (
        int(base_range.get("min", 1))
        if isinstance(base_range, Mapping)
        else 1
    )
    plan["default_initial_ignition_range"] = {
        "min": min_sources,
        "max": len(plan["ignitions"]),
    }
    plan["ignition_selection_mode"] = "curated_gallery_augmented_initial_only"
    plan["ignition_selection_version"] = 1
    plan["base_plan_id"] = spec.base_plan_id
    plan["gallery_added_object_ids"] = list(spec.added_object_ids)
    plan["gallery_preferred_source_category"] = spec.preferred_source_category
    if spec.replace_base_ignitions:
        plan["gallery_replaced_base_ignitions"] = True

    plan_hash = _plan_hash(plan)
    plan_id = (
        f"{spec.scene_id}_{spec.fire_type}_medium_gallery_augmented_{plan_hash}"
    )
    plan["plan_hash"] = plan_hash
    plan["plan_id"] = plan_id
    validate_gallery_plan(plan)
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scenes_root = Path(args.scenes_root)
    records = []
    for spec in SPECS:
        plan = build_plan(spec, scenes_root=scenes_root)
        output = scenes_root / spec.scene_id / "plans" / f"{plan['plan_id']}.json"
        if not args.dry_run:
            _write_json_if_changed(output, plan)
        record = {
            "scene_id": spec.scene_id,
            "fire_type": spec.fire_type,
            "plan_id": plan["plan_id"],
            "source_count": len(plan["ignitions"]),
            "added_object_ids": list(spec.added_object_ids),
            "added_categories": [
                item["category"]
                for item in plan["ignitions"]
                if int(item["object_id"]) in spec.added_object_ids
            ],
            "preferred_source_category": spec.preferred_source_category,
            "output": str(output),
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
