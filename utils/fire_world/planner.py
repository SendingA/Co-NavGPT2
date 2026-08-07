"""Plan a deterministic fire scenario for one scene.

Stage 2 of the fire_world pipeline. Reads ``inventory.json`` (output of
stage 1) and emits ``plan.json`` describing the scenario to simulate. The
plan is the single source of truth consumed by stage 3 (propagation).

Determinism guarantees:
    - Same (scene_id, fire_type, intensity, seed, template_version,
      optional num_ignitions) ALWAYS produces the same semantic plan_id,
      stable plan_hash, and ignitions.
    - The planner emits t=0 initial objects only. An explicit count
      participates in the plan hash; otherwise the intensity preset
      deterministically supplies the count.
    - Objects ignited later are selected by the propagation solver, never by
      a planner-authored secondary list.

Future-proofing:
    The current implementation picks ignitions from a small set of
    hand-written templates. A later PR can add an LLM mode behind
    ``--use_llm`` that fills the same JSON schema. The downstream stages
    only see the schema, so they don't need to change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .templates import (
    IGNITION_SELECTION_VERSION,
    INTENSITIES,
    TEMPLATE_VERSION,
    TEMPLATES,
    _default_propagation_rules,
    build_template_ignitions,
)
from .plan_ids import semantic_plan_id


PLAN_SCHEMA_VERSION = 4


# ---------------------------------------------------------------------------
# Plan id
# ---------------------------------------------------------------------------
def plan_hash_for(
    scene_id: str,
    fire_type: str,
    intensity: str,
    seed: int,
    template_version: int = TEMPLATE_VERSION,
    num_ignitions: Optional[int] = None,
) -> str:
    """Return the stable 12-hex identity component for a plan."""
    key = (
        f"{scene_id}|{fire_type}|{intensity}|{seed}|tpl{template_version}"
        f"|initial-only-v{IGNITION_SELECTION_VERSION}"
    )
    if num_ignitions is not None:
        requested = _validate_num_ignitions(num_ignitions)
        key += f"|n{requested}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def plan_id_for(
    scene_id: str,
    fire_type: str,
    intensity: str,
    seed: int,
    template_version: int = TEMPLATE_VERSION,
    num_ignitions: Optional[int] = None,
) -> str:
    """Return ``scene_type_intensity_hash`` for human-readable lookup."""
    plan_hash = plan_hash_for(
        scene_id,
        fire_type,
        intensity,
        seed,
        template_version=template_version,
        num_ignitions=num_ignitions,
    )
    return semantic_plan_id(scene_id, fire_type, intensity, plan_hash)


def _validate_num_ignitions(num_ignitions: int) -> int:
    """Return a positive exact initial-source count or raise a clear error."""
    if isinstance(num_ignitions, bool):
        raise ValueError("num_ignitions must be a positive integer, not bool")
    try:
        requested = int(num_ignitions)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"num_ignitions must be a positive integer, got {num_ignitions!r}"
        ) from exc
    if requested != num_ignitions or requested < 1:
        raise ValueError(
            f"num_ignitions must be a positive integer, got {num_ignitions!r}"
        )
    return requested


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------
def build_plan(
    inventory: Dict,
    fire_type: str,
    intensity: str,
    seed: int,
    num_ignitions: Optional[int] = None,
) -> Dict:
    if fire_type not in TEMPLATES:
        raise ValueError(
            f"unknown fire_type {fire_type!r}; "
            f"available: {sorted(TEMPLATES.keys())}"
        )
    if intensity not in INTENSITIES:
        raise ValueError(
            f"unknown intensity {intensity!r}; "
            f"available: {sorted(INTENSITIES.keys())}"
        )
    requested_count = (
        None
        if num_ignitions is None
        else _validate_num_ignitions(num_ignitions)
    )
    if fire_type == "multi_origin" and requested_count == 1:
        raise ValueError(
            "fire_type='multi_origin' requires num_ignitions >= 2"
        )

    preset = INTENSITIES[intensity]
    rng = np.random.default_rng(seed)

    ignitions: List[Dict] = build_template_ignitions(
        inventory,
        fire_type,
        rng,
        preset,
        requested_count,
    )
    if not ignitions:
        raise RuntimeError(
            f"template {fire_type!r} could not pick any ignition objects "
            f"from {len(inventory['objects'])} candidates; "
            "consider running scene_scan with a richer dataset shard."
        )
    initial_count = sum(
        ignition["ignite_time_s"] == 0.0
        and ignition.get("ignition_role") == "initial"
        for ignition in ignitions
    )
    if requested_count is not None and initial_count != requested_count:
        raise RuntimeError(
            f"template {fire_type!r} generated {initial_count} initial "
            f"sources for num_ignitions={requested_count}"
        )
    if initial_count != len(ignitions):
        raise RuntimeError(
            f"template {fire_type!r} emitted non-initial ignition entries; "
            "the planner contract permits t=0 initial sources only"
        )

    rules = _default_propagation_rules(intensity)
    plan_hash = plan_hash_for(
        inventory["scene_id"],
        fire_type,
        intensity,
        seed,
        num_ignitions=requested_count,
    )

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": semantic_plan_id(
            inventory["scene_id"],
            fire_type,
            intensity,
            plan_hash,
        ),
        "plan_hash": plan_hash,
        "scene_id": inventory["scene_id"],
        "scene_glb": inventory.get("scene_glb"),
        "world_aabb": inventory["world_aabb"],
        "fire_type": fire_type,
        "intensity": intensity,
        "seed": int(seed),
        "template_version": int(TEMPLATE_VERSION),
        "duration_s": float(preset.duration_s),
        "num_initial_ignitions": initial_count,
        "ignition_selection_mode": "initial_only",
        "ignition_selection_version": IGNITION_SELECTION_VERSION,
        "ignitions": ignitions,
        "propagation_rules": rules,
    }
    if requested_count is not None:
        plan["num_initial_ignitions_requested"] = requested_count
    return plan


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_inventory(scene_id: str, scenes_root: Path) -> Dict:
    p = scenes_root / scene_id / "inventory.json"
    if not p.exists():
        raise FileNotFoundError(
            f"inventory.json not found for scene {scene_id} (expected {p}). "
            "Run utils.fire_world.scene_scan first."
        )
    return json.loads(p.read_text())


def write_plan(
    inventory: Dict,
    fire_type: str,
    intensity: str,
    seed: int,
    plans_root: Path,
    num_ignitions: Optional[int] = None,
) -> Path:
    plan = build_plan(
        inventory,
        fire_type,
        intensity,
        seed,
        num_ignitions=num_ignitions,
    )
    out_dir = plans_root / inventory["scene_id"] / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{plan['plan_id']}.json"
    out.write_text(json.dumps(plan, indent=2))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Generate plan.json for a fire scenario from an inventory."
    )
    parser.add_argument("--scene", required=True,
                        help="scene short id matching the inventory dir, "
                             "e.g. TEEsavR23oF.")
    parser.add_argument("--fire_type", required=True,
                        choices=sorted(TEMPLATES.keys()))
    parser.add_argument("--intensity", default="medium",
                        choices=sorted(INTENSITIES.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_ignitions",
        type=int,
        default=None,
        help="exact number of initial ignition objects lit at t=0. Later "
             "object ignition is decided by the propagation solver. Omit to "
             "draw the initial count from the intensity preset.",
    )
    parser.add_argument("--scenes_root", default="scenes")
    parser.add_argument("--plans_root", default="scenes")
    parser.add_argument("--print_only", action="store_true",
                        help="don't write to disk; just print the plan json")
    args = parser.parse_args()

    inventory = load_inventory(args.scene, Path(args.scenes_root))
    plan = build_plan(
        inventory,
        args.fire_type,
        args.intensity,
        args.seed,
        num_ignitions=args.num_ignitions,
    )

    if args.print_only:
        print(json.dumps(plan, indent=2))
        return 0

    out = write_plan(
        inventory,
        args.fire_type,
        args.intensity,
        args.seed,
        Path(args.plans_root),
        num_ignitions=args.num_ignitions,
    )
    count_summary = f"initial_ignitions={plan['num_initial_ignitions']}"
    print(f"[planner] {plan['scene_id']} {plan['fire_type']} "
          f"{plan['intensity']} seed={plan['seed']} -> "
          f"plan_id={plan['plan_id']}  {count_summary}")
    for ig in plan["ignitions"]:
        print(f"  - t={ig['ignite_time_s']:6.1f}s  obj#{ig['object_id']:>3} "
              f"{ig['category']:>12s}  T={ig['source_temp_c']:.0f}C  "
              f"r={ig['source_radius_m']}m  fuel={ig['fuel_kg']}kg")
    print(f"[planner] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
