"""Plan a deterministic fire scenario for one scene.

Stage 2 of the fire_world pipeline. Reads ``inventory.json`` (output of
stage 1) and emits ``plan.json`` describing the scenario to simulate. The
plan is the single source of truth consumed by stage 3 (propagation).

Determinism guarantees:
    - Same (scene_id, fire_type, intensity, seed, template_version) ALWAYS
      produces the same plan_id and the same ignitions.
    - plan_id = sha1("scene|type|intensity|seed|template_version")[:12].

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
    INTENSITIES,
    TEMPLATE_VERSION,
    TEMPLATES,
    _default_propagation_rules,
)


PLAN_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Plan id
# ---------------------------------------------------------------------------
def plan_id_for(
    scene_id: str,
    fire_type: str,
    intensity: str,
    seed: int,
    template_version: int = TEMPLATE_VERSION,
) -> str:
    key = f"{scene_id}|{fire_type}|{intensity}|{seed}|tpl{template_version}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------
def build_plan(
    inventory: Dict,
    fire_type: str,
    intensity: str,
    seed: int,
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
    preset = INTENSITIES[intensity]
    rng = np.random.default_rng(seed)

    ignitions: List[Dict] = TEMPLATES[fire_type](inventory, rng, preset)
    if not ignitions:
        raise RuntimeError(
            f"template {fire_type!r} could not pick any ignition objects "
            f"from {len(inventory['objects'])} candidates; "
            "consider running scene_scan with a richer dataset shard."
        )

    rules = _default_propagation_rules(intensity)
    pid = plan_id_for(inventory["scene_id"], fire_type, intensity, seed)

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": pid,
        "scene_id": inventory["scene_id"],
        "scene_glb": inventory.get("scene_glb"),
        "world_aabb": inventory["world_aabb"],
        "fire_type": fire_type,
        "intensity": intensity,
        "seed": int(seed),
        "template_version": int(TEMPLATE_VERSION),
        "duration_s": float(preset.duration_s),
        "ignitions": ignitions,
        "propagation_rules": rules,
    }


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
) -> Path:
    plan = build_plan(inventory, fire_type, intensity, seed)
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
    parser.add_argument("--scenes_root", default="scenes")
    parser.add_argument("--plans_root", default="scenes")
    parser.add_argument("--print_only", action="store_true",
                        help="don't write to disk; just print the plan json")
    args = parser.parse_args()

    inventory = load_inventory(args.scene, Path(args.scenes_root))
    plan = build_plan(inventory, args.fire_type, args.intensity, args.seed)

    if args.print_only:
        print(json.dumps(plan, indent=2))
        return 0

    out = write_plan(inventory, args.fire_type, args.intensity, args.seed,
                     Path(args.plans_root))
    print(f"[planner] {plan['scene_id']} {plan['fire_type']} "
          f"{plan['intensity']} seed={plan['seed']} -> "
          f"plan_id={plan['plan_id']}  ignitions={len(plan['ignitions'])}")
    for ig in plan["ignitions"]:
        print(f"  - t={ig['ignite_time_s']:6.1f}s  obj#{ig['object_id']:>3} "
              f"{ig['category']:>12s}  T={ig['source_temp_c']:.0f}C  "
              f"r={ig['source_radius_m']}m  fuel={ig['fuel_kg']}kg")
    print(f"[planner] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
