"""Smoke tests for utils.fire_world.planner.

Exercises:
  * deterministic plan_id under fixed inputs
  * all four templates produce non-empty plans on the val_mini fixture
  * same-floor constraint: secondary ignitions stay within +/-1.5 m of
    the primary's Y coordinate

Run with::

    python scripts/test_fire_planner.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.planner import (  # noqa: E402
    PLAN_SCHEMA_VERSION,
    build_plan,
    load_inventory,
    plan_id_for,
    write_plan,
)
from utils.fire_world.templates import INTENSITIES, TEMPLATES  # noqa: E402


SCENE = "TEEsavR23oF"
SCENES_ROOT = ROOT / "scenes"


def test_deterministic_plan_id() -> None:
    a = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    b = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    c = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 43)
    assert a == b, "plan_id changed under identical inputs"
    assert a != c, "plan_id should differ under different seeds"
    print(f"deterministic plan_id: {a} (seed=42) vs {c} (seed=43) -> OK")


def test_plan_id_in_payload() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    plan = build_plan(inv, "kitchen_grease_fire", "medium", 42)
    expected = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    assert plan["plan_id"] == expected
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["scene_id"] == SCENE
    assert plan["world_aabb"] is not None
    assert "propagation_rules" in plan and plan["propagation_rules"]
    print(f"plan payload: plan_id={plan['plan_id']} ignitions={len(plan['ignitions'])}")


def test_all_templates() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    for name in TEMPLATES:
        for intensity in INTENSITIES:
            plan = build_plan(inv, name, intensity, seed=11)
            assert plan["ignitions"], f"empty ignitions for {name}/{intensity}"
            for ig in plan["ignitions"]:
                assert ig["source_temp_c"] > 0
                assert ig["source_radius_m"] > 0
                assert ig["fuel_kg"] > 0
                assert ig["ignite_time_s"] >= 0
            print(f"  {name}/{intensity}: {len(plan['ignitions'])} ignitions")
    print("all templates: OK")


def test_same_floor_constraint() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    for name, intensity, seed in [
        ("kitchen_grease_fire", "medium", 42),
        ("bedroom_textile", "severe", 7),
        ("living_room_electric", "severe", 1),
        ("multi_origin", "severe", 0),
    ]:
        plan = build_plan(inv, name, intensity, seed)
        ys = [ig["position"][1] for ig in plan["ignitions"]]
        if len(ys) > 1:
            spread = max(ys) - min(ys)
            assert spread <= 1.5 + 1e-3, (
                f"ignitions span {spread:.2f} m vertically "
                f"({name}/{intensity}/seed={seed})"
            )
            print(f"  {name}/{intensity} seed={seed} y-spread={spread:.2f} m")
    print("same-floor constraint: OK")


def test_write_and_reread() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    out = write_plan(inv, "kitchen_grease_fire", "medium", 42, plans_root=SCENES_ROOT)
    plan = json.loads(out.read_text())
    assert plan["plan_id"] == plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    print(f"write_plan: OK -> {out}")


def main() -> int:
    test_deterministic_plan_id()
    test_plan_id_in_payload()
    test_all_templates()
    test_same_floor_constraint()
    test_write_and_reread()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
