"""Smoke tests for utils.fire_world.planner.

Exercises:
  * deterministic plan_id under fixed inputs
  * all four templates produce non-empty plans on the val_mini fixture
  * every plan contains initial t=0 objects only
  * multi-origin initial objects remain on one floor

Run with::

    python scripts/test_fire_planner.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

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
            assert len(plan["ignitions"]) == plan["num_initial_ignitions"]
            for ig in plan["ignitions"]:
                assert ig["source_temp_c"] > 0
                assert ig["source_radius_m"] > 0
                assert ig["fuel_kg"] > 0
                assert ig["ignite_time_s"] == 0
                assert ig["ignition_role"] == "initial"
                assert "parent_object_id" not in ig
            print(f"  {name}/{intensity}: {len(plan['ignitions'])} ignitions")
    print("all templates: OK")


def test_initial_only_constraint() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    for name, intensity, seed in [
        ("kitchen_grease_fire", "medium", 42),
        ("bedroom_textile", "severe", 7),
        ("living_room_electric", "severe", 1),
        ("multi_origin", "severe", 0),
    ]:
        plan = build_plan(inv, name, intensity, seed)
        initials = plan["ignitions"]
        assert len(initials) == plan["num_initial_ignitions"]
        assert all(ig["ignition_role"] == "initial" for ig in initials)
        assert all(ig["ignite_time_s"] == 0.0 for ig in initials)
        assert all("parent_object_id" not in ig for ig in initials)

        if name == "multi_origin":
            initial_ys = [
                ignition["position"][1] for ignition in initials
            ]
            spread = max(initial_ys) - min(initial_ys)
            assert spread <= 1.5 + 1e-3, (
                f"multi_origin initial sources span {spread:.2f} m "
                "vertically"
            )
        print(
            f"  {name}/{intensity} seed={seed} "
            f"initials={len(initials)} initial-only=OK"
        )
    print("initial-only planner constraints: OK")


def test_write_and_reread() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    with TemporaryDirectory() as temp_dir:
        out = write_plan(
            inv,
            "kitchen_grease_fire",
            "medium",
            42,
            plans_root=Path(temp_dir),
        )
        plan = json.loads(out.read_text())
    assert plan["plan_id"] == plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    print(f"write_plan temporary round-trip: OK -> {out.name}")


def main() -> int:
    test_deterministic_plan_id()
    test_plan_id_in_payload()
    test_all_templates()
    test_initial_only_constraint()
    test_write_and_reread()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
