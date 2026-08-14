"""Smoke tests for utils.fire_world.planner.

Exercises:
  * deterministic plan_id under fixed inputs
  * all four templates produce non-empty plans on the val_mini fixture
  * every plan contains initial t=0 objects only
  * multi-origin initial objects span 4-6 semantic areas with 1-2 per area

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
    plan_hash_for,
    plan_id_for,
    write_plan,
)
from utils.fire_world.templates import (  # noqa: E402
    INTENSITIES,
    TEMPLATES,
    _initial_candidate_capacity,
    _inventory_pool,
)


SCENE = "TEEsavR23oF"
SCENES_ROOT = ROOT / "scenes"


def test_deterministic_plan_id() -> None:
    a = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    b = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    c = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 43)
    assert a == b, "plan_id changed under identical inputs"
    assert a != c, "plan_id should differ under different seeds"
    assert a.startswith(f"{SCENE}_kitchen_grease_fire_medium_")
    assert a.endswith(plan_hash_for(
        SCENE, "kitchen_grease_fire", "medium", 42
    ))
    print(f"deterministic plan_id: {a} (seed=42) vs {c} (seed=43) -> OK")


def test_plan_id_in_payload() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    plan = build_plan(inv, "kitchen_grease_fire", "medium", 42)
    expected = plan_id_for(SCENE, "kitchen_grease_fire", "medium", 42)
    assert plan["plan_id"] == expected
    assert plan["plan_hash"] == plan_hash_for(
        SCENE, "kitchen_grease_fire", "medium", 42
    )
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert plan["scene_id"] == SCENE
    assert plan["world_aabb"] is not None
    assert "propagation_rules" in plan and plan["propagation_rules"]
    print(f"plan payload: plan_id={plan['plan_id']} ignitions={len(plan['ignitions'])}")


def test_all_templates() -> None:
    inv = load_inventory(SCENE, SCENES_ROOT)
    objects = _inventory_pool(inv)
    for name in TEMPLATES:
        for intensity, preset in INTENSITIES.items():
            capacity = _initial_candidate_capacity(objects, name)
            if capacity < preset.n_ignitions_min:
                try:
                    build_plan(inv, name, intensity, seed=11)
                except RuntimeError:
                    print(
                        f"  {name}/{intensity}: infeasible as expected "
                        f"(capacity={capacity}, need={preset.n_ignitions_min})"
                    )
                    continue
                raise AssertionError(
                    f"{name}/{intensity} should reject capacity {capacity}"
                )
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
        ("living_room_electric", "medium", 1),
        ("multi_origin", "severe", 0),
    ]:
        plan = build_plan(inv, name, intensity, seed)
        initials = plan["ignitions"]
        assert len(initials) == plan["num_initial_ignitions"]
        assert all(ig["ignition_role"] == "initial" for ig in initials)
        assert all(ig["ignite_time_s"] == 0.0 for ig in initials)
        assert all("parent_object_id" not in ig for ig in initials)

        if name == "multi_origin":
            policy = plan["multi_origin_area_policy"]
            counts = policy["sources_per_area"]
            assert 4 <= len(counts) <= 6
            assert all(1 <= count <= 2 for count in counts.values())
            assert sum(counts.values()) == len(initials)
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
