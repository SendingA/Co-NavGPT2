"""Smoke-tests for utils.fire_world.scene_scan.

These run without habitat_sim: the path that requires sim is opt-in, so the
core inventory logic is fully testable against fixture data already in the
repo.

Run with::

    python scripts/test_scene_scan.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.scene_scan import (  # noqa: E402
    GoalObject,
    build_inventory,
    lookup_material,
    write_inventory,
)


def test_material_table() -> None:
    f, sy = lookup_material("BED")
    assert 0.0 <= f <= 1.0 and 0.0 <= sy <= 1.0, "out of range"
    f0, sy0 = lookup_material("definitely_not_a_real_category")
    assert (f0, sy0) == lookup_material("_default"), \
        "fallback to default missing"
    print("material_table: OK")


def test_build_inventory_TEEsavR23oF() -> None:
    inv = build_inventory(
        scene_id="TEEsavR23oF",
        scene_dataset_root=ROOT / "data" / "scene_datasets" / "hm3d_v0.2",
        objectgoal_root=ROOT / "data" / "datasets" / "objectnav_hm3d_v2",
    )
    assert inv["scene_id"] == "TEEsavR23oF"
    assert inv["world_aabb"] is not None and len(inv["world_aabb"]) == 6, \
        "world_aabb missing"
    assert inv["world_aabb_source"] in {"navmesh", "object_union"}
    assert len(inv["objects"]) >= 1, "no goal objects parsed"
    cats = {o["category"] for o in inv["objects"]}
    # The val_mini shard for this scene must have at least one goal object.
    assert "bed" in cats or "chair" in cats or "tv_monitor" in cats, \
        f"expected canonical goal cats, got {sorted(cats)}"
    sem = inv["semantic_summary"]
    assert sem["n_instances"] > 0, "semantic.txt parse failed"
    assert sem["global"], "no semantic categories aggregated"
    # Sanity: each object has consistent aabb.
    for o in inv["objects"]:
        amin, amax = o["aabb_min"], o["aabb_max"]
        assert all(amin[i] <= amax[i] for i in range(3)), o
    print(f"build_inventory: OK ({len(inv['objects'])} objects, "
          f"{sem['n_instances']} semantic instances)")


def test_write_inventory(tmp: Path) -> None:
    out = write_inventory(
        scene_id="TEEsavR23oF",
        out_root=tmp,
        scene_dataset_root=ROOT / "data" / "scene_datasets" / "hm3d_v0.2",
        objectgoal_root=ROOT / "data" / "datasets" / "objectnav_hm3d_v2",
    )
    assert out.exists() and out.stat().st_size > 100, "inventory not written"
    inv = json.loads(out.read_text())
    assert "objects" in inv and "_walkable_mask_in_memory" not in inv, \
        "in-memory mask leaked into the JSON"
    print(f"write_inventory: OK -> {out}")


def main() -> int:
    test_material_table()
    test_build_inventory_TEEsavR23oF()
    tmp = ROOT / "outputs" / "test_scene_scan_tmp"
    if tmp.exists():
        for p in tmp.rglob("*"):
            if p.is_file():
                p.unlink()
    test_write_inventory(tmp)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
