"""Smoke tests for utils.fire_world.scene_scan (schema v2).

These run without habitat_sim: the new scan path uses only numpy +
trimesh-free GLB parsing. Test fixture is the val_mini scene
00800-TEEsavR23oF already shipped in the repo.

Run with::

    python scripts/test_scene_scan.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.scene_scan import (  # noqa: E402
    STRUCTURAL_CATEGORIES,
    build_inventory,
    lookup_material,
    write_inventory,
)


def test_material_table() -> None:
    f, sy = lookup_material("BED")
    assert 0.0 <= f <= 1.0 and 0.0 <= sy <= 1.0, "out of range"
    f0, sy0 = lookup_material("definitely_not_a_real_category")
    assert (f0, sy0) == lookup_material("_default")
    # Structural categories must be flagged non-flammable.
    for c in STRUCTURAL_CATEGORIES:
        assert lookup_material(c) == (0.0, 0.0), c
    print("material_table: OK")


def test_build_inventory_v2() -> None:
    inv, voxels = build_inventory(
        scene_id="TEEsavR23oF",
        scene_dataset_root=ROOT / "data" / "scene_datasets" / "hm3d_v0.2",
        objectgoal_root=ROOT / "data" / "datasets" / "objectnav_hm3d_v2",
    )
    assert inv["schema_version"] == 2
    assert inv["scene_id"] == "TEEsavR23oF"
    assert inv["world_aabb"] is not None and len(inv["world_aabb"]) == 6
    assert inv["world_aabb_source"] == "semantic_glb"
    assert len(inv["instances"]) > 200, \
        f"v2 inventory should recover hundreds of instances, got {len(inv['instances'])}"
    cats = {it["category"] for it in inv["instances"]}
    # Structural categories must be present (proves we are looking at the
    # full scene, not just goal objects).
    assert "wall" in cats and "floor" in cats and "ceiling" in cats, \
        f"missing structural categories; got {sorted(cats)[:10]}"
    # Floor clustering should have found the two HM3D floors.
    assert len(inv["floors"]) >= 1
    # Each instance has a consistent AABB.
    for it in inv["instances"]:
        amin, amax = it["aabb_min"], it["aabb_max"]
        assert all(amin[i] <= amax[i] for i in range(3)), it
    # Structural voxels should have been rasterised.
    assert voxels["walls"] is not None and voxels["walls"].any()
    assert voxels["ceilings"] is not None and voxels["ceilings"].any()
    print(f"build_inventory v2: OK ({len(inv['instances'])} instances, "
          f"{inv['build_summary']['n_structural_instances']} structural, "
          f"{inv['build_summary']['n_flammable_instances']} flammable, "
          f"{len(inv['floors'])} floor(s))")


def test_write_inventory(tmp: Path) -> None:
    out = write_inventory(
        scene_id="TEEsavR23oF",
        out_root=tmp,
        scene_dataset_root=ROOT / "data" / "scene_datasets" / "hm3d_v0.2",
        objectgoal_root=ROOT / "data" / "datasets" / "objectnav_hm3d_v2",
        save_structural_voxels=True,
    )
    assert out.exists() and out.stat().st_size > 1000
    inv = json.loads(out.read_text())
    assert "instances" in inv
    assert inv["structural"]["wall_voxel_path"], "wall path missing"
    wp = Path(inv["structural"]["wall_voxel_path"])
    assert wp.exists() and wp.stat().st_size > 0, "wall mask not written"
    print(f"write_inventory: OK -> {out}  ({wp.stat().st_size // 1024} KiB walls)")


def main() -> int:
    test_material_table()
    test_build_inventory_v2()
    tmp = ROOT / "outputs" / "test_scene_scan_tmp"
    if tmp.exists():
        for p in tmp.rglob("*"):
            if p.is_file():
                p.unlink()
        for d in sorted(tmp.rglob("*"), reverse=True):
            if d.is_dir():
                d.rmdir()
        tmp.rmdir() if tmp.exists() else None
    test_write_inventory(tmp)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
