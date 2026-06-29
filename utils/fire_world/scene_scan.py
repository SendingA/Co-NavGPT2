"""Build a deterministic ``inventory.json`` (schema v2) for an HM3D scene.

This rewrite of the scan stage uses the per-vertex colour information in
``*.semantic.glb`` (decoded with the sRGB OETF, see
``utils/fire_world/hm3d_semantic.py``) to recover **every** instance, not
just the 5-6 goal categories that the v1 scan harvested from
ObjectGoal-NAV.

Output schema v2:

    {
      "schema_version": 2,
      "scene_id": "TEEsavR23oF",
      "scene_dir": "data/scene_datasets/hm3d_v0.2/val/00800-TEEsavR23oF",
      "scene_glb": "...basis.glb",
      "objectgoal_shard": "data/datasets/.../{scene}.json.gz" | null,
      "world_aabb": [xmin, ymin, zmin, xmax, ymax, zmax],
      "world_aabb_source": "semantic_glb" | "object_union" | "navmesh",
      "voxel_m": 0.10,
      "floors": [{"id": int, "y": float, "y_min": float, "y_max": float}, ...],
      "instances": [
        {"instance_id", "category", "color_hex", "region_id",
         "aabb_min", "aabb_max", "centroid",
         "n_vertices", "n_faces",
         "structural": bool, "is_goal": bool, "goal_object_id": int|null,
         "flammability", "smoke_yield",
         "floor_id": int|null}
      ],
      "structural": {
        "wall_voxel_path":     "scenes/{id}/structural/walls.npy",
        "floor_voxel_path":    "scenes/{id}/structural/floors.npy",
        "ceiling_voxel_path":  "scenes/{id}/structural/ceilings.npy",
        "voxel_m": 0.10,
        "origin": [x, y, z]
      },
      "semantic_summary": { ... per-region histograms ... }
    }

Rationale:
    - ``instances`` is the world model used everywhere downstream (planner,
      voxel propagation, top-down rendering).
    - ``structural`` carries pre-rasterised obstacle masks for walls,
      floors, ceilings; the propagation engine uses them for zero-flux
      boundary conditions and ceiling jets.
    - The legacy ``objects`` field is kept as a *backward-compat* alias
      pointing to the goal-flagged instances so older planners (template
      v1) keep running. New code should read ``instances`` instead.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .hm3d_semantic import (
    InstanceGeom,
    aggregate_instances,
    read_semantic_txt,
)


# ---------------------------------------------------------------------------
# Material table - per-category fire properties.
# ---------------------------------------------------------------------------
# Numbers are dimensionless [0, 1]: flammability ~ relative ignition
# speed, smoke_yield ~ relative dense smoke produced when burning.
MATERIAL_TABLE: Dict[str, Tuple[float, float]] = {
    # furniture (cloth / wood / foam)
    "bed":         (0.75, 0.80),
    "couch":       (0.80, 0.85),
    "sofa":        (0.80, 0.85),
    "armchair":    (0.75, 0.75),
    "chair":       (0.55, 0.50),
    "stool":       (0.50, 0.45),
    "ottoman":     (0.55, 0.55),
    "wardrobe":    (0.60, 0.55),
    "cabinet":     (0.55, 0.50),
    "shelf":       (0.55, 0.45),
    "bookshelf":   (0.70, 0.55),
    "dresser":     (0.55, 0.50),
    "nightstand":  (0.50, 0.40),
    "desk":        (0.55, 0.45),
    "table":       (0.55, 0.45),
    "side table":  (0.55, 0.45),
    "coffee table": (0.55, 0.45),
    "drawer":      (0.55, 0.45),
    # textiles
    "curtain":     (0.85, 0.75),
    "blanket":     (0.80, 0.60),
    "pillow":      (0.80, 0.55),
    "rug":         (0.65, 0.55),
    "carpet":      (0.65, 0.55),
    "towel":       (0.75, 0.55),
    "clothes":     (0.75, 0.55),
    "plush toy":   (0.70, 0.55),
    # paper / books
    "book":        (0.75, 0.50),
    "stack of papers": (0.85, 0.55),
    "magazine":    (0.85, 0.55),
    # plants
    "potted plant": (0.50, 0.45),
    "plant":        (0.50, 0.45),
    # appliances / electronics
    "tv":          (0.40, 0.55),
    "tv_monitor":  (0.40, 0.55),
    "monitor":     (0.40, 0.55),
    "computer":    (0.50, 0.55),
    "table lamp":  (0.40, 0.40),
    "lamp":        (0.40, 0.40),
    "chandelier":  (0.30, 0.30),
    "stove":       (0.85, 0.70),
    "ventilation hood": (0.20, 0.30),
    "refrigerator": (0.30, 0.40),
    # bathroom / utilities (porcelain / metal / water)
    "toilet":      (0.05, 0.05),
    "bathtub":     (0.05, 0.05),
    "sink":        (0.05, 0.05),
    "bottle of soap": (0.10, 0.10),
    "toilet paper":   (0.85, 0.55),
    # structural -> not flammable; tagged in STRUCTURAL_CATEGORIES below
    "_default":    (0.30, 0.30),
}

STRUCTURAL_CATEGORIES = {
    "wall", "floor", "ceiling", "door", "door frame", "window",
    "window frame", "stairs", "staircase", "balustrade", "handrail",
    "handle", "column", "beam", "wall hanging decoration", "picture",
    "moulding",
}


def lookup_material(category: str) -> Tuple[float, float]:
    cat = category.lower().strip()
    if cat in STRUCTURAL_CATEGORIES:
        return (0.0, 0.0)
    return MATERIAL_TABLE.get(cat, MATERIAL_TABLE["_default"])


# ---------------------------------------------------------------------------
# Floor clustering (1D k-means-lite on instance-y).
# ---------------------------------------------------------------------------
def cluster_floors(instances: List[InstanceGeom],
                   y_gap_m: float = 1.5) -> List[Dict]:
    """Group instances into floors by their y_min, then pick a stable
    ``floor_y`` for each group as the median of the instance y_min values."""
    if not instances:
        return []
    floor_inst = [i for i in instances if i.category.lower() == "floor"]
    if floor_inst:
        ys = sorted(i.aabb_min[1] for i in floor_inst)
    else:
        ys = sorted(i.aabb_min[1] for i in instances)
    # Chain merge: clusters bounded by y_gap_m gaps.
    clusters: List[List[float]] = []
    for y in ys:
        if not clusters or y - clusters[-1][-1] > y_gap_m:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    floors: List[Dict] = []
    for fi, cl in enumerate(clusters):
        y_floor = float(np.median(cl))
        floors.append({
            "id": fi,
            "y": float(y_floor),
            "y_min": float(min(cl) - 0.10),
            "y_max": float(min(cl) + 3.20),  # typical floor-to-ceiling
        })
    # The next-floor's y is also an upper bound for this floor.
    for fi in range(len(floors) - 1):
        floors[fi]["y_max"] = float(floors[fi + 1]["y"]) - 0.10
    return floors


def assign_floor(inst: InstanceGeom, floors: List[Dict]) -> Optional[int]:
    if not floors:
        return None
    cy = 0.5 * (inst.aabb_min[1] + inst.aabb_max[1])
    best, best_d = None, float("inf")
    for f in floors:
        if f["y_min"] - 0.20 <= cy <= f["y_max"] + 0.20:
            d = abs(cy - f["y"])
            if d < best_d:
                best, best_d = f["id"], d
    if best is None:
        # fall back to closest by y centre
        for f in floors:
            d = abs(cy - f["y"])
            if d < best_d:
                best, best_d = f["id"], d
    return int(best) if best is not None else None


# ---------------------------------------------------------------------------
# ObjectGoal lookup (used to flag is_goal and reuse object_id).
# ---------------------------------------------------------------------------
def _scene_short(s: str) -> str:
    """Normalise an HM3D scene reference to its short id.

    Accepts:
        Nfvxx8J5NCo
        00880-Nfvxx8J5NCo
        data/.../00880-Nfvxx8J5NCo
        data/.../00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb
    """
    s = s.split("/")[-1]
    s = s.replace(".basis.glb", "").replace(".glb", "")
    # Strip the optional HM3D numeric prefix like "00880-".
    if "-" in s:
        head, tail = s.split("-", 1)
        if head.isdigit():
            s = tail
    return s


def _find_objectgoal_shard(scene_short: str, objectgoal_root: Path,
                           splits: Iterable[str]) -> Optional[Path]:
    for sp in splits:
        cand = objectgoal_root / sp / "content" / f"{scene_short}.json.gz"
        if cand.exists():
            return cand
    return None


def _read_objectgoal_ids(shard: Path) -> Dict[int, Dict]:
    if shard.suffix == ".gz":
        d = json.loads(gzip.open(shard, "rt").read())
    else:
        d = json.loads(shard.read_text())
    out: Dict[int, Dict] = {}
    for cat_key, goals in d.get("goals_by_category", {}).items():
        cat = cat_key.split("_", 1)[-1] if "_" in cat_key else cat_key
        for g in goals:
            oid = g.get("object_id")
            if oid is not None:
                out[int(oid)] = {
                    "object_category": g.get("object_category", cat),
                    "position": g.get("position"),
                }
    return out


# ---------------------------------------------------------------------------
# Structural voxel rasterisation.
# ---------------------------------------------------------------------------
def rasterise_structural_voxels(
    instances: List[InstanceGeom],
    world_aabb: List[float],
    voxel_m: float,
    floor_slab_cells: int = 2,
    ceiling_slab_cells: int = 2,
) -> Dict[str, np.ndarray]:
    """Stamp wall / floor / ceiling instance AABBs into 3D voxel masks.

    HM3D's ``floor`` and ``ceiling`` semantic instances are single big
    meshes whose AABB spans the full storey height: a floor instance's
    AABB y-extent typically covers ``[y_floor, y_floor + 2.0 m]``, not
    just the ~5 cm slab a real floor occupies. Stamping that AABB
    verbatim would mark the entire room volume as ``floor`` and then
    propagation's structural-mask logic would zero-flux every voxel
    inside the room, which is exactly the failure mode users saw:
    fire trapped in a single voxel, no spread along the floor.

    To recover sensible slabs we only stamp the **bottom** ``floor_slab_cells``
    voxels of a floor instance's AABB (and the top ``ceiling_slab_cells``
    voxels of a ceiling instance) - that's where the actual surface
    is. Walls and other vertical structure keep their full AABB.
    """
    amin = np.array(world_aabb[:3], dtype=np.float64)
    amax = np.array(world_aabb[3:], dtype=np.float64)
    extent = np.maximum(amax - amin, voxel_m)
    Nx, Ny, Nz = (np.ceil(extent / voxel_m).astype(int))
    walls = np.zeros((Nx, Ny, Nz), dtype=bool)
    floors = np.zeros_like(walls)
    ceilings = np.zeros_like(walls)
    for inst in instances:
        cat = inst.category.lower()
        i0 = np.maximum(np.floor((inst.aabb_min - amin) / voxel_m).astype(int), 0)
        i1 = np.minimum(np.ceil((inst.aabb_max - amin) / voxel_m).astype(int),
                        np.array([Nx, Ny, Nz]))
        if np.any(i1 <= i0):
            continue
        if cat in {"wall", "door", "door frame", "window", "window frame",
                   "balustrade", "handrail", "moulding", "column", "beam"}:
            sl = (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2]))
            walls[sl] = True
        elif cat in {"floor", "stairs", "staircase"}:
            # Stamp only the bottom slab of the AABB. Stairs use the
            # same logic; for a staircase the inferred slab will follow
            # the bottom of the bounding box, which is close enough to
            # the lowest tread for our coarse 0.10-0.15 m grid.
            y_top = min(i0[1] + int(floor_slab_cells), int(i1[1]))
            sl = (slice(i0[0], i1[0]),
                  slice(i0[1], y_top),
                  slice(i0[2], i1[2]))
            floors[sl] = True
        elif cat == "ceiling":
            # Top slab only.
            y_bot = max(i1[1] - int(ceiling_slab_cells), int(i0[1]))
            sl = (slice(i0[0], i1[0]),
                  slice(y_bot, i1[1]),
                  slice(i0[2], i1[2]))
            ceilings[sl] = True
    return {
        "walls": walls,
        "floors": floors,
        "ceilings": ceilings,
        "shape": (int(Nx), int(Ny), int(Nz)),
        "origin": amin.tolist(),
        "voxel_m": float(voxel_m),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_inventory(
    scene_id: str,
    scene_dataset_root: Path = Path("data/scene_datasets/hm3d_v0.2"),
    objectgoal_root: Path = Path("data/datasets/objectnav_hm3d_v2"),
    splits: Tuple[str, ...] = ("val_mini", "val", "train"),
    voxel_m: float = 0.10,
    progress: bool = False,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    """Return ``(inventory_dict, structural_voxels_dict)``.

    The dict is JSON-serialisable; the voxel dict is kept separate so
    callers can decide whether to persist the (potentially large)
    boolean masks to .npy files alongside the JSON.
    """
    scene_short = _scene_short(scene_id)
    candidates = list(scene_dataset_root.rglob(f"{scene_short}.basis.glb"))
    if not candidates:
        raise FileNotFoundError(
            f"could not locate {scene_short}.basis.glb under {scene_dataset_root}"
        )
    scene_dir = candidates[0].parent
    scene_glb = scene_dir / f"{scene_short}.basis.glb"
    semantic_glb = scene_dir / f"{scene_short}.semantic.glb"
    semantic_txt = scene_dir / f"{scene_short}.semantic.txt"

    # 1) ObjectGoal lookup (optional)
    shard = _find_objectgoal_shard(scene_short, objectgoal_root, splits)
    goal_by_id: Dict[int, Dict] = _read_objectgoal_ids(shard) if shard else {}

    # 2) Per-instance geometry from semantic.glb
    if not semantic_glb.exists():
        raise FileNotFoundError(f"missing {semantic_glb}")
    if not semantic_txt.exists():
        raise FileNotFoundError(f"missing {semantic_txt}")
    instances, agg_summary = aggregate_instances(
        semantic_glb, semantic_txt, progress=progress
    )

    # 3) World AABB from the union of all instances
    if instances:
        all_min = np.min([i.aabb_min for i in instances], axis=0) - 0.50
        all_max = np.max([i.aabb_max for i in instances], axis=0) + 0.50
        world_aabb = all_min.tolist() + all_max.tolist()
        world_aabb_source = "semantic_glb"
    else:
        world_aabb = None
        world_aabb_source = "unknown"

    # 4) Floor clustering
    floors = cluster_floors(instances)

    # 5) Build per-instance dicts
    instance_dicts: List[Dict] = []
    for inst in instances:
        cat = inst.category.lower()
        is_struct = cat in STRUCTURAL_CATEGORIES
        flammability, smoke_yield = lookup_material(cat)
        floor_id = assign_floor(inst, floors)
        d = inst.as_dict()
        d.update({
            "structural": bool(is_struct),
            "is_goal": int(inst.instance_id) in goal_by_id,
            "goal_object_id": (
                int(inst.instance_id) if int(inst.instance_id) in goal_by_id else None
            ),
            "flammability": float(flammability),
            "smoke_yield": float(smoke_yield),
            "floor_id": int(floor_id) if floor_id is not None else None,
        })
        instance_dicts.append(d)

    # 6) Structural voxel masks
    structural_voxels = (
        rasterise_structural_voxels(instances, world_aabb, voxel_m)
        if world_aabb is not None else
        {"walls": None, "floors": None, "ceilings": None,
         "shape": None, "origin": None, "voxel_m": voxel_m}
    )

    # 7) Semantic summary (region histograms) - kept for the LLM planner
    by_hex, rows = read_semantic_txt(semantic_txt)
    per_region: Dict[str, Counter] = defaultdict(Counter)
    global_counter: Counter = Counter()
    for r in rows:
        per_region[str(r.region_id)][r.category.lower()] += 1
        global_counter[r.category.lower()] += 1

    # 8) Backward-compat 'objects' alias = goal-flagged instances only
    legacy_objects: List[Dict] = []
    for d in instance_dicts:
        if not d["is_goal"]:
            continue
        if d["structural"]:
            continue
        legacy_objects.append({
            "object_id": d["instance_id"],
            "category": d["category"],
            "position": (
                goal_by_id[d["instance_id"]]["position"]
                if d["instance_id"] in goal_by_id else d["centroid"]
            ),
            "aabb_min": d["aabb_min"],
            "aabb_max": d["aabb_max"],
            "flammability": d["flammability"],
            "smoke_yield": d["smoke_yield"],
            "notes": "auto from semantic_glb (v2)",
        })

    inventory: Dict = {
        "schema_version": 2,
        "scene_id": scene_short,
        "scene_dir": str(scene_dir),
        "scene_glb": str(scene_glb),
        "semantic_glb": str(semantic_glb),
        "semantic_txt": str(semantic_txt),
        "objectgoal_shard": str(shard) if shard else None,
        "world_aabb": world_aabb,
        "world_aabb_source": world_aabb_source,
        "voxel_m": float(voxel_m),
        "floors": floors,
        "instances": instance_dicts,
        "objects": legacy_objects,
        "structural": {
            "voxel_m": float(voxel_m),
            "shape": list(structural_voxels["shape"]) if structural_voxels["shape"] else None,
            "origin": structural_voxels["origin"],
            # Paths are filled in by write_inventory once we know out_dir.
            "wall_voxel_path": None,
            "floor_voxel_path": None,
            "ceiling_voxel_path": None,
        },
        "semantic_summary": {
            "n_instances": int(len(rows)),
            "global": dict(global_counter),
            "per_region": {r: dict(c) for r, c in per_region.items()},
        },
        "build_summary": {
            **agg_summary,
            "n_floors": len(floors),
            "n_structural_instances": int(sum(1 for d in instance_dicts if d["structural"])),
            "n_flammable_instances": int(sum(
                1 for d in instance_dicts
                if (not d["structural"]) and d["flammability"] >= 0.30
            )),
        },
    }
    return inventory, structural_voxels


def write_inventory(
    scene_id: str,
    out_root: Path = Path("scenes"),
    save_structural_voxels: bool = True,
    **kwargs,
) -> Path:
    inv, voxels = build_inventory(scene_id, **kwargs)
    out_dir = out_root / inv["scene_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_structural_voxels and voxels.get("shape"):
        struct_dir = out_dir / "structural"
        struct_dir.mkdir(exist_ok=True)
        np.save(struct_dir / "walls.npy", voxels["walls"])
        np.save(struct_dir / "floors.npy", voxels["floors"])
        np.save(struct_dir / "ceilings.npy", voxels["ceilings"])
        inv["structural"]["wall_voxel_path"] = str(struct_dir / "walls.npy")
        inv["structural"]["floor_voxel_path"] = str(struct_dir / "floors.npy")
        inv["structural"]["ceiling_voxel_path"] = str(struct_dir / "ceilings.npy")

    out_path = out_dir / "inventory.json"
    out_path.write_text(json.dumps(inv, indent=2))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    p = argparse.ArgumentParser(
        description="Build an inventory.json (schema v2) for an HM3D scene."
    )
    p.add_argument("--scene", required=True)
    p.add_argument("--scene_dataset_root", default="data/scene_datasets/hm3d_v0.2")
    p.add_argument("--objectgoal_root", default="data/datasets/objectnav_hm3d_v2")
    p.add_argument("--out_root", default="scenes")
    p.add_argument("--voxel_m", type=float, default=0.10)
    p.add_argument("--no_structural_voxels", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out = write_inventory(
        scene_id=args.scene,
        out_root=Path(args.out_root),
        scene_dataset_root=Path(args.scene_dataset_root),
        objectgoal_root=Path(args.objectgoal_root),
        voxel_m=args.voxel_m,
        progress=not args.quiet,
        save_structural_voxels=not args.no_structural_voxels,
    )
    inv = json.loads(out.read_text())
    bs = inv["build_summary"]
    print(f"[scene_scan v2] {inv['scene_id']}: "
          f"{bs['n_instances_recovered']}/{bs['n_instances_in_txt']} instances "
          f"({bs['n_structural_instances']} structural, "
          f"{bs['n_flammable_instances']} flammable), "
          f"{bs['n_floors']} floor(s)")
    if inv["world_aabb"]:
        bb = inv["world_aabb"]
        ext = [round(bb[i + 3] - bb[i], 2) for i in range(3)]
        print(f"  world aabb: min={[round(x, 2) for x in bb[:3]]} "
              f"max={[round(x, 2) for x in bb[3:]]} extent={ext}")
    if inv["floors"]:
        print(f"  floors: " + ", ".join(
            f"#{f['id']} y={f['y']:.2f}m" for f in inv["floors"]
        ))
    print(f"[scene_scan v2] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
