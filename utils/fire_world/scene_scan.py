"""Build a deterministic ``inventory.json`` for a Habitat HM3D scene.

The inventory is the **single source of truth** for downstream stages
(planner, propagation). It is intentionally small and human-readable so
it can live in the repo as a fixture and be reviewed in a PR diff.

Why not parse ``*.semantic.glb``?
    HM3D v0.2 ships per-instance vertex colours but does not ship the
    ``.scn`` descriptor that Habitat needs to populate
    ``sim.semantic_scene.objects``, and the colour <-> instance ID
    mapping in the GLB does not match ``*.semantic.txt`` directly. Going
    through that path is brittle and slow (60s+ per scene with trimesh).

What we use instead - all small, all already present in the repo:
  1. The ObjectGoal dataset shard ``data/datasets/.../{scene}.json.gz``
     gives us, for every goal-category object in the scene:
        - ``position`` (world XYZ),
        - ``object_id`` (the HM3D instance id we keep as primary key),
        - ``object_category`` (chair / bed / sofa / toilet / tv_monitor),
        - ``view_points`` (used to derive a tight 2D footprint).
  2. The Habitat ``*.semantic.txt`` table gives us the full per-instance
     category histogram for the scene, even for non-goal objects, so the
     downstream LLM planner can ground its prompt with what is actually
     in the building (kitchens, bathrooms, etc.).
  3. The ``*.basis.navmesh`` (loaded via habitat_sim) gives us the floor
     polygon from which we derive a per-floor world AABB and a 2D
     occupancy mask later used by the propagation engine.

The output schema is documented at the top of ``write_inventory``.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Lightweight schema (kept as plain dicts for forward-compat / JSON I/O).
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class GoalObject:
    object_id: int
    category: str
    position: Tuple[float, float, float]
    aabb_min: Tuple[float, float, float]
    aabb_max: Tuple[float, float, float]
    flammability: float
    smoke_yield: float
    notes: str = ""

    def as_dict(self) -> Dict:
        return {
            "object_id": int(self.object_id),
            "category": self.category,
            "position": list(map(float, self.position)),
            "aabb_min": list(map(float, self.aabb_min)),
            "aabb_max": list(map(float, self.aabb_max)),
            "flammability": float(self.flammability),
            "smoke_yield": float(self.smoke_yield),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Material table (per-category flammability + smoke yield).
# ---------------------------------------------------------------------------
# Numbers are dimensionless [0, 1]: flammability ~ relative ignition speed,
# smoke_yield ~ relative dense smoke produced when burning. These are
# intentionally rough; the LLM planner can override on a per-fire basis.
MATERIAL_TABLE: Dict[str, Tuple[float, float]] = {
    # furniture (cloth / wood / foam dominated -> high)
    "bed":         (0.75, 0.80),
    "couch":       (0.80, 0.85),
    "sofa":        (0.80, 0.85),
    "chair":       (0.55, 0.50),
    # bathroom / utilities (porcelain / metal / water -> low)
    "toilet":      (0.05, 0.05),
    "bathtub":     (0.05, 0.05),
    "sink":        (0.05, 0.05),
    # appliances (mostly metal but with cabling / plastic)
    "tv_monitor":  (0.40, 0.55),
    "tv":          (0.40, 0.55),
    "monitor":     (0.40, 0.55),
    "stove":       (0.85, 0.70),  # source of grease fires
    "refrigerator": (0.30, 0.40),
    # plants
    "potted plant": (0.50, 0.45),
    "plant":        (0.50, 0.45),
    # default fallback
    "_default":    (0.30, 0.30),
}


def lookup_material(category: str) -> Tuple[float, float]:
    cat = category.lower().strip()
    return MATERIAL_TABLE.get(cat, MATERIAL_TABLE["_default"])


# ---------------------------------------------------------------------------
# ObjectGoal dataset reader.
# ---------------------------------------------------------------------------
def _scene_short_id(scene_id_or_path: str) -> str:
    """Normalise either ``00800-TEEsavR23oF`` or its glb path to short id."""
    s = scene_id_or_path.split("/")[-1]
    s = s.replace(".basis.glb", "").replace(".glb", "")
    return s


def _find_dataset_shard(scene_short: str, splits: List[str], dataset_root: Path) -> Optional[Path]:
    for split in splits:
        cand = dataset_root / split / "content" / f"{scene_short}.json.gz"
        if cand.exists():
            return cand
    return None


def _read_objectgoal_shard(path: Path) -> Dict:
    if path.suffix == ".gz":
        return json.loads(gzip.open(path, "rt").read())
    return json.loads(path.read_text())


def _aabb_from_view_points(view_points: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate an object AABB from the agent view-points around it.

    The view-points sit on a circle around the object at agent height.
    Radius ~ 1.0 m by HM3D convention (success_distance = 0.2 m + agent
    radius). We collapse them to an XZ AABB; the Y span comes from the
    object's height clamped between [floor, floor+2.5 m] which is enough
    for 'is this voxel inside the object'.
    """
    if not view_points:
        return np.zeros(3), np.zeros(3)
    pts = np.array(
        [vp["agent_state"]["position"] for vp in view_points], dtype=np.float64
    )
    # Tight 2D box: median +/- a fixed shrink so AABB sits inside the
    # view-circle, i.e. closer to the object centroid than to its viewers.
    xs, zs = pts[:, 0], pts[:, 2]
    cx, cz = float(np.median(xs)), float(np.median(zs))
    rx = max(0.3, 0.6 * (xs.max() - xs.min()) / 2)
    rz = max(0.3, 0.6 * (zs.max() - zs.min()) / 2)
    y_min = float(np.percentile(pts[:, 1], 5)) - 0.10
    y_max = y_min + 1.20  # heuristic furniture height
    return (
        np.array([cx - rx, y_min, cz - rz]),
        np.array([cx + rx, y_max, cz + rz]),
    )


def _harvest_goal_objects(
    shard: Dict,
    scene_short: str,
) -> List[GoalObject]:
    seen: Dict[int, GoalObject] = {}
    for cat_key, goals in shard.get("goals_by_category", {}).items():
        # cat_key like 'TEEsavR23oF.basis.glb_bed' -> trailing token is cat.
        cat = cat_key.split("_", 1)[-1] if "_" in cat_key else cat_key
        for g in goals:
            oid = g.get("object_id")
            if oid is None or oid in seen:
                continue
            pos = tuple(g.get("position", [0.0, 0.0, 0.0]))
            aabb_min, aabb_max = _aabb_from_view_points(g.get("view_points", []))
            if np.allclose(aabb_min, aabb_max):
                # Fallback: tiny box around the position.
                aabb_min = np.array(pos) - 0.30
                aabb_max = np.array(pos) + 0.30
            f, sy = lookup_material(cat)
            seen[oid] = GoalObject(
                object_id=int(oid),
                category=str(cat),
                position=pos,
                aabb_min=tuple(aabb_min.tolist()),
                aabb_max=tuple(aabb_max.tolist()),
                flammability=f,
                smoke_yield=sy,
                notes="from objectgoal goals_by_category",
            )
    return sorted(seen.values(), key=lambda o: o.object_id)


# ---------------------------------------------------------------------------
# Semantic.txt category histogram (helps the LLM planner ground the scene).
# ---------------------------------------------------------------------------
_TXT_LINE_RE = re.compile(r"^(\d+),([0-9A-Fa-f]{6}),\"([^\"]+)\",(\d+)$")


def _read_semantic_categories(semantic_txt: Path) -> Dict:
    if not semantic_txt.exists():
        return {"per_region": {}, "global": {}, "n_instances": 0}
    per_region: Dict[str, Counter] = defaultdict(Counter)
    global_counter: Counter = Counter()
    n = 0
    for line in semantic_txt.read_text().splitlines():
        m = _TXT_LINE_RE.match(line.strip())
        if not m:
            continue
        n += 1
        _id, _hex, cat, region = m.groups()
        global_counter[cat.lower()] += 1
        per_region[str(region)][cat.lower()] += 1
    return {
        "per_region": {r: dict(c) for r, c in per_region.items()},
        "global": dict(global_counter),
        "n_instances": n,
    }


# ---------------------------------------------------------------------------
# Navmesh footprint -> 2D occupancy + world AABB.
# ---------------------------------------------------------------------------
def _navmesh_footprint(
    navmesh_path: Path,
    scene_glb: Path,
    scene_dataset_cfg: Path,
    voxel_xy: float = 0.10,
    use_sim: bool = False,
) -> Optional[Dict]:
    """Build a 2D walkable footprint from a Habitat navmesh.

    Two paths:
        - default (``use_sim=False``): try ``habitat_sim.PathFinder`` standalone.
          Fast (no rendering) but HM3D v0.2 navmeshes sometimes refuse to
          load this way; we silently fall back to "no footprint".
        - ``use_sim=True``: spin up a full ``habitat_sim.Simulator`` so its
          pathfinder is guaranteed to be populated. This is **slow on WSL**
          (cold EGL+CUDA context can take 60-120 s) and is opt-in.

    Returns:
      dict with keys aabb_min / aabb_max / origin_xz / voxel_xy / shape /
      floors_y / walkable_mask (numpy uint8 HxW; 1 = walkable),
      or None if neither path worked.
    """
    if not navmesh_path.exists():
        print(f"[scene_scan] navmesh not found: {navmesh_path}")
        return None

    samples: Optional[np.ndarray] = None
    bb: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    try:
        import habitat_sim  # type: ignore
    except Exception as e:  # pragma: no cover - habitat optional
        print(f"[scene_scan] habitat_sim not importable ({e}); skipping footprint")
        return None

    if not use_sim:
        pn = habitat_sim.PathFinder()
        pn.load_nav_mesh(str(navmesh_path))
        if pn.is_loaded:
            bb = pn.get_bounds()
            samples = _sample_navmesh(pn)
        else:
            print(f"[scene_scan] standalone navmesh load failed: {navmesh_path}; "
                  "rerun with --use_sim to use the full simulator path")
            return None
    else:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = str(scene_glb)
        sim_cfg.scene_dataset_config_file = str(scene_dataset_cfg)
        sim_cfg.gpu_device_id = 0
        ac = habitat_sim.agent.AgentConfiguration()
        cfg = habitat_sim.Configuration(sim_cfg, [ac])
        sim = habitat_sim.Simulator(cfg)
        try:
            pn = sim.pathfinder
            if not pn.is_loaded:
                print(f"[scene_scan] sim navmesh not loaded; aborting footprint")
                return None
            bb = pn.get_bounds()
            samples = _sample_navmesh(pn)
        finally:
            sim.close()

    if samples is None or bb is None:
        return None

    aabb_min = np.array(bb[0], dtype=np.float64)
    aabb_max = np.array(bb[1], dtype=np.float64)

    # Floors: cluster Y coords with a 1D histogram.
    ys = samples[:, 1]
    hist, edges = np.histogram(ys, bins=64)
    floors: List[float] = []
    threshold = max(50, hist.max() // 10)
    for i, c in enumerate(hist):
        if c >= threshold:
            floors.append(float(0.5 * (edges[i] + edges[i + 1])))
    merged: List[float] = []
    for f in sorted(set(round(f, 2) for f in floors)):
        if not merged or abs(f - merged[-1]) > 0.5:
            merged.append(f)
    floors = merged

    nx = max(1, int(np.ceil((aabb_max[0] - aabb_min[0]) / voxel_xy)))
    nz = max(1, int(np.ceil((aabb_max[2] - aabb_min[2]) / voxel_xy)))
    walkable = np.zeros((nx, nz), dtype=np.uint8)
    ix = np.clip(((samples[:, 0] - aabb_min[0]) / voxel_xy).astype(int), 0, nx - 1)
    iz = np.clip(((samples[:, 2] - aabb_min[2]) / voxel_xy).astype(int), 0, nz - 1)
    walkable[ix, iz] = 1

    return {
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "origin_xz": [float(aabb_min[0]), float(aabb_min[2])],
        "voxel_xy": float(voxel_xy),
        "shape": [int(nx), int(nz)],
        "floors_y": floors,
        "walkable_mask": walkable,
    }


def _sample_navmesh(pn, n_samples: int = 50000) -> np.ndarray:
    samples: List[List[float]] = []
    for _ in range(n_samples):
        p = pn.get_random_navigable_point()
        if not np.isfinite(p).all():
            continue
        samples.append(list(p))
    if not samples:
        return np.zeros((0, 3))
    return np.asarray(samples, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_inventory(
    scene_id: str,
    scene_dataset_root: Path = Path("data/scene_datasets/hm3d_v0.2"),
    objectgoal_root: Path = Path("data/datasets/objectnav_hm3d_v2"),
    splits: Tuple[str, ...] = ("val_mini", "val", "train"),
    voxel_xy: float = 0.10,
    use_sim: bool = False,
    scene_dataset_cfg: Optional[Path] = None,
) -> Dict:
    """Scan a single scene and return its inventory dict (no I/O)."""
    scene_short = _scene_short_id(scene_id)

    candidates = list(scene_dataset_root.rglob(f"{scene_short}.basis.glb"))
    if not candidates:
        raise FileNotFoundError(
            f"could not locate {scene_short}.basis.glb under {scene_dataset_root}"
        )
    scene_dir = candidates[0].parent
    scene_glb = scene_dir / f"{scene_short}.basis.glb"
    semantic_txt = scene_dir / f"{scene_short}.semantic.txt"
    navmesh = scene_dir / f"{scene_short}.basis.navmesh"

    if scene_dataset_cfg is None:
        scene_dataset_cfg = scene_dataset_root / "hm3d_annotated_basis.scene_dataset_config.json"

    shard_path = _find_dataset_shard(scene_short, list(splits), objectgoal_root)
    goal_objects: List[GoalObject] = []
    if shard_path is not None:
        shard = _read_objectgoal_shard(shard_path)
        goal_objects = _harvest_goal_objects(shard, scene_short)

    semantic_summary = _read_semantic_categories(semantic_txt)
    footprint = _navmesh_footprint(
        navmesh,
        scene_glb=scene_glb,
        scene_dataset_cfg=scene_dataset_cfg,
        voxel_xy=voxel_xy,
        use_sim=use_sim,
    )

    # If the navmesh path failed, derive a coarse world_aabb from the goal
    # objects so the rest of the pipeline still has a bounding box to work
    # with. This is intentionally a *loose* fallback: 1 m padding around the
    # union of object AABBs.
    if footprint is None and goal_objects:
        all_min = np.min([np.array(o.aabb_min) for o in goal_objects], axis=0)
        all_max = np.max([np.array(o.aabb_max) for o in goal_objects], axis=0)
        all_min -= 1.0
        all_max += 1.0
        world_aabb = all_min.tolist() + all_max.tolist()
    elif footprint is not None:
        world_aabb = footprint["aabb_min"] + footprint["aabb_max"]
    else:
        world_aabb = None

    inventory = {
        "schema_version": 1,
        "scene_id": scene_short,
        "scene_dir": str(scene_dir),
        "scene_glb": str(scene_glb),
        "objectgoal_shard": str(shard_path) if shard_path else None,
        "world_aabb": world_aabb,
        "world_aabb_source": (
            "navmesh" if footprint is not None else
            "object_union" if goal_objects else "unknown"
        ),
        "footprint": (
            {k: v for k, v in footprint.items() if k != "walkable_mask"}
            if footprint
            else None
        ),
        "objects": [o.as_dict() for o in goal_objects],
        "semantic_summary": semantic_summary,
        "_walkable_mask_in_memory": footprint.get("walkable_mask")
        if footprint is not None
        else None,
    }
    return inventory


def write_inventory(
    scene_id: str,
    out_root: Path = Path("scenes"),
    **kwargs,
) -> Path:
    """Build and persist an inventory.json (+ optional walkable mask)."""
    inv = build_inventory(scene_id, **kwargs)
    scene_short = inv["scene_id"]
    out_dir = out_root / scene_short
    out_dir.mkdir(parents=True, exist_ok=True)

    walkable = inv.pop("_walkable_mask_in_memory", None)
    if walkable is not None and inv["footprint"] is not None:
        mask_path = out_dir / "walkable_mask.npy"
        np.save(mask_path, walkable)
        inv["footprint"]["walkable_mask_path"] = str(mask_path)

    out_path = out_dir / "inventory.json"
    out_path.write_text(json.dumps(inv, indent=2))
    return out_path


def _cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build inventory.json for a Habitat HM3D scene."
    )
    parser.add_argument("--scene", required=True,
                        help="scene short id, e.g. 00800-TEEsavR23oF or "
                             "TEEsavR23oF (the prefix is auto-resolved).")
    parser.add_argument("--scene_dataset_root", default="data/scene_datasets/hm3d_v0.2")
    parser.add_argument("--objectgoal_root", default="data/datasets/objectnav_hm3d_v2")
    parser.add_argument("--out_root", default="scenes")
    parser.add_argument("--voxel_xy", type=float, default=0.10)
    parser.add_argument("--use_sim", action="store_true",
                        help="spin up a habitat_sim.Simulator to extract the "
                             "navmesh footprint. Slow on WSL; off by default.")
    args = parser.parse_args()

    out = write_inventory(
        scene_id=args.scene,
        out_root=Path(args.out_root),
        scene_dataset_root=Path(args.scene_dataset_root),
        objectgoal_root=Path(args.objectgoal_root),
        voxel_xy=args.voxel_xy,
        use_sim=args.use_sim,
    )
    inv = json.loads(out.read_text())
    n_obj = len(inv["objects"])
    n_cat = len(inv["semantic_summary"].get("global", {}))
    print(f"[scene_scan] {inv['scene_id']}: "
          f"{n_obj} goal objects, {n_cat} semantic categories")
    if inv["world_aabb"]:
        bb = inv["world_aabb"]
        print(f"  world aabb: min={[round(x,2) for x in bb[:3]]} "
              f"max={[round(x,2) for x in bb[3:]]} "
              f"(extent {[round(bb[i+3]-bb[i],2) for i in range(3)]})")
    if inv["footprint"]:
        print(f"  navmesh: shape={inv['footprint']['shape']} "
              f"floors_y={inv['footprint']['floors_y']}")
    print(f"[scene_scan] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
