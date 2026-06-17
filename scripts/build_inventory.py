"""Build ``scenes/<scene>/inventory.json`` (and structural voxel masks)
from one or more HM3D scenes.

Examples
--------

Single scene by short id::

    python scripts/build_inventory.py --scene 00880-Nfvxx8J5NCo

By short id without the ``00880-`` prefix (auto-resolved against
``data/scene_datasets/hm3d_v0.2``)::

    python scripts/build_inventory.py --scene Nfvxx8J5NCo

By directory::

    python scripts/build_inventory.py --scene_dir \\
        data/scene_datasets/hm3d_v0.2/val/00880-Nfvxx8J5NCo

Multiple scenes::

    python scripts/build_inventory.py \\
        --scene 00800-TEEsavR23oF 00880-Nfvxx8J5NCo

All ``val`` scenes::

    python scripts/build_inventory.py --all val

The output goes to ``scenes/<scene_short>/`` by default::

    scenes/Nfvxx8J5NCo/
      inventory.json
      structural/
        walls.npy
        floors.npy
        ceilings.npy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.scene_scan import write_inventory  # noqa: E402


def _resolve_scene_args(
    scene_args: List[str],
    scene_dirs: List[str],
    all_split: Optional[str],
    scene_dataset_root: Path,
) -> List[str]:
    """Normalise everything to a list of "short ids" understood by
    ``scene_scan.write_inventory``."""
    short_ids: List[str] = []

    for s in scene_args:
        # Already a short id (Foo) or prefixed (00800-Foo); both work.
        short_ids.append(s.strip())

    for d in scene_dirs:
        p = Path(d).resolve()
        glb = list(p.glob("*.basis.glb"))
        if not glb:
            print(f"[build_inventory] WARN: no .basis.glb in {p}, skipping")
            continue
        # The first glb tells us the short id (matches the scene folder
        # name without the leading numeric prefix).
        short_ids.append(glb[0].stem.replace(".basis", ""))

    if all_split:
        split_dir = scene_dataset_root / all_split
        if not split_dir.exists():
            raise FileNotFoundError(f"split not found: {split_dir}")
        for entry in sorted(split_dir.iterdir()):
            if not entry.is_dir():
                continue
            glb = list(entry.glob("*.basis.glb"))
            if glb:
                short_ids.append(glb[0].stem.replace(".basis", ""))

    if not short_ids:
        raise SystemExit(
            "no scenes supplied. Use --scene, --scene_dir, or --all <split>."
        )
    # Dedupe, keep order.
    seen = set()
    uniq: List[str] = []
    for s in short_ids:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--scene", nargs="+", default=[],
                   help="One or more HM3D short ids (e.g. 00880-Nfvxx8J5NCo "
                        "or just Nfvxx8J5NCo).")
    p.add_argument("--scene_dir", nargs="+", default=[],
                   help="One or more scene directories containing "
                        "<id>.basis.glb / .semantic.glb / .semantic.txt.")
    p.add_argument("--all", choices=["val", "val_mini", "train"], default=None,
                   help="Scan every scene in this HM3D split.")
    p.add_argument("--scene_dataset_root", default="data/scene_datasets/hm3d_v0.2")
    p.add_argument("--objectgoal_root", default="data/datasets/objectnav_hm3d_v2")
    p.add_argument("--out_root", default="scenes")
    p.add_argument("--voxel_m", type=float, default=0.10,
                   help="Voxel edge length for structural masks (default 0.10 m).")
    p.add_argument("--no_structural_voxels", action="store_true",
                   help="Skip wall/floor/ceiling .npy export "
                        "(inventory.json still records the categories).")
    p.add_argument("--quiet", action="store_true",
                   help="Don't print primitive-by-primitive progress.")
    args = p.parse_args(argv)

    scene_dataset_root = Path(args.scene_dataset_root)
    short_ids = _resolve_scene_args(
        args.scene, args.scene_dir, args.all, scene_dataset_root
    )
    print(f"[build_inventory] {len(short_ids)} scene(s) to process: "
          + ", ".join(short_ids[:6])
          + (f" ... (+{len(short_ids) - 6} more)" if len(short_ids) > 6 else ""))

    n_ok = 0
    for sid in short_ids:
        t0 = time.time()
        try:
            out = write_inventory(
                scene_id=sid,
                out_root=Path(args.out_root),
                scene_dataset_root=scene_dataset_root,
                objectgoal_root=Path(args.objectgoal_root),
                voxel_m=args.voxel_m,
                progress=not args.quiet,
                save_structural_voxels=not args.no_structural_voxels,
            )
        except FileNotFoundError as e:
            print(f"[build_inventory] {sid}: SKIP ({e})")
            continue
        except Exception as e:
            print(f"[build_inventory] {sid}: FAIL ({type(e).__name__}: {e})")
            continue
        elapsed = time.time() - t0

        inv = json.loads(out.read_text())
        bs = inv["build_summary"]
        msg = (
            f"[build_inventory] {sid}: "
            f"{bs['n_instances_recovered']}/{bs['n_instances_in_txt']} instances "
            f"({bs['n_structural_instances']} structural, "
            f"{bs['n_flammable_instances']} flammable), "
            f"{bs['n_floors']} floor(s), "
            f"{elapsed:.1f}s -> {out}"
        )
        print(msg)
        if inv["world_aabb"]:
            bb = inv["world_aabb"]
            ext = [round(bb[i + 3] - bb[i], 2) for i in range(3)]
            print(f"  world aabb extent: {ext} m")
        if inv["floors"]:
            floors_str = ", ".join(
                f"#{f['id']} y={f['y']:.2f}m" for f in inv["floors"]
            )
            print(f"  floors: {floors_str}")
        n_ok += 1

    print(f"[build_inventory] done: {n_ok}/{len(short_ids)} succeeded")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
