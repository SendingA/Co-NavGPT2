#!/usr/bin/env python3
"""Regenerate FireWorld plan JSONs from the installed scene inventories.

This is deliberately a plan-only command. It writes semantic plans under
``scenes/<scene>/plans`` and a reproducibility manifest under ``outputs``;
it never starts the substantially more expensive propagation/timeline stage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fire_world.pipeline_runner import (  # noqa: E402
    ALL_FIRE_TYPES,
    ALL_INTENSITIES,
    append_jsonl,
    atomic_write_json,
    build_scene_plans,
    make_run_id,
    parse_seeds,
    parse_selection,
    scenario_matrix,
    utc_now,
)
from utils.fire_world.templates import TEMPLATE_VERSION  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the deterministic FireWorld plan matrix from "
            "existing scenes/*/inventory.json files without baking timelines"
        )
    )
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument(
        "--output-root",
        default="outputs/fire_plan_regeneration",
        help="run manifests and summaries; plan JSONs stay under scenes-root",
    )
    parser.add_argument("--fire-types", default="all")
    parser.add_argument("--intensities", default="all")
    parser.add_argument("--seeds", default="42")
    parser.add_argument(
        "--scene-list",
        default=None,
        help="optional file with one scene ID per line",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero if any requested template is infeasible",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def _load_scene_filter(path: Optional[str]) -> Optional[Set[str]]:
    if path is None:
        return None
    values = {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError("--scene-list contains no scene IDs")
    return values


def _discover_inventories(
    scenes_root: Path,
    selected: Optional[Set[str]],
) -> List[Path]:
    paths = sorted(scenes_root.glob("*/inventory.json"))
    if selected is not None:
        by_scene = {path.parent.name: path for path in paths}
        missing = sorted(selected - set(by_scene))
        if missing:
            raise ValueError(f"scene inventories not found: {missing}")
        paths = [by_scene[scene_id] for scene_id in sorted(selected)]
    if not paths:
        raise ValueError(f"no inventories found under {scenes_root}")
    return paths


def _summarise(
    records: Sequence[Dict],
    *,
    scene_count: int,
    combinations_per_scene: int,
    dry_run: bool,
) -> Dict:
    status_counts = Counter(str(record["status"]) for record in records)
    source_counts = defaultdict(Counter)
    for record in records:
        count = record.get("num_initial_ignitions")
        if count is not None:
            source_counts[str(record["intensity"])][int(count)] += 1
    return {
        "template_version": int(TEMPLATE_VERSION),
        "dry_run": bool(dry_run),
        "scene_count": int(scene_count),
        "combinations_per_scene": int(combinations_per_scene),
        "requested_plan_count": int(scene_count * combinations_per_scene),
        "feasible_plan_count": int(
            sum(
                count
                for status, count in status_counts.items()
                if status != "skipped_infeasible"
            )
        ),
        "skipped_infeasible": int(status_counts["skipped_infeasible"]),
        "status_counts": dict(sorted(status_counts.items())),
        "source_count_histogram": {
            intensity: {
                str(count): frequency
                for count, frequency in sorted(histogram.items())
            }
            for intensity, histogram in sorted(source_counts.items())
        },
        "updated_at": utc_now(),
    }


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    scenes_root = Path(args.scenes_root)
    output_root = Path(args.output_root)
    fire_types = parse_selection(
        args.fire_types, ALL_FIRE_TYPES, "--fire-types"
    )
    intensities = parse_selection(
        args.intensities, ALL_INTENSITIES, "--intensities"
    )
    seeds = parse_seeds(args.seeds)
    combinations = scenario_matrix(fire_types, intensities, seeds)
    inventories = _discover_inventories(
        scenes_root, _load_scene_filter(args.scene_list)
    )

    records: List[Dict] = []
    for index, inventory_path in enumerate(inventories, start=1):
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        scene_id = inventory_path.parent.name
        if str(inventory.get("scene_id")) != scene_id:
            raise ValueError(
                f"inventory scene_id {inventory.get('scene_id')!r} does not "
                f"match directory {scene_id!r}"
            )
        _, scene_records = build_scene_plans(
            inventory=inventory,
            combinations=combinations,
            scenes_root=scenes_root,
            dry_run=args.dry_run,
        )
        records.extend(scene_records)
        feasible = sum(
            record["status"] != "skipped_infeasible"
            for record in scene_records
        )
        print(
            f"[plan regeneration] {index}/{len(inventories)} {scene_id}: "
            f"feasible={feasible}/{len(combinations)}",
            flush=True,
        )

    summary = _summarise(
        records,
        scene_count=len(inventories),
        combinations_per_scene=len(combinations),
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        run_id = args.run_id or make_run_id("plans-v11")
        run_dir = output_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_write_json(
            run_dir / "run_config.json",
            {
                "run_id": run_id,
                "created_at": utc_now(),
                "command": "regenerate_fire_plans",
                "config": vars(args),
                "template_version": int(TEMPLATE_VERSION),
            },
        )
        for record in records:
            append_jsonl(run_dir / "manifest.jsonl", record)
        summary["run_id"] = run_id
        summary["run_dir"] = str(run_dir)
        atomic_write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))

    if args.strict and summary["skipped_infeasible"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
