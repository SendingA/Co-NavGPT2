#!/usr/bin/env python3
"""Prepare every required FireWorld asset for one HM3D scene."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fire_world.pipeline_runner import (
    ALL_FIRE_TYPES,
    ALL_INTENSITIES,
    append_jsonl,
    atomic_write_json,
    build_scene_plans,
    estimate_timeline_bytes,
    load_or_build_inventory,
    make_run_id,
    parse_seeds,
    parse_selection,
    record_config,
    scenario_matrix,
    summarise_records,
    task_log_path,
    worker_prepare_timeline,
    write_scene_asset_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build inventory, structural masks, deterministic plans and "
            "validated FireWorld timelines for one scene."
        )
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--fire-types",
        default="all",
        help="all or comma-separated FireWorld template names",
    )
    parser.add_argument(
        "--intensities",
        default="all",
        help="all or comma-separated light,medium,severe",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="one or more comma-separated deterministic seeds",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/scene_datasets/hm3d_v0.2",
    )
    parser.add_argument(
        "--objectgoal-root",
        default="data/datasets/objectnav_hm3d_v2",
    )
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--out-root", default="outputs/fire_world")
    parser.add_argument("--scan-voxel-m", type=float, default=0.10)
    parser.add_argument("--voxel-m", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--save-dt", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and plan in memory, print the matrix, write nothing",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="do not reuse a structurally valid existing timeline",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild timelines even when existing assets validate",
    )
    parser.add_argument(
        "--force-scan",
        action="store_true",
        help="rebuild inventory and structural masks",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="skip SHA256 calculation for generated/resumed timelines",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero if any requested template is infeasible",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def _config(args: argparse.Namespace) -> Dict:
    return {
        "scene": args.scene,
        "fire_types": args.fire_types,
        "intensities": args.intensities,
        "seeds": args.seeds,
        "dataset_root": args.dataset_root,
        "objectgoal_root": args.objectgoal_root,
        "scenes_root": args.scenes_root,
        "out_root": args.out_root,
        "scan_voxel_m": args.scan_voxel_m,
        "voxel_m": args.voxel_m,
        "dt": args.dt,
        "save_dt": args.save_dt,
        "dry_run": args.dry_run,
        "resume": not args.no_resume,
        "force": args.force,
        "force_scan": args.force_scan,
        "checksum": not args.no_checksum,
        "strict": args.strict,
    }


def main(argv: List[str] = None) -> int:
    args = _parser().parse_args(argv)
    fire_types = parse_selection(
        args.fire_types, ALL_FIRE_TYPES, "--fire-types"
    )
    intensities = parse_selection(
        args.intensities, ALL_INTENSITIES, "--intensities"
    )
    seeds = parse_seeds(args.seeds)
    combinations = scenario_matrix(fire_types, intensities, seeds)
    dataset_root = Path(args.dataset_root)
    objectgoal_root = Path(args.objectgoal_root)
    scenes_root = Path(args.scenes_root)
    out_root = Path(args.out_root)
    records: List[Dict] = []

    inventory, inventory_record = load_or_build_inventory(
        scene_id=args.scene,
        dataset_root=dataset_root,
        objectgoal_root=objectgoal_root,
        scenes_root=scenes_root,
        scan_voxel_m=args.scan_voxel_m,
        resume=not args.no_resume,
        force_scan=args.force_scan,
        dry_run=args.dry_run,
    )
    records.append(inventory_record)
    tasks, plan_records = build_scene_plans(
        inventory=inventory,
        combinations=combinations,
        scenes_root=scenes_root,
        dry_run=args.dry_run,
    )
    records.extend(plan_records)
    for task in tasks:
        plan = task.get("plan")
        if plan is None:
            plan = json.loads(Path(task["plan_path"]).read_text())
        task["estimate"] = estimate_timeline_bytes(
            plan, args.voxel_m, args.dt, args.save_dt
        )

    estimated_bytes = sum(
        task["estimate"]["timeline_uncompressed_bytes"] for task in tasks
    )
    skipped = sum(
        record.get("status") == "skipped_infeasible"
        for record in plan_records
    )
    if args.dry_run:
        payload = {
            "scene_id": args.scene,
            "requested_scenarios": len(combinations),
            "feasible_scenarios": len(tasks),
            "skipped_infeasible": skipped,
            "timeline_uncompressed_bytes": estimated_bytes,
            "tasks": [
                {
                    "fire_type": task["fire_type"],
                    "intensity": task["intensity"],
                    "seed": task["seed"],
                    "plan_id": task["plan_id"],
                    "estimate": task["estimate"],
                }
                for task in tasks
            ],
            "records": records,
        }
        print(json.dumps(payload, indent=2))
        return 1 if args.strict and skipped else 0

    run_id = args.run_id or make_run_id("scene")
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    record_config(
        run_dir=run_dir,
        command="prepare_fire_world_scene",
        config=_config(args),
    )
    manifest_path = run_dir / "manifest.jsonl"
    for record in records:
        append_jsonl(manifest_path, record)

    timeline_records: List[Dict] = []
    for index, task in enumerate(tasks, start=1):
        print(
            f"[scene pipeline] {index}/{len(tasks)} "
            f"{task['fire_type']}/{task['intensity']} "
            f"seed={task['seed']} plan={task['plan_id']}",
            flush=True,
        )
        payload = {
            "task": task,
            "log_path": str(task_log_path(run_dir, task)),
            "scenes_root": str(scenes_root),
            "out_root": str(out_root),
            "voxel_m": args.voxel_m,
            "dt": args.dt,
            "save_dt": args.save_dt,
            "resume": not args.no_resume,
            "force": args.force,
            "checksum": not args.no_checksum,
        }
        record = worker_prepare_timeline(payload)
        timeline_records.append(record)
        records.append(record)
        append_jsonl(manifest_path, record)
        print(
            f"  -> {record['status']} in {record['elapsed_s']}s",
            flush=True,
        )

    index_path = write_scene_asset_index(
        args.scene, timeline_records, out_root
    )
    summary = summarise_records(records)
    summary.update(
        {
            "run_id": run_id,
            "scene_id": args.scene,
            "requested_scenarios": len(combinations),
            "feasible_scenarios": len(tasks),
            "skipped_infeasible": skipped,
            "timeline_uncompressed_bytes": estimated_bytes,
            "asset_index": str(index_path),
        }
    )
    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["failure_count"]:
        return 1
    if args.strict and skipped:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
