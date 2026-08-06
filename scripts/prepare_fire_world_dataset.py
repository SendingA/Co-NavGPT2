#!/usr/bin/env python3
"""Prepare FireWorld assets across an HM3D dataset split."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fire_world.pipeline_runner import (
    ALL_FIRE_TYPES,
    ALL_INTENSITIES,
    append_jsonl,
    atomic_write_json,
    build_scene_plans,
    discover_dataset_scenes,
    estimate_timeline_bytes,
    load_or_build_inventory,
    make_run_id,
    parse_seeds,
    parse_selection,
    record_config,
    safe_worker_count,
    scenario_matrix,
    summarise_records,
    task_log_path,
    validate_timeline,
    worker_prepare_timeline,
    write_scene_asset_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover HM3D scenes and prepare the complete deterministic "
            "FireWorld template/intensity asset matrix."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="data/scene_datasets/hm3d_v0.2",
    )
    parser.add_argument(
        "--splits",
        default="val",
        help="comma-separated dataset splits, for example val,train",
    )
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
    parser.add_argument("--seeds", default="42")
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
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="limit complete scenes after deterministic sorting",
    )
    parser.add_argument(
        "--scene-list",
        default=None,
        help="optional text file containing one scene short ID per line",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument("--no-checksum", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero when any scenario is infeasible",
    )
    parser.add_argument(
        "--disk-safety-factor",
        type=float,
        default=1.05,
        help="multiply missing uncompressed timeline bytes for disk preflight",
    )
    parser.add_argument(
        "--allow-low-disk",
        action="store_true",
        help="continue even when conservative uncompressed estimate exceeds free space",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def _scene_filter(path: Optional[str]) -> Optional[Set[str]]:
    if path is None:
        return None
    values = {
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError("--scene-list contains no scene IDs")
    return values


def _config(args: argparse.Namespace) -> Dict:
    return {
        "dataset_root": args.dataset_root,
        "splits": args.splits,
        "fire_types": args.fire_types,
        "intensities": args.intensities,
        "seeds": args.seeds,
        "objectgoal_root": args.objectgoal_root,
        "scenes_root": args.scenes_root,
        "out_root": args.out_root,
        "scan_voxel_m": args.scan_voxel_m,
        "voxel_m": args.voxel_m,
        "dt": args.dt,
        "save_dt": args.save_dt,
        "jobs": args.jobs,
        "max_scenes": args.max_scenes,
        "scene_list": args.scene_list,
        "dry_run": args.dry_run,
        "resume": not args.no_resume,
        "force": args.force,
        "force_scan": args.force_scan,
        "checksum": not args.no_checksum,
        "strict": args.strict,
        "disk_safety_factor": args.disk_safety_factor,
        "allow_low_disk": args.allow_low_disk,
    }


def _print_size(label: str, value: int) -> None:
    gib = value / float(1024 ** 3)
    print(f"[dataset pipeline] {label}: {gib:.2f} GiB", flush=True)


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes must be >= 1")
    if args.disk_safety_factor <= 0.0:
        raise ValueError("--disk-safety-factor must be positive")

    dataset_root = Path(args.dataset_root)
    objectgoal_root = Path(args.objectgoal_root)
    scenes_root = Path(args.scenes_root)
    out_root = Path(args.out_root)
    splits = tuple(
        part.strip() for part in args.splits.split(",") if part.strip()
    )
    fire_types = parse_selection(
        args.fire_types, ALL_FIRE_TYPES, "--fire-types"
    )
    intensities = parse_selection(
        args.intensities, ALL_INTENSITIES, "--intensities"
    )
    seeds = parse_seeds(args.seeds)
    combinations = scenario_matrix(fire_types, intensities, seeds)
    selected_ids = _scene_filter(args.scene_list)
    discovery = discover_dataset_scenes(dataset_root, splits)
    if selected_ids is not None:
        discovery = [
            item for item in discovery if item.scene_id in selected_ids
        ]
        discovered_ids = {item.scene_id for item in discovery}
        missing_ids = sorted(selected_ids - discovered_ids)
        if missing_ids:
            raise ValueError(
                f"scene IDs not found in selected splits: {missing_ids}"
            )

    complete = sorted(
        (item for item in discovery if item.complete),
        key=lambda item: item.scene_id,
    )
    incomplete = sorted(
        (item for item in discovery if not item.complete),
        key=lambda item: item.scene_id,
    )
    if args.max_scenes is not None:
        complete = complete[: args.max_scenes]
    allowed_complete_ids = {item.scene_id for item in complete}
    discovery_for_run = [
        item
        for item in discovery
        if not item.complete or item.scene_id in allowed_complete_ids
    ]

    run_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None
    if not args.dry_run:
        run_id = args.run_id or make_run_id("dataset")
        run_dir = out_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = run_dir / "manifest.jsonl"
        record_config(
            run_dir=run_dir,
            command="prepare_fire_world_dataset",
            config=_config(args),
            discovery=discovery_for_run,
        )
    else:
        run_id = "dry-run"

    records: List[Dict] = []
    for item in incomplete:
        record = {
            "stage": "discovery",
            "scene_id": item.scene_id,
            "split": item.split,
            "status": "skipped_missing_semantic_assets",
            "reason": f"missing {list(item.missing)}",
            "scene_dir": item.scene_dir,
        }
        records.append(record)
        if manifest_path is not None:
            append_jsonl(manifest_path, record)

    print(
        f"[dataset pipeline] discovered={len(discovery)} "
        f"complete_selected={len(complete)} incomplete={len(incomplete)} "
        f"scenarios_per_scene={len(combinations)}",
        flush=True,
    )

    tasks: List[Dict] = []
    for index, item in enumerate(complete, start=1):
        print(
            f"[dataset pipeline] planning scene {index}/{len(complete)} "
            f"{item.scene_id}",
            flush=True,
        )
        try:
            inventory, inventory_record = load_or_build_inventory(
                scene_id=item.scene_id,
                dataset_root=dataset_root,
                objectgoal_root=objectgoal_root,
                scenes_root=scenes_root,
                scan_voxel_m=args.scan_voxel_m,
                resume=not args.no_resume,
                force_scan=args.force_scan,
                dry_run=args.dry_run,
            )
            scene_tasks, plan_records = build_scene_plans(
                inventory=inventory,
                combinations=combinations,
                scenes_root=scenes_root,
                dry_run=args.dry_run,
            )
            for task in scene_tasks:
                plan = task.get("plan")
                if plan is None:
                    plan = json.loads(Path(task["plan_path"]).read_text())
                task["estimate"] = estimate_timeline_bytes(
                    plan, args.voxel_m, args.dt, args.save_dt
                )
            scene_records = [inventory_record, *plan_records]
            tasks.extend(scene_tasks)
        except Exception as exc:
            scene_records = [
                {
                    "stage": "inventory",
                    "scene_id": item.scene_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            ]
        records.extend(scene_records)
        if manifest_path is not None:
            for record in scene_records:
                append_jsonl(manifest_path, record)

    skipped_infeasible = sum(
        record.get("status") == "skipped_infeasible" for record in records
    )
    inventory_failures = sum(
        record.get("stage") == "inventory"
        and record.get("status") == "failed"
        for record in records
    )
    total_uncompressed = sum(
        task["estimate"]["timeline_uncompressed_bytes"] for task in tasks
    )

    # Validate existing canonical outputs before estimating missing disk.
    existing_valid = 0
    missing_uncompressed = 0
    for task in tasks:
        plan = task.get("plan")
        if plan is None:
            plan = json.loads(Path(task["plan_path"]).read_text())
        timeline = out_root / task["scene_id"] / task["plan_id"] / "timeline.npz"
        valid, _, _ = validate_timeline(
            timeline, plan, args.voxel_m, args.dt, args.save_dt
        )
        if valid and not args.force and not args.no_resume:
            existing_valid += 1
        else:
            missing_uncompressed += task["estimate"][
                "timeline_uncompressed_bytes"
            ]

    preflight = {
        "run_id": run_id,
        "dataset_scene_folders": len(discovery),
        "complete_scenes_selected": len(complete),
        "incomplete_scene_folders": len(incomplete),
        "requested_scenarios": len(complete) * len(combinations),
        "feasible_tasks": len(tasks),
        "skipped_infeasible": skipped_infeasible,
        "inventory_failures": inventory_failures,
        "existing_valid_timelines": existing_valid,
        "missing_timelines": len(tasks) - existing_valid,
        "timeline_uncompressed_bytes": total_uncompressed,
        "missing_uncompressed_bytes": missing_uncompressed,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    _print_size("all timeline upper bound", total_uncompressed)
    _print_size("missing timeline upper bound", missing_uncompressed)

    if args.dry_run:
        return 1 if inventory_failures or (
            args.strict and skipped_infeasible
        ) else 0

    out_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(out_root)
    required_disk = int(
        missing_uncompressed * float(args.disk_safety_factor)
    )
    disk_preflight = {
        "free_bytes": int(disk.free),
        "required_conservative_bytes": required_disk,
        "disk_safety_factor": float(args.disk_safety_factor),
        "passes": required_disk <= disk.free,
    }
    atomic_write_json(run_dir / "preflight.json", {
        **preflight,
        "disk": disk_preflight,
    })
    _print_size("free disk", disk.free)
    _print_size("conservative required disk", required_disk)
    if required_disk > disk.free and not args.allow_low_disk:
        failure = {
            "stage": "preflight",
            "status": "failed",
            "reason": (
                f"required {required_disk} bytes exceeds "
                f"{disk.free} free bytes"
            ),
        }
        records.append(failure)
        append_jsonl(manifest_path, failure)
        summary = summarise_records(records)
        summary.update({**preflight, "disk": disk_preflight})
        atomic_write_json(run_dir / "summary.json", summary)
        print("[dataset pipeline] aborting before propagation: low disk")
        return 1

    effective_jobs, memory_preflight = safe_worker_count(tasks, args.jobs)
    atomic_write_json(run_dir / "resource_plan.json", memory_preflight)
    print(
        f"[dataset pipeline] propagation workers "
        f"{effective_jobs}/{args.jobs} after memory cap",
        flush=True,
    )

    payloads: List[Dict] = []
    for task in tasks:
        payloads.append(
            {
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
        )

    timeline_records: List[Dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=effective_jobs) as executor:
        future_to_task = {
            executor.submit(worker_prepare_timeline, payload): payload["task"]
            for payload in payloads
        }
        for completed, future in enumerate(
            as_completed(future_to_task), start=1
        ):
            task = future_to_task[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "stage": "timeline",
                    "scene_id": task["scene_id"],
                    "fire_type": task["fire_type"],
                    "intensity": task["intensity"],
                    "seed": task["seed"],
                    "template_version": task["template_version"],
                    "plan_id": task["plan_id"],
                    "status": "failed",
                    "reason": f"worker failure: {type(exc).__name__}: {exc}",
                }
            timeline_records.append(record)
            records.append(record)
            append_jsonl(manifest_path, record)
            print(
                f"[dataset pipeline] {completed}/{len(payloads)} "
                f"{record.get('scene_id')} "
                f"{record.get('fire_type')}/{record.get('intensity')} "
                f"-> {record.get('status')} "
                f"({record.get('elapsed_s', 0)}s)",
                flush=True,
            )

    for scene_id in sorted({task["scene_id"] for task in tasks}):
        write_scene_asset_index(scene_id, timeline_records, out_root)

    summary = summarise_records(records)
    summary.update(
        {
            **preflight,
            "disk": disk_preflight,
            "resources": memory_preflight,
            "wall_time_s": round(time.time() - started, 3),
            "run_dir": str(run_dir),
        }
    )
    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failure_count"]:
        return 1
    if args.strict and skipped_infeasible:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
