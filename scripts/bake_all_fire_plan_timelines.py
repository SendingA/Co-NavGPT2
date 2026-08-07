#!/usr/bin/env python3
"""Bake ``timeline.npz`` for every existing FireWorld plan JSON.

Unlike ``prepare_fire_world_dataset.py``, this command never regenerates a
template matrix.  It treats every persisted ``scenes/*/plans/*.json`` file as
authoritative, including historical template versions, and fills the matching
canonical ``outputs/fire_world/<scene>/<plan_id>/timeline.npz`` path.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.fire_world.pipeline_runner import (
    ALL_FIRE_TYPES,
    ALL_INTENSITIES,
    append_jsonl,
    atomic_write_json,
    discover_existing_plan_tasks,
    make_run_id,
    parse_selection,
    prepare_timeline_task,
    summarise_records,
    task_log_path,
    utc_now,
    validate_timeline,
    write_scene_asset_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bake validated timelines for every existing fire plan",
    )
    parser.add_argument("--scenes-root", default="scenes")
    parser.add_argument("--out-root", default="outputs/fire_world")
    parser.add_argument("--voxel-m", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--save-dt", type=float, default=1.0)
    parser.add_argument(
        "--fire-types",
        default="all",
        help="comma-separated fire types or 'all'",
    )
    parser.add_argument(
        "--intensities",
        default="all",
        help="comma-separated intensities or 'all'",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=50.0,
        help="stop before a task if it could reduce free disk below this reserve",
    )
    parser.add_argument(
        "--projected-compression-ratio",
        type=float,
        default=0.60,
        help=(
            "preflight ratio applied to uncompressed missing timelines; "
            "actual free space is checked again before every task"
        ),
    )
    parser.add_argument(
        "--allow-projected-low-disk",
        action="store_true",
        help="start even when the compressed projection plus reserve exceeds free disk",
    )
    parser.add_argument(
        "--max-plans",
        type=int,
        default=None,
        help="deterministic prefix limit for smoke tests",
    )
    parser.add_argument("--run-id", default=None)
    return parser


def _gib(value: int) -> float:
    return float(value) / float(1024 ** 3)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("voxel_m", "dt", "save_dt"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_free_gib < 0.0:
        raise ValueError("--min-free-gib must be non-negative")
    if not 0.0 < args.projected_compression_ratio <= 1.0:
        raise ValueError("--projected-compression-ratio must be in (0, 1]")
    if args.max_plans is not None and args.max_plans < 1:
        raise ValueError("--max-plans must be at least one")


def _existing_status(
    tasks: Sequence[Dict],
    out_root: Path,
    *,
    voxel_m: float,
    dt: float,
    save_dt: float,
    force: bool,
    resume: bool,
) -> Dict:
    valid_count = 0
    missing_uncompressed = 0
    total_uncompressed = 0
    for task in tasks:
        plan = json.loads(Path(task["plan_path"]).read_text())
        estimate = task["estimate"]
        total_uncompressed += int(estimate["timeline_uncompressed_bytes"])
        timeline = (
            out_root / task["scene_id"] / task["plan_id"] / "timeline.npz"
        )
        valid, _, _ = validate_timeline(
            timeline,
            plan,
            voxel_m=voxel_m,
            dt=dt,
            save_dt=save_dt,
        )
        if valid and resume and not force:
            valid_count += 1
        else:
            missing_uncompressed += int(
                estimate["timeline_uncompressed_bytes"]
            )
    return {
        "plan_count": len(tasks),
        "existing_valid_timelines": valid_count,
        "missing_timelines": len(tasks) - valid_count,
        "timeline_uncompressed_bytes": total_uncompressed,
        "missing_uncompressed_bytes": missing_uncompressed,
    }


def _write_progress(
    run_dir: Path,
    *,
    total: int,
    completed: int,
    records: Sequence[Dict],
    out_root: Path,
) -> None:
    atomic_write_json(
        run_dir / "progress.json",
        {
            "updated_at": utc_now(),
            "total": int(total),
            "completed": int(completed),
            "remaining": int(max(0, total - completed)),
            "free_bytes": int(shutil.disk_usage(out_root).free),
            **summarise_records(records),
        },
    )


def main(argv: Sequence[str] = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    fire_types = parse_selection(
        args.fire_types,
        ALL_FIRE_TYPES,
        "--fire-types",
    )
    intensities = parse_selection(
        args.intensities,
        ALL_INTENSITIES,
        "--intensities",
    )
    scenes_root = Path(args.scenes_root)
    out_root = Path(args.out_root)
    tasks, discovery_records = discover_existing_plan_tasks(
        scenes_root,
        voxel_m=args.voxel_m,
        dt=args.dt,
        save_dt=args.save_dt,
    )
    tasks = [
        task
        for task in tasks
        if task["fire_type"] in fire_types
        and task["intensity"] in intensities
    ]
    if args.max_plans is not None:
        tasks = tasks[: args.max_plans]
    selected_paths = {task["plan_path"] for task in tasks}
    discovery_records = [
        record
        for record in discovery_records
        if record.get("plan_path") in selected_paths
        or (
            record.get("status") == "failed"
            and record.get("fire_type") in fire_types
            and record.get("intensity") in intensities
        )
    ]

    discovery_failures = [
        record
        for record in discovery_records
        if record.get("status") == "failed"
    ]
    status = _existing_status(
        tasks,
        out_root,
        voxel_m=args.voxel_m,
        dt=args.dt,
        save_dt=args.save_dt,
        force=args.force,
        resume=not args.no_resume,
    )
    disk = shutil.disk_usage(out_root if out_root.exists() else out_root.parent)
    reserve_bytes = int(args.min_free_gib * 1024 ** 3)
    projected_bytes = int(
        status["missing_uncompressed_bytes"]
        * args.projected_compression_ratio
    )
    preflight = {
        **status,
        "plan_discovery_failures": len(discovery_failures),
        "voxel_m": float(args.voxel_m),
        "dt": float(args.dt),
        "save_dt": float(args.save_dt),
        "fire_types": list(fire_types),
        "intensities": list(intensities),
        "free_bytes": int(disk.free),
        "reserve_bytes": reserve_bytes,
        "projected_compression_ratio": float(
            args.projected_compression_ratio
        ),
        "projected_missing_disk_bytes": projected_bytes,
        "projected_disk_passes": projected_bytes + reserve_bytes <= disk.free,
    }
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    print(
        "[all-plan propagation] plans={} ready={} missing={} "
        "projected={:.2f}GiB free={:.2f}GiB reserve={:.2f}GiB".format(
            status["plan_count"],
            status["existing_valid_timelines"],
            status["missing_timelines"],
            _gib(projected_bytes),
            _gib(disk.free),
            _gib(reserve_bytes),
        ),
        flush=True,
    )
    if args.dry_run:
        return 1 if discovery_failures else 0
    if discovery_failures:
        print(
            "[all-plan propagation] aborting: invalid plan/inventory inputs",
            flush=True,
        )
        return 1
    if (
        projected_bytes + reserve_bytes > disk.free
        and not args.allow_projected_low_disk
    ):
        print(
            "[all-plan propagation] aborting: projected compressed output "
            "would violate the disk reserve",
            flush=True,
        )
        return 1

    run_id = args.run_id or make_run_id("all-existing-plans")
    run_dir = out_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.jsonl"
    atomic_write_json(
        run_dir / "run_config.json",
        {
            "run_id": run_id,
            "created_at": utc_now(),
            "command": "bake_all_fire_plan_timelines",
            "config": vars(args),
        },
    )
    atomic_write_json(run_dir / "preflight.json", preflight)
    for record in discovery_records:
        append_jsonl(manifest_path, record)

    timeline_records: List[Dict] = []
    started = time.time()
    stopped_for_disk = False
    for index, task in enumerate(tasks, start=1):
        free_bytes = shutil.disk_usage(out_root).free
        # A newly written NPZ cannot exceed its uncompressed timeline payload.
        # Keeping that full per-task bound in addition to the user reserve
        # prevents a poorly compressing scene from filling the filesystem.
        task_bound = int(task["estimate"]["timeline_uncompressed_bytes"])
        plan = json.loads(Path(task["plan_path"]).read_text())
        timeline = (
            out_root / task["scene_id"] / task["plan_id"] / "timeline.npz"
        )
        already_valid, _, _ = validate_timeline(
            timeline,
            plan,
            voxel_m=args.voxel_m,
            dt=args.dt,
            save_dt=args.save_dt,
        )
        needs_write = args.force or args.no_resume or not already_valid
        if needs_write and free_bytes < reserve_bytes + task_bound:
            record = {
                "stage": "timeline",
                "scene_id": task["scene_id"],
                "plan_id": task["plan_id"],
                "status": "blocked_disk",
                "reason": (
                    f"free {free_bytes} < reserve {reserve_bytes} + "
                    f"single-task bound {task_bound}"
                ),
                "timestamp": utc_now(),
            }
            timeline_records.append(record)
            append_jsonl(manifest_path, record)
            stopped_for_disk = True
            print(
                f"[all-plan propagation] stopped before {index}/{len(tasks)} "
                f"for disk reserve",
                flush=True,
            )
            break

        log_path = task_log_path(run_dir, task)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        task_started = time.time()
        try:
            with log_path.open("a", encoding="utf-8") as log:
                from contextlib import redirect_stderr, redirect_stdout

                with redirect_stdout(log), redirect_stderr(log):
                    record = prepare_timeline_task(
                        task=task,
                        scenes_root=scenes_root,
                        out_root=out_root,
                        voxel_m=args.voxel_m,
                        dt=args.dt,
                        save_dt=args.save_dt,
                        resume=not args.no_resume,
                        force=args.force,
                        checksum=args.checksum,
                    )
        except Exception as exc:
            record = {
                "stage": "timeline",
                "scene_id": task["scene_id"],
                "fire_type": task["fire_type"],
                "intensity": task["intensity"],
                "seed": task["seed"],
                "template_version": task["template_version"],
                "plan_id": task["plan_id"],
                "plan_path": task["plan_path"],
                "timeline_path": str(timeline),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.time() - task_started, 3),
                "timestamp": utc_now(),
            }
        record["log_path"] = str(log_path)
        timeline_records.append(record)
        append_jsonl(manifest_path, record)
        print(
            "[all-plan propagation] {}/{} {} {} -> {} ({:.1f}s), "
            "free={:.1f}GiB".format(
                index,
                len(tasks),
                task["scene_id"],
                task["plan_id"],
                record["status"],
                float(record.get("elapsed_s", 0.0)),
                _gib(shutil.disk_usage(out_root).free),
            ),
            flush=True,
        )
        _write_progress(
            run_dir,
            total=len(tasks),
            completed=index,
            records=timeline_records,
            out_root=out_root,
        )

    for scene_id in sorted({task["scene_id"] for task in tasks}):
        write_scene_asset_index(scene_id, timeline_records, out_root)

    valid_count = 0
    invalid: List[Dict] = []
    for task in tasks:
        plan = json.loads(Path(task["plan_path"]).read_text())
        timeline = (
            out_root / task["scene_id"] / task["plan_id"] / "timeline.npz"
        )
        valid, reason, details = validate_timeline(
            timeline,
            plan,
            voxel_m=args.voxel_m,
            dt=args.dt,
            save_dt=args.save_dt,
        )
        if valid:
            valid_count += 1
        else:
            invalid.append(
                {
                    "scene_id": task["scene_id"],
                    "plan_id": task["plan_id"],
                    "timeline_path": str(timeline),
                    "reason": reason,
                    "details": details,
                }
            )
    completeness = {
        "updated_at": utc_now(),
        "plan_count": len(tasks),
        "valid_timeline_count": valid_count,
        "invalid_timeline_count": len(invalid),
        "complete": valid_count == len(tasks),
        "stopped_for_disk": stopped_for_disk,
        "invalid": invalid,
        "free_bytes": int(shutil.disk_usage(out_root).free),
        "wall_time_s": round(time.time() - started, 3),
    }
    atomic_write_json(run_dir / "completeness.json", completeness)
    summary = {
        **summarise_records([*discovery_records, *timeline_records]),
        **completeness,
        "run_dir": str(run_dir),
    }
    atomic_write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if completeness["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
