#!/usr/bin/env python3
"""Keep one active standard FireWorld plan per semantic plan group.

The operation is deliberately recoverable: superseded plan JSONs and matching
timeline directories are moved into a run-scoped backup, and every move is
recorded in ``manifest.json``. Custom fire types (including route-constrast
safe-detour plans) are never candidates for pruning.

The default mode is a read-only dry run. Pass ``--apply`` to perform moves.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.templates import INTENSITIES, TEMPLATES


STANDARD_FIRE_TYPES = frozenset(TEMPLATES)
STANDARD_INTENSITIES = frozenset(INTENSITIES)
CURATION_KEYS = (
    "curation",
    "route_validation",
    "episode_selection",
    "goal_protection",
    "protected_goal",
)


@dataclass(frozen=True)
class PlanRecord:
    path: Path
    payload: Dict[str, Any]

    @property
    def plan_id(self) -> str:
        return str(self.payload["plan_id"])

    @property
    def group(self) -> Tuple[str, str, str]:
        return (
            str(self.payload["scene_id"]),
            str(self.payload["fire_type"]),
            str(self.payload["intensity"]),
        )

    @property
    def is_standard(self) -> bool:
        return (
            self.payload.get("fire_type") in STANDARD_FIRE_TYPES
            and self.payload.get("intensity") in STANDARD_INTENSITIES
        )

    @property
    def is_curated(self) -> bool:
        return any(key in self.payload for key in CURATION_KEYS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_records(scenes_root: Path) -> List[PlanRecord]:
    records: List[PlanRecord] = []
    errors: List[str] = []
    for path in sorted(scenes_root.glob("*/plans/*.json")):
        try:
            payload = json.loads(path.read_text())
            for key in ("plan_id", "scene_id", "fire_type", "intensity"):
                if key not in payload:
                    raise ValueError(f"missing required key {key!r}")
            if path.stem != str(payload["plan_id"]):
                raise ValueError(
                    f"filename stem {path.stem!r} != plan_id "
                    f"{payload['plan_id']!r}"
                )
            records.append(PlanRecord(path=path, payload=payload))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError(
            "refusing to prune because plan validation failed:\n"
            + "\n".join(errors)
        )
    return records


def _keeper_sort_key(record: PlanRecord) -> Tuple[int, int, int, str]:
    payload = record.payload
    return (
        int(payload.get("template_version", -1)),
        int(payload.get("schema_version", -1)),
        int(payload.get("seed") == 42),
        record.plan_id,
    )


def select_keepers(
    records: Iterable[PlanRecord],
) -> Tuple[List[PlanRecord], List[PlanRecord], List[PlanRecord]]:
    """Return ``(keepers, superseded, protected_custom)``.

    A curated standard plan is preferred as the sole keeper for its group. If
    multiple curated plans share a standard group, pruning aborts rather than
    making an unsafe automatic choice. Non-standard plans are always protected.
    """
    groups: Dict[Tuple[str, str, str], List[PlanRecord]] = defaultdict(list)
    protected_custom: List[PlanRecord] = []
    for record in records:
        if record.is_standard:
            groups[record.group].append(record)
        else:
            protected_custom.append(record)

    keepers: List[PlanRecord] = []
    superseded: List[PlanRecord] = []
    for group, members in sorted(groups.items()):
        curated = [record for record in members if record.is_curated]
        if len(curated) > 1:
            raise RuntimeError(
                f"standard group {group!r} contains multiple curated plans: "
                + ", ".join(record.plan_id for record in curated)
            )
        keeper = curated[0] if curated else max(members, key=_keeper_sort_key)
        keepers.append(keeper)
        superseded.extend(record for record in members if record != keeper)
    return keepers, superseded, protected_custom


def _relative_under(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"path {path} is outside configured root {root}") from exc


def _move_spec(source: Path, destination: Path, kind: str) -> Dict[str, str]:
    return {
        "kind": kind,
        "source": str(source),
        "destination": str(destination),
    }


def _build_move_specs(
    superseded: Sequence[PlanRecord],
    scenes_root: Path,
    outputs_root: Path,
    backup_run_root: Path,
) -> List[Dict[str, str]]:
    moves: List[Dict[str, str]] = []
    for record in sorted(superseded, key=lambda item: str(item.path)):
        plan_relative = _relative_under(record.path, scenes_root)
        moves.append(
            _move_spec(
                record.path,
                backup_run_root / "scenes" / plan_relative,
                "plan",
            )
        )
        timeline_dir = (
            outputs_root / str(record.payload["scene_id"]) / record.plan_id
        )
        if timeline_dir.exists():
            timeline_relative = _relative_under(timeline_dir, outputs_root)
            moves.append(
                _move_spec(
                    timeline_dir,
                    backup_run_root
                    / "outputs"
                    / "fire_world"
                    / timeline_relative,
                    "timeline",
                )
            )
    return moves


def _asset_index_updates(
    outputs_root: Path,
    active_plan_ids: frozenset[str],
) -> List[Dict[str, Any]]:
    updates: List[Dict[str, Any]] = []
    for index_path in sorted(outputs_root.glob("*/asset_index.json")):
        payload = json.loads(index_path.read_text())
        assets = list(payload.get("assets", []))
        kept_assets = []
        for asset in assets:
            plan_id = str(asset.get("plan_id", ""))
            scene_id = str(payload.get("scene_id", index_path.parent.name))
            timeline_dir = outputs_root / scene_id / plan_id
            if plan_id in active_plan_ids and timeline_dir.exists():
                kept_assets.append(asset)
        if kept_assets != assets:
            updated_payload = dict(payload)
            updated_payload["assets"] = kept_assets
            updated_payload["updated_at"] = _utc_now()
            updates.append(
                {
                    "path": str(index_path),
                    "before_count": len(assets),
                    "after_count": len(kept_assets),
                    "payload": updated_payload,
                }
            )
    return updates


def build_prune_plan(
    scenes_root: Path,
    outputs_root: Path,
    backup_run_root: Path,
) -> Dict[str, Any]:
    records = _read_records(scenes_root)
    keepers, superseded, protected_custom = select_keepers(records)
    moves = _build_move_specs(
        superseded, scenes_root, outputs_root, backup_run_root
    )
    active_ids = frozenset(
        record.plan_id for record in (*keepers, *protected_custom)
    )
    asset_updates = _asset_index_updates(outputs_root, active_ids)
    group_sizes: Dict[str, int] = defaultdict(int)
    for record in records:
        if record.is_standard:
            group_sizes["|".join(record.group)] += 1
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "scenes_root": str(scenes_root),
        "outputs_root": str(outputs_root),
        "backup_run_root": str(backup_run_root),
        "standard_fire_types": sorted(STANDARD_FIRE_TYPES),
        "standard_plan_count_before": sum(
            record.is_standard for record in records
        ),
        "standard_group_count": len(keepers),
        "duplicate_group_count": sum(size > 1 for size in group_sizes.values()),
        "keeper_plan_ids": sorted(record.plan_id for record in keepers),
        "superseded_plan_ids": sorted(
            record.plan_id for record in superseded
        ),
        "protected_custom_plan_ids": sorted(
            record.plan_id for record in protected_custom
        ),
        "moves": moves,
        "asset_index_updates": asset_updates,
    }


def _validate_preflight(plan: Dict[str, Any], backup_run_root: Path) -> None:
    if backup_run_root.exists():
        raise RuntimeError(
            f"backup run directory already exists: {backup_run_root}"
        )
    destinations = set()
    for move in plan["moves"]:
        source = Path(move["source"])
        destination = Path(move["destination"])
        if not source.exists():
            raise RuntimeError(f"move source disappeared: {source}")
        if destination in destinations or destination.exists():
            raise RuntimeError(f"move destination already exists: {destination}")
        destinations.add(destination)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def apply_prune_plan(plan: Dict[str, Any], backup_run_root: Path) -> Dict[str, Any]:
    _validate_preflight(plan, backup_run_root)
    backup_run_root.mkdir(parents=True)
    serializable_preflight = dict(plan)
    serializable_preflight["asset_index_updates"] = [
        {key: value for key, value in update.items() if key != "payload"}
        for update in plan["asset_index_updates"]
    ]
    _write_json(backup_run_root / "preflight.json", serializable_preflight)

    completed_moves: List[Dict[str, str]] = []
    for move in plan["moves"]:
        source = Path(move["source"])
        destination = Path(move["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        completed_moves.append(move)

    asset_results: List[Dict[str, Any]] = []
    for update in plan["asset_index_updates"]:
        index_path = Path(update["path"])
        scene_id = index_path.parent.name
        backup_path = (
            backup_run_root / "asset_indexes" / scene_id / "asset_index.json"
        )
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(index_path, backup_path)
        _write_json(index_path, update["payload"])
        asset_results.append(
            {
                "path": str(index_path),
                "backup": str(backup_path),
                "before_count": update["before_count"],
                "after_count": update["after_count"],
            }
        )

    result = {
        "schema_version": 1,
        "completed_at": _utc_now(),
        "status": "applied",
        "standard_plan_count_before": plan["standard_plan_count_before"],
        "standard_plan_count_after": plan["standard_group_count"],
        "standard_group_count": plan["standard_group_count"],
        "duplicate_group_count_before": plan["duplicate_group_count"],
        "protected_custom_plan_count": len(plan["protected_custom_plan_ids"]),
        "protected_custom_plan_ids": plan["protected_custom_plan_ids"],
        "superseded_plan_count": len(plan["superseded_plan_ids"]),
        "moved_timeline_count": sum(
            move["kind"] == "timeline" for move in completed_moves
        ),
        "moves": completed_moves,
        "asset_index_updates": asset_results,
        "restore_instruction": (
            "Move each manifest entry from destination back to source, then "
            "restore the backed-up asset indexes."
        ),
    }
    _write_json(backup_run_root / "manifest.json", result)
    return result


def _summary(plan: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    return {
        "dry_run": dry_run,
        "standard_plan_count_before": plan["standard_plan_count_before"],
        "standard_group_count": plan["standard_group_count"],
        "duplicate_group_count": plan["duplicate_group_count"],
        "keeper_plan_count": len(plan["keeper_plan_ids"]),
        "superseded_plan_count": len(plan["superseded_plan_ids"]),
        "protected_custom_plan_count": len(plan["protected_custom_plan_ids"]),
        "timeline_move_count": sum(
            move["kind"] == "timeline" for move in plan["moves"]
        ),
        "asset_index_update_count": len(plan["asset_index_updates"]),
        "backup_run_root": plan["backup_run_root"],
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, default=Path("scenes"))
    parser.add_argument(
        "--outputs-root", type=Path, default=Path("outputs/fire_world")
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("outputs/fire_plan_pruning_backup"),
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("prune_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform recoverable moves; without this flag the run is read-only",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    backup_run_root = args.backup_root / args.run_id
    plan = build_prune_plan(
        args.scenes_root, args.outputs_root, backup_run_root
    )
    print(json.dumps(_summary(plan, dry_run=not args.apply), indent=2))
    if args.apply:
        result = apply_prune_plan(plan, backup_run_root)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "manifest": str(backup_run_root / "manifest.json"),
                    "superseded_plan_count": result["superseded_plan_count"],
                    "moved_timeline_count": result["moved_timeline_count"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
