"""Resumable orchestration for complete FireWorld asset preparation.

The existing FireWorld stages remain the source of truth:

1. :mod:`utils.fire_world.scene_scan` builds ``inventory.json`` and
   structural masks.
2. :mod:`utils.fire_world.planner` builds deterministic plan JSON files.
3. :mod:`utils.fire_world.propagation` bakes ``timeline.npz``.

This module adds dataset-scale concerns around those stages: discovery,
preflight, atomic timeline installation, lightweight NPZ validation,
checksums, resumability and machine-readable run records. Canonical runtime
paths are deliberately unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import tempfile
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .planner import build_plan
from .propagation import run_propagation
from .scene_scan import build_inventory, write_inventory
from .templates import INTENSITIES, TEMPLATES, TEMPLATE_VERSION


ALL_FIRE_TYPES: Tuple[str, ...] = tuple(sorted(TEMPLATES))
ALL_INTENSITIES: Tuple[str, ...] = tuple(INTENSITIES)
TIMELINE_FIELDS: Tuple[str, ...] = ("flame", "smoke", "temp")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for manifests."""
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "fireworld") -> str:
    """Create a sortable run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def atomic_write_json(path: Path, value: object) -> None:
    """Write JSON through a same-directory temporary file and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def append_jsonl(path: Path, record: Dict) -> None:
    """Append one durable manifest record from the coordinator process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_selection(
    raw: str,
    available: Sequence[str],
    option_name: str,
) -> Tuple[str, ...]:
    """Parse ``all`` or a comma-separated CLI selection."""
    allowed = tuple(available)
    if raw.strip().lower() == "all":
        return allowed
    selected = tuple(
        part.strip() for part in raw.split(",") if part.strip()
    )
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(
            f"{option_name} contains unsupported values {unknown}; "
            f"choose from {list(allowed)} or 'all'"
        )
    if not selected:
        raise ValueError(f"{option_name} must not be empty")
    return selected


def parse_seeds(raw: str) -> Tuple[int, ...]:
    """Parse one or more comma-separated deterministic seeds."""
    seeds: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return tuple(dict.fromkeys(seeds))


@dataclass(frozen=True)
class SceneDiscovery:
    """Dataset assets required by :func:`scene_scan.build_inventory`."""

    scene_id: str
    split: str
    scene_dir: str
    basis_glb: str
    semantic_glb: str
    semantic_txt: str
    complete: bool
    missing: Tuple[str, ...]


def discover_dataset_scenes(
    dataset_root: Path,
    splits: Sequence[str],
) -> List[SceneDiscovery]:
    """Discover scene folders and explicitly classify incomplete assets."""
    records: List[SceneDiscovery] = []
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(
            path for path in split_dir.iterdir() if path.is_dir()
        ):
            scene_id = scene_dir.name.split("-", 1)[-1]
            basis = scene_dir / f"{scene_id}.basis.glb"
            semantic_glb = scene_dir / f"{scene_id}.semantic.glb"
            semantic_txt = scene_dir / f"{scene_id}.semantic.txt"
            required = (basis, semantic_glb, semantic_txt)
            missing = tuple(path.name for path in required if not path.exists())
            records.append(
                SceneDiscovery(
                    scene_id=scene_id,
                    split=split,
                    scene_dir=str(scene_dir),
                    basis_glb=str(basis),
                    semantic_glb=str(semantic_glb),
                    semantic_txt=str(semantic_txt),
                    complete=not missing,
                    missing=missing,
                )
            )
    return records


def _inventory_valid(path: Path, scene_id: str) -> Tuple[bool, str]:
    if not path.is_file():
        return False, "missing inventory.json"
    try:
        inventory = json.loads(path.read_text())
    except Exception as exc:
        return False, f"invalid inventory JSON: {exc}"
    if inventory.get("scene_id") != scene_id:
        return False, "inventory scene_id mismatch"
    if int(inventory.get("schema_version", 0)) < 2:
        return False, "inventory schema_version is older than 2"
    if not inventory.get("world_aabb") or not (
        inventory.get("instances") or inventory.get("objects")
    ):
        return False, "inventory lacks world_aabb or object instances"
    structural = inventory.get("structural") or {}
    for key in (
        "wall_voxel_path",
        "floor_voxel_path",
        "ceiling_voxel_path",
    ):
        raw_path = structural.get(key)
        if not raw_path or not Path(raw_path).is_file():
            return False, f"inventory structural asset missing: {key}"
    return True, "valid"


def load_or_build_inventory(
    scene_id: str,
    dataset_root: Path,
    objectgoal_root: Path,
    scenes_root: Path,
    scan_voxel_m: float,
    resume: bool,
    force_scan: bool,
    dry_run: bool,
) -> Tuple[Dict, Dict]:
    """Load a valid inventory or build it once for a scene."""
    started = time.time()
    inventory_path = scenes_root / scene_id / "inventory.json"
    valid, reason = _inventory_valid(inventory_path, scene_id)
    if resume and not force_scan and valid:
        inventory = json.loads(inventory_path.read_text())
        return inventory, {
            "stage": "inventory",
            "scene_id": scene_id,
            "status": "resumed",
            "path": str(inventory_path),
            "reason": reason,
            "elapsed_s": round(time.time() - started, 3),
            "timestamp": utc_now(),
        }

    if dry_run:
        inventory, _ = build_inventory(
            scene_id=scene_id,
            scene_dataset_root=dataset_root,
            objectgoal_root=objectgoal_root,
            voxel_m=scan_voxel_m,
            progress=False,
        )
        status = "would_generate"
        output_path = str(inventory_path)
    else:
        output_path_obj = write_inventory(
            scene_id=scene_id,
            out_root=scenes_root,
            scene_dataset_root=dataset_root,
            objectgoal_root=objectgoal_root,
            voxel_m=scan_voxel_m,
            progress=False,
            save_structural_voxels=True,
        )
        inventory = json.loads(output_path_obj.read_text())
        output_path = str(output_path_obj)
        status = "generated"

    return inventory, {
        "stage": "inventory",
        "scene_id": scene_id,
        "status": status,
        "path": output_path,
        "reason": reason if inventory_path.exists() else "",
        "elapsed_s": round(time.time() - started, 3),
        "timestamp": utc_now(),
    }


def scenario_matrix(
    fire_types: Sequence[str],
    intensities: Sequence[str],
    seeds: Sequence[int],
) -> List[Tuple[str, str, int]]:
    """Return the stable Cartesian scenario matrix."""
    return [
        (fire_type, intensity, int(seed))
        for fire_type, intensity, seed in product(
            fire_types, intensities, seeds
        )
    ]


def discover_existing_plan_tasks(
    scenes_root: Path,
    *,
    voxel_m: float,
    dt: float,
    save_dt: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Enumerate every persisted plan JSON without regenerating plans.

    Dataset-matrix preparation intentionally targets only the current
    template/seed matrix.  This helper instead treats every JSON already under
    ``scenes/<scene>/plans`` as an explicit asset request, including retained
    historical template versions.
    """

    scenes_root = Path(scenes_root)
    tasks: List[Dict] = []
    records: List[Dict] = []
    for plan_path in sorted(scenes_root.glob("*/plans/*.json")):
        scene_id = plan_path.parents[1].name
        base = {
            "stage": "plan_discovery",
            "scene_id": scene_id,
            "plan_path": str(plan_path),
            "timestamp": utc_now(),
        }
        try:
            plan = json.loads(plan_path.read_text())
            # Preserve filterable metadata even when a later validation step
            # fails, so a targeted bake reports only failures in its scope.
            if "fire_type" in plan:
                base["fire_type"] = str(plan["fire_type"])
            if "intensity" in plan:
                base["intensity"] = str(plan["intensity"])
            plan_id = str(plan["plan_id"])
            if plan_path.stem != plan_id:
                raise ValueError(
                    f"filename {plan_path.stem!r} != plan_id {plan_id!r}"
                )
            if str(plan.get("scene_id")) != scene_id:
                raise ValueError(
                    f"plan scene_id {plan.get('scene_id')!r} != {scene_id!r}"
                )
            inventory_path = scenes_root / scene_id / "inventory.json"
            valid_inventory, inventory_reason = _inventory_valid(
                inventory_path,
                scene_id,
            )
            if not valid_inventory:
                raise ValueError(inventory_reason)
            estimate = estimate_timeline_bytes(
                plan,
                voxel_m=voxel_m,
                dt=dt,
                save_dt=save_dt,
            )
            task = {
                "scene_id": scene_id,
                "fire_type": str(plan["fire_type"]),
                "intensity": str(plan["intensity"]),
                "seed": int(plan["seed"]),
                "template_version": int(plan["template_version"]),
                "plan_id": plan_id,
                "plan_path": str(plan_path),
                "inventory_path": str(inventory_path),
                "estimate": estimate,
            }
            tasks.append(task)
            records.append(
                {
                    **base,
                    "status": "ready",
                    "plan_id": plan_id,
                    "fire_type": task["fire_type"],
                    "intensity": task["intensity"],
                    "seed": task["seed"],
                    "template_version": task["template_version"],
                    "estimate": estimate,
                }
            )
        except Exception as exc:
            records.append(
                {
                    **base,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return tasks, records


def _atomic_write_plan(path: Path, plan: Dict) -> None:
    atomic_write_json(path, plan)


def build_scene_plans(
    inventory: Dict,
    combinations: Sequence[Tuple[str, str, int]],
    scenes_root: Path,
    dry_run: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """Build every feasible deterministic plan for one inventory."""
    tasks: List[Dict] = []
    records: List[Dict] = []
    scene_id = str(inventory["scene_id"])
    for fire_type, intensity, seed in combinations:
        started = time.time()
        base_record = {
            "stage": "plan",
            "scene_id": scene_id,
            "fire_type": fire_type,
            "intensity": intensity,
            "seed": int(seed),
            "template_version": int(TEMPLATE_VERSION),
            "timestamp": utc_now(),
        }
        try:
            plan = build_plan(
                inventory=inventory,
                fire_type=fire_type,
                intensity=intensity,
                seed=int(seed),
                num_ignitions=None,
            )
        except (RuntimeError, ValueError) as exc:
            records.append(
                {
                    **base_record,
                    "status": "skipped_infeasible",
                    "reason": str(exc),
                    "elapsed_s": round(time.time() - started, 3),
                }
            )
            continue

        plan_path = (
            scenes_root / scene_id / "plans" / f"{plan['plan_id']}.json"
        )
        if not dry_run:
            existing_same = False
            if plan_path.is_file():
                try:
                    existing_same = json.loads(plan_path.read_text()) == plan
                except Exception:
                    existing_same = False
            if not existing_same:
                _atomic_write_plan(plan_path, plan)
            plan_status = "resumed" if existing_same else "generated"
        else:
            plan_status = "would_generate"

        record = {
            **base_record,
            "status": plan_status,
            "plan_id": plan["plan_id"],
            "plan_path": str(plan_path),
            "duration_s": float(plan["duration_s"]),
            "num_initial_ignitions": int(plan["num_initial_ignitions"]),
            "elapsed_s": round(time.time() - started, 3),
        }
        records.append(record)
        tasks.append(
            {
                "scene_id": scene_id,
                "fire_type": fire_type,
                "intensity": intensity,
                "seed": int(seed),
                "template_version": int(TEMPLATE_VERSION),
                "plan_id": str(plan["plan_id"]),
                "plan_path": str(plan_path),
                "plan": plan if dry_run else None,
                "inventory_path": str(
                    scenes_root / scene_id / "inventory.json"
                ),
                "inventory": inventory if dry_run else None,
            }
        )
    return tasks, records


def _read_npy_header(
    archive: zipfile.ZipFile,
    member: str,
) -> Tuple[Tuple[int, ...], np.dtype]:
    """Read only a compressed NPY member's header, not its array body."""
    from numpy.lib import format as npformat

    with archive.open(member, "r") as handle:
        version = npformat.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = npformat.read_array_header_1_0(handle)
        elif version in ((2, 0), (3, 0)):
            shape, _, dtype = npformat.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NPY header version {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def expected_timeline_layout(
    plan: Dict,
    voxel_m: float,
    dt: float,
    save_dt: float,
) -> Tuple[Tuple[int, int, int], int]:
    """Return expected spatial shape and saved-frame count."""
    aabb = np.asarray(plan["world_aabb"], dtype=np.float64)
    extent = np.maximum(aabb[3:] - aabb[:3], 1e-3)
    shape = tuple(int(math.ceil(value / voxel_m)) for value in extent)
    n_steps = int(math.ceil(float(plan["duration_s"]) / dt))
    save_every = max(1, int(round(save_dt / dt)))
    n_frames = 1 + n_steps // save_every
    return shape, n_frames


def estimate_timeline_bytes(
    plan: Dict,
    voxel_m: float,
    dt: float,
    save_dt: float,
) -> Dict[str, int]:
    """Estimate uncompressed timeline and conservative process memory."""
    shape, n_frames = expected_timeline_layout(
        plan, voxel_m=voxel_m, dt=dt, save_dt=save_dt
    )
    voxels = int(np.prod(shape, dtype=np.int64))
    timeline_bytes = int(n_frames * voxels * len(TIMELINE_FIELDS) * 2)
    # Propagation keeps many float32/bool work arrays in addition to the
    # three float16 timeline buffers. This is conservative enough to gate
    # concurrency while remaining independent of implementation internals.
    working_bytes = int(voxels * 128)
    return {
        "nx": shape[0],
        "ny": shape[1],
        "nz": shape[2],
        "voxels": voxels,
        "n_frames": n_frames,
        "timeline_uncompressed_bytes": timeline_bytes,
        "estimated_peak_memory_bytes": timeline_bytes + working_bytes,
    }


def validate_timeline(
    timeline_path: Path,
    plan: Dict,
    voxel_m: float,
    dt: float,
    save_dt: float,
) -> Tuple[bool, str, Dict]:
    """Validate timeline structure without inflating large field arrays."""
    if not timeline_path.is_file():
        return False, "timeline.npz is missing", {}
    if timeline_path.stat().st_size <= 0:
        return False, "timeline.npz is empty", {}

    expected_shape, expected_frames = expected_timeline_layout(
        plan, voxel_m=voxel_m, dt=dt, save_dt=save_dt
    )
    try:
        headers: Dict[str, Dict] = {}
        with zipfile.ZipFile(timeline_path, "r") as archive:
            names = set(archive.namelist())
            required = {
                "flame.npy",
                "smoke.npy",
                "temp.npy",
                "times.npy",
            }
            missing = sorted(required - names)
            if missing:
                return False, f"timeline members missing: {missing}", {}
            for field in TIMELINE_FIELDS:
                shape, dtype = _read_npy_header(
                    archive, f"{field}.npy"
                )
                headers[field] = {
                    "shape": list(shape),
                    "dtype": str(dtype),
                }
                if shape != (expected_frames, *expected_shape):
                    return (
                        False,
                        f"{field} shape {shape} != "
                        f"{(expected_frames, *expected_shape)}",
                        headers,
                    )
                if dtype != np.dtype(np.float16):
                    return (
                        False,
                        f"{field} dtype {dtype} is not float16",
                        headers,
                    )

        # Semantic plan-ID migration updates the lightweight sidecar without
        # recompressing multi-gigabyte voxel arrays. Match runtime.py by
        # preferring that sidecar when it is available.
        sidecar_path = timeline_path.with_name("timeline_meta.json")
        if sidecar_path.is_file():
            meta = json.loads(sidecar_path.read_text())
            with np.load(timeline_path, allow_pickle=False) as payload:
                times = np.asarray(payload["times"], dtype=np.float64)
        elif "meta_json.npy" in names:
            with np.load(timeline_path, allow_pickle=False) as payload:
                times = np.asarray(payload["times"], dtype=np.float64)
                meta = json.loads(str(payload["meta_json"]))
        elif "meta.npy" in names:
            with np.load(timeline_path, allow_pickle=True) as payload:
                times = np.asarray(payload["times"], dtype=np.float64)
                legacy_meta = payload["meta"]
                raw_meta = (
                    legacy_meta.item()
                    if legacy_meta.shape == ()
                    else legacy_meta.reshape(-1)[0]
                )
                meta = (
                    raw_meta
                    if isinstance(raw_meta, dict)
                    else json.loads(str(raw_meta))
                )
        else:
            return False, "timeline metadata is missing", headers
        if times.shape != (expected_frames,):
            return False, f"times shape {times.shape} is invalid", headers
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            return False, "times are non-finite or non-monotonic", headers
        if not np.isclose(times[0], 0.0):
            return False, "timeline does not start at t=0", headers
        if not np.isclose(times[-1], float(plan["duration_s"])):
            return (
                False,
                f"timeline ends at {times[-1]} instead of "
                f"{plan['duration_s']}",
                headers,
            )
        if str(meta.get("plan_id")) != str(plan["plan_id"]):
            return False, "timeline metadata plan_id mismatch", headers
        if not np.isclose(float(meta.get("voxel_m", -1.0)), voxel_m):
            return False, "timeline metadata voxel_m mismatch", headers
        if not np.isclose(float(meta.get("dt", -1.0)), dt):
            return False, "timeline metadata dt mismatch", headers
        if not np.isclose(float(meta.get("save_dt", -1.0)), save_dt):
            return False, "timeline metadata save_dt mismatch", headers
        if int(meta.get("n_frames", -1)) != expected_frames:
            return False, "timeline metadata n_frames mismatch", headers
    except Exception as exc:
        return False, f"timeline validation error: {exc}", {}

    details = {
        "shape": [expected_frames, *expected_shape],
        "n_frames": expected_frames,
        "duration_s": float(plan["duration_s"]),
        "size_bytes": int(timeline_path.stat().st_size),
    }
    return True, "valid", details


class PlanLock:
    """Small cross-process lock based on exclusive file creation."""

    def __init__(self, path: Path, stale_after_s: float = 24 * 3600) -> None:
        self.path = path
        self.stale_after_s = float(stale_after_s)
        self.acquired = False

    def _owner_is_alive(self) -> bool:
        try:
            payload = json.loads(self.path.read_text())
        except Exception:
            return False
        if payload.get("hostname") != socket.gethostname():
            return True
        pid = int(payload.get("pid", -1))
        return pid > 0 and Path(f"/proc/{pid}").exists()

    def __enter__(self) -> "PlanLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_after_s or not self._owner_is_alive():
                self.path.unlink()
        try:
            fd = os.open(
                str(self.path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError as exc:
            raise RuntimeError(f"asset is locked by {self.path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "created_at": utc_now(),
                },
                handle,
            )
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False


def prepare_timeline_task(
    task: Dict,
    scenes_root: Path,
    out_root: Path,
    voxel_m: float,
    dt: float,
    save_dt: float,
    resume: bool,
    force: bool,
    checksum: bool,
) -> Dict:
    """Bake, validate and atomically install one plan timeline."""
    started = time.time()
    scene_id = str(task["scene_id"])
    plan_id = str(task["plan_id"])
    plan_path = Path(task.get("plan_path") or (
        scenes_root / scene_id / "plans" / f"{plan_id}.json"
    ))
    inventory_path = Path(task.get("inventory_path") or (
        scenes_root / scene_id / "inventory.json"
    ))
    plan = task.get("plan") or json.loads(plan_path.read_text())
    inventory = task.get("inventory") or json.loads(inventory_path.read_text())
    final_dir = out_root / scene_id / plan_id
    timeline_path = final_dir / "timeline.npz"
    estimate = estimate_timeline_bytes(plan, voxel_m, dt, save_dt)
    base = {
        "stage": "timeline",
        "scene_id": scene_id,
        "fire_type": task["fire_type"],
        "intensity": task["intensity"],
        "seed": int(task["seed"]),
        "template_version": int(task["template_version"]),
        "plan_id": plan_id,
        "plan_path": str(plan_path),
        "timeline_path": str(timeline_path),
        "estimate": estimate,
        "timestamp": utc_now(),
    }

    valid, reason, details = validate_timeline(
        timeline_path, plan, voxel_m, dt, save_dt
    )
    if valid and resume and not force:
        record = {
            **base,
            "status": "resumed",
            "reason": reason,
            "timeline": details,
            "elapsed_s": round(time.time() - started, 3),
        }
        if checksum:
            record["sha256"] = sha256_file(timeline_path)
        return record

    lock_path = out_root / ".locks" / f"{scene_id}__{plan_id}.lock"
    with PlanLock(lock_path):
        # Recheck after acquiring the lock in case another worker completed.
        valid, reason, details = validate_timeline(
            timeline_path, plan, voxel_m, dt, save_dt
        )
        if valid and resume and not force:
            record = {
                **base,
                "status": "resumed",
                "reason": reason,
                "timeline": details,
                "elapsed_s": round(time.time() - started, 3),
            }
            if checksum:
                record["sha256"] = sha256_file(timeline_path)
            return record

        staging_dir = (
            out_root
            / ".staging"
            / f"{scene_id}__{plan_id}__{uuid.uuid4().hex}"
        )
        staging_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            run_propagation(
                inventory=inventory,
                plan=plan,
                voxel_m=voxel_m,
                dt=dt,
                save_dt=save_dt,
                out_dir=staging_dir,
                keep_in_memory=False,
                verbose=True,
            )
            staged_timeline = staging_dir / "timeline.npz"
            staged_valid, staged_reason, staged_details = validate_timeline(
                staged_timeline, plan, voxel_m, dt, save_dt
            )
            if not staged_valid:
                raise RuntimeError(
                    f"new timeline failed validation: {staged_reason}"
                )

            if final_dir.exists():
                quarantine = (
                    out_root
                    / ".replaced"
                    / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    / scene_id
                    / plan_id
                )
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(final_dir), str(quarantine))
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(staging_dir), str(final_dir))
        except Exception:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            raise

    record = {
        **base,
        "status": "generated" if not valid else "replaced",
        "reason": reason,
        "timeline": staged_details,
        "elapsed_s": round(time.time() - started, 3),
    }
    if checksum:
        record["sha256"] = sha256_file(timeline_path)
    return record


def worker_prepare_timeline(payload: Dict) -> Dict:
    """ProcessPool-compatible wrapper with per-task log capture."""
    import contextlib
    import traceback

    task = payload["task"]
    log_path = Path(payload["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
            log_handle
        ):
            print(
                f"[pipeline] start scene={task['scene_id']} "
                f"plan={task['plan_id']} at {utc_now()}",
                flush=True,
            )
            try:
                record = prepare_timeline_task(
                    task=task,
                    scenes_root=Path(payload["scenes_root"]),
                    out_root=Path(payload["out_root"]),
                    voxel_m=float(payload["voxel_m"]),
                    dt=float(payload["dt"]),
                    save_dt=float(payload["save_dt"]),
                    resume=bool(payload["resume"]),
                    force=bool(payload["force"]),
                    checksum=bool(payload["checksum"]),
                )
                record["log_path"] = str(log_path)
                print(
                    f"[pipeline] finish status={record['status']} "
                    f"elapsed={record['elapsed_s']}s at {utc_now()}",
                    flush=True,
                )
                return record
            except Exception as exc:
                traceback.print_exc(file=log_handle)
                return {
                    "stage": "timeline",
                    "scene_id": task["scene_id"],
                    "fire_type": task["fire_type"],
                    "intensity": task["intensity"],
                    "seed": int(task["seed"]),
                    "template_version": int(task["template_version"]),
                    "plan_id": task["plan_id"],
                    "plan_path": task["plan_path"],
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.time() - started, 3),
                    "timestamp": utc_now(),
                    "log_path": str(log_path),
                }


def available_memory_bytes() -> int:
    """Return Linux MemAvailable, falling back to physical pages."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    if hasattr(os, "sysconf"):
        return int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        )
    return 0


def safe_worker_count(tasks: Sequence[Dict], requested: int) -> Tuple[int, Dict]:
    """Conservatively cap propagation workers by estimated peak memory."""
    requested = max(1, int(requested))
    peaks = [
        int(task.get("estimate", {}).get("estimated_peak_memory_bytes", 0))
        for task in tasks
    ]
    maximum_peak = max(peaks, default=0)
    available = available_memory_bytes()
    if maximum_peak <= 0 or available <= 0:
        safe = 1
    else:
        safe = max(1, int((available * 0.65) // maximum_peak))
    effective = min(requested, safe)
    return effective, {
        "requested_workers": requested,
        "effective_workers": effective,
        "available_memory_bytes": available,
        "largest_estimated_peak_memory_bytes": maximum_peak,
        "memory_safety_fraction": 0.65,
    }


def write_scene_asset_index(
    scene_id: str,
    timeline_records: Iterable[Dict],
    out_root: Path,
) -> Path:
    """Merge successful timeline records into a per-scene lookup index."""
    path = out_root / scene_id / "asset_index.json"
    existing: Dict = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    assets = {
        str(item["plan_id"]): item
        for item in existing.get("assets", [])
        if item.get("plan_id")
    }
    for record in timeline_records:
        if record.get("scene_id") != scene_id:
            continue
        if record.get("status") not in {"generated", "replaced", "resumed"}:
            continue
        assets[str(record["plan_id"])] = {
            "plan_id": record["plan_id"],
            "fire_type": record["fire_type"],
            "intensity": record["intensity"],
            "seed": int(record["seed"]),
            "template_version": int(record["template_version"]),
            "plan_path": record["plan_path"],
            "timeline_path": record["timeline_path"],
            "sha256": record.get("sha256"),
            "timeline": record.get("timeline", {}),
            "status": record["status"],
        }
    payload = {
        "schema_version": 1,
        "scene_id": scene_id,
        "updated_at": utc_now(),
        "assets": sorted(
            assets.values(),
            key=lambda item: (
                item["fire_type"],
                item["intensity"],
                item["seed"],
            ),
        ),
    }
    atomic_write_json(path, payload)
    return path


def summarise_records(records: Sequence[Dict]) -> Dict:
    """Build compact stage/status counts for run completion."""
    stage_status: Dict[str, Dict[str, int]] = {}
    for record in records:
        stage = str(record.get("stage", "unknown"))
        status = str(record.get("status", "unknown"))
        per_stage = stage_status.setdefault(stage, {})
        per_stage[status] = per_stage.get(status, 0) + 1
    failures = [
        record
        for record in records
        if record.get("status") == "failed"
    ]
    return {
        "updated_at": utc_now(),
        "record_count": len(records),
        "counts": stage_status,
        "failure_count": len(failures),
        "failures": failures,
    }


def task_log_path(run_dir: Path, task: Dict) -> Path:
    """Return a collision-free per-plan log path."""
    label = (
        f"{task['fire_type']}__{task['intensity']}"
        f"__seed{task['seed']}__{task['plan_id']}.log"
    )
    return run_dir / "logs" / str(task["scene_id"]) / label


def record_config(
    run_dir: Path,
    command: str,
    config: Dict,
    discovery: Optional[Sequence[SceneDiscovery]] = None,
) -> Path:
    """Write immutable-enough run provenance before expensive work."""
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "command": command,
        "template_version": int(TEMPLATE_VERSION),
        "config": config,
    }
    if discovery is not None:
        payload["discovery"] = [asdict(item) for item in discovery]
    path = run_dir / "run_config.json"
    atomic_write_json(path, payload)
    return path
