#!/usr/bin/env python3
"""Launch reproducible normal or FireWorld planner baselines.

The default ``controlled`` matrix follows ``docs/benchmark_experiment_plan``:

* compare every global planner with FMM fixed; and
* compare every local planner with ``co_ut`` fixed.

Use ``--matrix cartesian`` only when every global/local combination is
actually required. Runs are sequential because Habitat, PointNav and the VLM
normally share one GPU. Each run receives an isolated artifact directory and
is considered complete only when ``main.py`` exits successfully and reports
the requested episode count.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation_resume import load_metric_resume


GLOBAL_PLANNERS = ("nearest", "co_ut", "fill", "random", "gpt")
LOCAL_PLANNERS = ("fmm", "astar", "rl", "pointnav")
DEFAULT_LOCAL_PLANNERS = ("fmm", "astar", "pointnav")
DATASET_IDS = ("objectnav", "person")
CONDITIONS = ("normal", "fire-none", "fire-sensed", "fire-oracle")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    task_config: str
    data_path: str

    @property
    def resolved_data_path(self) -> Path:
        value = self.data_path.replace("{split}", "val")
        return (PROJECT_ROOT / value).resolve()


DATASETS: Dict[str, DatasetSpec] = {
    "objectnav": DatasetSpec(
        dataset_id="objectnav",
        task_config="multi_objectnav_hm3d.yaml",
        data_path=(
            "data/datasets/objectnav_hm3d_v2/{split}/{split}.json.gz"
        ),
    ),
    "person": DatasetSpec(
        dataset_id="person",
        task_config="person_objectnav_hm3d.yaml",
        data_path=(
            "data/datasets/objectnav_hm3d_person_v1/"
            "{split}/{split}.json.gz"
        ),
    ),
}


@dataclass(frozen=True)
class Baseline:
    global_planner: str
    local_planner: str


@dataclass(frozen=True)
class RunSpec:
    dataset_id: str
    global_planner: str
    local_planner: str
    seed: int
    episodes: int
    condition: str = "normal"

    @property
    def run_id(self) -> str:
        return (
            f"{self.dataset_id}__{self.condition}"
            f"__G-{self.global_planner}"
            f"__L-{self.local_planner}"
            f"__s-{self.seed}"
            f"__n-{self.episodes}"
        )


class BenchmarkConfigurationError(ValueError):
    """Raised before any navigation process is started."""


_RUN_ID_RE = re.compile(
    r"^(?P<dataset>objectnav|person)"
    r"__(?P<condition>normal|fire-none|fire-sensed|fire-oracle)"
    r"__G-(?P<global>nearest|co_ut|fill|random|gpt)"
    r"__L-(?P<local>fmm|astar|rl|pointnav)"
    r"__s-(?P<seed>-?\d+)__n-(?P<episodes>\d+)$"
)


def parse_run_id(run_id: str) -> RunSpec:
    """Recover a :class:`RunSpec` from the launcher's stable run ID."""
    match = _RUN_ID_RE.fullmatch(str(run_id))
    if match is None:
        raise BenchmarkConfigurationError(
            f"invalid benchmark run ID: {run_id!r}"
        )
    return RunSpec(
        dataset_id=match.group("dataset"),
        global_planner=match.group("global"),
        local_planner=match.group("local"),
        seed=int(match.group("seed")),
        episodes=int(match.group("episodes")),
        condition=match.group("condition"),
    )


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sha256(dataset_path: Path) -> str:
    """Hash the root manifest and every content shard in stable order."""

    files = [dataset_path]
    files.extend(sorted((dataset_path.parent / "content").glob("*.json.gz")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(dataset_path.parent).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_baselines(
    global_planners: Sequence[str],
    local_planners: Sequence[str],
    *,
    matrix: str,
    reference_global: str,
    reference_local: str,
) -> List[Baseline]:
    """Build a deterministic, duplicate-free planner matrix."""

    unknown_global = sorted(set(global_planners) - set(GLOBAL_PLANNERS))
    unknown_local = sorted(set(local_planners) - set(LOCAL_PLANNERS))
    if unknown_global:
        raise BenchmarkConfigurationError(
            f"unknown global planners: {', '.join(unknown_global)}"
        )
    if unknown_local:
        raise BenchmarkConfigurationError(
            f"unknown local planners: {', '.join(unknown_local)}"
        )
    if reference_global not in GLOBAL_PLANNERS:
        raise BenchmarkConfigurationError(
            f"unknown reference global planner: {reference_global}"
        )
    if reference_local not in LOCAL_PLANNERS:
        raise BenchmarkConfigurationError(
            f"unknown reference local planner: {reference_local}"
        )
    if not global_planners or not local_planners:
        raise BenchmarkConfigurationError(
            "at least one global and one local planner are required"
        )

    if matrix == "controlled":
        candidates = [
            *(Baseline(name, reference_local) for name in global_planners),
            *(Baseline(reference_global, name) for name in local_planners),
        ]
    elif matrix == "cartesian":
        candidates = [
            Baseline(global_name, local_name)
            for global_name in global_planners
            for local_name in local_planners
        ]
    else:
        raise BenchmarkConfigurationError(
            f"unknown matrix {matrix!r}; choose controlled or cartesian"
        )

    unique: List[Baseline] = []
    seen = set()
    for baseline in candidates:
        key = (baseline.global_planner, baseline.local_planner)
        if key not in seen:
            unique.append(baseline)
            seen.add(key)
    return unique


def build_run_specs(
    datasets: Sequence[str],
    baselines: Sequence[Baseline],
    seeds: Sequence[int],
    episodes: int,
    condition: str = "normal",
) -> List[RunSpec]:
    if episodes <= 0:
        raise BenchmarkConfigurationError("--episodes must be positive")
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise BenchmarkConfigurationError(
            f"unknown datasets: {', '.join(unknown)}"
        )
    if not datasets:
        raise BenchmarkConfigurationError("at least one dataset is required")
    if not seeds:
        raise BenchmarkConfigurationError("at least one seed is required")
    if condition not in CONDITIONS:
        raise BenchmarkConfigurationError(
            f"unknown condition {condition!r}; choose from {CONDITIONS}"
        )

    return [
        RunSpec(
            dataset_id=dataset_id,
            global_planner=baseline.global_planner,
            local_planner=baseline.local_planner,
            seed=int(seed),
            episodes=int(episodes),
            condition=condition,
        )
        for dataset_id in datasets
        for seed in seeds
        for baseline in baselines
    ]


def count_dataset_episodes(dataset_path: Path) -> int:
    """Count root episodes or sharded ``content/*.json.gz`` episodes."""

    if not dataset_path.is_file():
        raise BenchmarkConfigurationError(
            f"dataset manifest does not exist: {dataset_path}"
        )
    try:
        with gzip.open(dataset_path, "rt", encoding="utf-8") as stream:
            root_payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigurationError(
            f"cannot read dataset manifest {dataset_path}: {exc}"
        ) from exc

    root_episodes = root_payload.get("episodes", [])
    if root_episodes:
        return len(root_episodes)

    content_dir = dataset_path.parent / "content"
    shards = sorted(content_dir.glob("*.json.gz"))
    if not shards:
        return 0
    total = 0
    for shard in shards:
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as stream:
                total += len(json.load(stream).get("episodes", []))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkConfigurationError(
                f"cannot read dataset shard {shard}: {exc}"
            ) from exc
    return total


def _resolve_executable(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.parent != Path("."):
        if not path.is_file():
            raise BenchmarkConfigurationError(
                f"Python executable does not exist: {path}"
            )
        return str(path.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise BenchmarkConfigurationError(
            f"Python executable is not on PATH: {value}"
        )
    return resolved


def _resolve_project_file(value: Optional[str], label: str) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise BenchmarkConfigurationError(f"{label} does not exist: {path}")
    return path


def preflight(
    run_specs: Sequence[RunSpec],
    *,
    python_executable: str,
    pointnav_checkpoint: str,
    rl_checkpoint: Optional[str],
    dry_run: bool,
    fire_type: str = "multi_origin",
    fire_intensity: str = "medium",
    fire_scenes_root: str = "scenes",
    fire_out_root: str = "outputs/fire_world",
) -> Tuple[dict, List[str]]:
    """Validate all shared assets before the first long-running baseline."""

    resolved_python = _resolve_executable(python_executable)
    main_path = PROJECT_ROOT / "main.py"
    if not main_path.is_file():
        raise BenchmarkConfigurationError(f"main.py is missing: {main_path}")

    requested_datasets = sorted({run.dataset_id for run in run_specs})
    dataset_metadata = {}
    max_requested = max(run.episodes for run in run_specs)
    for dataset_id in requested_datasets:
        dataset = DATASETS[dataset_id]
        task_path = PROJECT_ROOT / "configs" / dataset.task_config
        if not task_path.is_file():
            raise BenchmarkConfigurationError(
                f"task config does not exist: {task_path}"
            )
        episode_count = count_dataset_episodes(dataset.resolved_data_path)
        if episode_count < max_requested:
            raise BenchmarkConfigurationError(
                f"dataset {dataset_id!r} has {episode_count} episodes, "
                f"fewer than requested {max_requested}"
            )
        dataset_metadata[dataset_id] = {
            **asdict(dataset),
            "resolved_data_path": str(dataset.resolved_data_path),
            "available_episodes": episode_count,
            "sha256": _dataset_sha256(dataset.resolved_data_path),
        }

    local_planners = {run.local_planner for run in run_specs}
    assets = {}
    if "pointnav" in local_planners:
        checkpoint = _resolve_project_file(
            pointnav_checkpoint,
            "PointNav checkpoint",
        )
        assets["pointnav_checkpoint"] = {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": _file_sha256(checkpoint),
        }
    if "rl" in local_planners:
        if rl_checkpoint is None:
            raise BenchmarkConfigurationError(
                "--local-planners includes rl, but --rl-checkpoint was not "
                "provided; the supplementary map-RL baseline requires a "
                "trained metadata-bearing checkpoint"
            )
        checkpoint = _resolve_project_file(
            rl_checkpoint,
            "RL local-planner checkpoint",
        )
        assets["rl_checkpoint"] = {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": _file_sha256(checkpoint),
        }

    warnings = []
    if any(run.global_planner == "gpt" for run in run_specs):
        if not os.environ.get("OPENAI_API_KEY"):
            message = (
                "OPENAI_API_KEY is not set, but the matrix includes gpt; "
                "exclude gpt or export the key before a real run"
            )
            if dry_run:
                warnings.append(message)
            else:
                raise BenchmarkConfigurationError(message)

    fire_assets = {}
    if any(run.condition != "normal" for run in run_specs):
        from utils.fire_world.plan_selection import (
            discover_runnable_fire_scenes,
        )

        requested_scene_ids = set()
        for dataset_id in requested_datasets:
            content_dir = DATASETS[dataset_id].resolved_data_path.parent / "content"
            requested_scene_ids.update(
                path.name.removesuffix(".json.gz")
                for path in content_dir.glob("*.json.gz")
            )
        selections = discover_runnable_fire_scenes(
            intensity=fire_intensity,
            fire_type=fire_type,
            scenes_root=fire_scenes_root,
            out_root=fire_out_root,
            scene_ids=requested_scene_ids,
        )
        missing = sorted(requested_scene_ids - set(selections))
        if missing:
            raise BenchmarkConfigurationError(
                "FireWorld timelines are missing for dataset scenes: "
                + ", ".join(missing)
            )
        fire_assets = {
            "fire_type": fire_type,
            "intensity": fire_intensity,
            "scenes_root": str(fire_scenes_root),
            "out_root": str(fire_out_root),
            "scene_count": len(selections),
            "plans": {
                scene_id: selection.plan_id
                for scene_id, selection in sorted(selections.items())
            },
        }

    return {
        "python": resolved_python,
        "main": str(main_path),
        "datasets": dataset_metadata,
        "assets": assets,
        "fire_assets": fire_assets,
    }, warnings


_PROTECTED_MAIN_FLAGS = {
    "--task_config",
    "--config",
    "--dataset_path",
    "--scenes_dir",
    "--max_episodes",
    "--start_episode",
    "--resume_metrics_path",
    "--seed",
    "--num_agents",
    "--nav_mode",
    "--local_planner",
    "--dump_location",
    "-d",
    "--visualize",
    "--print_images",
    "--fire_world",
    "--risk_enabled",
    "--risk_source",
    "--fire_world_plan_id",
    "--fire_world_fire_type",
    "--fire_world_intensity",
    "--fire_world_scenes_root",
    "--fire_world_out_root",
    "--fire_clock_mode",
    "--fire_render_backend",
    "--fire_render_device",
    "--fire_fast",
    "--fire_world_n_steps",
    "--fire_world_render_scale",
    "--fire_dump_dir",
    "--fire_save_every",
    "--fire_save_npz",
    "--fire_show_window",
    "--risk_dump_dir",
    "--risk_run_id",
    "--risk_save_every",
    "--risk_save_traces",
    "--pointnav_checkpoint",
    "--rl_local_checkpoint",
}

_NON_SEMANTIC_ARTIFACT_FLAGS = {
    "--visualize",
    "--print_images",
    "--fire_save_every",
    "--fire_save_npz",
    "--fire_show_window",
    "--risk_save_every",
    "--risk_save_traces",
}


def _without_artifact_flags(command: Sequence[str]) -> List[str]:
    """Remove output-only switches when comparing resumable run commands."""

    normalized = []
    index = 0
    while index < len(command):
        token = command[index]
        flag = token.split("=", 1)[0]
        if flag in _NON_SEMANTIC_ARTIFACT_FLAGS:
            index += 1 if "=" in token else 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def _artifact_only_manifest_change(existing: dict, requested: dict) -> bool:
    """Allow an interrupted run to adopt stricter artifact suppression."""

    existing_payload = dict(existing)
    requested_payload = dict(requested)
    existing_payload.pop("fingerprint", None)
    requested_payload.pop("fingerprint", None)
    try:
        existing_payload["command"] = _without_artifact_flags(
            existing_payload["command"]
        )
        requested_payload["command"] = _without_artifact_flags(
            requested_payload["command"]
        )
    except (KeyError, TypeError):
        return False
    return existing_payload == requested_payload


def validate_extra_main_args(extra_args: Sequence[str]) -> None:
    conflicting = sorted(
        {
            token.split("=", 1)[0]
            for token in extra_args
            if token.split("=", 1)[0] in _PROTECTED_MAIN_FLAGS
        }
    )
    if conflicting:
        raise BenchmarkConfigurationError(
            "--main-args cannot override launcher-owned flags: "
            + ", ".join(conflicting)
        )


def build_command(
    run: RunSpec,
    run_dir: Path,
    *,
    python_executable: str,
    pointnav_checkpoint: str,
    pointnav_device: str,
    pointnav_deterministic: int,
    rl_checkpoint: Optional[str],
    rl_device: str,
    rl_deterministic: int,
    num_agents: int,
    extra_main_args: Sequence[str],
    fire_type: str = "multi_origin",
    fire_intensity: str = "medium",
    fire_scenes_root: str = "scenes",
    fire_out_root: str = "outputs/fire_world",
    fire_render_backend: str = "torch",
    fire_render_device: str = "cuda:0",
) -> List[str]:
    dataset = DATASETS[run.dataset_id]
    command = [
        python_executable,
        str(PROJECT_ROOT / "main.py"),
        "--task_config",
        dataset.task_config,
        "--max_episodes",
        str(run.episodes),
        "--seed",
        str(run.seed),
        "--num_agents",
        str(num_agents),
        "--nav_mode",
        run.global_planner,
        "--local_planner",
        run.local_planner,
        "--visualize",
        "0",
        "--print_images",
        "0",
        "--dump_location",
        str(run_dir / "navigation"),
    ]
    if run.condition == "normal":
        command.extend(["--fire_world", "0", "--risk_enabled", "0"])
    else:
        risk_source = run.condition.removeprefix("fire-")
        command.extend(
            [
                "--fire_world",
                "1",
                "--fire_world_plan_id",
                "auto",
                "--fire_world_fire_type",
                fire_type,
                "--fire_world_intensity",
                fire_intensity,
                "--fire_world_scenes_root",
                fire_scenes_root,
                "--fire_world_out_root",
                fire_out_root,
                "--fire_clock_mode",
                "step",
                "--fire_fast",
                "1",
                "--fire_world_n_steps",
                "24",
                "--fire_world_render_scale",
                "0.5",
                "--fire_render_backend",
                fire_render_backend,
                "--fire_render_device",
                fire_render_device,
                "--fire_save_every",
                "0",
                "--fire_save_npz",
                "0",
                "--fire_show_window",
                "0",
                "--risk_enabled",
                "1",
                "--risk_source",
                risk_source,
                "--fire_dump_dir",
                str(run_dir / "fire_sensors"),
                "--risk_dump_dir",
                str(run_dir / "risk"),
                "--risk_run_id",
                run.run_id,
                "--risk_save_every",
                "0",
                "--risk_save_traces",
                "0",
            ]
        )
    if run.local_planner == "pointnav":
        command.extend(
            [
                "--pointnav_checkpoint",
                pointnav_checkpoint,
                "--pointnav_device",
                pointnav_device,
                "--pointnav_deterministic",
                str(int(pointnav_deterministic)),
            ]
        )
    elif run.local_planner == "rl":
        if rl_checkpoint is None:
            raise BenchmarkConfigurationError(
                "RL command requested without --rl-checkpoint"
            )
        command.extend(
            [
                "--rl_local_checkpoint",
                rl_checkpoint,
                "--rl_local_device",
                rl_device,
                "--rl_local_deterministic",
                str(int(rl_deterministic)),
            ]
        )
    command.extend(extra_main_args)
    return command


_PROGRESS_RE = re.compile(r"---\((\d+)/(\d+)\)")
_METRIC_RE = re.compile(
    r"(?:^|,\s*)([A-Za-z0-9_./-]+):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def parse_final_aggregate(log_path: Path) -> Optional[dict]:
    """Extract the final aggregate row emitted by ``main.py``."""

    if not log_path.is_file():
        return None
    last_match = None
    for line in log_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        progress = _PROGRESS_RE.search(line)
        if progress:
            last_match = (line, progress)
    if last_match is None:
        return None

    line, progress = last_match
    metrics = {
        name: float(value)
        for name, value in _METRIC_RE.findall(
            line[: progress.start()].strip()
        )
    }
    return {
        "episodes_completed": int(progress.group(1)),
        "episodes_planned": int(progress.group(2)),
        "metrics": metrics,
        "source": "stdout.log",
    }


def _git_provenance() -> dict:
    def run_git(*args: str) -> Optional[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty": bool(run_git("status", "--porcelain")),
    }


def _record_previous_status(run_dir: Path) -> int:
    previous = _read_json(run_dir / "status.json")
    if previous is None:
        return 1
    history_path = run_dir / "status_history.jsonl"
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(previous, sort_keys=True) + "\n")
    return int(previous.get("attempt", 0)) + 1


def _resume_candidate(run: RunSpec, run_dir: Path) -> Optional[dict]:
    """Select the most advanced valid exact or legacy metric checkpoint."""
    candidates = [
        run_dir / "navigation" / "metrics" / "resume_state.json",
        run_dir / "metrics" / "aggregate.json",
    ]
    valid = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            state = load_metric_resume(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            state.episodes_planned != run.episodes
            or state.episodes_completed <= 0
            or state.episodes_completed >= run.episodes
        ):
            continue
        valid.append(
            {
                "episodes_completed": state.episodes_completed,
                "start_episode": state.episodes_completed + 1,
                "metrics_path": str(path.resolve()),
                "precision": state.precision,
                "exact": state.precision != "legacy_3_decimal_average",
            }
        )
    if not valid:
        return None
    return max(
        valid,
        key=lambda item: (
            item["episodes_completed"],
            int(item["exact"]),
        ),
    )


def _effective_command(
    command: Sequence[str],
    resume: Optional[dict],
) -> List[str]:
    effective = list(command)
    if resume is not None:
        effective.extend(
            [
                "--start_episode",
                str(resume["start_episode"]),
                "--resume_metrics_path",
                resume["metrics_path"],
            ]
        )
    return effective


def execute_run(
    run: RunSpec,
    run_dir: Path,
    command: Sequence[str],
    *,
    force: bool,
    resume_incomplete: bool = True,
) -> dict:
    """Execute one run and return its terminal status."""

    manifest_payload = {
        "schema_version": 1,
        "run": asdict(run),
        "run_id": run.run_id,
        "condition": run.condition,
        "command": list(command),
        "working_directory": str(PROJECT_ROOT),
    }
    manifest_payload["fingerprint"] = _stable_hash(manifest_payload)
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    existing_manifest = _read_json(manifest_path)
    existing_status = _read_json(status_path)

    if existing_manifest is not None:
        if (
            existing_manifest.get("fingerprint")
            != manifest_payload["fingerprint"]
            and not _artifact_only_manifest_change(
                existing_manifest, manifest_payload
            )
            and not force
        ):
            raise BenchmarkConfigurationError(
                f"run directory contains a different command: {run_dir}; "
                "choose another --study-id or pass --force"
            )
        if (
            not force
            and existing_status is not None
            and existing_status.get("status") == "completed"
        ):
            return {
                **existing_status,
                "launcher_result": "resumed",
            }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    resume = (
        _resume_candidate(run, run_dir)
        if resume_incomplete and not force
        else None
    )
    effective_command = _effective_command(command, resume)
    attempt = _record_previous_status(run_dir)
    _write_json(manifest_path, manifest_payload)
    (run_dir / "command.txt").write_text(
        shlex.join(effective_command) + "\n",
        encoding="utf-8",
    )

    started_at = _now()
    running_status = {
        "schema_version": 1,
        "run_id": run.run_id,
        "status": "running",
        "attempt": attempt,
        "started_at": started_at,
        "requested_episodes": run.episodes,
        "resume": resume,
    }
    _write_json(status_path, running_status)

    log_path = run_dir / "stdout.log"
    if log_path.is_file():
        archive_path = (
            run_dir
            / "attempts"
            / f"attempt_{attempt - 1:02d}_stdout.log"
        )
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, archive_path)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("MPLCONFIGDIR", "/tmp/conav-matplotlib")
    start_time = time.monotonic()
    return_code = None
    interrupted = False
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            effective_command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    log_stream.write(line)
                    log_stream.flush()
                    print(f"[{run.run_id}] {line}", end="", flush=True)
            return_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            process.terminate()
            try:
                return_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()

    aggregate = parse_final_aggregate(log_path)
    observed_episodes = (
        None if aggregate is None else aggregate["episodes_completed"]
    )
    complete = (
        not interrupted
        and return_code == 0
        and observed_episodes == run.episodes
        and aggregate["episodes_planned"] == run.episodes
    )
    if aggregate is not None:
        aggregate_path = run_dir / "metrics" / "aggregate.json"
        existing_aggregate = _read_json(aggregate_path)
        if (
            existing_aggregate is None
            or int(existing_aggregate.get("episodes_completed", 0))
            <= aggregate["episodes_completed"]
        ):
            _write_json(aggregate_path, aggregate)

    terminal_status = {
        **running_status,
        "status": (
            "interrupted"
            if interrupted
            else ("completed" if complete else "failed")
        ),
        "finished_at": _now(),
        "elapsed_seconds": round(time.monotonic() - start_time, 3),
        "return_code": return_code,
        "observed_episodes": observed_episodes,
        "aggregate_available": aggregate is not None,
        "launcher_result": "executed",
    }
    if return_code == 0 and not complete:
        terminal_status["error"] = (
            "main.py exited successfully but did not report the exact "
            f"requested episode count {run.episodes}"
        )
    _write_json(status_path, terminal_status)
    return terminal_status


def _environment_payload(python_executable: str) -> dict:
    return {
        "created_at": _now(),
        "project_root": str(PROJECT_ROOT),
        "python": python_executable,
        "launcher_python_version": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": _git_provenance(),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run normal or FireWorld ObjectNav/person planner baselines with "
            "isolated, resumable outputs."
        )
    )
    parser.add_argument(
        "--condition",
        choices=CONDITIONS,
        default="normal",
        help="normal or registered FireWorld risk condition",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_IDS,
        default=list(DATASET_IDS),
    )
    parser.add_argument(
        "--global-planners",
        nargs="+",
        choices=GLOBAL_PLANNERS,
        default=list(GLOBAL_PLANNERS),
    )
    parser.add_argument(
        "--local-planners",
        nargs="+",
        choices=LOCAL_PLANNERS,
        default=list(DEFAULT_LOCAL_PLANNERS),
    )
    parser.add_argument(
        "--matrix",
        choices=("controlled", "cartesian"),
        default="controlled",
        help=(
            "controlled freezes FMM for global comparisons and co_ut for "
            "local comparisons; cartesian runs every selected pair"
        ),
    )
    parser.add_argument(
        "--reference-global",
        choices=GLOBAL_PLANNERS,
        default="co_ut",
    )
    parser.add_argument(
        "--reference-local",
        choices=LOCAL_PLANNERS,
        default="fmm",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--fire-type", default="multi_origin")
    parser.add_argument("--fire-intensity", default="medium")
    parser.add_argument("--fire-scenes-root", default="scenes")
    parser.add_argument("--fire-out-root", default="outputs/fire_world")
    parser.add_argument(
        "--fire-render-backend",
        choices=("torch", "numpy", "auto"),
        default="torch",
    )
    parser.add_argument("--fire-render-device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        default="outputs/benchmarks",
        help="study directories are created below this path",
    )
    parser.add_argument(
        "--study-id",
        default=None,
        help="default: <condition>_planner_baselines_<episodes>ep",
    )
    parser.add_argument(
        "--only-run-ids",
        nargs="+",
        default=None,
        help=(
            "execute only these run IDs from an existing study while "
            "preserving its master manifest and completeness scope"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--pointnav-checkpoint",
        default="data/ddppo-models/gibson-2plus-resnet50.pth",
    )
    parser.add_argument("--pointnav-device", default="cuda:0")
    parser.add_argument(
        "--pointnav-deterministic",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--rl-checkpoint", default=None)
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument(
        "--rl-deterministic",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="preflight and print commands without creating output files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run completed matching runs and overwrite their live log",
    )
    parser.add_argument(
        "--restart-incomplete",
        action="store_true",
        help=(
            "disable automatic episode resume and restart incomplete runs "
            "from episode 1"
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed run",
    )
    parser.add_argument(
        "--main-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "additional main.py arguments; place this option last. Core "
            "dataset/planner/episode/output flags cannot be overridden"
        ),
    )
    return parser


def _study_manifest(
    args: argparse.Namespace,
    run_specs: Sequence[RunSpec],
    preflight_payload: dict,
) -> dict:
    payload = {
        "schema_version": 1,
        "study_id": args.study_id,
        "condition": args.condition,
        "matrix": args.matrix,
        "episodes_per_run": args.episodes,
        "datasets": list(args.datasets),
        "global_planners": list(args.global_planners),
        "local_planners": list(args.local_planners),
        "reference_global": args.reference_global,
        "reference_local": args.reference_local,
        "seeds": list(args.seeds),
        "num_agents": args.num_agents,
        "run_ids": [run.run_id for run in run_specs],
        "preflight": preflight_payload,
        "extra_main_args": list(args.main_args),
        "fire": {
            "fire_type": args.fire_type,
            "intensity": args.fire_intensity,
            "scenes_root": args.fire_scenes_root,
            "out_root": args.fire_out_root,
            "render_backend": args.fire_render_backend,
            "render_device": args.fire_render_device,
            "clock_mode": "step",
        },
    }
    payload["fingerprint"] = _stable_hash(payload)
    return payload


def _prepare_study(
    study_dir: Path,
    manifest: dict,
    python_executable: str,
    *,
    force: bool,
) -> None:
    manifest_path = study_dir / "study_manifest.json"
    existing = _read_json(manifest_path)
    if (
        existing is not None
        and existing.get("fingerprint") != manifest["fingerprint"]
        and not force
    ):
        raise BenchmarkConfigurationError(
            f"study {study_dir.name!r} already has a different matrix; "
            "choose another --study-id or pass --force"
        )
    study_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["created_at"] = (
        existing.get("created_at") if existing is not None else _now()
    )
    payload["updated_at"] = _now()
    _write_json(manifest_path, payload)
    _write_json(
        study_dir / "environment.json",
        _environment_payload(python_executable),
    )


def _write_completeness(
    study_dir: Path,
    run_specs: Sequence[RunSpec],
    results: Dict[str, dict],
) -> dict:
    statuses = {
        run.run_id: results.get(
            run.run_id,
            {"status": "not_started"},
        )
        for run in run_specs
    }
    summary = {
        "generated_at": _now(),
        "planned_runs": len(run_specs),
        "completed_runs": sum(
            status.get("status") == "completed"
            for status in statuses.values()
        ),
        "failed_runs": sum(
            status.get("status") in {"failed", "interrupted"}
            for status in statuses.values()
        ),
        "not_started_runs": sum(
            status.get("status") == "not_started"
            for status in statuses.values()
        ),
        "runs": statuses,
    }
    _write_json(study_dir / "reports" / "completeness.json", summary)
    return summary


def _existing_study_runs(
    study_dir: Path,
    requested_run_ids: Sequence[str],
) -> Tuple[List[RunSpec], List[RunSpec], dict]:
    """Resolve an ordered run subset without replacing the master manifest."""
    manifest_path = study_dir / "study_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        raise BenchmarkConfigurationError(
            "--only-run-ids requires an existing study manifest: "
            f"{manifest_path}"
        )
    manifest_ids = manifest.get("run_ids")
    if not isinstance(manifest_ids, list) or not manifest_ids:
        raise BenchmarkConfigurationError(
            f"study manifest has no run_ids: {manifest_path}"
        )
    if len(manifest_ids) != len(set(manifest_ids)):
        raise BenchmarkConfigurationError(
            f"study manifest contains duplicate run IDs: {manifest_path}"
        )

    master_runs = [parse_run_id(run_id) for run_id in manifest_ids]
    known = {run.run_id: run for run in master_runs}
    requested = list(requested_run_ids)
    if not requested:
        raise BenchmarkConfigurationError(
            "--only-run-ids needs at least one run ID"
        )
    if len(requested) != len(set(requested)):
        raise BenchmarkConfigurationError(
            "--only-run-ids contains duplicate run IDs"
        )
    unknown = [run_id for run_id in requested if run_id not in known]
    if unknown:
        raise BenchmarkConfigurationError(
            "run IDs are not part of the existing study: "
            + ", ".join(unknown)
        )
    selected = [known[run_id] for run_id in requested]
    return master_runs, selected, manifest


def _existing_results(
    study_dir: Path,
    run_specs: Sequence[RunSpec],
) -> Dict[str, dict]:
    results = {}
    for run in run_specs:
        status = _read_json(study_dir / "runs" / run.run_id / "status.json")
        results[run.run_id] = (
            status if status is not None else {"status": "not_started"}
        )
    return results


def run_launcher(args: argparse.Namespace) -> int:
    if args.num_agents <= 0:
        raise BenchmarkConfigurationError("--num-agents must be positive")
    validate_extra_main_args(args.main_args)
    args.study_id = (
        args.study_id
        if args.study_id is not None
        else f"{args.condition}_planner_baselines_{args.episodes}ep"
    )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    study_dir = output_root.resolve() / args.study_id

    selecting_existing = args.only_run_ids is not None
    if selecting_existing:
        master_run_specs, run_specs, existing_manifest = (
            _existing_study_runs(study_dir, args.only_run_ids)
        )
        manifest_num_agents = int(
            existing_manifest.get("num_agents", args.num_agents)
        )
        if manifest_num_agents != args.num_agents:
            raise BenchmarkConfigurationError(
                "--num-agents does not match the existing study manifest"
            )
        baselines = [
            Baseline(run.global_planner, run.local_planner)
            for run in run_specs
        ]
    else:
        baselines = build_baselines(
            args.global_planners,
            args.local_planners,
            matrix=args.matrix,
            reference_global=args.reference_global,
            reference_local=args.reference_local,
        )
        run_specs = build_run_specs(
            args.datasets,
            baselines,
            args.seeds,
            args.episodes,
            condition=args.condition,
        )
        master_run_specs = run_specs

    preflight_payload, warnings = preflight(
        run_specs,
        python_executable=args.python,
        pointnav_checkpoint=args.pointnav_checkpoint,
        rl_checkpoint=args.rl_checkpoint,
        dry_run=args.dry_run,
        fire_type=args.fire_type,
        fire_intensity=args.fire_intensity,
        fire_scenes_root=args.fire_scenes_root,
        fire_out_root=args.fire_out_root,
    )
    args.python = preflight_payload["python"]

    selected_datasets = sorted({run.dataset_id for run in run_specs})
    selected_seeds = sorted({run.seed for run in run_specs})
    episode_budgets = sorted({run.episodes for run in run_specs})
    episode_label = (
        str(episode_budgets[0])
        if len(episode_budgets) == 1
        else "/".join(str(value) for value in episode_budgets)
    )
    print(
        f"[benchmark] matrix={args.matrix} "
        f"baselines={len(baselines)} datasets={len(selected_datasets)} "
        f"seeds={len(selected_seeds)} runs={len(run_specs)} "
        f"episodes/run={episode_label}"
    )
    if selecting_existing:
        print(
            "[benchmark] existing-study subset "
            f"{len(run_specs)}/{len(master_run_specs)} runs; "
            "master manifest will be preserved"
        )
    print(
        "[benchmark] dataset episodes: "
        + ", ".join(
            f"{dataset_id}="
            f"{preflight_payload['datasets'][dataset_id]['available_episodes']}"
            for dataset_id in selected_datasets
        )
    )
    for warning in warnings:
        print(f"[benchmark] WARNING: {warning}")

    commands = {}
    for run in run_specs:
        run_dir = study_dir / "runs" / run.run_id
        commands[run.run_id] = build_command(
            run,
            run_dir,
            python_executable=args.python,
            pointnav_checkpoint=args.pointnav_checkpoint,
            pointnav_device=args.pointnav_device,
            pointnav_deterministic=args.pointnav_deterministic,
            rl_checkpoint=args.rl_checkpoint,
            rl_device=args.rl_device,
            rl_deterministic=args.rl_deterministic,
            num_agents=args.num_agents,
            extra_main_args=args.main_args,
            fire_type=args.fire_type,
            fire_intensity=args.fire_intensity,
            fire_scenes_root=args.fire_scenes_root,
            fire_out_root=args.fire_out_root,
            fire_render_backend=args.fire_render_backend,
            fire_render_device=args.fire_render_device,
        )

    if args.dry_run:
        for index, run in enumerate(run_specs, start=1):
            print(
                f"[{index:02d}/{len(run_specs):02d}] {run.run_id}\n"
                f"  {shlex.join(commands[run.run_id])}"
            )
        return 0

    if selecting_existing:
        results = _existing_results(study_dir, master_run_specs)
    else:
        manifest = _study_manifest(args, run_specs, preflight_payload)
        _prepare_study(
            study_dir,
            manifest,
            args.python,
            force=args.force,
        )
        results: Dict[str, dict] = {}
    for index, run in enumerate(run_specs, start=1):
        print(f"[benchmark] run {index}/{len(run_specs)}: {run.run_id}")
        run_dir = study_dir / "runs" / run.run_id
        try:
            result = execute_run(
                run,
                run_dir,
                commands[run.run_id],
                force=args.force,
                resume_incomplete=not args.restart_incomplete,
            )
        except BenchmarkConfigurationError as exc:
            result = {
                "run_id": run.run_id,
                "status": "failed",
                "error": str(exc),
                "launcher_result": "configuration_error",
            }
            print(f"[benchmark] ERROR: {exc}", file=sys.stderr)
        results[run.run_id] = result
        if result.get("launcher_result") == "resumed":
            print(f"[benchmark] resumed completed run: {run.run_id}")
        elif result.get("status") != "completed":
            print(
                f"[benchmark] failed run: {run.run_id}",
                file=sys.stderr,
            )
            if args.fail_fast or result.get("status") == "interrupted":
                break

    completeness = _write_completeness(
        study_dir,
        master_run_specs,
        results,
    )
    print(
        "[benchmark] complete "
        f"{completeness['completed_runs']}/{completeness['planned_runs']} "
        f"runs; report={study_dir / 'reports' / 'completeness.json'}"
    )
    return (
        0
        if completeness["completed_runs"] == completeness["planned_runs"]
        else 1
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        return run_launcher(args)
    except BenchmarkConfigurationError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
