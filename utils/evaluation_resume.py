"""Exact episode-progress checkpoints for resumable benchmark evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class MetricResumeState:
    episodes_completed: int
    episodes_planned: int
    metric_sums: Dict[str, float]
    precision: str
    metric_contract: Optional[str] = None
    last_episode_id: Optional[str] = None
    last_scene_id: Optional[str] = None


def _metric_map(values, *, label: str) -> Dict[str, float]:
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a JSON object")
    result = {}
    for key, value in values.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label}[{key!r}] must be finite")
        result[str(key)] = numeric
    return result


def load_metric_resume(path) -> MetricResumeState:
    """Load an exact state or migrate a legacy rounded aggregate."""
    resume_path = Path(path)
    with resume_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    completed = int(payload.get("episodes_completed", 0))
    planned = int(payload.get("episodes_planned", 0))
    if completed <= 0:
        raise ValueError("resume state must contain completed episodes")
    if planned <= 0 or completed > planned:
        raise ValueError("resume episode counts are inconsistent")

    if "metric_sums" in payload:
        sums = _metric_map(payload["metric_sums"], label="metric_sums")
        precision = str(payload.get("precision", "exact"))
    elif "metrics" in payload:
        averages = _metric_map(payload["metrics"], label="metrics")
        sums = {
            name: value * completed
            for name, value in averages.items()
        }
        precision = "legacy_3_decimal_average"
    else:
        raise ValueError(
            "resume state needs metric_sums or legacy aggregate metrics"
        )

    return MetricResumeState(
        episodes_completed=completed,
        episodes_planned=planned,
        metric_sums=sums,
        precision=precision,
        metric_contract=(
            None
            if payload.get("metric_contract") is None
            else str(payload["metric_contract"])
        ),
        last_episode_id=(
            None
            if payload.get("last_episode_id") is None
            else str(payload["last_episode_id"])
        ),
        last_scene_id=(
            None
            if payload.get("last_scene_id") is None
            else str(payload["last_scene_id"])
        ),
    )


def write_metric_resume(
    path,
    *,
    episodes_completed: int,
    episodes_planned: int,
    metric_sums,
    precision: str,
    metric_contract: Optional[str] = None,
    last_episode_id,
    last_scene_id,
) -> None:
    """Atomically persist exact cumulative sums after one episode."""
    completed = int(episodes_completed)
    planned = int(episodes_planned)
    sums = _metric_map(dict(metric_sums), label="metric_sums")
    if completed <= 0 or planned <= 0 or completed > planned:
        raise ValueError("resume episode counts are inconsistent")
    averages = {
        name: value / completed
        for name, value in sums.items()
    }
    payload = {
        "schema_version": 1,
        "episodes_completed": completed,
        "episodes_planned": planned,
        "metric_sums": sums,
        "metrics": averages,
        "precision": str(precision),
        "metric_contract": (
            None if metric_contract is None else str(metric_contract)
        ),
        "last_episode_id": (
            None if last_episode_id is None else str(last_episode_id)
        ),
        "last_scene_id": (
            None if last_scene_id is None else str(last_scene_id)
        ),
        "updated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def advance_episode_iterator(env, episodes_completed: int):
    """Leave Habitat's current episode at the first unfinished episode."""
    last_completed = None
    for _ in range(int(episodes_completed)):
        last_completed = env.current_episode
        env.current_episode = next(env.episode_iterator)
    return last_completed


__all__ = [
    "MetricResumeState",
    "advance_episode_iterator",
    "load_metric_resume",
    "write_metric_resume",
]
