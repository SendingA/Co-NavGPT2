"""Lightweight GPT trace logger for behaviour-cloning the frontier assigner.

Activated by setting the ``GNN_TRACE_DIR`` environment variable. When set,
``log_decision`` appends a single JSONL line per global decision step,
containing the serialised feature tensors plus the GPT assignment label.

Files are written under ``$GNN_TRACE_DIR/<timestamp>_<pid>.jsonl`` so that
parallel workers (e.g. ``main_vec.py``) do not race.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Sequence

import numpy as np

from utils.graph_builder import build_features, features_to_jsonable

_FILE_HANDLE = None
_FILE_PATH: Optional[str] = None


def is_enabled() -> bool:
    return bool(os.environ.get("GNN_TRACE_DIR"))


def _ensure_handle():
    global _FILE_HANDLE, _FILE_PATH
    if _FILE_HANDLE is not None:
        return _FILE_HANDLE
    out_dir = os.environ.get("GNN_TRACE_DIR")
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fname = f"trace_{int(time.time())}_{os.getpid()}.jsonl"
    _FILE_PATH = os.path.join(out_dir, fname)
    _FILE_HANDLE = open(_FILE_PATH, "a", buffering=1)  # line-buffered
    print(f"[gpt_trace] writing traces to {_FILE_PATH}")
    return _FILE_HANDLE


def log_decision(
    target_point_list: Sequence[Sequence[int]],
    target_score: Optional[Sequence[float]],
    target_edge_map: Optional[np.ndarray],
    pose_pred: Sequence[Sequence[float]],
    map_size: int,
    goal_name: str,
    gpt_response: Dict[str, str],
    extra: Optional[Dict] = None,
) -> None:
    """Persist one (features, gpt_label) pair if tracing is enabled."""
    handle = _ensure_handle()
    if handle is None:
        return

    feats = build_features(target_point_list, target_score, target_edge_map, pose_pred, map_size)
    if feats is None:
        return

    # Convert {"robot_i": "frontier_j"} -> integer labels per robot
    num_robots = len(pose_pred)
    labels = []
    valid = True
    num_frontiers = len(target_point_list)
    for i in range(num_robots):
        key = f"robot_{i}"
        if key not in gpt_response:
            valid = False
            break
        try:
            idx = int(str(gpt_response[key]).split("_")[-1])
        except (ValueError, IndexError):
            valid = False
            break
        if idx < 0 or idx >= num_frontiers:
            valid = False
            break
        labels.append(idx)

    if not valid:
        return

    record = {
        "ts": time.time(),
        "goal_name": goal_name,
        "map_size": map_size,
        "num_robots": num_robots,
        "num_frontiers": num_frontiers,
        "features": features_to_jsonable(feats),
        "labels": labels,
    }
    if extra:
        record["extra"] = extra
    handle.write(json.dumps(record) + "\n")


def close():
    global _FILE_HANDLE
    if _FILE_HANDLE is not None:
        try:
            _FILE_HANDLE.close()
        finally:
            _FILE_HANDLE = None
