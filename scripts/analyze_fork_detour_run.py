#!/usr/bin/env python3
"""Compare oracle-risk and risk-blind action traces for a fork scenario."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.tune_fire_route_scenarios import _scene_grid  # noqa: E402
from utils.fire_world.fine_tuning import (  # noqa: E402
    CURATED_FIRE_PROFILES,
    radial_hazard_map,
    route_overlay,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--oracle-actions", required=True)
    parser.add_argument("--oracle-risk-summary", required=True)
    parser.add_argument("--oracle-log", required=True)
    parser.add_argument("--baseline-actions", required=True)
    parser.add_argument("--baseline-risk-summary", required=True)
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution-m", type=float, default=0.10)
    parser.add_argument(
        "--scene-dataset-config",
        default=(
            "data/scene_datasets/hm3d_v0.2/"
            "hm3d_annotated_basis.scene_dataset_config.json"
        ),
    )
    return parser


def _read(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paths(actions: Mapping[str, object]) -> Sequence[np.ndarray]:
    result = []
    for agent_id in range(int(actions["num_agents"])):
        result.append(np.asarray([
            step["actions"][agent_id]["position_after"]
            for step in actions["steps"]
        ], dtype=np.float64))
    return result


def _path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path[:, [0, 2]], axis=0), axis=1).sum())


def _cell_set(path: np.ndarray, resolution_m: float) -> set:
    cells = np.rint(path[:, [0, 2]] / float(resolution_m)).astype(np.int64)
    return set(map(tuple, cells))


def _near_fraction(source: set, target: set, radius_cells: int) -> float:
    if not source:
        return 0.0
    radius2 = int(radius_cells) ** 2
    return sum(
        any((row - other_row) ** 2 + (col - other_col) ** 2 <= radius2
            for other_row, other_col in target)
        for row, col in source
    ) / len(source)


def _team_overlap(paths: Sequence[np.ndarray]) -> Dict[str, float]:
    first = _cell_set(paths[0], 0.25)
    second = _cell_set(paths[1], 0.25)
    intersection = first & second
    union = first | second
    return {
        "resolution_m": 0.25,
        "shared_cells": len(intersection),
        "jaccard": len(intersection) / max(1, len(union)),
        "agent_0_shared_fraction": len(intersection) / max(1, len(first)),
        "agent_1_shared_fraction": len(intersection) / max(1, len(second)),
        "agent_0_within_0_5m_of_agent_1": _near_fraction(first, second, 2),
        "agent_1_within_0_5m_of_agent_0": _near_fraction(second, first, 2),
    }


def _log_metrics(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(
        r"distance_to_goal:\s*([0-9.]+), success:\s*([0-9.]+), "
        r"spl:\s*([0-9.]+).*?num_steps:\s*([0-9.]+)",
        text,
    )
    if not matches:
        raise RuntimeError(f"no final task metrics found in {path}")
    distance, success, spl, steps = matches[-1]
    return {
        "distance_to_goal": float(distance),
        "success": float(success),
        "spl": float(spl),
        "num_steps": int(float(steps)),
    }


def _run_metrics(
    actions: Mapping[str, object],
    risk_summary: Mapping[str, object],
    log_path: Path,
    ignition_positions: np.ndarray,
) -> Dict[str, object]:
    paths = _paths(actions)
    agents = []
    for agent_id, path in enumerate(paths):
        risk = np.asarray([
            step["actions"][agent_id]["risk_after"]
            for step in actions["steps"]
        ], dtype=np.float64)
        hard = np.asarray([
            step["actions"][agent_id]["hard_unsafe_after"]
            for step in actions["steps"]
        ], dtype=bool)
        distances = np.linalg.norm(
            path[:, None, [0, 2]] - ignition_positions[None, :, [0, 2]],
            axis=2,
        )
        agents.append({
            "agent_id": agent_id,
            "path_length_m": _path_length(path),
            "displacement_m": float(np.linalg.norm(
                path[-1, [0, 2]] - path[0, [0, 2]]
            )),
            "max_risk": float(risk.max(initial=0.0)),
            "mean_risk": float(risk.mean()) if risk.size else 0.0,
            "hard_unsafe_samples": int(hard.sum()),
            "min_ignition_centre_clearance_m": float(distances.min()),
        })
    team = risk_summary.get("team", {})
    return {
        "planner_source": actions["planner_source"],
        "task": _log_metrics(log_path),
        "safe_success": float(risk_summary.get("safe_success", 0.0)),
        "CHE": float(team.get("CHE", 0.0)),
        "critical_violations": int(team.get("critical_violations", 0)),
        "total_wall_time_s": float(actions["total_wall_time_s"]),
        "agents": agents,
        "trajectory_overlap": _team_overlap(paths),
        "initial_agent_separation_m": float(np.linalg.norm(
            paths[0][0, [0, 2]] - paths[1][0, [0, 2]]
        )),
    }


def _draw_path(image: np.ndarray, cells: np.ndarray, color) -> None:
    if cells.size == 0:
        return
    points = np.stack([cells[:, 1], cells[:, 0]], axis=1).astype(np.int32)
    cv2.polylines(image, [points], False, color, thickness=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    candidate = _read((ROOT / args.candidate_report).resolve())
    plan = _read((ROOT / args.plan_path).resolve())
    oracle_actions = _read((ROOT / args.oracle_actions).resolve())
    oracle_summary = _read((ROOT / args.oracle_risk_summary).resolve())
    baseline_actions = _read((ROOT / args.baseline_actions).resolve())
    baseline_summary = _read((ROOT / args.baseline_risk_summary).resolve())
    ignition_positions = np.asarray(
        [item["position"] for item in plan["ignitions"]], dtype=np.float64
    )
    oracle = _run_metrics(
        oracle_actions,
        oracle_summary,
        (ROOT / args.oracle_log).resolve(),
        ignition_positions,
    )
    baseline = _run_metrics(
        baseline_actions,
        baseline_summary,
        (ROOT / args.baseline_log).resolve(),
        ignition_positions,
    )
    oracle_primary = oracle["agents"][0]
    baseline_primary = baseline["agents"][0]
    oracle_path = _paths(oracle_actions)[0]
    baseline_path = _paths(baseline_actions)[0]
    oracle_cells = _cell_set(oracle_path, 0.25)
    baseline_cells = _cell_set(baseline_path, 0.25)
    actual_union = oracle_cells | baseline_cells
    comparison = {
        "extra_steps_for_oracle": (
            oracle["task"]["num_steps"] - baseline["task"]["num_steps"]
        ),
        "oracle_to_baseline_path_length_ratio": (
            oracle_primary["path_length_m"]
            / max(baseline_primary["path_length_m"], 1e-9)
        ),
        "primary_max_risk_reduction": (
            baseline_primary["max_risk"] - oracle_primary["max_risk"]
        ),
        "CHE_reduction": baseline["CHE"] - oracle["CHE"],
        "critical_violations_avoided": (
            baseline["critical_violations"] - oracle["critical_violations"]
        ),
        "actual_primary_route_divergence": (
            1.0 - len(oracle_cells & baseline_cells) / max(1, len(actual_union))
        ),
    }
    report = {
        "schema_version": 1,
        "scene_id": candidate["scene_id"],
        "plan_id": plan["plan_id"],
        "ignition_sources": [
            {
                "object_id": int(item["object_id"]),
                "category": str(item["category"]),
                "position": [float(value) for value in item["position"]],
            }
            for item in plan["ignitions"]
        ],
        "episode_id": candidate["episode_id"],
        "object_category": candidate["object_category"],
        "experiment_type": "controlled_known_goal_route_contrast",
        "offline_counterfactual": {
            "short_unsafe_length_m": candidate["contrast"]["blind"]["length_m"],
            "short_unsafe_max_risk": candidate["contrast"]["blind"]["max_risk"],
            "long_safe_length_m": candidate["contrast"]["aware"]["length_m"],
            "long_safe_max_risk": candidate["contrast"]["aware"]["max_risk"],
            "detour_ratio": candidate["contrast"]["detour_ratio"],
            "path_divergence": candidate["contrast"]["path_divergence"],
        },
        "oracle": oracle,
        "risk_blind_baseline": baseline,
        "comparison": comparison,
        "overlay_legend": {
            "blue": "offline shortest unsafe route",
            "green": "offline risk-aware safe route",
            "orange": "actual risk_source=none primary trajectory",
            "cyan": "actual oracle risk-aware primary trajectory",
            "purple": "actual oracle secondary-agent trajectory",
        },
    }

    inventory = _read(ROOT / "scenes" / str(candidate["scene_id"]) / "inventory.json")
    grid = _scene_grid(
        inventory,
        (ROOT / args.scene_dataset_config).resolve(),
        float(args.resolution_m),
        float(candidate["start_position"][1]),
    )
    try:
        profile = CURATED_FIRE_PROFILES[str(candidate["profile"])]
        ignition_cells = [
            grid.frame.world_to_grid(position)
            for position in ignition_positions
        ]
        risk = np.zeros(grid.traversible.shape, dtype=np.float32)
        for ignition_cell in ignition_cells:
            source_risk, _ = radial_hazard_map(
                grid.traversible.shape,
                ignition_cell,
                resolution_m=float(args.resolution_m),
                core_radius_m=profile.synthetic_core_radius_m,
                risk_radius_m=profile.synthetic_risk_radius_m,
            )
            risk = np.maximum(risk, source_risk)
        contrast = SimpleNamespace(
            blind=SimpleNamespace(cells=tuple(map(tuple, candidate["contrast"]["blind"]["cells"]))),
            aware=SimpleNamespace(cells=tuple(map(tuple, candidate["contrast"]["aware"]["cells"]))),
        )
        image = route_overlay(
            grid.traversible,
            risk,
            contrast,
            start=candidate["start_cell"],
            goal=candidate["chosen_goal_cell"],
            ignitions=ignition_cells,
        )
        baseline_cells_grid = grid.frame.world_to_grid(baseline_path)
        oracle_paths = _paths(oracle_actions)
        oracle_primary_grid = grid.frame.world_to_grid(oracle_paths[0])
        oracle_secondary_grid = grid.frame.world_to_grid(oracle_paths[1])
        _draw_path(image, baseline_cells_grid, (255, 145, 20))
        _draw_path(image, oracle_primary_grid, (0, 255, 255))
        _draw_path(image, oracle_secondary_grid, (190, 70, 220))
        scale = max(1, int(round(0.40 / float(args.resolution_m))))
        image = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
    finally:
        grid.simulator.close()

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "run_comparison.json"
    image_path = output_dir / "run_comparison.png"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cv2.imwrite(str(image_path), image)
    print(f"[fork-analysis] report={report_path}")
    print(f"[fork-analysis] overlay={image_path}")
    print(
        "[fork-analysis] "
        f"extra_steps={comparison['extra_steps_for_oracle']} "
        f"critical_avoided={comparison['critical_violations_avoided']} "
        f"CHE_reduction={comparison['CHE_reduction']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
