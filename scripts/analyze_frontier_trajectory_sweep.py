#!/usr/bin/env python3
"""Rank dense two-agent paths from the fixed-plan frontier strategy sweep."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCREEN_ROOT = Path(
    "outputs/fire_cost_experiments/frontier_trajectory_strategy_sweep"
)
SCENARIOS = {
    "bed": {
        "plan": "scenes/Nfvxx8J5NCo/plans/"
        "Nfvxx8J5NCo_route_contrast_stable_50d97ff18bd9.json",
        "baseline": "outputs/fire_cost_experiments/fork_detour_nfv_three_source/"
        "risk_none/risk/fork-detour-three-source-none-images/rank_000/ep_0000/"
        "action_list.json",
        "co_ut_actions": "outputs/fire_cost_experiments/fork_detour_nfv_three_source/"
        "oracle/risk/fork-detour-three-source-oracle-alpha4-images/rank_000/"
        "ep_0000/action_list.json",
        "co_ut_risk": "outputs/fire_cost_experiments/fork_detour_nfv_three_source/"
        "oracle/risk/fork-detour-three-source-oracle-alpha4-images/rank_000/"
        "ep_0000/risk_summary.json",
        "co_ut_metrics": "outputs/fire_cost_experiments/fork_detour_nfv_three_source/"
        "oracle/navigation/metrics/resume_state.json",
    },
    "person": {
        "plan": "scenes/Nfvxx8J5NCo/plans/"
        "Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab.json",
        "baseline": "outputs/fire_cost_experiments/person_fork_detour_nfv_three_source/"
        "risk_none/risk/person-fork-three-source-none-images/rank_000/ep_0000/"
        "action_list.json",
        "co_ut_actions": "outputs/fire_cost_experiments/"
        "person_fork_detour_nfv_three_source/oracle/risk/"
        "person-fork-three-source-oracle-alpha4-images/rank_000/ep_0000/"
        "action_list.json",
        "co_ut_risk": "outputs/fire_cost_experiments/"
        "person_fork_detour_nfv_three_source/oracle/risk/"
        "person-fork-three-source-oracle-alpha4-images/rank_000/ep_0000/"
        "risk_summary.json",
        "co_ut_metrics": "outputs/fire_cost_experiments/"
        "person_fork_detour_nfv_three_source/oracle/navigation/metrics/"
        "resume_state.json",
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(SCREEN_ROOT))
    parser.add_argument("--output-dir", default=None)
    return parser


def _read(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def action_paths(payload: Mapping[str, object]) -> Sequence[np.ndarray]:
    paths = []
    for agent_id in range(int(payload["num_agents"])):
        points = []
        for step in payload["steps"]:
            action = next(
                item
                for item in step["actions"]
                if int(item["agent_id"]) == agent_id
            )
            points.append(action["position_after"])
        paths.append(np.asarray(points, dtype=np.float64))
    return paths


def _path_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(
        np.diff(path[:, [0, 2]], axis=0), axis=1
    ).sum())


def _moving_path(path: np.ndarray, epsilon_m: float = 0.02) -> np.ndarray:
    if len(path) == 0:
        return path.copy()
    keep = [0]
    for index in range(1, len(path)):
        if np.linalg.norm(
            path[index, [0, 2]] - path[keep[-1], [0, 2]]
        ) > float(epsilon_m):
            keep.append(index)
    return path[np.asarray(keep, dtype=np.int64)]


def _cell_sequence(path: np.ndarray, resolution_m: float = 0.25):
    cells = np.rint(path[:, [0, 2]] / float(resolution_m)).astype(np.int64)
    return [tuple(map(int, cell)) for cell in cells]


def path_geometry(path: np.ndarray) -> Dict[str, float]:
    moving = _moving_path(path)
    length = _path_length(moving)
    step_lengths = (
        np.linalg.norm(np.diff(moving[:, [0, 2]], axis=0), axis=1)
        if len(moving) > 1 else np.empty(0, dtype=np.float64)
    )
    if len(step_lengths):
        cumulative = np.cumsum(step_lengths)
        initial_segment_count = min(
            int(np.searchsorted(cumulative, 2.0, side="left")) + 1,
            len(step_lengths),
        )
        initial_path = moving[:initial_segment_count + 1]
        initial_length = float(step_lengths[:initial_segment_count].sum())
        initial_displacement = float(np.linalg.norm(
            initial_path[-1, [0, 2]] - initial_path[0, [0, 2]]
        ))
        initial_progress_ratio = initial_displacement / max(
            initial_length, 1e-9
        )
    else:
        initial_progress_ratio = 0.0
    displacement = (
        float(np.linalg.norm(moving[-1, [0, 2]] - moving[0, [0, 2]]))
        if len(moving) else 0.0
    )
    extent = np.ptp(moving[:, [0, 2]], axis=0) if len(moving) else np.zeros(2)
    cells = _cell_sequence(moving)
    unique_cells = set(cells)
    headings = (
        np.arctan2(
            np.diff(moving[:, 0]),
            np.diff(moving[:, 2]),
        )
        if len(moving) > 1 else np.empty(0)
    )
    if len(headings) > 1:
        turns = np.diff(headings)
        turns = (turns + math.pi) % (2.0 * math.pi) - math.pi
        accumulated_turn = float(np.abs(turns).sum())
    else:
        accumulated_turn = 0.0
    return {
        "path_length_m": length,
        "displacement_m": displacement,
        "straightness": displacement / max(length, 1e-9),
        "spatial_span_m": float(np.linalg.norm(extent)),
        "bbox_area_m2": float(extent[0] * extent[1]),
        "moving_pose_count": int(len(moving)),
        "unique_0_25m_cell_count": int(len(unique_cells)),
        "self_revisit_fraction": float(
            1.0 - len(unique_cells) / max(1, len(cells))
        ),
        "accumulated_turn_rad": accumulated_turn,
        "turn_rad_per_m": accumulated_turn / max(length, 1e-9),
        "initial_2m_progress_ratio": initial_progress_ratio,
    }


def _near_fraction(source: set, target: set, radius_cells: int) -> float:
    if not source:
        return 0.0
    radius2 = int(radius_cells) ** 2
    return float(sum(
        any(
            (row - other_row) ** 2 + (col - other_col) ** 2 <= radius2
            for other_row, other_col in target
        )
        for row, col in source
    ) / len(source))


def team_geometry(paths: Sequence[np.ndarray]) -> Dict[str, float]:
    first = set(_cell_sequence(_moving_path(paths[0])))
    second = set(_cell_sequence(_moving_path(paths[1])))
    shared = first & second
    union = first | second
    pairwise = np.linalg.norm(
        paths[0][:, None, [0, 2]] - paths[1][None, :, [0, 2]],
        axis=2,
    )
    return {
        "shared_0_25m_cells": int(len(shared)),
        "cell_jaccard": float(len(shared) / max(1, len(union))),
        "agent_1_near_agent_0_fraction_0_5m": _near_fraction(
            second, first, 2
        ),
        "minimum_pairwise_distance_m": float(pairwise.min()),
    }


def route_divergence(path: np.ndarray, baseline: np.ndarray) -> float:
    candidate_cells = set(_cell_sequence(_moving_path(path)))
    baseline_cells = set(_cell_sequence(_moving_path(baseline)))
    return float(
        1.0
        - len(candidate_cells & baseline_cells)
        / max(1, len(candidate_cells | baseline_cells))
    )


def _risk_by_agent(actions: Mapping[str, object], agent_id: int) -> np.ndarray:
    values = []
    for step in actions["steps"]:
        action = next(
            item for item in step["actions"]
            if int(item["agent_id"]) == agent_id
        )
        values.append(float(action.get("risk_after", 0.0)))
    return np.asarray(values, dtype=np.float64)


def summarize_run(
    *,
    target: str,
    strategy: str,
    actions_path: Path,
    risk_path: Path,
    metrics_path: Path,
    baseline_path: Path,
) -> Dict[str, object]:
    actions = _read(actions_path)
    risk = _read(risk_path)
    metrics_payload = _read(metrics_path)
    metrics = metrics_payload.get("metrics", metrics_payload)
    paths = action_paths(actions)
    baseline = action_paths(_read(baseline_path))[0]
    agents = {
        str(agent_id): path_geometry(path)
        for agent_id, path in enumerate(paths)
    }
    for agent_id in range(len(paths)):
        agent_risk = _risk_by_agent(actions, agent_id)
        agents[str(agent_id)]["maximum_risk"] = float(
            agent_risk.max(initial=0.0)
        )
    primary_length_ratio = (
        agents["0"]["path_length_m"] / max(_path_length(baseline), 1e-9)
    )
    team_risk = risk.get("team", {})
    safe_success = float(risk.get(
        "safe_success", metrics.get("risk/safe_success", 0.0)
    ))
    critical = int(team_risk.get("critical_violations", 0))
    success = float(metrics.get("success", 0.0))
    steps = int(round(float(metrics.get("num_steps", actions["num_steps"]))))
    divergence = route_divergence(paths[0], baseline)
    detour_evidence = bool(
        success >= 1.0
        and safe_success >= 1.0
        and critical == 0
        and divergence >= 0.50
        and primary_length_ratio >= 1.05
    )
    team = team_geometry(paths)
    spatially_relaxed = bool(
        agents["1"]["spatial_span_m"] >= 3.0
        and agents["1"]["self_revisit_fraction"] <= 0.35
        and team["agent_1_near_agent_0_fraction_0_5m"] <= 0.15
    )
    return {
        "target": target,
        "strategy": strategy,
        "actions_path": str(actions_path),
        "risk_summary_path": str(risk_path),
        "metrics_path": str(metrics_path),
        "task": {
            "success": success,
            "safe_success": safe_success,
            "critical_violations": critical,
            "num_steps": steps,
            "spl": float(metrics.get("spl", 0.0)),
            "CHE": float(team_risk.get("CHE", metrics.get("risk/che", 0.0))),
        },
        "primary_detour": {
            "route_divergence_from_risk_none": divergence,
            "path_length_ratio_to_risk_none": float(primary_length_ratio),
            "evidence_passed": detour_evidence,
        },
        "agents": agents,
        "team_geometry": team,
        "spatially_relaxed": spatially_relaxed,
        "eligible": bool(detour_evidence and steps <= 250),
    }


def _normalize(values, *, higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    span = float(array.max() - array.min())
    if span <= 1e-12:
        return np.ones_like(array)
    result = (array - array.min()) / span
    return result if higher_is_better else 1.0 - result


def rank_runs(runs: Sequence[Dict[str, object]]) -> Sequence[Dict[str, object]]:
    grouped: Dict[str, list] = {}
    for run in runs:
        grouped.setdefault(str(run["target"]), []).append(run)
    for target_runs in grouped.values():
        components = {
            "secondary_span": _normalize([
                run["agents"]["1"]["spatial_span_m"] for run in target_runs
            ], higher_is_better=True),
            "secondary_straightness": _normalize([
                run["agents"]["1"]["straightness"] for run in target_runs
            ], higher_is_better=True),
            "secondary_revisit": _normalize([
                run["agents"]["1"]["self_revisit_fraction"]
                for run in target_runs
            ], higher_is_better=False),
            "secondary_turning": _normalize([
                run["agents"]["1"]["turn_rad_per_m"] for run in target_runs
            ], higher_is_better=False),
            "team_near_overlap": _normalize([
                run["team_geometry"]["agent_1_near_agent_0_fraction_0_5m"]
                for run in target_runs
            ], higher_is_better=False),
        }
        for index, run in enumerate(target_runs):
            run["trajectory_score"] = float(
                0.30 * components["secondary_span"][index]
                + 0.20 * components["secondary_straightness"][index]
                + 0.15 * components["secondary_revisit"][index]
                + 0.10 * components["secondary_turning"][index]
                + 0.25 * components["team_near_overlap"][index]
            ) if run["eligible"] else 0.0
        ordered = sorted(
            target_runs,
            key=lambda run: (
                bool(run["eligible"]),
                bool(run["spatially_relaxed"]),
                float(run["trajectory_score"]),
            ),
            reverse=True,
        )
        for rank, run in enumerate(ordered, start=1):
            run["rank_within_target"] = rank
    return sorted(runs, key=lambda run: (
        str(run["target"]), int(run["rank_within_target"])
    ))


def _screen_paths(
    root: Path,
    target: str,
    strategy: str,
    *,
    run_label: Optional[str] = None,
):
    label = run_label or f"{target}__{strategy}"
    case_root = root / "runs" / label
    run_id = f"trajectory-sweep-{label}"
    episode_root = case_root / "risk" / run_id / "rank_000" / "ep_0000"
    return (
        episode_root / "action_list.json",
        episode_root / "risk_summary.json",
        case_root / "navigation" / "metrics" / "resume_state.json",
    )


def discover_runs(root: Path):
    runs = []
    missing = []
    for target, scenario in SCENARIOS.items():
        baseline = ROOT / scenario["baseline"]
        references = [("co_ut", ROOT / scenario["co_ut_actions"],
                       ROOT / scenario["co_ut_risk"],
                       ROOT / scenario["co_ut_metrics"])]
        references.extend(
            (strategy, *_screen_paths(root, target, strategy))
            for strategy in ("nearest", "fill", "random")
        )
        if target == "bed":
            references.append((
                "random_ep4",
                *_screen_paths(root, "bed_ep4", "random"),
            ))
            for distance in (4, 6):
                references.append((
                    f"random_ep4_d{distance}",
                    *_screen_paths(
                        root,
                        "bed_ep4",
                        "random",
                        run_label=f"bed_ep4__random_d{distance}",
                    ),
                ))
        else:
            for distance in (4, 6):
                references.append((
                    f"random_d{distance}",
                    *_screen_paths(
                        root,
                        "person",
                        "random",
                        run_label=f"person__random_d{distance}",
                    ),
                ))
        for strategy, actions, risk, metrics in references:
            paths = (actions, risk, metrics, baseline)
            if not all(path.is_file() for path in paths):
                missing.append({
                    "target": target,
                    "strategy": strategy,
                    "missing": [str(path) for path in paths if not path.is_file()],
                })
                continue
            runs.append(summarize_run(
                target=target,
                strategy=strategy,
                actions_path=actions,
                risk_path=risk,
                metrics_path=metrics,
                baseline_path=baseline,
            ))
    return rank_runs(runs), missing


def save_plot(runs, output_path: Path) -> None:
    colors = {
        "nearest": "#1b9e77",
        "co_ut": "#d95f02",
        "fill": "#7570b3",
        "random": "#1f78b4",
        "random_ep4": "#e7298a",
        "random_ep4_d4": "#66a61e",
        "random_ep4_d6": "#e6ab02",
        "random_d4": "#66a61e",
        "random_d6": "#e6ab02",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), squeeze=False)
    for row, target in enumerate(("bed", "person")):
        plan = _read(ROOT / SCENARIOS[target]["plan"])
        ignitions = np.asarray(
            [item["position"] for item in plan["ignitions"]], dtype=np.float64
        )
        for agent_id in range(2):
            axis = axes[row, agent_id]
            for run in [item for item in runs if item["target"] == target]:
                actions = _read(Path(run["actions_path"]))
                path = action_paths(actions)[agent_id]
                strategy = str(run["strategy"])
                axis.plot(
                    path[:, 0], path[:, 2], linewidth=2.0,
                    color=colors.get(strategy, "black"),
                    label=(
                        f"{strategy}: span={run['agents'][str(agent_id)]['spatial_span_m']:.1f}m, "
                        f"revisit={run['agents'][str(agent_id)]['self_revisit_fraction']:.2f}"
                    ),
                )
            axis.scatter(
                ignitions[:, 0], ignitions[:, 2], marker="X", s=70,
                color="#e31a1c", edgecolors="#7f0000", label="ignitions",
            )
            axis.set_title(f"{target.title()} Agent {agent_id}")
            axis.set_xlabel("Habitat world x (m) / image column")
            axis.set_ylabel("Habitat world z (m) / image row")
            axis.set_aspect("equal", adjustable="datalim")
            axis.invert_yaxis()
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="best")
    figure.suptitle("Dense native trajectories by frontier assignment strategy")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    root = (ROOT / args.root).resolve()
    output_dir = (
        (ROOT / args.output_dir).resolve()
        if args.output_dir else root / "reports"
    )
    runs, missing = discover_runs(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "selection_contract": {
            "dense_native_action_lists": True,
            "maximum_steps": 250,
            "safe_detour_required": True,
            "secondary_minimum_spatial_span_m": 3.0,
            "secondary_maximum_self_revisit_fraction": 0.35,
            "maximum_agent_1_near_agent_0_fraction_0_5m": 0.15,
        },
        "runs": runs,
        "missing_runs": missing,
    }
    report_path = output_dir / "strategy_ranking.json"
    plot_path = output_dir / "strategy_trajectories.png"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_plot(runs, plot_path)
    print(f"[trajectory-analysis] report={report_path}")
    print(f"[trajectory-analysis] plot={plot_path}")
    for target in ("bed", "person"):
        ranked = [run for run in runs if run["target"] == target]
        if ranked:
            best = ranked[0]
            print(
                f"[trajectory-analysis] target={target} best={best['strategy']} "
                f"eligible={best['eligible']} relaxed={best['spatially_relaxed']} "
                f"score={best['trajectory_score']:.3f}"
            )
    if missing:
        print(f"[trajectory-analysis] missing_runs={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
