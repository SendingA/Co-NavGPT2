#!/usr/bin/env python3
"""Compare the selected dense trajectories before and after moving agent 1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_frontier_trajectory_sweep import (
    action_paths,
    path_geometry,
    route_divergence,
    team_geometry,
)


DEFAULT_SELECTED_ROOT = Path(
    "outputs/fire_cost_experiments/frontier_trajectory_open_start/selected"
)
CASES = {
    "bed": {
        "old_actions": (
            "outputs/fire_cost_experiments/frontier_trajectory_strategy_sweep/"
            "selected/runs/bed_ep4__random/risk/trajectory-sweep-bed_ep4__random/"
            "rank_000/ep_0000/action_list.json"
        ),
        "new_label": "bed_open_egress__random",
        "risk_none_actions": (
            "outputs/fire_cost_experiments/fork_detour_nfv_three_source/"
            "risk_none/risk/fork-detour-three-source-none-images/"
            "rank_000/ep_0000/action_list.json"
        ),
        "geometry": (
            "outputs/fire_cost_experiments/frontier_trajectory_open_start/"
            "bed_open_egress/geometry/scenario_geometry.json"
        ),
    },
    "person": {
        "old_actions": (
            "outputs/fire_cost_experiments/frontier_trajectory_open_start/"
            "selected/runs/person_open_forward__random/risk/"
            "trajectory-sweep-person_open_forward__random/"
            "rank_000/ep_0000/action_list.json"
        ),
        "new_label": "person_north_yawm90__fill",
        "risk_none_actions": (
            "outputs/fire_cost_experiments/person_fork_detour_nfv_three_source/"
            "risk_none/risk/person-fork-three-source-none-images/"
            "rank_000/ep_0000/action_list.json"
        ),
        "geometry": (
            "outputs/fire_cost_experiments/frontier_trajectory_open_start/"
            "person_north_yawm90/geometry/scenario_geometry.json"
        ),
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-root", default=str(DEFAULT_SELECTED_ROOT))
    parser.add_argument("--output-dir", default=None)
    return parser


def _read(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_paths(selected_root: Path, label: str):
    case_root = selected_root / "runs" / label
    episode = (
        case_root / "risk" / f"trajectory-sweep-{label}" /
        "rank_000" / "ep_0000"
    )
    return (
        episode / "action_list.json",
        episode / "risk_summary.json",
        case_root / "navigation" / "metrics" / "resume_state.json",
    )


def _summarize(actions_path: Path) -> Mapping[str, object]:
    actions = _read(actions_path)
    paths = action_paths(actions)
    return {
        "actions_path": str(actions_path),
        "num_steps": int(actions["num_steps"]),
        "start_positions": [path[0].tolist() for path in paths],
        "end_positions": [path[-1].tolist() for path in paths],
        "agents": {
            str(agent_id): path_geometry(path)
            for agent_id, path in enumerate(paths)
        },
        "team_geometry": team_geometry(paths),
    }


def _task_summary(metrics_path: Path, risk_path: Path) -> Mapping[str, object]:
    metrics_payload = _read(metrics_path)
    metrics = metrics_payload.get("metrics", metrics_payload)
    risk = _read(risk_path)
    team_risk = risk.get("team", {})
    return {
        "success": float(metrics.get("success", 0.0)),
        "safe_success": float(
            risk.get("safe_success", metrics.get("risk/safe_success", 0.0))
        ),
        "critical_steps": int(team_risk.get(
            "critical_steps", team_risk.get("critical_violations", 0)
        )),
        "spl": float(metrics.get("spl", 0.0)),
        "num_steps": int(round(float(metrics.get("num_steps", 0.0)))),
        "CHE_per_step": float(team_risk.get(
            "CHE_per_step",
            team_risk.get(
                "CHE",
                metrics.get("risk/che_per_step", metrics.get("risk/che", 0.0)),
            ),
        )),
    }


def _detour_summary(
    actions_path: Path,
    risk_none_actions_path: Path,
) -> Mapping[str, float]:
    primary = action_paths(_read(actions_path))[0]
    baseline = action_paths(_read(risk_none_actions_path))[0]
    primary_length = float(path_geometry(primary)["path_length_m"])
    baseline_length = float(path_geometry(baseline)["path_length_m"])
    return {
        "route_divergence_from_risk_none": route_divergence(
            primary, baseline
        ),
        "path_length_ratio_to_risk_none": (
            primary_length / max(baseline_length, 1e-9)
        ),
    }


def _delta(old: Mapping[str, object], new: Mapping[str, object]):
    old_agent = old["agents"]["1"]
    new_agent = new["agents"]["1"]
    old_team = old["team_geometry"]
    new_team = new["team_geometry"]
    return {
        "agent_1_spatial_span_m": (
            new_agent["spatial_span_m"] - old_agent["spatial_span_m"]
        ),
        "agent_1_displacement_m": (
            new_agent["displacement_m"] - old_agent["displacement_m"]
        ),
        "agent_1_straightness": (
            new_agent["straightness"] - old_agent["straightness"]
        ),
        "agent_1_self_revisit_fraction": (
            new_agent["self_revisit_fraction"] -
            old_agent["self_revisit_fraction"]
        ),
        "agent_1_turn_rad_per_m": (
            new_agent["turn_rad_per_m"] - old_agent["turn_rad_per_m"]
        ),
        "agent_1_initial_2m_progress_ratio": (
            new_agent["initial_2m_progress_ratio"] -
            old_agent["initial_2m_progress_ratio"]
        ),
        "agent_1_near_agent_0_fraction_0_5m": (
            new_team["agent_1_near_agent_0_fraction_0_5m"] -
            old_team["agent_1_near_agent_0_fraction_0_5m"]
        ),
    }


def _save_plot(report: Mapping[str, object], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)
    for column, target in enumerate(("bed", "person")):
        axis = axes[0, column]
        case = report["cases"][target]
        old_paths = action_paths(_read(Path(case["old"]["actions_path"])))
        new_paths = action_paths(_read(Path(case["new"]["actions_path"])))
        geometry = _read(ROOT / CASES[target]["geometry"])
        ignitions = np.asarray(
            [item["position"] for item in geometry["ignition_sources"]],
            dtype=np.float64,
        )
        axis.plot(
            old_paths[1][:, 0], old_paths[1][:, 2],
            linestyle="--", linewidth=1.7, color="#91bfdb",
            label="Agent 1 old start",
        )
        axis.plot(
            new_paths[0][:, 0], new_paths[0][:, 2],
            linewidth=2.0, color="#ef6548", label="Agent 0 moved-start run",
        )
        axis.plot(
            new_paths[1][:, 0], new_paths[1][:, 2],
            linewidth=2.4, color="#0571b0", label="Agent 1 separated start",
        )
        axis.scatter(
            ignitions[:, 0], ignitions[:, 2], marker="X", s=80,
            color="#fdae61", edgecolors="#a50026", label="ignitions",
        )
        new_agent = case["new"]["agents"]["1"]
        axis.set_title(
            f"{target.title()}: Agent 1 span {new_agent['spatial_span_m']:.2f} m, "
            f"initial progress {new_agent['initial_2m_progress_ratio']:.1%}"
        )
        axis.set_xlabel("Habitat world x (m)")
        axis.set_ylabel("Habitat world z (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.invert_yaxis()
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Dense executed trajectories before and after moving Agent 1")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    selected_root = (ROOT / args.selected_root).resolve()
    output_dir = (
        (ROOT / args.output_dir).resolve()
        if args.output_dir else selected_root / "reports"
    )
    report = {
        "schema_version": 1,
        "coordinate_convention": (
            "Habitat world x/z; plots invert z only for top-down display"
        ),
        "selection_contract": {
            "dense_native_action_lists_preserved": True,
            "safe_success_required": True,
            "critical_steps_required": 0,
            "agent_1_near_agent_0_fraction_0_5m_max": 0.15,
            "agent_1_spatial_span_m_min": 4.0,
            "agent_1_straightness_min": 0.65,
            "agent_1_self_revisit_fraction_max": 0.15,
            "agent_1_initial_2m_progress_ratio_min": 0.50,
        },
        "cases": {},
    }
    for target, spec in CASES.items():
        actions_path, risk_path, metrics_path = _new_paths(
            selected_root, str(spec["new_label"])
        )
        required = [
            ROOT / spec["old_actions"], actions_path, risk_path,
            metrics_path, ROOT / spec["geometry"],
            ROOT / spec["risk_none_actions"],
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{target} missing outputs: {missing}")
        old = _summarize(ROOT / spec["old_actions"])
        new = dict(_summarize(actions_path))
        new["task"] = _task_summary(metrics_path, risk_path)
        new["primary_detour"] = _detour_summary(
            actions_path, ROOT / spec["risk_none_actions"]
        )
        geometry = _read(ROOT / spec["geometry"])
        new["start_selection"] = {
            key: geometry["secondary"][key]
            for key in (
                "navmesh_clearance_m",
                "primary_route_clearance_m",
                "start_separation_m",
                "ignition_clearance_m",
            )
        }
        report["cases"][target] = {
            "old": old,
            "new": new,
            "new_minus_old": _delta(old, new),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "open_start_comparison.json"
    plot_path = output_dir / "open_start_trajectory_comparison.png"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_plot(report, plot_path)
    print(f"[open-start-analysis] report={report_path}")
    print(f"[open-start-analysis] plot={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
