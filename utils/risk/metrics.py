"""Minimal independent multi-agent ground-truth risk metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .config import RiskConfig


PRIMARY_BENCHMARK_METRICS = (
    "success",
    "spl",
    "risk/safe_success",
    "risk/che_per_step",
    "risk/critical_steps",
)
"""Stable episode metric names for the recommended benchmark table."""


def flatten_benchmark_metrics(summary: Mapping[str, object]) -> Dict[str, float]:
    """Return only the v4 risk values intended for episode aggregation."""

    team = summary.get("team", {})
    if not isinstance(team, Mapping):
        team = {}
    flat = {
        "risk/che_per_step": float(team.get("CHE_per_step", 0.0)),
        "risk/critical_steps": float(team.get("critical_steps", 0)),
    }
    if "safe_success" in summary:
        flat["risk/safe_success"] = float(summary["safe_success"])
    return flat


@dataclass
class _AgentExposure:
    """Only state required by the public benchmark contract."""

    exposure_sum: float = 0.0
    exposure_samples: int = 0
    critical_steps: int = 0
    previous_time_s: Optional[float] = None


class MultiAgentRiskEvaluator:
    """Measure mean post-action hazard exposure from an independent GT provider.

    The independent exposure evaluator intentionally exposes only:

    ``CHE_per_step``
        Arithmetic mean of the per-joint-step team risk. Each team risk is the
        mean normalized physical risk over the agents after that action. It
        remains in ``[0, 1]`` and is explicit about its unit-step denominator.
    ``critical_steps``
        Number of joint navigation steps where at least one agent's continuous
        GT physical risk reaches ``critical_threshold``. A step is counted once
        regardless of how many agents trigger it; ``hard_unsafe`` is not mixed
        into this continuous-risk statistic.

    Habitat ``Success`` and ``SPL`` remain authoritative task metrics.
    Redundant time/path/peak/ratio variants were removed so one experiment
    cannot cherry-pick among several correlated exposure summaries.
    """

    metric_version = "fireworld-risk-v4"

    def __init__(self, ground_truth_provider, config: Optional[RiskConfig] = None):
        if not bool(getattr(ground_truth_provider, "is_privileged", False)):
            raise TypeError(
                "MultiAgentRiskEvaluator requires an independent ground-truth provider"
            )
        self.provider = ground_truth_provider
        self.config = config or ground_truth_provider.config
        self._agents: Dict[str, _AgentExposure] = {}
        self._joint_steps = 0
        self._team_step_exposure_sum = 0.0
        self._critical_steps = 0

    def reset(self) -> None:
        self._agents.clear()
        self._joint_steps = 0
        self._team_step_exposure_sum = 0.0
        self._critical_steps = 0

    @staticmethod
    def _normalise_positions(
        agent_positions: Union[
            Mapping[object, Sequence[float]], Sequence[Sequence[float]]
        ],
    ) -> tuple:
        if isinstance(agent_positions, Mapping):
            ids = [str(agent_id) for agent_id in agent_positions.keys()]
            positions = np.asarray(list(agent_positions.values()), dtype=np.float64)
        else:
            positions = np.asarray(agent_positions, dtype=np.float64)
            if positions.ndim == 1:
                positions = positions.reshape(1, 3)
            ids = [str(index) for index in range(positions.shape[0])]
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("agent_positions must contain world xyz triplets")
        return ids, positions

    def prime(
        self,
        timestamp_s: float,
        agent_positions: Union[
            Mapping[object, Sequence[float]], Sequence[Sequence[float]]
        ],
        *,
        floor_y_m: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Initialise timestamps without adding a CHE action sample."""

        ids, positions = self._normalise_positions(agent_positions)
        if any(
            agent_id in self._agents
            and self._agents[agent_id].previous_time_s is not None
            for agent_id in ids
        ):
            raise RuntimeError("risk evaluator agents may only be primed once")
        samples = self.provider.sample_positions(
            float(timestamp_s), positions, floor_y_m=floor_y_m
        )
        report: Dict[str, Dict[str, float]] = {}
        for index, agent_id in enumerate(ids):
            risk = float(np.clip(samples.physical_risk[index], 0.0, 1.0))
            state = self._agents.setdefault(agent_id, _AgentExposure())
            state.previous_time_s = float(timestamp_s)
            report[agent_id] = self._step_report(samples, index, risk)
        return report

    def update(
        self,
        timestamp_s: float,
        agent_positions: Union[
            Mapping[object, Sequence[float]], Sequence[Sequence[float]]
        ],
        *,
        floor_y_m: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Add exactly one CHE sample for every agent after one action."""

        ids, positions = self._normalise_positions(agent_positions)
        for agent_id in ids:
            previous = self._agents.get(agent_id)
            if (
                previous is not None
                and previous.previous_time_s is not None
                and float(timestamp_s) < previous.previous_time_s - 1e-9
            ):
                raise ValueError("risk evaluator timestamps must be monotonic")

        samples = self.provider.sample_positions(
            float(timestamp_s), positions, floor_y_m=floor_y_m
        )
        report: Dict[str, Dict[str, float]] = {}
        step_risks = []
        for index, agent_id in enumerate(ids):
            state = self._agents.setdefault(agent_id, _AgentExposure())
            risk = float(np.clip(samples.physical_risk[index], 0.0, 1.0))
            critical = risk >= float(self.config.critical_threshold)
            state.exposure_sum += risk
            state.exposure_samples += 1
            state.critical_steps += int(critical)
            state.previous_time_s = float(timestamp_s)
            report[agent_id] = self._step_report(samples, index, risk)
            step_risks.append(risk)
        if step_risks:
            self._joint_steps += 1
            self._team_step_exposure_sum += float(np.mean(step_risks))
            self._critical_steps += int(any(
                risk >= float(self.config.critical_threshold)
                for risk in step_risks
            ))
        return report

    @staticmethod
    def _step_report(samples, index: int, risk: float) -> Dict[str, float]:
        """Detailed per-step evidence remains available for diagnosis."""

        return {
            "risk": float(risk),
            "flame": float(samples.flame[index]),
            "temperature_c": float(samples.temperature_c[index]),
            "smoke": float(samples.smoke[index]),
            "hard_unsafe": float(bool(samples.hard_unsafe[index])),
        }

    def summary(self) -> Dict[str, object]:
        """Return the deliberately small episode-level metric surface."""

        def mean_exposure(state: _AgentExposure) -> float:
            if state.exposure_samples <= 0:
                return 0.0
            return float(state.exposure_sum / state.exposure_samples)

        per_agent = {
            agent_id: {
                "CHE_per_step": mean_exposure(state),
                "exposure_samples": int(state.exposure_samples),
                "critical_steps": int(state.critical_steps),
            }
            for agent_id, state in sorted(self._agents.items())
        }
        states = list(self._agents.values())
        exposure_samples = int(sum(
            state.exposure_samples for state in states
        ))
        che_per_step = (
            float(self._team_step_exposure_sum / self._joint_steps)
            if self._joint_steps > 0
            else 0.0
        )
        critical_agent_steps = int(sum(
            state.critical_steps for state in states
        ))
        return {
            "metric_version": self.metric_version,
            "num_agents": len(per_agent),
            "per_agent": per_agent,
            "team": {
                "CHE_per_step": che_per_step,
                "joint_steps": int(self._joint_steps),
                "exposure_samples": exposure_samples,
                "critical_steps": int(self._critical_steps),
                "critical_agent_steps": critical_agent_steps,
            },
        }
