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
    "risk/che",
)
"""Stable episode metric names for the recommended benchmark table."""


@dataclass
class _AgentExposure:
    """Only state required by the public benchmark contract."""

    che: float = 0.0
    critical_violations: int = 0
    previous_time_s: Optional[float] = None


class MultiAgentRiskEvaluator:
    """Accumulate VULCAN-style CHE from an independent GT provider.

    The public risk benchmark intentionally exposes only:

    ``CHE``
        Discrete cumulative hazard exposure from VULCAN Eq. (14), summed once
        per executed action and agent.
    ``critical_violations``
        Diagnostic count used to derive ``SafeSuccess``. It is retained in
        ``risk_summary.json`` but is not promoted as a primary benchmark
        column.

    Habitat ``Success`` and ``SPL`` remain authoritative task metrics.
    Redundant time/path/peak/ratio variants were removed so one experiment
    cannot cherry-pick among several correlated exposure summaries.
    """

    metric_version = "fireworld-risk-v2"

    def __init__(self, ground_truth_provider, config: Optional[RiskConfig] = None):
        if not bool(getattr(ground_truth_provider, "is_privileged", False)):
            raise TypeError(
                "MultiAgentRiskEvaluator requires an independent ground-truth provider"
            )
        self.provider = ground_truth_provider
        self.config = config or ground_truth_provider.config
        self._agents: Dict[str, _AgentExposure] = {}

    def reset(self) -> None:
        self._agents.clear()

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
        for index, agent_id in enumerate(ids):
            state = self._agents.setdefault(agent_id, _AgentExposure())
            risk = float(np.clip(samples.physical_risk[index], 0.0, 1.0))
            hard_unsafe = bool(samples.hard_unsafe[index])
            critical = (
                risk >= float(self.config.critical_threshold) or hard_unsafe
            )
            state.che += risk
            state.critical_violations += int(critical)
            state.previous_time_s = float(timestamp_s)
            report[agent_id] = self._step_report(samples, index, risk)
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

        per_agent = {
            agent_id: {
                "CHE": float(state.che),
                "critical_violations": int(state.critical_violations),
            }
            for agent_id, state in sorted(self._agents.items())
        }
        states = list(self._agents.values())
        return {
            "metric_version": self.metric_version,
            "num_agents": len(per_agent),
            "per_agent": per_agent,
            "team": {
                "CHE": float(sum(state.che for state in states)),
                "critical_violations": int(sum(
                    state.critical_violations for state in states
                )),
            },
        }
