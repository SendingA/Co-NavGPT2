"""Focused tests for independent multi-agent exposure metrics."""
from __future__ import annotations

import unittest

import numpy as np

from utils.risk.config import RiskConfig
from utils.risk.metrics import (
    MultiAgentRiskEvaluator,
    flatten_benchmark_metrics,
)
from utils.risk.model import RiskPointSamples


class _ScheduledGroundTruthProvider:
    is_privileged = True

    def __init__(self, config: RiskConfig):
        self.config = config
        self.schedule = {
            0: np.array([0.2, 0.1], dtype=np.float32),
            1: np.array([0.8, 0.1], dtype=np.float32),
            2: np.array([0.9, 0.95], dtype=np.float32),
        }

    def sample_positions(self, timestamp_s, positions, floor_y_m=None):
        del floor_y_m
        risks = self.schedule[int(timestamp_s)][:len(positions)]
        count = len(risks)
        zeros = np.zeros(count, dtype=np.float32)
        return RiskPointSamples(
            flame=zeros,
            temperature_c=np.full(count, 25.0, dtype=np.float32),
            temperature=zeros,
            smoke=zeros,
            physical_risk=risks,
            hard_unsafe=np.zeros(count, dtype=bool),
            confidence=np.ones(count, dtype=np.float32),
        )


class RiskMetricTests(unittest.TestCase):
    def test_independent_multi_agent_exposure_summary(self) -> None:
        config = RiskConfig(danger_threshold=0.6, critical_threshold=0.85)
        evaluator = MultiAgentRiskEvaluator(
            _ScheduledGroundTruthProvider(config), config
        )
        evaluator.update(0.0, [[0, 0, 0], [0, 0, 1]])
        evaluator.update(1.0, [[1, 0, 0], [1, 0, 1]])
        evaluator.update(2.0, [[2, 0, 0], [2, 0, 1]])

        report = evaluator.summary()
        team = report["team"]
        self.assertEqual(report["metric_version"], "fireworld-risk-v4")
        self.assertEqual(report["num_agents"], 2)
        self.assertEqual(
            set(team),
            {
                "CHE_per_step",
                "joint_steps",
                "exposure_samples",
                "critical_steps",
                "critical_agent_steps",
            },
        )
        self.assertEqual(team["joint_steps"], 3)
        self.assertEqual(team["exposure_samples"], 6)
        self.assertAlmostEqual(team["CHE_per_step"], 3.05 / 6.0, places=6)
        self.assertEqual(team["critical_steps"], 1)
        self.assertEqual(team["critical_agent_steps"], 2)
        self.assertAlmostEqual(
            report["per_agent"]["0"]["CHE_per_step"], 1.9 / 3.0, places=6
        )
        self.assertEqual(report["per_agent"]["0"]["exposure_samples"], 3)

    def test_evaluator_rejects_non_ground_truth_provider(self) -> None:
        with self.assertRaisesRegex(TypeError, "ground-truth"):
            MultiAgentRiskEvaluator(object(), RiskConfig())

    def test_prime_does_not_count_reset_as_che_sample(self) -> None:
        config = RiskConfig(danger_threshold=0.6, critical_threshold=0.85)
        evaluator = MultiAgentRiskEvaluator(
            _ScheduledGroundTruthProvider(config), config
        )
        evaluator.prime(0.0, [[0, 0, 0], [0, 0, 1]])
        primed_team = evaluator.summary()["team"]
        self.assertEqual(primed_team["CHE_per_step"], 0.0)
        self.assertEqual(primed_team["joint_steps"], 0)
        self.assertEqual(primed_team["exposure_samples"], 0)
        evaluator.update(1.0, [[1, 0, 0], [1, 0, 1]])
        evaluator.update(2.0, [[2, 0, 0], [2, 0, 1]])
        team = evaluator.summary()["team"]
        self.assertEqual(team["exposure_samples"], 4)
        self.assertEqual(team["joint_steps"], 2)
        self.assertAlmostEqual(team["CHE_per_step"], 2.75 / 4.0, places=6)
        with self.assertRaisesRegex(RuntimeError, "primed once"):
            evaluator.prime(3.0, [[3, 0, 0], [3, 0, 1]])

    def test_mapping_agent_ids_are_preserved(self) -> None:
        config = RiskConfig()
        evaluator = MultiAgentRiskEvaluator(
            _ScheduledGroundTruthProvider(config), config
        )
        step = evaluator.update(
            0.0,
            {"robot-red": [0, 0, 0], "robot-blue": [0, 0, 1]},
        )
        self.assertEqual(set(step), {"robot-red", "robot-blue"})
        self.assertEqual(set(evaluator.summary()["per_agent"]), set(step))

    def test_out_of_order_timestamp_does_not_mutate_summary(self) -> None:
        config = RiskConfig()
        evaluator = MultiAgentRiskEvaluator(
            _ScheduledGroundTruthProvider(config), config
        )
        evaluator.update(1.0, [[0, 0, 0], [0, 0, 1]])
        before = evaluator.summary()
        with self.assertRaisesRegex(ValueError, "monotonic"):
            evaluator.update(0.0, [[0, 0, 0], [0, 0, 1]])
        self.assertEqual(evaluator.summary(), before)

    def test_summary_does_not_publish_redundant_exposure_variants(self) -> None:
        evaluator = MultiAgentRiskEvaluator(
            _ScheduledGroundTruthProvider(RiskConfig()), RiskConfig()
        )
        evaluator.update(0.0, [[0, 0, 0], [0, 0, 1]])
        serialized = repr(evaluator.summary())
        for removed in (
            "CHE_raw", "CHE_time", "CHE_mean", "path_risk",
            "peak_risk", "danger_time_ratio", "sample_count",
        ):
            self.assertNotIn(removed, serialized)

    def test_hard_unsafe_does_not_count_as_continuous_critical_step(self) -> None:
        class HardUnsafeOnlyProvider(_ScheduledGroundTruthProvider):
            def sample_positions(self, timestamp_s, positions, floor_y_m=None):
                samples = super().sample_positions(
                    timestamp_s, positions, floor_y_m=floor_y_m
                )
                return RiskPointSamples(
                    flame=np.ones_like(samples.flame),
                    temperature_c=samples.temperature_c,
                    temperature=samples.temperature,
                    smoke=samples.smoke,
                    physical_risk=np.full_like(samples.physical_risk, 0.2),
                    hard_unsafe=np.ones_like(samples.hard_unsafe),
                    confidence=samples.confidence,
                )

        config = RiskConfig(critical_threshold=0.8)
        evaluator = MultiAgentRiskEvaluator(HardUnsafeOnlyProvider(config), config)
        evaluator.update(0.0, [[0, 0, 0], [0, 0, 1]])
        team = evaluator.summary()["team"]
        self.assertEqual(team["critical_steps"], 0)
        self.assertEqual(team["critical_agent_steps"], 0)

    def test_flattened_v4_metrics_omit_early_stop(self) -> None:
        flattened = flatten_benchmark_metrics({
            "team": {
                "CHE_per_step": 0.25,
                "critical_steps": 3,
                "early_stop": 1,
            },
            "safe_success": 0.75,
            "early_stop": {"triggered": True},
        })
        self.assertEqual(flattened, {
            "risk/che_per_step": 0.25,
            "risk/critical_steps": 3.0,
            "risk/safe_success": 0.75,
        })


if __name__ == "__main__":
    unittest.main()
