"""Focused tests for independent multi-agent exposure metrics."""
from __future__ import annotations

import unittest

import numpy as np

from utils.risk.config import RiskConfig
from utils.risk.metrics import MultiAgentRiskEvaluator
from utils.risk.model import RiskPointSamples


class _ScheduledGroundTruthProvider:
    is_privileged = True

    def __init__(self, config: RiskConfig):
        self.config = config
        self.schedule = {
            0: np.array([0.2, 0.1], dtype=np.float32),
            1: np.array([0.8, 0.1], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
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
        self.assertEqual(report["metric_version"], "fireworld-risk-v2")
        self.assertEqual(report["num_agents"], 2)
        self.assertEqual(set(team), {"CHE", "critical_violations"})
        self.assertAlmostEqual(team["CHE"], 2.2, places=6)
        self.assertEqual(team["critical_violations"], 1)
        self.assertAlmostEqual(
            report["per_agent"]["0"]["CHE"], 1.9, places=6
        )

    def test_evaluator_rejects_non_ground_truth_provider(self) -> None:
        with self.assertRaisesRegex(TypeError, "ground-truth"):
            MultiAgentRiskEvaluator(object(), RiskConfig())

    def test_prime_does_not_count_reset_as_che_sample(self) -> None:
        config = RiskConfig(danger_threshold=0.6, critical_threshold=0.85)
        evaluator = MultiAgentRiskEvaluator(
            _ScheduledGroundTruthProvider(config), config
        )
        evaluator.prime(0.0, [[0, 0, 0], [0, 0, 1]])
        self.assertEqual(evaluator.summary()["team"]["CHE"], 0.0)
        evaluator.update(1.0, [[1, 0, 0], [1, 0, 1]])
        evaluator.update(2.0, [[2, 0, 0], [2, 0, 1]])
        team = evaluator.summary()["team"]
        self.assertAlmostEqual(team["CHE"], 1.9, places=6)
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


if __name__ == "__main__":
    unittest.main()
