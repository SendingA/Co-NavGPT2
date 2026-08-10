"""Headless tests for frontier risk reporting and team assignment."""

from __future__ import annotations

import unittest

import numpy as np

from utils.risk.frontier import (
    SeverityThresholds,
    UtilityWeights,
    assign_frontiers,
    build_frontier_risk_reports,
    frontier_risk_report,
    guard_frontier_assignments,
    risk_context_payload,
    score_frontiers,
)


class FrontierRiskReportTests(unittest.TestCase):
    def test_each_frontier_can_report_its_own_route_provenance(self) -> None:
        risk = np.zeros((5, 5), dtype=np.float32)
        labels = np.zeros((5, 5), dtype=np.int32)
        labels[1, 1] = 1
        labels[3, 3] = 2

        reports = build_frontier_risk_reports(
            labels,
            risk,
            frontier_points=((1, 1), (3, 3)),
            route_cells=(((0, 0), (1, 1)), ((4, 4), (3, 3))),
            route_is_proxy=(False, True),
        )

        self.assertEqual(
            [report.route_is_proxy for report in reports],
            [False, True],
        )

    def test_report_includes_distribution_route_confidence_and_hard_block(self):
        risk = np.zeros((5, 6), dtype=np.float32)
        risk[1, 1:4] = [0.10, 0.20, 0.30]
        risk[3, 3] = 0.90
        confidence = np.zeros_like(risk)
        confidence[1, 1:4] = [0.60, 0.80, 1.00]
        mask = np.zeros_like(risk, dtype=bool)
        mask[1, 1:4] = True

        report = frontier_risk_report(
            2,
            mask,
            risk,
            confidence,
            route_cells=[(0, 0), (1, 2), (3, 3), (100, 100)],
        )

        self.assertEqual(report.frontier_id, 2)
        self.assertEqual(report.point, (1, 2))
        self.assertEqual(report.cell_count, 3)
        self.assertAlmostEqual(report.mean_risk, 0.20, places=6)
        self.assertAlmostEqual(report.p95_risk, 0.29, places=6)
        self.assertAlmostEqual(report.max_risk, 0.30, places=6)
        self.assertAlmostEqual(report.route_risk, (0.0 + 0.2 + 0.9) / 3, places=6)
        self.assertAlmostEqual(report.route_max_risk, 0.90, places=6)
        self.assertAlmostEqual(report.confidence, 0.80, places=6)
        self.assertAlmostEqual(report.uncertainty, 0.20, places=6)
        self.assertTrue(report.hard_blocked)
        self.assertEqual(report.severity, "dangerous")

    def test_label_map_uses_zero_based_frontier_ids_and_explicit_points(self):
        labels = np.zeros((6, 6), dtype=np.int32)
        labels[1, 1:3] = 1
        labels[4, 3:6] = 2
        risk = np.zeros_like(labels, dtype=np.float32)
        risk[labels == 1] = 0.1
        risk[labels == 2] = 0.4

        reports = build_frontier_risk_reports(
            labels,
            risk,
            frontier_points=[(1, 1), (4, 4)],
        )

        self.assertEqual([report.frontier_id for report in reports], [0, 1])
        self.assertEqual([report.point for report in reports], [(1, 1), (4, 4)])
        self.assertEqual([report.severity for report in reports], ["safe", "moderate"])
        payload = risk_context_payload(reports)
        self.assertEqual(len(payload["hazard_report"]), 2)
        self.assertFalse(payload["hazard_report"][0]["hard_blocked"])

    def test_point_only_frontier_samples_risk_and_confidence(self):
        risk = np.zeros((4, 4), dtype=np.float32)
        risk[2, 3] = 0.5
        confidence = np.zeros_like(risk)
        confidence[2, 3] = 0.7

        report = frontier_risk_report(
            0,
            np.zeros_like(risk, dtype=bool),
            risk,
            confidence,
            frontier_point=(2, 3),
        )

        self.assertEqual(report.cell_count, 1)
        self.assertAlmostEqual(report.mean_risk, 0.5)
        self.assertAlmostEqual(report.confidence, 0.7)

    def test_explicit_hard_unsafe_map_cannot_be_averaged_away(self):
        risk = np.full((5, 5), 0.05, dtype=np.float32)
        mask = np.ones_like(risk, dtype=bool)
        hard = np.zeros_like(mask)
        hard[2, 2] = True

        report = frontier_risk_report(
            0,
            mask,
            risk,
            hard_unsafe_map=hard,
        )

        self.assertAlmostEqual(report.mean_risk, 0.05)
        self.assertTrue(report.hard_blocked)
        self.assertEqual(report.severity, "dangerous")

    def test_single_stacked_mask_accepts_direct_point_and_route(self):
        risk = np.zeros((4, 4), dtype=np.float32)
        risk[1, 1] = 0.2
        risk[2, 2] = 0.4
        masks = np.zeros((1, 4, 4), dtype=bool)
        masks[0, 1, 1] = True

        reports = build_frontier_risk_reports(
            masks,
            risk,
            frontier_points=(1, 1),
            route_cells=[(1, 1), (2, 2)],
        )

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].point, (1, 1))
        self.assertAlmostEqual(reports[0].route_risk, 0.3)

    def test_straight_line_proxy_affects_cost_but_not_hard_veto(self):
        risk = np.zeros((4, 4), dtype=np.float32)
        risk[2, 2] = 1.0
        mask = np.zeros_like(risk, dtype=bool)
        mask[0, 0] = True
        report = frontier_risk_report(
            0,
            mask,
            risk,
            route_cells=[(1, 1), (2, 2)],
            route_is_proxy=True,
        )
        self.assertTrue(report.route_is_proxy)
        self.assertGreater(report.route_risk, 0.0)
        self.assertFalse(report.hard_blocked)
        score = score_frontiers(
            [(0, 0)], [report], hard_risk_threshold=0.8
        )[0][0]
        self.assertTrue(score.feasible)


class FrontierAssignmentTests(unittest.TestCase):
    @staticmethod
    def _reports(risks, confidences=None, hard_index=None):
        risk_map = np.zeros((7, 7), dtype=np.float32)
        confidence_map = np.ones_like(risk_map)
        labels = np.zeros((7, 7), dtype=np.int32)
        points = [(1, 1), (1, 5), (5, 3)]
        for index, value in enumerate(risks):
            row, col = points[index]
            labels[row, col] = index + 1
            risk_map[row, col] = value
            if confidences is not None:
                confidence_map[row, col] = confidences[index]
        thresholds = SeverityThresholds(
            safe_max=0.25,
            moderate_max=0.55,
            hard_max=(0.85 if hard_index is not None else 0.95),
        )
        return build_frontier_risk_reports(
            labels,
            risk_map,
            confidence_map,
            frontier_points=points[: len(risks)],
            thresholds=thresholds,
        )

    def test_score_penalizes_risk_and_uncertainty(self):
        reports = self._reports([0.1, 0.1], confidences=[1.0, 0.2])
        scores = score_frontiers(
            {0: (1, 3)},
            reports,
            information_gain={0: 10, 1: 10},
        )[0]

        self.assertGreater(scores[0].utility, scores[1].utility)
        self.assertEqual(scores[0].normalized_distance, scores[1].normalized_distance)
        self.assertLess(scores[0].uncertainty, scores[1].uncertainty)

    def test_assignment_uses_redundancy_to_split_equal_frontiers(self):
        reports = self._reports([0.1, 0.1])
        assignments = assign_frontiers(
            {0: (1, 3), 1: (1, 3)},
            reports,
            information_gain={0: 10, 1: 10},
            weights=UtilityWeights(redundancy=2.0),
            allow_shared=True,
        )

        # Both split variants have equal team utility; lexicographic tie-break
        # makes the result reproducible across runs/platforms.
        self.assertEqual(assignments, {0: 0, 1: 1})

    def test_hard_threshold_filters_before_assignment(self):
        reports = self._reports([0.90], hard_index=0)
        assignments = assign_frontiers(
            [(1, 1), (2, 2)],
            reports,
        )

        self.assertEqual(assignments, {0: None, 1: None})

    def test_route_risk_can_outweigh_nearer_frontier(self):
        risk = np.zeros((8, 8), dtype=np.float32)
        labels = np.zeros_like(risk, dtype=np.int32)
        labels[1, 2] = 1
        labels[1, 6] = 2
        risk[1, 2] = 0.05
        risk[1, 6] = 0.05
        risk[3, 2:5] = 0.70
        reports = build_frontier_risk_reports(
            labels,
            risk,
            frontier_points=[(1, 2), (1, 6)],
            route_cells={
                0: [(3, 2), (3, 3), (3, 4)],
                1: [(0, 4), (0, 5), (0, 6)],
            },
        )

        assignment = assign_frontiers(
            {0: (1, 1)},
            reports,
            information_gain={0: 10, 1: 10},
        )
        self.assertEqual(assignment, {0: 1})

    def test_guard_rejects_vlm_hard_block_and_uses_safe_fallback(self):
        reports = self._reports([0.1, 0.9], hard_index=1)
        guarded = guard_frontier_assignments(
            {
                "robot_0": "frontier_1",
                "robot_1": "not-a-frontier",
            },
            reports,
            fallback_assignments={0: 0, 1: 0},
            expected_robot_ids=[0, 1],
        )

        self.assertEqual(guarded.assignments, {0: 0, 1: 0})
        self.assertEqual(guarded.rejected[0], "hard_blocked")
        self.assertEqual(guarded.rejected[1], "missing_or_invalid")

        malformed = guard_frontier_assignments(
            None,
            reports,
            fallback_assignments={0: 0},
            expected_robot_ids=[0],
        )
        self.assertEqual(malformed.assignments, {0: 0})
        self.assertEqual(malformed.rejected, {0: "missing_or_invalid"})


if __name__ == "__main__":
    unittest.main()
