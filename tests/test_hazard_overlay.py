"""Headless regressions for the risk overlay on the obstacle-map panel."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from utils.visualization import (
    Visualize,
    overlay_hazard_on_obstacle_map,
)


class HazardOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.full((9, 10, 3), 220, dtype=np.uint8)
        self.obstacles = np.zeros((9, 10), dtype=bool)
        self.obstacles[1:4, 1:3] = True
        self.base[self.obstacles] = (40, 40, 40)

    def test_zero_risk_is_exact_identity_and_inputs_are_immutable(self) -> None:
        risk = np.zeros(self.obstacles.shape, dtype=np.float32)
        hard = np.zeros(self.obstacles.shape, dtype=bool)
        base_before = self.base.copy()
        risk_before = risk.copy()
        hard_before = hard.copy()

        rendered = overlay_hazard_on_obstacle_map(
            self.base,
            risk,
            hard_unsafe_mask=hard,
            obstacle_mask=self.obstacles,
        )

        np.testing.assert_array_equal(rendered, self.base)
        np.testing.assert_array_equal(self.base, base_before)
        np.testing.assert_array_equal(risk, risk_before)
        np.testing.assert_array_equal(hard, hard_before)

    def test_gradient_hard_cells_and_obstacles_remain_distinct(self) -> None:
        risk = np.zeros(self.obstacles.shape, dtype=np.float32)
        risk[5, 2] = 0.25
        risk[5, 4] = 0.60
        risk[5, 6] = 1.00
        risk[self.obstacles] = 1.00
        hard = np.zeros(self.obstacles.shape, dtype=bool)
        hard[4:7, 7:10] = True

        rendered = overlay_hazard_on_obstacle_map(
            self.base,
            risk,
            hard_unsafe_mask=hard,
            obstacle_mask=self.obstacles,
        )

        # Increasing risk removes green while keeping a strong red channel.
        self.assertGreater(rendered[5, 2, 1], rendered[5, 4, 1])
        self.assertGreater(rendered[5, 4, 1], rendered[5, 6, 1])
        self.assertGreater(rendered[5, 6, 2], rendered[5, 6, 1])
        # The centre of a hard region is visibly magenta, not ordinary red.
        self.assertGreater(rendered[5, 8, 0], 100)
        self.assertLess(rendered[5, 8, 1], 80)
        self.assertGreater(rendered[5, 8, 2], 180)
        # Physical obstacles are restored exactly even under maximum hazard.
        np.testing.assert_array_equal(
            rendered[self.obstacles], self.base[self.obstacles]
        )

    def test_shape_mismatch_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "planning_risk shape"):
            overlay_hazard_on_obstacle_map(
                self.base,
                np.zeros((8, 10), dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "hard_unsafe_mask shape"):
            overlay_hazard_on_obstacle_map(
                self.base,
                np.zeros((9, 10), dtype=np.float32),
                hard_unsafe_mask=np.zeros((8, 10), dtype=bool),
            )

    def test_visualize_adds_overlay_and_legend_only_when_supplied(self) -> None:
        shape = (20, 20)
        args = SimpleNamespace(
            num_agents=2,
            visualize=False,
            print_images=False,
            dump_location="unused",
            nav_mode="co_ut",
        )
        obstacle = np.zeros(shape, dtype=np.float32)
        obstacle[5:8, 6] = 1.0
        explored = np.ones(shape, dtype=np.float32)
        visited = [np.zeros(shape), np.zeros(shape)]
        edge = np.zeros(shape)
        edge[3:8, 14] = 1
        goals = [np.zeros(shape), np.zeros(shape)]
        goals[0][15, 15] = 1
        goals[1][4, 15] = 1
        top_view = np.full((20, 20, 3), 180, dtype=np.uint8)
        risk = np.zeros(shape, dtype=np.float32)
        risk[6:15, 8:13] = np.linspace(
            0.2, 1.0, 9, dtype=np.float32
        )[:, None]
        hard = np.zeros(shape, dtype=bool)
        hard[9:12, 10:12] = True
        common = (
            args,
            1,
            [[5, 5, 0.0], [12, 4, 1.0]],
            obstacle,
            explored,
            "chair",
            visited,
            edge,
            goals,
            top_view,
        )

        baseline = Visualize(*common)
        rendered = Visualize(
            *common,
            planning_risk=risk,
            hard_unsafe_mask=hard,
        )
        labelled = Visualize(
            *common,
            planning_risk=risk,
            hard_unsafe_mask=hard,
            hazard_display_label="GT display only",
        )

        self.assertEqual(rendered.shape, (537, 980, 3))
        self.assertFalse(np.array_equal(rendered, baseline))
        self.assertFalse(np.array_equal(labelled, rendered))
        # The camera/top-view panel itself is untouched by the hazard overlay.
        np.testing.assert_array_equal(
            rendered[50:530, 500:980], baseline[50:530, 500:980]
        )

    def test_evaluator_visualization_accessor_is_explicit_and_defensive(
        self,
    ) -> None:
        from utils.risk.runtime import RiskRuntime

        sentinel = object()
        runtime = object.__new__(RiskRuntime)
        runtime.gt_provider = mock.Mock()
        runtime.gt_provider.snapshot.return_value = sentinel

        result = runtime.evaluator_visualization_state(3.5)

        self.assertIs(result, sentinel)
        runtime.gt_provider.snapshot.assert_called_once_with(3.5)

    def test_main_keeps_none_source_gt_out_of_both_planners(self) -> None:
        root = Path(__file__).resolve().parents[1]
        module = ast.parse((root / "main.py").read_text())
        source = ast.unparse(module)
        self.assertIn("evaluator_visualization_state", source)
        self.assertIn("GT display only", source)
        calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Visualize"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertIn("planning_risk", keywords)
        self.assertIn("hard_unsafe_mask", keywords)
        self.assertEqual(
            ast.unparse(keywords["planning_risk"]),
            "visualization_risk",
        )
        self.assertEqual(
            ast.unparse(keywords["hard_unsafe_mask"]),
            "visualization_hard_unsafe",
        )

        risk_context_calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RiskPlanningContext"
        ]
        self.assertEqual(len(risk_context_calls), 1)
        context_keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in risk_context_calls[0].keywords
        }
        self.assertEqual(context_keywords["planning_risk"], "planning_risk")
        self.assertNotIn("visualization", ast.unparse(risk_context_calls[0]))

        local_calls = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_risk_map"
        ]
        self.assertEqual(len(local_calls), 1)
        self.assertEqual(ast.unparse(local_calls[0].args[0]), "planning_risk")
        self.assertNotIn("visualization", ast.unparse(local_calls[0]))


if __name__ == "__main__":
    unittest.main()
