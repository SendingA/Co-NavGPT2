"""Toy-map tests for the hazard-aware fast-marching planner."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

try:
    from utils.fmm_planner import FMMPlanner
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    if exc.name != "skfmm":
        raise
    FMMPlanner = None


def _descending_path(planner, start, goal, max_steps=1000):
    """Extract an 8-connected path by descending an FMM distance field."""
    cell = tuple(start)
    goal = tuple(goal)
    path = [cell]
    for _ in range(max_steps):
        if cell == goal:
            break
        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (
                    0 <= nxt[0] < planner.traversible.shape[0]
                    and 0 <= nxt[1] < planner.traversible.shape[1]
                ):
                    continue
                if planner.traversible[nxt] > 0:
                    candidates.append(nxt)
        if not candidates:
            break
        nxt = min(candidates, key=lambda item: planner.fmm_dist[item])
        if planner.fmm_dist[nxt] >= planner.fmm_dist[cell] - 1e-8:
            break
        cell = nxt
        path.append(cell)
    return path


@unittest.skipUnless(FMMPlanner is not None, "skfmm is not installed")
class RiskAwareFMMTests(unittest.TestCase):
    def setUp(self):
        self.shape = (41, 41)
        self.traversible = np.ones(self.shape, dtype=np.float32)
        self.start = (20, 5)
        self.goal = (20, 35)
        self.goal_map = np.zeros(self.shape, dtype=np.uint8)
        self.goal_map[self.goal] = 1

    def test_alpha_zero_is_exact_legacy_distance(self):
        baseline = FMMPlanner(self.traversible)
        baseline.set_multi_goal(self.goal_map)

        ignored_risk = np.ones(self.shape, dtype=np.float32)
        alpha_zero = FMMPlanner(
            self.traversible,
            risk_map=ignored_risk,
            risk_alpha=0.0,
        )
        alpha_zero.set_multi_goal(self.goal_map)

        np.testing.assert_allclose(
            alpha_zero.fmm_dist, baseline.fmm_dist, rtol=0.0, atol=0.0
        )

    def test_soft_risk_selects_safe_detour(self):
        risk = np.zeros(self.shape, dtype=np.float32)
        risk[16:25, 16:27] = 1.0

        baseline = FMMPlanner(self.traversible)
        baseline.set_multi_goal(self.goal_map)
        baseline_path = _descending_path(baseline, self.start, self.goal)

        safer = FMMPlanner(
            self.traversible, risk_map=risk, risk_alpha=2.0
        )
        safer.set_multi_goal(self.goal_map)
        safer_path = _descending_path(safer, self.start, self.goal)

        self.assertEqual(baseline_path[-1], self.goal)
        self.assertEqual(safer_path[-1], self.goal)
        self.assertGreater(sum(risk[cell] for cell in baseline_path), 0.0)
        self.assertEqual(sum(risk[cell] for cell in safer_path), 0.0)

    def test_updated_hazard_map_replans_to_opposite_safe_corridor(self):
        upper_hazard = np.zeros(self.shape, dtype=np.float32)
        upper_hazard[8:22, 15:28] = 1.0
        lower_hazard = np.zeros(self.shape, dtype=np.float32)
        lower_hazard[19:33, 15:28] = 1.0

        paths = []
        for risk in (upper_hazard, lower_hazard):
            planner = FMMPlanner(
                self.traversible, risk_map=risk, risk_alpha=3.0
            )
            planner.set_multi_goal(self.goal_map)
            path = _descending_path(planner, self.start, self.goal)
            self.assertEqual(path[-1], self.goal)
            self.assertEqual(sum(risk[cell] for cell in path), 0.0)
            paths.append(path)

        upper_route_rows = [row for row, col in paths[0] if 14 <= col <= 28]
        lower_route_rows = [row for row, col in paths[1] if 14 <= col <= 28]
        self.assertGreater(np.mean(upper_route_rows), 20.0)
        self.assertLess(np.mean(lower_route_rows), 20.0)

    def test_hard_mask_blocks_cells_and_routes_through_gap(self):
        hard_mask = np.zeros(self.shape, dtype=bool)
        hard_mask[10:31, 20] = True
        hard_mask[10, 20] = False

        planner = FMMPlanner(
            self.traversible, hard_unsafe_mask=hard_mask
        )
        planner.set_multi_goal(self.goal_map)
        path = _descending_path(planner, self.start, self.goal)

        self.assertEqual(path[-1], self.goal)
        self.assertFalse(any(hard_mask[cell] for cell in path))
        self.assertTrue(np.all(planner.traversible[hard_mask] == 0))

    def test_goal_cell_is_safely_admitted_at_hard_mask_boundary(self):
        hard_mask = np.zeros(self.shape, dtype=bool)
        hard_mask[self.goal] = True
        planner = FMMPlanner(
            self.traversible, hard_unsafe_mask=hard_mask
        )

        planner.set_multi_goal(self.goal_map)
        path = _descending_path(planner, self.start, self.goal)

        self.assertEqual(path[-1], self.goal)
        self.assertEqual(planner.traversible[self.goal], 1.0)
        self.assertFalse(planner.hard_unsafe_mask[self.goal])

    def test_agent_inside_hard_region_gets_escape_waypoint(self):
        hard_mask = np.zeros(self.shape, dtype=bool)
        hard_mask[18:23, 3:8] = True
        planner = FMMPlanner(
            self.traversible, hard_unsafe_mask=hard_mask
        )

        escape_goal = planner.prepare_emergency_escape(self.start)

        self.assertIsNotNone(escape_goal)
        self.assertEqual(int(np.sum(escape_goal)), 1)
        escape_cell = tuple(np.argwhere(escape_goal == 1)[0])
        self.assertFalse(hard_mask[escape_cell])
        self.assertFalse(planner.hard_unsafe_mask[self.start])
        planner.set_multi_goal(escape_goal)
        escape_path = _descending_path(planner, self.start, escape_cell)
        self.assertEqual(escape_path[-1], escape_cell)

    def test_risk_grid_shape_must_match_navigation_grid(self):
        with self.assertRaisesRegex(ValueError, "risk_map shape"):
            FMMPlanner(
                self.traversible,
                risk_map=np.zeros((10, 10), dtype=np.float32),
                risk_alpha=1.0,
            )


class AgentRiskAPIContractTests(unittest.TestCase):
    def test_agent_variants_expose_matching_setter_signature(self):
        root = Path(__file__).resolve().parents[1]
        signatures = []
        for relative_path in (
            "agents/vlm_agents.py",
            "agents/vlm_multi_agents.py",
        ):
            module = ast.parse((root / relative_path).read_text())
            agent_class = next(
                node
                for node in module.body
                if isinstance(node, ast.ClassDef) and node.name == "VLM_Agent"
            )
            setter = next(
                node
                for node in agent_class.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "set_risk_map"
            )
            signatures.append([argument.arg for argument in setter.args.args])

            act_method = next(
                node
                for node in agent_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "act"
            )
            self.assertIn(
                "if not getattr(self, 'risk_navigation_enabled', False)",
                ast.unparse(act_method),
            )

        self.assertEqual(signatures[0], signatures[1])
        self.assertEqual(
            signatures[0],
            [
                "self",
                "risk_map",
                "hard_unsafe_mask",
                "risk_alpha",
                "enabled",
            ],
        )


if __name__ == "__main__":
    unittest.main()
