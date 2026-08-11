"""Deterministic regressions for arbitrary-episode fire-plan generation."""
from __future__ import annotations

import unittest

import numpy as np

from utils.fire_world.episode_plan import (
    EpisodeGoalGeometry,
    EpisodeSourceCandidate,
    combined_radial_hazard_map,
    episode_scene_id,
    native_goal_geometry,
    resolve_episode,
    select_best_source_combination,
    source_clearance_metrics,
)
from utils.fire_world.fine_tuning import RouteContrastThresholds


def _candidate(object_id: int, cell, progress_m: float):
    position = (float(cell[0]), 0.0, float(cell[1]))
    return EpisodeSourceCandidate(
        instance={
            "instance_id": object_id,
            "category": "table",
            "centroid": list(position),
            "flammability": 0.8,
        },
        cell=tuple(cell),
        position=position,
        distance_to_blind_path_m=0.0,
        distance_to_start_m=4.0,
        nearest_goal_centre_clearance_m=8.0,
        nearest_goal_viewpoint_clearance_m=8.0,
        blind_path_progress_m=float(progress_m),
    )


def _fork_grid():
    grid = np.ones((15, 25), dtype=np.uint8)
    grid[[0, -1], :] = 0
    grid[:, [0, -1]] = 0
    grid[1:14, 12] = 0
    grid[2, 12] = 1
    grid[7, 12] = 1
    return grid


class EpisodeResolutionTests(unittest.TestCase):
    def test_reused_episode_id_requires_category(self):
        dataset = {"episodes": [
            {"episode_id": "5", "object_category": "bed"},
            {"episode_id": "5", "object_category": "chair"},
        ]}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            resolve_episode(dataset, "5")
        selected = resolve_episode(dataset, "5", object_category="bed")
        self.assertEqual(selected["object_category"], "bed")

    def test_scene_id_and_native_viewpoints_keep_episode_floor(self):
        episode = {
            "scene_id": "hm3d/001-SceneA/SceneA.basis.glb",
        }
        self.assertEqual(episode_scene_id(episode), "SceneA")
        dataset = {"goals_by_category": {
            "SceneA.basis.glb_person": [{
                "position": [3.0, 0.0, 4.0],
                "view_points": [
                    {"agent_state": {"position": [2.0, 0.0, 4.0]}},
                    {"agent_state": {"position": [2.0, 2.0, 4.0]}},
                ],
            }],
        }}
        geometry = native_goal_geometry(
            dataset,
            scene_id="SceneA",
            object_category="person",
            floor_y_m=0.0,
        )
        self.assertEqual(geometry.centres.shape, (1, 3))
        self.assertEqual(geometry.viewpoints.tolist(), [[2.0, 0.0, 4.0]])

    def test_clearance_checks_start_centres_and_every_viewpoint(self):
        geometry = EpisodeGoalGeometry(
            centres=np.asarray([[3.0, 0.0, 4.0]]),
            viewpoints=np.asarray([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        )
        metrics = source_clearance_metrics(
            [0.0, 0.5, 0.0],
            start_position=[0.0, 0.0, 1.0],
            goal_geometry=geometry,
        )
        self.assertAlmostEqual(metrics["distance_to_start_m"], 1.0)
        self.assertAlmostEqual(metrics["nearest_goal_centre_clearance_m"], 5.0)
        self.assertAlmostEqual(
            metrics["nearest_goal_viewpoint_clearance_m"], 2.0
        )


class SourceCombinationTests(unittest.TestCase):
    def test_combined_map_unions_three_sources(self):
        risk, hard = combined_radial_hazard_map(
            (15, 25),
            [(7, 10), (7, 12), (7, 14)],
            resolution_m=1.0,
            core_radius_m=0.7,
            risk_radius_m=2.1,
        )
        self.assertTrue(all(hard[cell] for cell in [(7, 10), (7, 12), (7, 14)]))
        self.assertEqual(float(risk[7, 12]), 1.0)

    def test_three_source_union_selects_long_safe_fork(self):
        diagnostics = {}
        result = select_best_source_combination(
            _fork_grid(),
            start=(7, 2),
            goals=[(7, 22)],
            candidates=[
                _candidate(1, (7, 10), 8.0),
                _candidate(2, (7, 12), 10.0),
                _candidate(3, (7, 14), 12.0),
            ],
            source_count=3,
            resolution_m=1.0,
            core_radius_m=0.7,
            risk_radius_m=2.1,
            risk_alpha=4.0,
            thresholds=RouteContrastThresholds(
                min_detour_ratio=1.10, max_detour_ratio=2.0
            ),
            max_goal_risk=0.35,
            min_source_spacing_m=1.0,
            max_combinations=10,
            diagnostics=diagnostics,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.contrast.accepted)
        self.assertEqual([item.object_id for item in result.sources], [1, 2, 3])
        self.assertGreater(result.contrast.detour_ratio, 1.10)
        self.assertEqual(result.contrast.aware.hard_unsafe_cells, 0)
        self.assertEqual(diagnostics["evaluated_combinations"], 1)

    def test_single_corridor_is_rejected_when_fire_removes_only_route(self):
        grid = np.zeros((9, 21), dtype=np.uint8)
        grid[4, 1:20] = 1
        diagnostics = {}
        result = select_best_source_combination(
            grid,
            start=(4, 1),
            goals=[(4, 19)],
            candidates=[
                _candidate(1, (4, 8), 7.0),
                _candidate(2, (4, 10), 9.0),
                _candidate(3, (4, 12), 11.0),
            ],
            source_count=3,
            resolution_m=1.0,
            core_radius_m=0.7,
            risk_radius_m=2.1,
            risk_alpha=4.0,
            thresholds=RouteContrastThresholds(),
            max_goal_risk=0.35,
            min_source_spacing_m=1.0,
            max_combinations=10,
            diagnostics=diagnostics,
        )
        self.assertIsNone(result)
        self.assertEqual(diagnostics["rejection_counts"]["no_route"], 1)


if __name__ == "__main__":
    unittest.main()
