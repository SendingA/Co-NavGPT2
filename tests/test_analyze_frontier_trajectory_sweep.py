import unittest

import numpy as np

from scripts.analyze_frontier_trajectory_sweep import (
    path_geometry,
    rank_runs,
    route_divergence,
    team_geometry,
)


def _xyz(points):
    return np.asarray([[x, 0.2, z] for x, z in points], dtype=np.float64)


class FrontierTrajectoryAnalysisTests(unittest.TestCase):
    def test_path_geometry_ignores_stationary_actions(self):
        path = _xyz([
            (0.0, 0.0),
            (0.0, 0.0),
            (0.25, 0.0),
            (0.50, 0.0),
            (0.75, 0.0),
            (1.00, 0.0),
            (1.00, 0.0),
        ])
        geometry = path_geometry(path)
        self.assertEqual(geometry["moving_pose_count"], 5)
        self.assertAlmostEqual(geometry["path_length_m"], 1.0)
        self.assertAlmostEqual(geometry["spatial_span_m"], 1.0)
        self.assertAlmostEqual(geometry["straightness"], 1.0)
        self.assertAlmostEqual(geometry["self_revisit_fraction"], 0.0)
        self.assertAlmostEqual(geometry["initial_2m_progress_ratio"], 1.0)

    def test_loop_has_revisit_and_more_turning_than_open_path(self):
        open_path = _xyz([(0, 0), (0.5, 0), (1.0, 0), (1.5, 0)])
        loop = _xyz([
            (0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5),
            (0, 0), (0.5, 0),
        ])
        open_geometry = path_geometry(open_path)
        loop_geometry = path_geometry(loop)
        self.assertGreater(loop_geometry["self_revisit_fraction"], 0.0)
        self.assertGreater(
            loop_geometry["turn_rad_per_m"],
            open_geometry["turn_rad_per_m"],
        )
        self.assertLess(
            loop_geometry["initial_2m_progress_ratio"],
            open_geometry["initial_2m_progress_ratio"],
        )

    def test_team_overlap_and_route_divergence_are_spatial(self):
        first = _xyz([(0, 0), (0.25, 0), (0.50, 0)])
        separate = _xyz([(0, 2), (0.25, 2), (0.50, 2)])
        same = first.copy()
        self.assertEqual(team_geometry([first, separate])["shared_0_25m_cells"], 0)
        self.assertEqual(team_geometry([first, same])["cell_jaccard"], 1.0)
        self.assertEqual(route_divergence(first, same), 0.0)
        self.assertEqual(route_divergence(first, separate), 1.0)

    def test_rank_prefers_eligible_relaxed_run(self):
        template = {
            "target": "bed",
            "eligible": True,
            "spatially_relaxed": True,
            "agents": {
                "1": {
                    "spatial_span_m": 5.0,
                    "straightness": 0.8,
                    "self_revisit_fraction": 0.1,
                    "turn_rad_per_m": 0.2,
                }
            },
            "team_geometry": {
                "agent_1_near_agent_0_fraction_0_5m": 0.0,
            },
        }
        good = {**template, "strategy": "fill"}
        bad = {
            **template,
            "strategy": "nearest",
            "eligible": False,
            "spatially_relaxed": False,
            "agents": {"1": {
                "spatial_span_m": 1.0,
                "straightness": 0.2,
                "self_revisit_fraction": 0.7,
                "turn_rad_per_m": 2.0,
            }},
            "team_geometry": {
                "agent_1_near_agent_0_fraction_0_5m": 0.8,
            },
        }
        ranked = rank_runs([bad, good])
        self.assertEqual(ranked[0]["strategy"], "fill")
        self.assertEqual(ranked[0]["rank_within_target"], 1)
        self.assertGreater(ranked[0]["trajectory_score"], 0.0)
        self.assertEqual(ranked[1]["trajectory_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
