"""Tests for curated FireWorld route-contrast fine-tuning."""
from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.fire_world.fine_tuning import (
    CURATED_FIRE_PROFILES,
    RouteContrastThresholds,
    build_curated_plan,
    curated_plan_hash,
    evaluate_route_contrast,
    radial_hazard_map,
    route_overlay,
    shortest_grid_path,
    write_curated_plan,
)
from scripts.tune_fire_route_scenarios import _write_episode_dataset


def _two_route_grid():
    traversible = np.ones((15, 25), dtype=np.uint8)
    traversible[[0, -1], :] = 0
    traversible[:, [0, -1]] = 0
    traversible[1:14, 12] = 0
    traversible[2, 12] = 1
    traversible[7, 12] = 1
    return traversible, (7, 2), (7, 22), (7, 12)


def _inventory():
    return {
        "scene_id": "SceneA",
        "scene_glb": "data/SceneA.basis.glb",
        "world_aabb": [0.0, 0.0, 0.0, 3.0, 2.0, 3.0],
    }


def _instance():
    return {
        "instance_id": 17,
        "category": "chair",
        "centroid": [1.5, 0.4, 1.5],
        "flammability": 0.8,
    }


class GridRouteTests(unittest.TestCase):
    def test_blind_shortest_path_uses_direct_gap(self) -> None:
        traversible, start, goal, ignition = _two_route_grid()
        result = shortest_grid_path(traversible, start, [goal])

        self.assertIsNotNone(result)
        cells, _, _ = result
        self.assertIn(ignition, cells)

    def test_risk_aware_path_detours_around_hard_fire(self) -> None:
        traversible, start, goal, ignition = _two_route_grid()
        risk, hard = radial_hazard_map(
            traversible.shape,
            ignition,
            resolution_m=1.0,
            core_radius_m=1.0,
            risk_radius_m=2.5,
        )
        contrast = evaluate_route_contrast(
            traversible,
            start,
            [goal],
            risk,
            hard,
            risk_alpha=4.0,
            thresholds=RouteContrastThresholds(
                min_detour_ratio=1.10,
                max_detour_ratio=1.80,
            ),
        )

        self.assertIsNotNone(contrast)
        self.assertTrue(contrast.accepted, contrast.reasons)
        self.assertIn(ignition, contrast.blind.cells)
        self.assertNotIn(ignition, contrast.aware.cells)
        self.assertGreater(contrast.detour_ratio, 1.10)
        self.assertGreaterEqual(contrast.exposure_reduction, 0.70)
        self.assertEqual(contrast.aware.hard_unsafe_cells, 0)

    def test_non_route_fire_is_rejected(self) -> None:
        traversible, start, goal, _ = _two_route_grid()
        risk, hard = radial_hazard_map(
            traversible.shape,
            (12, 3),
            resolution_m=1.0,
            core_radius_m=0.5,
            risk_radius_m=1.5,
        )
        contrast = evaluate_route_contrast(
            traversible, start, [goal], risk, hard
        )

        self.assertIsNotNone(contrast)
        self.assertFalse(contrast.accepted)
        self.assertIn("blind_path_not_dangerous", contrast.reasons)

    def test_overlay_contains_distinct_route_colours(self) -> None:
        traversible, start, goal, ignition = _two_route_grid()
        risk, hard = radial_hazard_map(
            traversible.shape,
            ignition,
            resolution_m=1.0,
            core_radius_m=1.0,
            risk_radius_m=2.5,
        )
        contrast = evaluate_route_contrast(
            traversible,
            start,
            [goal],
            risk,
            hard,
            thresholds=RouteContrastThresholds(
                min_detour_ratio=1.10,
                max_detour_ratio=1.80,
            ),
        )
        image = route_overlay(
            traversible,
            risk,
            contrast,
            start=start,
            goal=goal,
            ignition=ignition,
        )

        self.assertEqual(image.shape, traversible.shape + (3,))
        self.assertTrue(np.any(np.all(image == (30, 100, 255), axis=2)))
        self.assertTrue(np.any(np.all(image == (30, 220, 80), axis=2)))
        self.assertTrue(np.array_equal(image[ignition], (255, 220, 0)))


class CuratedPlanTests(unittest.TestCase):
    def test_curated_dataset_contains_only_selected_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            source_root.mkdir()
            source = {
                "episodes": [
                    {
                        "episode_id": "5",
                        "object_category": "sofa",
                        "start_position": [1.0, 0.0, 1.0],
                    },
                    {
                        "episode_id": "5",
                        "object_category": "sofa",
                        "start_position": [2.0, 0.0, 2.0],
                    },
                ],
                "goals_by_category": {"SceneA.basis.glb_sofa": []},
            }
            with gzip.open(
                source_root / "SceneA.json.gz", "wt", encoding="utf-8"
            ) as handle:
                json.dump(source, handle)
            candidate = SimpleNamespace(
                scene_id="SceneA",
                episode_id="5",
                profile_name="stable",
                object_category="sofa",
                start_position=[2.0, 0.0, 2.0],
            )

            path = _write_episode_dataset(
                source_root,
                root / "processed",
                candidate,
                {"plan_id": "SceneA_route_contrast_stable_hash"},
            )

            with gzip.open(path, "rt", encoding="utf-8") as handle:
                root_result = json.load(handle)
            shard_path = path.parent / "content" / "SceneA.json.gz"
            with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(root_result["episodes"], [])
            self.assertNotIn("goals_by_category", root_result)
            self.assertEqual(
                root_result["content_scenes_path"],
                "{data_path}/content/{scene}.json.gz",
            )
            self.assertEqual(
                [episode["episode_id"] for episode in result["episodes"]],
                ["5"],
            )
            self.assertEqual(
                result["episodes"][0]["start_position"],
                [2.0, 0.0, 2.0],
            )
            self.assertIn("goals_by_category", result)
            self.assertEqual(
                result["fire_route_scenario"]["plan_id"],
                "SceneA_route_contrast_stable_hash",
            )
            self.assertEqual(
                root_result["fire_route_scenario"],
                result["fire_route_scenario"],
            )

    def test_profiles_keep_distinct_route_acceptance_contracts(self) -> None:
        stable = CURATED_FIRE_PROFILES["stable"]
        dynamic = CURATED_FIRE_PROFILES["dynamic"]

        self.assertEqual(stable.thresholds.min_detour_ratio, 1.15)
        self.assertEqual(dynamic.thresholds.min_detour_ratio, 1.05)
        self.assertEqual(dynamic.propagation_rules[
            "object_max_spread_radius_m"
        ], 0.85)
        self.assertEqual(dynamic.propagation_rules[
            "object_bbox_fill_speed_m_per_s"
        ], 0.0)

    def test_complete_payload_has_content_addressed_plan_id(self) -> None:
        curation = {"episode_id": "4", "start_cell": [2, 3]}
        first = build_curated_plan(
            _inventory(),
            _instance(),
            CURATED_FIRE_PROFILES["stable"],
            seed=7,
            curation=curation,
        )
        second = build_curated_plan(
            _inventory(),
            _instance(),
            CURATED_FIRE_PROFILES["stable"],
            seed=7,
            curation=curation,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["plan_hash"], curated_plan_hash(first))
        self.assertTrue(first["plan_id"].startswith(
            "SceneA_route_contrast_stable_"
        ))
        self.assertEqual(first["num_initial_ignitions"], 1)
        self.assertEqual(first["ignitions"][0]["ignite_time_s"], 0.0)
        self.assertEqual(first["ignitions"][0]["sustain_s"], 300.0)
        self.assertEqual(
            first["propagation_rules"]["limit_flame_to_source_envelope"],
            1,
        )

    def test_hash_changes_when_curation_changes(self) -> None:
        first = build_curated_plan(
            _inventory(),
            _instance(),
            CURATED_FIRE_PROFILES["stable"],
            seed=7,
            curation={"episode_id": "4"},
        )
        second = build_curated_plan(
            _inventory(),
            _instance(),
            CURATED_FIRE_PROFILES["stable"],
            seed=7,
            curation={"episode_id": "5"},
        )

        self.assertNotEqual(first["plan_hash"], second["plan_hash"])
        self.assertNotEqual(first["plan_id"], second["plan_id"])

    def test_plan_write_is_idempotent_and_refuses_collision(self) -> None:
        plan = build_curated_plan(
            _inventory(),
            _instance(),
            CURATED_FIRE_PROFILES["dynamic"],
            seed=3,
            curation={"episode_id": "2"},
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = write_curated_plan(plan, root)
            second = write_curated_plan(plan, root)
            self.assertEqual(first, second)

            first.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing"):
                write_curated_plan(plan, root)


if __name__ == "__main__":
    unittest.main()
