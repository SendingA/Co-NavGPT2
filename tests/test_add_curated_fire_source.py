import unittest

from scripts.add_curated_fire_source import (
    build_augmented_plan,
    quaternion_coeffs_to_matrix,
)
from utils.fire_world.fine_tuning import curated_plan_hash


class AddCuratedFireSourceTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "schema_version": 4,
            "scene_id": "scene",
            "fire_type": "route_contrast",
            "intensity": "stable",
            "ignition_selection_mode": "curated_route_contrast",
            "ignition_selection_version": 1,
            "num_initial_ignitions": 1,
            "ignitions": [{
                "object_id": 1,
                "category": "chair",
                "position": [0.0, 0.0, 0.0],
                "ignite_time_s": 0.0,
                "source_radius_m": 0.58,
                "source_temp_c": 820.0,
                "fuel_kg": 1.0,
                "smoke_yield": 0.48,
                "sustain_s": 300.0,
                "floor_spread_scale": 1.0,
                "ignition_role": "initial",
            }],
            "propagation_rules": {"flammable_threshold": 0.4},
            "curation": {"ignitions": []},
            "plan_hash": "000000000000",
            "plan_id": "scene_route_contrast_stable_000000000000",
        }
        self.instance = {
            "instance_id": 48,
            "category": "oven and stove",
            "centroid": [-9.113, 0.63, 2.962],
            "flammability": 0.85,
            "structural": False,
            "floor_id": 0,
        }

    def test_adds_content_addressed_source_without_mutating_base(self):
        result = build_augmented_plan(
            self.base,
            self.instance,
            map_cell=(250.4, 138.4),
            red_box_map_cells=(220, 125, 282, 165),
            reference_image="reference.png",
        )
        self.assertEqual(result["num_initial_ignitions"], 2)
        self.assertEqual(result["ignitions"][-1]["object_id"], 48)
        self.assertEqual(result["plan_hash"], curated_plan_hash(result))
        self.assertTrue(result["plan_id"].endswith(result["plan_hash"]))
        self.assertEqual(len(self.base["ignitions"]), 1)

    def test_rejects_structural_or_duplicate_source(self):
        duplicate = dict(self.instance, instance_id=1)
        with self.assertRaisesRegex(ValueError, "already"):
            build_augmented_plan(
                self.base, duplicate, map_cell=(1, 1),
                red_box_map_cells=(0, 0, 2, 2), reference_image="x.png",
            )
        structural = dict(self.instance, structural=True)
        with self.assertRaisesRegex(ValueError, "structural"):
            build_augmented_plan(
                self.base, structural, map_cell=(1, 1),
                red_box_map_cells=(0, 0, 2, 2), reference_image="x.png",
            )

    def test_habitat_quaternion_coefficients_are_normalized(self):
        matrix = quaternion_coeffs_to_matrix([0.0, 0.73961, 0.0, 0.67303])
        identity = matrix.T @ matrix
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(identity[row, col], row == col, places=6)


if __name__ == "__main__":
    unittest.main()
