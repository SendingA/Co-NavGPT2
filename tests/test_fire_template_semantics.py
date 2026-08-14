"""Semantic-category contracts for FireWorld scenario templates."""

import json
import unittest
from pathlib import Path

import numpy as np

from utils.fire_world.hm3d_semantic import read_semantic_txt
from utils.fire_world.planner import build_plan
from utils.fire_world.scene_scan import MATERIAL_TABLE
from utils.fire_world.templates import (
    MULTI_ORIGIN_AREAS_MAX,
    MULTI_ORIGIN_AREAS_MIN,
    TEMPLATE_CATEGORY_GROUPS,
    TEMPLATES,
    _initial_candidate_capacity,
    _inventory_pool,
    _pick_explicit_initials,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE_DATASETS = ROOT / "data" / "scene_datasets"
SCENES = ROOT / "scenes"


def _semantic_rows_by_scene():
    result = {}
    for path in SCENE_DATASETS.rglob("*.semantic.txt"):
        _, rows = read_semantic_txt(path)
        scene_id = path.name.removesuffix(".semantic.txt")
        result[scene_id] = rows
    return result


class FireTemplateSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows_by_scene = _semantic_rows_by_scene()
        cls.semantic_categories = {
            row.category.lower().strip()
            for rows in cls.rows_by_scene.values()
            for row in rows
        }

    def test_every_template_literal_is_a_real_semantic_category(self):
        self.assertEqual(len(self.rows_by_scene), 36)
        for template_name, groups in TEMPLATE_CATEGORY_GROUPS.items():
            for group_name, categories in groups.items():
                with self.subTest(template=template_name, group=group_name):
                    self.assertTrue(categories)
                    self.assertEqual(len(categories), len(set(categories)))
                    self.assertTrue(set(categories) <= self.semantic_categories)
        self.assertNotIn("tv_monitor", self.semantic_categories)

    def test_every_template_literal_has_explicit_material_properties(self):
        referenced = {
            category
            for groups in TEMPLATE_CATEGORY_GROUPS.values()
            for categories in groups.values()
            for category in categories
        }
        self.assertTrue(referenced <= set(MATERIAL_TABLE))
        self.assertEqual(MATERIAL_TABLE["oven and stove"], (0.85, 0.70))
        self.assertNotIn("tv_monitor", MATERIAL_TABLE)

    def test_multi_origin_is_area_based_without_category_groups(self):
        self.assertNotIn("multi_origin", TEMPLATE_CATEGORY_GROUPS)

    def test_eight_source_multi_origin_is_distributed_across_areas(self):
        inventory = json.loads(
            (SCENES / "Nfvxx8J5NCo" / "inventory.json").read_text()
        )
        inventory_by_id = {
            int(item["instance_id"]): item
            for item in inventory["instances"]
        }
        plan = build_plan(
            inventory,
            fire_type="multi_origin",
            intensity="severe",
            seed=7,
            num_ignitions=8,
        )

        self.assertEqual(len(plan["ignitions"]), 8)
        policy = plan["multi_origin_area_policy"]
        self.assertGreaterEqual(
            len(policy["selected_area_ids"]), MULTI_ORIGIN_AREAS_MIN
        )
        self.assertLessEqual(
            len(policy["selected_area_ids"]), MULTI_ORIGIN_AREAS_MAX
        )
        self.assertEqual(sum(policy["sources_per_area"].values()), 8)
        self.assertTrue(
            all(
                ignition["ignition_role"] == "initial"
                and ignition["ignite_time_s"] == 0.0
                and "parent_object_id" not in ignition
                and ignition["region_id"]
                == inventory_by_id[ignition["object_id"]]["region_id"]
                and inventory_by_id[ignition["object_id"]]["flammability"]
                >= 0.4
                for ignition in plan["ignitions"]
            )
        )

    def test_initial_selection_cannot_use_unlisted_category(self):
        objects = [
            {
                "object_id": 1,
                "category": "unknown",
                "position": [0.0, 0.0, 0.0],
                "flammability": 1.0,
            },
            {
                "object_id": 2,
                "category": "chair",
                "position": [0.5, 0.0, 0.0],
                "flammability": 0.8,
            },
        ]
        rng = np.random.default_rng(1)
        with self.assertRaisesRegex(RuntimeError, "only 0 eligible"):
            _pick_explicit_initials(
                objects, "living_room_electric", 1, rng
            )

    def test_generated_plan_categories_match_semantic_ids_and_template_groups(self):
        for inventory_path in sorted(SCENES.glob("*/inventory.json")):
            inventory = json.loads(inventory_path.read_text())
            scene_id = inventory["scene_id"]
            semantic_by_id = {
                row.instance_id: row.category.lower().strip()
                for row in self.rows_by_scene[scene_id]
            }
            inventory_by_id = {
                int(item["instance_id"]): item
                for item in inventory["instances"]
            }
            for template_name in TEMPLATES:
                with self.subTest(scene=scene_id, template=template_name):
                    requested = 4 if template_name == "multi_origin" else 1
                    capacity = _initial_candidate_capacity(
                        _inventory_pool(inventory), template_name
                    )
                    if capacity < requested:
                        with self.assertRaises(RuntimeError):
                            build_plan(
                                inventory,
                                fire_type=template_name,
                                intensity="severe",
                                seed=7,
                                num_ignitions=requested,
                            )
                        continue
                    plan = build_plan(
                        inventory,
                        fire_type=template_name,
                        intensity="severe",
                        seed=7,
                        num_ignitions=requested,
                    )
                    allowed_groups = TEMPLATE_CATEGORY_GROUPS.get(template_name)
                    for ignition in plan["ignitions"]:
                        category = ignition["category"].lower()
                        self.assertEqual(
                            semantic_by_id[ignition["object_id"]],
                            category,
                        )
                        if allowed_groups is not None:
                            allowed = (
                                set(allowed_groups["primary"])
                                | set(allowed_groups["fallback"])
                            )
                            self.assertIn(category, allowed)
                        else:
                            item = inventory_by_id[ignition["object_id"]]
                            self.assertGreaterEqual(
                                float(item["flammability"]), 0.4
                            )
                            self.assertEqual(
                                ignition["region_id"], item["region_id"]
                            )
                        self.assertEqual(
                            ignition["ignition_role"], "initial"
                        )
                        self.assertEqual(ignition["ignite_time_s"], 0.0)
                        self.assertNotIn("parent_object_id", ignition)

    def test_current_template_plan_matches_inventory_and_semantic_geometry(self):
        scene_id = "Nfvxx8J5NCo"
        inventory = json.loads(
            (SCENES / scene_id / "inventory.json").read_text()
        )
        plan = build_plan(
            inventory,
            fire_type="bedroom_textile",
            intensity="severe",
            seed=42,
        )
        inventory_by_id = {
            int(item["instance_id"]): item for item in inventory["instances"]
        }
        semantic_by_id = {
            row.instance_id: row.category.lower().strip()
            for row in self.rows_by_scene[scene_id]
        }

        for ignition in plan["ignitions"]:
            object_id = int(ignition["object_id"])
            item = inventory_by_id[object_id]
            self.assertEqual(ignition["category"].lower(), semantic_by_id[object_id])
            self.assertEqual(ignition["category"].lower(), item["category"].lower())
            position = np.asarray(ignition["position"], dtype=np.float64)
            aabb_min = np.asarray(item["aabb_min"], dtype=np.float64)
            aabb_max = np.asarray(item["aabb_max"], dtype=np.float64)
            self.assertTrue(np.all(position >= aabb_min))
            self.assertTrue(np.all(position <= aabb_max))


if __name__ == "__main__":
    unittest.main()
