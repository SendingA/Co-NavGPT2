"""Semantic-category contracts for FireWorld scenario templates."""

import json
import unittest
from pathlib import Path

import numpy as np

from utils.fire_world.hm3d_semantic import read_semantic_txt
from utils.fire_world.planner import build_plan
from utils.fire_world.scene_scan import MATERIAL_TABLE
from utils.fire_world.templates import (
    TEMPLATE_CATEGORY_GROUPS,
    TEMPLATES,
    _pick_primary,
    _pick_secondary,
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

    def test_primary_and_secondary_selection_cannot_use_unlisted_category(self):
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
        self.assertIsNone(
            _pick_primary(
                objects,
                rng,
                preferred_cats=["tv"],
                fallback_cats=["monitor"],
            )
        )
        secondaries = _pick_secondary(
            objects,
            objects[1],
            rng,
            radius_m=2.0,
            n=1,
            allowed_cats=("tv",),
        )
        self.assertEqual(secondaries, [])

    def test_generated_plan_categories_match_semantic_ids_and_template_groups(self):
        for inventory_path in sorted(SCENES.glob("*/inventory.json")):
            inventory = json.loads(inventory_path.read_text())
            scene_id = inventory["scene_id"]
            semantic_by_id = {
                row.instance_id: row.category.lower().strip()
                for row in self.rows_by_scene[scene_id]
            }
            for template_name in TEMPLATES:
                with self.subTest(scene=scene_id, template=template_name):
                    plan = build_plan(
                        inventory,
                        fire_type=template_name,
                        intensity="severe",
                        seed=7,
                    )
                    allowed_groups = TEMPLATE_CATEGORY_GROUPS.get(template_name)
                    for index, ignition in enumerate(plan["ignitions"]):
                        category = ignition["category"].lower()
                        self.assertEqual(
                            semantic_by_id[ignition["object_id"]],
                            category,
                        )
                        if allowed_groups is not None:
                            allowed = (
                                set(allowed_groups["primary"])
                                | set(allowed_groups["fallback"])
                                if index == 0
                                else set(allowed_groups["secondary"])
                            )
                            self.assertIn(category, allowed)

    def test_active_plan_matches_inventory_and_semantic_geometry(self):
        scene_id = "Nfvxx8J5NCo"
        inventory = json.loads(
            (SCENES / scene_id / "inventory.json").read_text()
        )
        plan = json.loads(
            (SCENES / scene_id / "plans" / "83679a07b632.json").read_text()
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

        corrected = next(
            ignition
            for ignition in plan["ignitions"]
            if ignition["category"] == "oven and stove"
        )
        self.assertEqual(corrected["object_id"], 48)
        self.assertTrue(
            np.allclose(
                corrected["position"],
                inventory_by_id[48]["centroid"],
                atol=1e-6,
            )
        )


if __name__ == "__main__":
    unittest.main()
