"""Initial-only ignition selection contracts for FireWorld plans."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from utils.fire_world.planner import (
    PLAN_SCHEMA_VERSION,
    build_plan,
    plan_hash_for,
    plan_id_for,
    scenario_rng_seed,
    write_plan,
)
from utils.fire_world.plan_ids import semantic_plan_id
from utils.fire_world.templates import (
    IGNITION_SELECTION_VERSION,
    MULTI_ORIGIN_AREAS_MAX,
    MULTI_ORIGIN_AREAS_MIN,
    MULTI_ORIGIN_SOURCES_PER_AREA_MAX,
    MULTI_ORIGIN_SOURCES_PER_AREA_MIN,
    TEMPLATE_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
SCENE_ID = "Nfvxx8J5NCo"


def _instance(
    object_id,
    category,
    x,
    z=0.0,
    flammability=0.8,
    region_id=None,
):
    instance = {
        "instance_id": object_id,
        "category": category,
        "centroid": [x, 0.5, z],
        "aabb_min": [x - 0.2, 0.0, z - 0.2],
        "aabb_max": [x + 0.2, 1.0, z + 0.2],
        "flammability": flammability,
        "smoke_yield": 0.6,
        "structural": False,
    }
    if region_id is not None:
        instance["region_id"] = region_id
    return instance


def _inventory(instances):
    return {
        "scene_id": "synthetic",
        "scene_glb": "synthetic.glb",
        "world_aabb": [-1.0, 0.0, -1.0, 12.0, 2.0, 2.0],
        "objects": [],
        "instances": instances,
    }


class FireIgnitionCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(
            (ROOT / "scenes" / SCENE_ID / "inventory.json").read_text()
        )

    @staticmethod
    def _assert_initial_only(plan, expected_count=None):
        ignitions = plan["ignitions"]
        if expected_count is not None:
            assert len(ignitions) == expected_count
        assert len(ignitions) == plan["num_initial_ignitions"]
        assert all(item["ignition_role"] == "initial" for item in ignitions)
        assert all(item["ignite_time_s"] == 0.0 for item in ignitions)
        assert len({item["object_id"] for item in ignitions}) == len(ignitions)
        for item in ignitions:
            assert "parent_object_id" not in item
            assert "secondary_rank" not in item

    def test_omitted_count_uses_initial_only_intensity_preset(self):
        key = (
            f"{SCENE_ID}|bedroom_textile|severe|7|"
            f"tpl{TEMPLATE_VERSION}|"
            f"initial-only-v{IGNITION_SELECTION_VERSION}"
        )
        expected_hash = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        expected = semantic_plan_id(
            SCENE_ID,
            "bedroom_textile",
            "severe",
            expected_hash,
        )

        plan = build_plan(
            self.inventory,
            "bedroom_textile",
            "severe",
            seed=7,
        )

        self.assertEqual(plan["plan_id"], expected)
        self.assertEqual(plan["plan_hash"], expected_hash)
        self.assertEqual(
            plan_hash_for(
                SCENE_ID,
                "bedroom_textile",
                "severe",
                seed=7,
            ),
            expected_hash,
        )
        self.assertEqual(
            plan_id_for(
                SCENE_ID,
                "bedroom_textile",
                "severe",
                seed=7,
            ),
            expected,
        )
        self.assertNotIn("num_initial_ignitions_requested", plan)
        self.assertEqual(plan["ignition_selection_mode"], "initial_only")
        self.assertEqual(
            plan["ignition_selection_version"],
            IGNITION_SELECTION_VERSION,
        )
        self.assertGreaterEqual(plan["num_initial_ignitions"], 7)
        self.assertLessEqual(plan["num_initial_ignitions"], 10)
        self.assertEqual(
            plan["default_initial_ignition_range"],
            {"min": 7, "max": 10},
        )
        self._assert_initial_only(plan)

    def test_default_intensity_source_count_tiers(self):
        for intensity, expected_min, expected_max in (
            ("light", 2, 2),
            ("medium", 4, 6),
            ("severe", 7, 10),
        ):
            with self.subTest(intensity=intensity):
                plan = build_plan(
                    self.inventory,
                    "bedroom_textile",
                    intensity,
                    seed=19,
                )
                self.assertGreaterEqual(
                    plan["num_initial_ignitions"], expected_min
                )
                self.assertLessEqual(
                    plan["num_initial_ignitions"], expected_max
                )
                self.assertEqual(
                    plan["default_initial_ignition_range"],
                    {"min": expected_min, "max": expected_max},
                )
                self._assert_initial_only(plan)

    def test_scenario_rng_is_stable_and_scene_specific(self):
        seed_a = scenario_rng_seed(
            "scene_a", "bedroom_textile", "medium", 42
        )
        self.assertEqual(
            seed_a,
            scenario_rng_seed(
                "scene_a", "bedroom_textile", "medium", 42
            ),
        )
        self.assertNotEqual(
            seed_a,
            scenario_rng_seed(
                "scene_b", "bedroom_textile", "medium", 42
            ),
        )
        self.assertNotEqual(
            seed_a,
            scenario_rng_seed(
                "scene_a", "bedroom_textile", "severe", 42
            ),
        )

    def test_explicit_count_is_exact_total_plan_count(self):
        requested = 4
        plan = build_plan(
            self.inventory,
            "bedroom_textile",
            "severe",
            seed=7,
            num_ignitions=requested,
        )

        self.assertEqual(plan["num_initial_ignitions_requested"], requested)
        self._assert_initial_only(plan, expected_count=requested)

    def test_planner_does_not_require_or_schedule_nearby_children(self):
        scene = _inventory([
            _instance(1, "bed", 0.0),
            _instance(2, "bed", 10.0),
            # This nearby flammable object is deliberately outside the
            # bedroom initial category pool. The solver may ignite it later,
            # but the plan must neither require nor list it.
            _instance(3, "chair", 0.4),
        ])
        plan = build_plan(
            scene,
            "bedroom_textile",
            "severe",
            seed=11,
            num_ignitions=2,
        )

        self._assert_initial_only(plan, expected_count=2)
        self.assertEqual(
            {item["object_id"] for item in plan["ignitions"]},
            {1, 2},
        )
        self.assertNotIn(3, {
            item["object_id"] for item in plan["ignitions"]
        })

    def test_count_is_deterministic_and_part_of_plan_id(self):
        plan_a = build_plan(
            self.inventory,
            "kitchen_grease_fire",
            "severe",
            seed=13,
            num_ignitions=1,
        )
        plan_b = build_plan(
            self.inventory,
            "kitchen_grease_fire",
            "severe",
            seed=13,
            num_ignitions=1,
        )
        plan_other_count = build_plan(
            self.inventory,
            "kitchen_grease_fire",
            "severe",
            seed=13,
            num_ignitions=2,
        )

        self.assertEqual(plan_a, plan_b)
        self.assertNotEqual(
            plan_a["plan_id"], plan_other_count["plan_id"]
        )
        expected_key = (
            f"{SCENE_ID}|kitchen_grease_fire|severe|13|"
            f"tpl{TEMPLATE_VERSION}|"
            f"initial-only-v{IGNITION_SELECTION_VERSION}|n1"
        )
        expected_hash = hashlib.sha1(
            expected_key.encode("utf-8")
        ).hexdigest()[:12]
        expected_id = semantic_plan_id(
            SCENE_ID,
            "kitchen_grease_fire",
            "severe",
            expected_hash,
        )
        self.assertEqual(plan_a["plan_id"], expected_id)
        self.assertEqual(plan_a["plan_hash"], expected_hash)

    def test_multi_origin_honors_exact_count_with_area_stratification(self):
        with self.assertRaisesRegex(RuntimeError, "cannot distribute"):
            build_plan(
                self.inventory,
                "multi_origin",
                "light",
                seed=7,
                num_ignitions=3,
            )

        plan = build_plan(
            self.inventory,
            "multi_origin",
            "light",
            seed=7,
            num_ignitions=8,
        )
        self._assert_initial_only(plan, expected_count=8)
        self.assertEqual(
            plan["ignition_selection_mode"],
            "area_stratified_initial_only",
        )
        policy = plan["multi_origin_area_policy"]
        area_counts = {
            int(area_id): int(count)
            for area_id, count in policy["sources_per_area"].items()
        }
        self.assertGreaterEqual(len(area_counts), MULTI_ORIGIN_AREAS_MIN)
        self.assertLessEqual(len(area_counts), MULTI_ORIGIN_AREAS_MAX)
        self.assertEqual(sum(area_counts.values()), 8)
        self.assertTrue(
            all(
                MULTI_ORIGIN_SOURCES_PER_AREA_MIN
                <= count
                <= MULTI_ORIGIN_SOURCES_PER_AREA_MAX
                for count in area_counts.values()
            )
        )
        observed = {}
        for ignition in plan["ignitions"]:
            region_id = int(ignition["region_id"])
            observed[region_id] = observed.get(region_id, 0) + 1
        self.assertEqual(observed, area_counts)

    def test_multi_origin_default_uses_four_to_six_areas(self):
        for intensity in ("light", "medium", "severe"):
            with self.subTest(intensity=intensity):
                plan = build_plan(
                    self.inventory,
                    "multi_origin",
                    intensity,
                    seed=31,
                )
                policy = plan["multi_origin_area_policy"]
                area_counts = list(policy["sources_per_area"].values())
                self.assertGreaterEqual(
                    len(area_counts), MULTI_ORIGIN_AREAS_MIN
                )
                self.assertLessEqual(
                    len(area_counts), MULTI_ORIGIN_AREAS_MAX
                )
                self.assertTrue(
                    all(
                        MULTI_ORIGIN_SOURCES_PER_AREA_MIN
                        <= count
                        <= MULTI_ORIGIN_SOURCES_PER_AREA_MAX
                        for count in area_counts
                    )
                )
                self.assertEqual(sum(area_counts), len(plan["ignitions"]))
                self.assertEqual(
                    plan["default_initial_ignition_range"],
                    {"min": 4, "max": 12},
                )

    def test_multi_origin_does_not_use_a_category_whitelist(self):
        scene = _inventory([
            _instance(1, "cabinet", 0.0, region_id=10),
            _instance(2, "lamp", 2.0, region_id=20),
            _instance(3, "laptop", 4.0, region_id=30),
            _instance(4, "curtain", 6.0, region_id=40),
        ])

        plan = build_plan(
            scene,
            "multi_origin",
            "light",
            seed=9,
        )

        self.assertEqual(
            {item["category"] for item in plan["ignitions"]},
            {"cabinet", "lamp", "laptop", "curtain"},
        )
        self.assertEqual(len(plan["ignitions"]), 4)

    def test_invalid_count_and_initial_shortage_fail_clearly(self):
        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError, "positive integer"
                ):
                    build_plan(
                        self.inventory,
                        "bedroom_textile",
                        "severe",
                        seed=7,
                        num_ignitions=invalid,
                    )

        shortage = _inventory([
            _instance(1, "bed", 0.0),
            _instance(2, "lamp", 0.5),
        ])
        with self.assertRaisesRegex(
            RuntimeError, "only 1 eligible initial ignition objects"
        ):
            build_plan(
                shortage,
                "bedroom_textile",
                "severe",
                seed=7,
                num_ignitions=2,
            )

    def test_write_plan_uses_initial_only_schema_and_metadata(self):
        with TemporaryDirectory() as temp_dir:
            path = write_plan(
                self.inventory,
                "living_room_electric",
                "severe",
                seed=5,
                plans_root=Path(temp_dir),
                num_ignitions=2,
            )
            plan = json.loads(path.read_text())

        self.assertEqual(path.stem, plan["plan_id"])
        self.assertEqual(plan["schema_version"], PLAN_SCHEMA_VERSION)
        self.assertEqual(plan["num_initial_ignitions_requested"], 2)
        self._assert_initial_only(plan, expected_count=2)


if __name__ == "__main__":
    unittest.main()
