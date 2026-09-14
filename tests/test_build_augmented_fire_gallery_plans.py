import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_augmented_fire_gallery_plans import (
    SPECS,
    _write_json_if_changed,
    build_plan,
    validate_gallery_plan,
)


class BuildAugmentedFireGalleryPlansTest(unittest.TestCase):
    def test_all_specs_build_stable_medium_augmented_plans(self):
        scenes_root = Path("scenes")
        for spec in SPECS:
            plan = build_plan(spec, scenes_root=scenes_root)
            validate_gallery_plan(plan)
            self.assertEqual(plan["intensity"], "medium")
            self.assertEqual(plan["base_plan_id"], spec.base_plan_id)
            self.assertEqual(
                plan["gallery_preferred_source_category"],
                spec.preferred_source_category,
            )
            self.assertEqual(
                plan["plan_id"],
                build_plan(spec, scenes_root=scenes_root)["plan_id"],
            )
            base_count = len(plan["ignitions"]) - len(spec.added_object_ids)
            if spec.replace_base_ignitions:
                self.assertEqual(base_count, 0)
                self.assertTrue(plan["gallery_replaced_base_ignitions"])
            else:
                self.assertGreater(base_count, 0)
                self.assertFalse(plan.get("gallery_replaced_base_ignitions", False))

    def test_validation_rejects_duplicate_sources(self):
        plan = build_plan(SPECS[0], scenes_root=Path("scenes"))
        plan["ignitions"].append(dict(plan["ignitions"][0]))
        plan["num_initial_ignitions"] += 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_gallery_plan(plan)

    def test_identical_plan_is_not_rewritten(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            payload = {"plan_id": "stable", "ignitions": [1, 2]}
            self.assertTrue(_write_json_if_changed(path, payload))
            first_mtime = path.stat().st_mtime_ns
            self.assertFalse(_write_json_if_changed(path, payload))
            self.assertEqual(path.stat().st_mtime_ns, first_mtime)


if __name__ == "__main__":
    unittest.main()
