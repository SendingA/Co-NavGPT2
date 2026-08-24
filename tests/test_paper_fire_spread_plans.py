import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_paper_fire_spread_plans import (
    ROOT,
    SPECS,
    build_plan,
    validate_spread_plan,
)


class PaperFireSpreadPlanTests(unittest.TestCase):
    def test_all_specs_build_deterministically_and_validate(self):
        expected_counts = {
            "kitchen_grease_fire": 4,
            "bedroom_textile": 11,
            "living_room_electric": 4,
            "multi_origin": 8,
        }
        for spec in SPECS:
            first = build_plan(spec, scenes_root=ROOT / "scenes")
            second = build_plan(spec, scenes_root=ROOT / "scenes")
            self.assertEqual(first, second)
            self.assertEqual(
                len(first["ignitions"]), expected_counts[spec.fire_type]
            )
            validate_spread_plan(first)

    def test_secondary_sources_reference_earlier_parents(self):
        for spec in SPECS:
            plan = build_plan(spec, scenes_root=ROOT / "scenes")
            by_id = {item["object_id"]: item for item in plan["ignitions"]}
            for ignition in plan["ignitions"]:
                parent_id = ignition.get("parent_object_id")
                if parent_id is None:
                    self.assertEqual(ignition["ignite_time_s"], 0.0)
                    continue
                self.assertLess(
                    by_id[parent_id]["ignite_time_s"],
                    ignition["ignite_time_s"],
                )
                self.assertGreater(ignition["spread_distance_m"], 0.0)

    def test_builder_does_not_modify_base_plan(self):
        spec = SPECS[0]
        base_path = (
            ROOT / "scenes" / spec.scene_id / "plans" / f"{spec.base_plan_id}.json"
        )
        before = base_path.read_bytes()
        build_plan(spec, scenes_root=ROOT / "scenes")
        self.assertEqual(base_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
