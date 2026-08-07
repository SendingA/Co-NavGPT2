"""Semantic FireWorld plan-ID generation and migration tests."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.migrate_fire_plan_ids import (
    apply_migration,
    discover_plan_renames,
)
from utils.fire_world.pipeline_runner import validate_timeline
from utils.fire_world.plan_ids import (
    extract_plan_hash,
    semantic_plan_id,
)


class SemanticPlanIdTests(unittest.TestCase):
    def test_semantic_id_preserves_meaning_and_hash(self):
        plan_id = semantic_plan_id(
            "Nfvxx8J5NCo",
            "kitchen grease fire",
            "Severe",
            "0123456789ab",
        )
        self.assertEqual(
            plan_id,
            "Nfvxx8J5NCo_kitchen_grease_fire_severe_0123456789ab",
        )
        self.assertEqual(extract_plan_hash(plan_id), "0123456789ab")
        self.assertEqual(extract_plan_hash("0123456789ab"), "0123456789ab")

    def test_rejects_invalid_hash(self):
        with self.assertRaisesRegex(ValueError, "12 lowercase hex"):
            semantic_plan_id("Scene", "type", "light", "not-a-hash")


class PlanIdMigrationTests(unittest.TestCase):
    def test_migrates_plan_timeline_and_sidecar_without_rebaking(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scenes_root = root / "scenes"
            outputs_root = root / "outputs" / "fire_world"
            plans_dir = scenes_root / "SceneA" / "plans"
            plans_dir.mkdir(parents=True)
            old_id = "0123456789ab"
            old_plan_path = plans_dir / f"{old_id}.json"
            plan = {
                "schema_version": 3,
                "plan_id": old_id,
                "scene_id": "SceneA",
                "fire_type": "bedroom_textile",
                "intensity": "medium",
                "world_aabb": [0.0, 0.0, 0.0, 0.4, 0.4, 0.4],
                "duration_s": 2.0,
            }
            old_plan_path.write_text(json.dumps(plan))

            timeline_dir = outputs_root / "SceneA" / old_id
            timeline_dir.mkdir(parents=True)
            fields = np.zeros((3, 2, 2, 2), dtype=np.float16)
            times = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
            embedded_meta = {
                "plan_id": old_id,
                "voxel_m": 0.2,
                "dt": 0.5,
                "save_dt": 1.0,
                "n_frames": 3,
            }
            np.savez_compressed(
                timeline_dir / "timeline.npz",
                flame=fields,
                smoke=fields,
                temp=fields,
                times=times,
                meta_json=np.array(json.dumps(embedded_meta)),
            )
            (timeline_dir / "timeline_meta.json").write_text(
                json.dumps(embedded_meta)
            )

            renames = discover_plan_renames(scenes_root, outputs_root)
            new_id = (
                "SceneA_bedroom_textile_medium_0123456789ab"
            )
            self.assertEqual(renames[0].new_plan_id, new_id)
            migration_map = outputs_root / "plan_id_migration.json"
            summary = apply_migration(
                renames,
                outputs_root,
                migration_map,
            )

            self.assertEqual(summary["plan_files_changed"], 1)
            self.assertEqual(summary["timeline_dirs_changed"], 1)
            new_plan_path = plans_dir / f"{new_id}.json"
            migrated_plan = json.loads(new_plan_path.read_text())
            self.assertEqual(migrated_plan["plan_id"], new_id)
            self.assertEqual(migrated_plan["plan_hash"], old_id)
            self.assertEqual(migrated_plan["schema_version"], 4)
            self.assertFalse(old_plan_path.exists())

            new_timeline = outputs_root / "SceneA" / new_id / "timeline.npz"
            self.assertTrue(new_timeline.is_file())
            sidecar = json.loads(
                new_timeline.with_name("timeline_meta.json").read_text()
            )
            self.assertEqual(sidecar["plan_id"], new_id)
            self.assertEqual(sidecar["plan_hash"], old_id)
            valid, reason, _ = validate_timeline(
                new_timeline,
                migrated_plan,
                voxel_m=0.2,
                dt=0.5,
                save_dt=1.0,
            )
            self.assertTrue(valid, reason)
            self.assertTrue(migration_map.is_file())
            first_map = json.loads(migration_map.read_text())
            self.assertEqual(first_map["plans"][0]["old_plan_id"], old_id)

            rerun = discover_plan_renames(scenes_root, outputs_root)
            second = apply_migration(rerun, outputs_root, migration_map)
            self.assertEqual(second["plan_files_changed"], 0)
            self.assertEqual(second["timeline_dirs_changed"], 0)
            second_map = json.loads(migration_map.read_text())
            self.assertEqual(second_map["plans"][0]["old_plan_id"], old_id)


if __name__ == "__main__":
    unittest.main()
