"""Recoverable FireWorld plan pruning contracts."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.prune_fire_plans import main


def _write_plan(
    scenes_root: Path,
    scene_id: str,
    plan_id: str,
    fire_type: str,
    intensity: str,
    template_version: int,
    **extra,
) -> Path:
    path = scenes_root / scene_id / "plans" / f"{plan_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_id": plan_id,
        "scene_id": scene_id,
        "fire_type": fire_type,
        "intensity": intensity,
        "template_version": template_version,
        "schema_version": 1,
        "seed": 42,
        **extra,
    }
    path.write_text(json.dumps(payload))
    return path


class PruneFirePlansTests(unittest.TestCase):
    def test_dry_run_then_apply_keeps_latest_and_custom_route_plan(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenes_root = root / "scenes"
            outputs_root = root / "outputs" / "fire_world"
            backup_root = root / "outputs" / "backup"
            scene_id = "test_scene"
            old_id = f"{scene_id}_multi_origin_medium_old"
            new_id = f"{scene_id}_multi_origin_medium_new"
            route_id = f"{scene_id}_route_contrast_stable_safe"
            old_path = _write_plan(
                scenes_root,
                scene_id,
                old_id,
                "multi_origin",
                "medium",
                11,
            )
            new_path = _write_plan(
                scenes_root,
                scene_id,
                new_id,
                "multi_origin",
                "medium",
                12,
            )
            route_path = _write_plan(
                scenes_root,
                scene_id,
                route_id,
                "route_contrast",
                "stable",
                10,
                curation={"purpose": "safe detour"},
            )
            old_timeline = outputs_root / scene_id / old_id
            old_timeline.mkdir(parents=True)
            (old_timeline / "timeline.npz").write_bytes(b"old timeline")
            route_timeline = outputs_root / scene_id / route_id
            route_timeline.mkdir(parents=True)
            (route_timeline / "timeline.npz").write_bytes(b"safe timeline")
            index_path = outputs_root / scene_id / "asset_index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scene_id": scene_id,
                        "assets": [
                            {"plan_id": old_id},
                            {"plan_id": route_id},
                        ],
                    }
                )
            )

            args = [
                "--scenes-root",
                str(scenes_root),
                "--outputs-root",
                str(outputs_root),
                "--backup-root",
                str(backup_root),
                "--run-id",
                "test_run",
            ]
            self.assertEqual(main(args), 0)
            self.assertTrue(old_path.exists())
            self.assertTrue(old_timeline.exists())
            self.assertFalse((backup_root / "test_run").exists())

            self.assertEqual(main([*args, "--apply"]), 0)
            self.assertFalse(old_path.exists())
            self.assertFalse(old_timeline.exists())
            self.assertTrue(new_path.exists())
            self.assertTrue(route_path.exists())
            self.assertTrue(route_timeline.exists())
            backed_plan = (
                backup_root
                / "test_run"
                / "scenes"
                / scene_id
                / "plans"
                / old_path.name
            )
            self.assertTrue(backed_plan.exists())
            self.assertTrue(
                (
                    backup_root
                    / "test_run"
                    / "outputs"
                    / "fire_world"
                    / scene_id
                    / old_id
                    / "timeline.npz"
                ).exists()
            )
            assets = json.loads(index_path.read_text())["assets"]
            self.assertEqual([item["plan_id"] for item in assets], [route_id])
            manifest = json.loads(
                (backup_root / "test_run" / "manifest.json").read_text()
            )
            self.assertEqual(manifest["superseded_plan_count"], 1)
            self.assertEqual(manifest["moved_timeline_count"], 1)
            self.assertEqual(
                manifest["protected_custom_plan_ids"], [route_id]
            )


if __name__ == "__main__":
    unittest.main()
