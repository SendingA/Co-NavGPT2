"""Focused tests for resumable FireWorld asset orchestration."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.fire_world.pipeline_runner import (
    ALL_FIRE_TYPES,
    ALL_INTENSITIES,
    build_scene_plans,
    discover_dataset_scenes,
    estimate_timeline_bytes,
    expected_timeline_layout,
    parse_seeds,
    parse_selection,
    scenario_matrix,
    validate_timeline,
)


class FireWorldPipelineRunnerTests(unittest.TestCase):
    def test_discovery_classifies_complete_and_incomplete_scene_dirs(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            complete = root / "val" / "00001-CompleteScene"
            incomplete = root / "val" / "00002-IncompleteScene"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            for suffix in ("basis.glb", "semantic.glb", "semantic.txt"):
                (complete / f"CompleteScene.{suffix}").write_bytes(b"x")
            (incomplete / "IncompleteScene.basis.glb").write_bytes(b"x")

            records = discover_dataset_scenes(root, ("val",))

        self.assertEqual(
            [record.scene_id for record in records],
            ["CompleteScene", "IncompleteScene"],
        )
        self.assertTrue(records[0].complete)
        self.assertFalse(records[1].complete)
        self.assertEqual(
            records[1].missing,
            (
                "IncompleteScene.semantic.glb",
                "IncompleteScene.semantic.txt",
            ),
        )

    def test_selection_and_matrix_are_stable_and_reject_unknown_values(self):
        self.assertEqual(
            parse_selection("all", ALL_FIRE_TYPES, "--fire-types"),
            ALL_FIRE_TYPES,
        )
        self.assertEqual(parse_seeds("42,7,42"), (42, 7))
        matrix = scenario_matrix(
            ("bedroom_textile", "multi_origin"),
            ("light", "severe"),
            (42,),
        )
        self.assertEqual(
            matrix,
            [
                ("bedroom_textile", "light", 42),
                ("bedroom_textile", "severe", 42),
                ("multi_origin", "light", 42),
                ("multi_origin", "severe", 42),
            ],
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_selection("unknown", ALL_INTENSITIES, "--intensities")

    @staticmethod
    def _plan():
        return {
            "plan_id": "testplan",
            "scene_id": "SyntheticScene",
            "world_aabb": [0.0, 0.0, 0.0, 0.4, 0.4, 0.4],
            "duration_s": 2.0,
        }

    def test_timeline_estimate_and_lightweight_validation(self):
        plan = self._plan()
        shape, frames = expected_timeline_layout(
            plan, voxel_m=0.2, dt=0.5, save_dt=1.0
        )
        self.assertEqual(shape, (2, 2, 2))
        self.assertEqual(frames, 3)
        estimate = estimate_timeline_bytes(
            plan, voxel_m=0.2, dt=0.5, save_dt=1.0
        )
        self.assertEqual(estimate["timeline_uncompressed_bytes"], 144)

        fields = np.zeros((3, 2, 2, 2), dtype=np.float16)
        times = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        meta = {
            "plan_id": "testplan",
            "voxel_m": 0.2,
            "dt": 0.5,
            "save_dt": 1.0,
            "n_frames": 3,
        }
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "timeline.npz"
            np.savez_compressed(
                path,
                flame=fields,
                smoke=fields,
                temp=fields,
                times=times,
                meta_json=np.array(json.dumps(meta)),
            )
            valid, reason, details = validate_timeline(
                path, plan, voxel_m=0.2, dt=0.5, save_dt=1.0
            )
            self.assertTrue(valid, reason)
            self.assertEqual(details["shape"], [3, 2, 2, 2])

            bad_meta = dict(meta, plan_id="wrong")
            np.savez_compressed(
                path,
                flame=fields,
                smoke=fields,
                temp=fields,
                times=times,
                meta_json=np.array(json.dumps(bad_meta)),
            )
            valid, reason, _ = validate_timeline(
                path, plan, voxel_m=0.2, dt=0.5, save_dt=1.0
            )
            self.assertFalse(valid)
            self.assertIn("plan_id mismatch", reason)

    def test_dry_run_builds_feasible_plans_without_writing(self):
        def instance(instance_id, category, x):
            return {
                "instance_id": instance_id,
                "object_id": instance_id,
                "category": category,
                "position": [x, 0.2, 0.2],
                "aabb_min": [x - 0.1, 0.0, 0.1],
                "aabb_max": [x + 0.1, 0.4, 0.3],
                "flammability": 0.8,
                "smoke_yield": 0.5,
                "structural": False,
                "floor_id": 0,
            }

        objects = [instance(1, "bed", 0.3), instance(2, "bed", 0.8)]
        inventory = {
            "schema_version": 2,
            "scene_id": "SyntheticScene",
            "scene_glb": "SyntheticScene.basis.glb",
            "world_aabb": [0.0, 0.0, 0.0, 1.2, 0.6, 0.6],
            "instances": objects,
            "objects": objects,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            tasks, records = build_scene_plans(
                inventory,
                scenario_matrix(
                    ("bedroom_textile",),
                    ("light", "medium", "severe"),
                    (42,),
                ),
                scenes_root=root,
                dry_run=True,
            )
            self.assertFalse(root.joinpath("SyntheticScene").exists())

        self.assertEqual(len(tasks), 3)
        self.assertEqual(
            {record["status"] for record in records},
            {"would_generate"},
        )
        self.assertTrue(all(task["plan"] is not None for task in tasks))


if __name__ == "__main__":
    unittest.main()
