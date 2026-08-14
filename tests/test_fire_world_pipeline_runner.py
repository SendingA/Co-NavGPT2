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
    discover_existing_plan_tasks,
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

    def test_existing_plan_discovery_keeps_persisted_versions(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            scene_root = root / "SceneA"
            plan_root = scene_root / "plans"
            structural_root = scene_root / "structural"
            plan_root.mkdir(parents=True)
            structural_root.mkdir()
            structural = {}
            for name in ("walls", "floors", "ceilings"):
                path = structural_root / f"{name}.npy"
                np.save(path, np.zeros((2, 2, 2), dtype=bool))
                structural[f"{name[:-1]}_voxel_path"] = str(path)
            inventory = {
                "schema_version": 2,
                "scene_id": "SceneA",
                "world_aabb": [0.0, 0.0, 0.0, 0.4, 0.4, 0.4],
                "instances": [{"instance_id": 1}],
                "structural": structural,
            }
            (scene_root / "inventory.json").write_text(json.dumps(inventory))
            for version in (7, 8):
                plan_id = f"SceneA_multi_origin_light_version{version}"
                plan = {
                    "plan_id": plan_id,
                    "scene_id": "SceneA",
                    "fire_type": "multi_origin",
                    "intensity": "light",
                    "seed": 42,
                    "template_version": version,
                    "duration_s": 2.0,
                    "world_aabb": inventory["world_aabb"],
                }
                (plan_root / f"{plan_id}.json").write_text(json.dumps(plan))

            tasks, records = discover_existing_plan_tasks(
                root,
                voxel_m=0.2,
                dt=0.5,
                save_dt=1.0,
            )

        self.assertEqual(len(tasks), 2)
        self.assertEqual(
            {task["template_version"] for task in tasks},
            {7, 8},
        )
        self.assertEqual({record["status"] for record in records}, {"ready"})
        self.assertTrue(all("estimate" in task for task in tasks))

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

            sidecar_meta = dict(meta, plan_id="semantic_testplan")
            path.with_name("timeline_meta.json").write_text(
                json.dumps(sidecar_meta)
            )
            semantic_plan = dict(plan, plan_id="semantic_testplan")
            valid, reason, _ = validate_timeline(
                path,
                semantic_plan,
                voxel_m=0.2,
                dt=0.5,
                save_dt=1.0,
            )
            self.assertTrue(valid, reason)

            np.savez_compressed(
                path,
                flame=fields,
                smoke=fields,
                temp=fields,
                times=times,
                meta=np.array([json.dumps(meta)], dtype=object),
            )
            valid, reason, _ = validate_timeline(
                path,
                semantic_plan,
                voxel_m=0.2,
                dt=0.5,
                save_dt=1.0,
            )
            self.assertTrue(valid, reason)

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

        objects = [
            instance(index, "bed", 0.1 * index)
            for index in range(1, 11)
        ]
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
