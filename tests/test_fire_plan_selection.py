"""Tests for scene-aware FireWorld plan selection."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from utils.fire_world.plan_selection import (
    discover_runnable_fire_scenes,
    find_scene_for_plan,
    scene_id_from_config,
    select_fire_plan,
)
from utils.risk.runtime import _active_fire_plan_id


class FirePlanSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.scenes_root = self.root / "scenes"
        self.out_root = self.root / "outputs" / "fire_world"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _add_plan(
        self,
        scene_id: str,
        fire_type: str,
        intensity: str,
        suffix: str,
        *,
        template_version: int,
        seed: int = 42,
        timeline: bool = True,
    ) -> str:
        plan_id = f"{scene_id}_{fire_type}_{intensity}_{suffix}"
        plan_path = self.scenes_root / scene_id / "plans" / f"{plan_id}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(
                {
                    "plan_id": plan_id,
                    "scene_id": scene_id,
                    "fire_type": fire_type,
                    "intensity": intensity,
                    "template_version": template_version,
                    "seed": seed,
                }
            ),
            encoding="utf-8",
        )
        if timeline:
            timeline_path = (
                self.out_root / scene_id / plan_id / "timeline.npz"
            )
            timeline_path.parent.mkdir(parents=True, exist_ok=True)
            timeline_path.touch()
        return plan_id

    def test_auto_defaults_to_multi_origin_and_latest_template(self) -> None:
        scene = "sceneA"
        self._add_plan(
            scene,
            "multi_origin",
            "medium",
            "older",
            template_version=3,
        )
        expected = self._add_plan(
            scene,
            "multi_origin",
            "medium",
            "latest",
            template_version=10,
        )
        self._add_plan(
            scene,
            "kitchen_grease_fire",
            "medium",
            "newer-template-but-lower-priority",
            template_version=99,
        )
        self._add_plan(
            scene,
            "multi_origin",
            "severe",
            "wrong-intensity",
            template_version=100,
        )

        selected = select_fire_plan(
            scene,
            scenes_root=self.scenes_root,
            out_root=self.out_root,
        )

        self.assertEqual(selected.plan_id, expected)
        self.assertEqual(selected.intensity, "medium")
        self.assertEqual(selected.template_version, 10)

    def test_auto_falls_back_and_ignores_unbaked_plan(self) -> None:
        scene = "sceneB"
        self._add_plan(
            scene,
            "multi_origin",
            "medium",
            "not-baked",
            template_version=20,
            timeline=False,
        )
        expected = self._add_plan(
            scene,
            "kitchen_grease_fire",
            "medium",
            "baked",
            template_version=2,
        )

        selected = select_fire_plan(
            scene,
            scenes_root=self.scenes_root,
            out_root=self.out_root,
        )

        self.assertEqual(selected.plan_id, expected)
        self.assertEqual(selected.fire_type, "kitchen_grease_fire")

    def test_explicit_selection_is_exact_and_requires_timeline(self) -> None:
        scene = "sceneC"
        explicit = self._add_plan(
            scene,
            "bedroom_textile",
            "light",
            "exact",
            template_version=1,
        )
        missing_timeline = self._add_plan(
            scene,
            "multi_origin",
            "medium",
            "missing",
            template_version=10,
            timeline=False,
        )

        selected = select_fire_plan(
            scene,
            plan_id=explicit,
            intensity="severe",
            fire_type="multi_origin",
            scenes_root=self.scenes_root,
            out_root=self.out_root,
        )
        self.assertEqual(selected.plan_id, explicit)
        with self.assertRaisesRegex(FileNotFoundError, "timeline"):
            select_fire_plan(
                scene,
                plan_id=missing_timeline,
                scenes_root=self.scenes_root,
                out_root=self.out_root,
            )

    def test_discovery_returns_only_runnable_scenes(self) -> None:
        ready_id = self._add_plan(
            "ready",
            "multi_origin",
            "medium",
            "ready",
            template_version=1,
        )
        self._add_plan(
            "unbaked",
            "multi_origin",
            "medium",
            "unbaked",
            template_version=1,
            timeline=False,
        )
        self._add_plan(
            "wrong-intensity",
            "multi_origin",
            "light",
            "light",
            template_version=1,
        )

        selections = discover_runnable_fire_scenes(
            scenes_root=self.scenes_root,
            out_root=self.out_root,
        )

        self.assertEqual(list(selections), ["ready"])
        self.assertEqual(selections["ready"].plan_id, ready_id)

    def test_scene_and_active_plan_helpers(self) -> None:
        plan_id = self._add_plan(
            "sceneD",
            "multi_origin",
            "medium",
            "lookup",
            template_version=1,
        )
        config = SimpleNamespace(
            habitat=SimpleNamespace(
                simulator=SimpleNamespace(
                    scene="/dataset/sceneD/sceneD.basis.glb"
                )
            )
        )
        self.assertEqual(scene_id_from_config(config), "sceneD")
        self.assertEqual(
            find_scene_for_plan(plan_id, scenes_root=self.scenes_root),
            "sceneD",
        )
        self.assertEqual(
            _active_fire_plan_id(
                SimpleNamespace(
                    fire_world_plan_id="configured",
                    fire_world_active_plan_id="selected",
                )
            ),
            "selected",
        )
        self.assertEqual(
            _active_fire_plan_id(
                SimpleNamespace(fire_world_plan_id="configured")
            ),
            "configured",
        )


if __name__ == "__main__":
    unittest.main()
