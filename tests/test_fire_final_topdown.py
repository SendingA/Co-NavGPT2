"""Headless tests for final FireWorld top-down coordinate and overlay logic."""
from __future__ import annotations

import unittest

import numpy as np

from scripts.render_fire_final_topdown import (
    _compose_overlay,
    _draw_person_marker,
    _fractal_noise,
    _person_goal_from_dataset,
    _sample_xz_field,
    _source_caption,
    _world_to_pixel,
)


class FireFinalTopdownTests(unittest.TestCase):
    def test_world_corners_map_to_image_corners(self) -> None:
        bounds = (-2.0, -1.0, 4.0, 3.0)
        self.assertEqual(_world_to_pixel(-2.0, -1.0, bounds, (401, 601)), (0, 0))
        self.assertEqual(
            _world_to_pixel(4.0, 3.0, bounds, (401, 601)),
            (600, 400),
        )

    def test_xz_sampling_preserves_axis_orientation(self) -> None:
        field_xz = np.zeros((3, 4), dtype=np.float32)
        field_xz[2, 3] = 1.0
        sampled = _sample_xz_field(
            field_xz,
            origin_xyz=(0.0, 0.0, 0.0),
            voxel_m=1.0,
            bounds_xz=(0.0, 0.0, 2.0, 3.0),
            image_hw=(4, 3),
        )
        self.assertEqual(int(np.argmax(sampled)), 11)
        self.assertEqual(sampled[-1, -1], 1.0)

    def test_overlay_changes_only_rendered_geometry(self) -> None:
        rgb = np.full((3, 3, 3), 100, dtype=np.uint8)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        flame = np.ones((3, 3), dtype=np.float32)
        smoke = np.ones((3, 3), dtype=np.float32)
        result = _compose_overlay(rgb, mask, flame, smoke)

        np.testing.assert_array_equal(result[0, 0], rgb[0, 0])
        self.assertGreater(int(result[1, 1, 0]), int(rgb[1, 1, 0]))
        self.assertLess(int(result[1, 1, 2]), int(rgb[1, 1, 2]))

    def test_overlay_keeps_underlying_scene_visible_at_full_hazard(self) -> None:
        dark = np.full((64, 64, 3), 45, dtype=np.uint8)
        light = np.full((64, 64, 3), 185, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=bool)
        flame = np.ones((64, 64), dtype=np.float32)
        smoke = np.ones((64, 64), dtype=np.float32)
        dark_result = _compose_overlay(dark, mask, flame, smoke)
        light_result = _compose_overlay(light, mask, flame, smoke)

        # If the overlay were opaque, the two different textures would become
        # identical.  Preserve a substantial portion of their contrast.
        retained_contrast = np.mean(
            light_result.astype(np.float32) - dark_result.astype(np.float32)
        )
        self.assertGreater(retained_contrast, 24.0)

    def test_smoke_is_visible_without_flame(self) -> None:
        rgb = np.full((64, 64, 3), 180, dtype=np.uint8)
        result = _compose_overlay(
            rgb,
            np.ones((64, 64), dtype=bool),
            np.zeros((64, 64), dtype=np.float32),
            np.full((64, 64), 0.30, dtype=np.float32),
        )
        self.assertLess(float(np.mean(result)), float(np.mean(rgb)) - 25.0)

    def test_visual_texture_is_deterministic_and_nonuniform(self) -> None:
        first = _fractal_noise((96, 128), seed=17)
        second = _fractal_noise((96, 128), seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(float(np.std(first)), 0.08)

    def test_source_caption_supports_legacy_plan_without_region(self) -> None:
        self.assertEqual(
            _source_caption(2, {"category": "shelf"}),
            "S2  shelf",
        )
        self.assertEqual(
            _source_caption(3, {"category": "lamp", "region_id": 5}),
            "S3  lamp  region 5",
        )

    def test_person_goal_and_marker_use_native_world_position(self) -> None:
        dataset = {
            "episodes": [
                {
                    "episode_id": "10",
                    "scene_id": "data/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb",
                    "object_category": "person",
                }
            ],
            "goals_by_category": {
                "Nfvxx8J5NCo.basis.glb_person": [
                    {
                        "object_id": "person_0",
                        "position": [1.0, 0.2, 1.0],
                    }
                ]
            },
        }
        goal = _person_goal_from_dataset(
            dataset,
            scene_id="Nfvxx8J5NCo",
            episode_id="10",
        )
        rgb = np.zeros((101, 101, 3), dtype=np.uint8)
        marked, pixel = _draw_person_marker(
            rgb,
            goal,
            bounds_xz=(0.0, 0.0, 2.0, 2.0),
        )
        self.assertEqual(pixel, [50, 50])
        self.assertGreater(int(np.sum(marked[45:56, 45:56])), 0)


if __name__ == "__main__":
    unittest.main()
