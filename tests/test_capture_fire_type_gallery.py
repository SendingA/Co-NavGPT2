import unittest

import numpy as np

from scripts.capture_fire_type_gallery import (
    Scenario,
    _candidate_positions,
    balanced_visibility_score,
    make_montage,
    parse_args,
    parse_preferred_camera,
    parse_preferred_object,
    parse_preferred_source,
    parse_scenario,
)


class _IdentityPathfinder:
    @staticmethod
    def snap_point(point):
        return np.asarray(point, dtype=np.float64)


def _outputs(
    flame_fraction: float, transmittance: float, *, visible_orange: bool = False
):
    height, width = 20, 20
    flame = np.zeros((height, width), dtype=np.float32)
    flame.flat[: int(height * width * flame_fraction)] = 1.0
    rgb = np.full((height, width, 3), 128, np.uint8)
    if visible_orange:
        rgb[flame > 0.03] = (230, 125, 35)
    return {
        "thermal_flame_mask": flame,
        "transmittance": np.full((height, width), transmittance, np.float32),
        "rgb_smoke": rgb,
    }


class CaptureFireTypeGalleryTest(unittest.TestCase):
    def test_parse_scenario_requires_known_type_and_three_fields(self):
        parsed = parse_scenario("multi_origin:scene:scene_multi_origin_medium_hash")
        self.assertEqual(
            parsed,
            Scenario("multi_origin", "scene", "scene_multi_origin_medium_hash"),
        )
        with self.assertRaises(Exception):
            parse_scenario("multi_origin:scene")
        with self.assertRaises(Exception):
            parse_scenario("unknown:scene:plan")

    def test_balanced_visibility_beats_no_fire_and_occluded_view(self):
        balanced = balanced_visibility_score(
            _outputs(0.03, 0.78, visible_orange=True)
        )["score"]
        no_fire = balanced_visibility_score(_outputs(0.0, 0.95))["score"]
        occluded = balanced_visibility_score(_outputs(0.2, 0.15))["score"]
        self.assertGreater(balanced, no_fire)
        self.assertGreater(balanced, occluded)

    def test_occluded_thermal_flame_is_not_counted_as_visible_rgb_fire(self):
        visible = balanced_visibility_score(
            _outputs(0.03, 0.78, visible_orange=True)
        )
        hidden = balanced_visibility_score(_outputs(0.03, 0.78))
        self.assertGreater(visible["rgb_fire_fraction"], 0.0)
        self.assertEqual(hidden["rgb_fire_fraction"], 0.0)
        self.assertGreater(visible["score"], hidden["score"])

    def test_montage_has_two_by_two_layout_and_labels(self):
        items = [
            (np.zeros((40, 60, 3), np.uint8), f"type {index}", "scene")
            for index in range(4)
        ]
        montage = make_montage(items, tile_size=(80, 50))
        self.assertEqual(montage.shape, (2 * (50 + 72), 2 * 80, 3))
        self.assertGreater(int(montage[:72].max()), 0)

    def test_pose_candidates_retain_complete_2_8_meter_view_ring(self):
        source = np.array([9.770256, 2.9, 4.948815], dtype=np.float64)
        candidates = _candidate_positions(_IdentityPathfinder(), source)
        clear_stove_view = np.array(
            [source[0] - 2.8, source[1], source[2]], dtype=np.float64
        )
        self.assertEqual(len(candidates), 64)
        self.assertTrue(
            any(np.allclose(point, clear_stove_view) for point in candidates)
        )

    def test_publication_defaults_are_full_scale_and_high_sample_count(self):
        args = parse_args([])
        self.assertEqual(args.render_scale, 1.0)
        self.assertEqual(args.n_steps, 64)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.smoke_density, 0.6)
        self.assertEqual(args.smoke_noise_strength, 0.24)
        self.assertEqual(args.flame_noise_strength, 0.75)
        self.assertEqual(args.flame_edge_break, 1.05)
        self.assertEqual(args.flame_glow_ksize, 21)

    def test_low_smoke_capture_controls_are_parsed(self):
        args = parse_args(
            ["--smoke-density", "0.35", "--smoke-noise-strength", "0.16"]
        )
        self.assertEqual(args.smoke_density, 0.35)
        self.assertEqual(args.smoke_noise_strength, 0.16)

    def test_preferred_source_and_flame_style_controls_are_parsed(self):
        args = parse_args(
            [
                "--preferred-source",
                "bedroom_textile:bed",
                "--flame-noise-strength",
                "1.05",
                "--flame-edge-break",
                "1.3",
                "--flame-color-jitter",
                "0.28",
                "--flame-glow-ksize",
                "11",
                "--flame-glow-gain",
                "0.1",
                "--flame-surface-reveal",
                "0.2",
                "--flame-highlight-compression",
                "1.4",
            ]
        )
        self.assertEqual(args.preferred_source, [("bedroom_textile", "bed")])
        self.assertEqual(args.flame_noise_strength, 1.05)
        self.assertEqual(args.flame_edge_break, 1.3)
        self.assertEqual(args.flame_glow_ksize, 11)
        self.assertEqual(args.flame_surface_reveal, 0.2)

    def test_preferred_source_requires_known_fire_type(self):
        self.assertEqual(
            parse_preferred_source("living_room_electric:tv"),
            ("living_room_electric", "tv"),
        )
        with self.assertRaises(Exception):
            parse_preferred_source("unknown:tv")

    def test_preferred_object_parses_exact_semantic_id(self):
        self.assertEqual(
            parse_preferred_object("living_room_electric:559"),
            ("living_room_electric", 559),
        )
        args = parse_args(
            ["--preferred-object", "kitchen_grease_fire:643"]
        )
        self.assertEqual(args.preferred_object, [("kitchen_grease_fire", 643)])
        with self.assertRaises(Exception):
            parse_preferred_object("living_room_electric:not-an-id")

    def test_preferred_camera_parses_reproducible_xyz(self):
        self.assertEqual(
            parse_preferred_camera("kitchen_grease_fire:6.97,2.064,4.949"),
            ("kitchen_grease_fire", (6.97, 2.064, 4.949)),
        )
        args = parse_args(
            ["--preferred-camera", "kitchen_grease_fire:6.97,2.064,4.949"]
        )
        self.assertEqual(
            args.preferred_camera,
            [("kitchen_grease_fire", (6.97, 2.064, 4.949))],
        )
        with self.assertRaises(Exception):
            parse_preferred_camera("kitchen_grease_fire:1,2")

    def test_preferred_view_target_uses_same_xyz_parser(self):
        args = parse_args(
            [
                "--preferred-view-target",
                "living_room_electric:-3.79,0.539,-4.877",
            ]
        )
        self.assertEqual(
            args.preferred_view_target,
            [("living_room_electric", (-3.79, 0.539, -4.877))],
        )


if __name__ == "__main__":
    unittest.main()
