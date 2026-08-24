import unittest

import numpy as np

from scripts.capture_fire_type_gallery import (
    Scenario,
    balanced_visibility_score,
    make_montage,
    parse_args,
    parse_scenario,
)


def _outputs(flame_fraction: float, transmittance: float):
    height, width = 20, 20
    flame = np.zeros((height, width), dtype=np.float32)
    flame.flat[: int(height * width * flame_fraction)] = 1.0
    return {
        "thermal_flame_mask": flame,
        "transmittance": np.full((height, width), transmittance, np.float32),
        "rgb_smoke": np.full((height, width, 3), 128, np.uint8),
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
        balanced = balanced_visibility_score(_outputs(0.03, 0.78))["score"]
        no_fire = balanced_visibility_score(_outputs(0.0, 0.95))["score"]
        occluded = balanced_visibility_score(_outputs(0.2, 0.15))["score"]
        self.assertGreater(balanced, no_fire)
        self.assertGreater(balanced, occluded)

    def test_montage_has_two_by_two_layout_and_labels(self):
        items = [
            (np.zeros((40, 60, 3), np.uint8), f"type {index}", "scene")
            for index in range(4)
        ]
        montage = make_montage(items, tile_size=(80, 50))
        self.assertEqual(montage.shape, (2 * (50 + 72), 2 * 80, 3))
        self.assertGreater(int(montage[:72].max()), 0)

    def test_publication_defaults_are_full_scale_and_high_sample_count(self):
        args = parse_args([])
        self.assertEqual(args.render_scale, 1.0)
        self.assertEqual(args.n_steps, 64)
        self.assertEqual(args.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
