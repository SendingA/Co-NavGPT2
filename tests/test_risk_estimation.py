"""Tests for the non-privileged smoke visibility proxy."""

from __future__ import annotations

import unittest

import numpy as np

from utils.risk.estimation import estimate_smoke_from_appearance_depth


class SmokeEstimationTests(unittest.TestCase):
    def test_uniform_grey_haze_scores_above_colourful_texture(self) -> None:
        height, width = 40, 48
        grey = np.full((height, width, 3), 190, dtype=np.uint8)
        yy, xx = np.indices((height, width))
        textured = np.stack(
            [
                ((xx % 2) * 255),
                ((yy % 2) * 255),
                (((xx + yy) % 2) * 255),
            ],
            axis=-1,
        ).astype(np.uint8)
        depth = np.full((height, width), 2.0, dtype=np.float32)

        grey_smoke, grey_conf = estimate_smoke_from_appearance_depth(
            grey, depth
        )
        texture_smoke, texture_conf = estimate_smoke_from_appearance_depth(
            textured, depth
        )

        self.assertGreater(float(grey_smoke.mean()), float(texture_smoke.mean()))
        self.assertLess(float(grey_conf.mean()), float(texture_conf.mean()))

    def test_invalid_depth_is_uncertain_and_conservatively_smoky(self) -> None:
        rgb = np.full((12, 15, 3), 128, dtype=np.uint8)
        depth = np.full((12, 15), 2.0, dtype=np.float32)
        depth[3:7, 4:9] = 0.0

        smoke, confidence = estimate_smoke_from_appearance_depth(rgb, depth)

        self.assertTrue(np.all(confidence[3:7, 4:9] == 0.05))
        self.assertGreaterEqual(
            float(smoke[3:7, 4:9].mean()),
            float(smoke[0:3, 0:4].mean()),
        )


if __name__ == "__main__":
    unittest.main()
