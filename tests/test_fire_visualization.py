"""Regression tests for FireWorld colors and diagnostic annotations."""

import unittest

import numpy as np
import supervision as sv

from utils.fire_sensors.config import SmokeConfig, VoxelSmokeConfig
from utils.fire_sensors.voxel_render import VoxelRenderParams
from utils.visualization import vis_result_fast


class FireVisualizationTests(unittest.TestCase):
    def test_default_smoke_is_gray_black_across_sensor_layers(self):
        expected = (72, 72, 72)

        self.assertEqual(SmokeConfig().smoke_color_rgb, expected)
        self.assertEqual(VoxelSmokeConfig().smoke_color_rgb, expected)
        self.assertEqual(VoxelRenderParams().smoke_color_rgb, expected)

    def test_fire_mask_does_not_cover_rendered_rgb(self):
        image = np.full((40, 64, 3), (20, 40, 80), dtype=np.uint8)
        fire_mask = np.zeros(image.shape[:2], dtype=bool)
        chair_mask = np.zeros(image.shape[:2], dtype=bool)
        fire_mask[5:30, 5:28] = True
        chair_mask[5:30, 36:59] = True
        detections = sv.Detections(
            xyxy=np.asarray(
                [[5, 5, 28, 30], [36, 5, 59, 30]],
                dtype=np.float32,
            ),
            mask=np.stack([fire_mask, chair_mask]),
            confidence=np.asarray([0.95, 0.80], dtype=np.float32),
            class_id=np.asarray([0, 1], dtype=np.int64),
        )

        annotated = vis_result_fast(
            image,
            detections,
            classes=["fire", "chair"],
            draw_bbox=False,
        )

        np.testing.assert_array_equal(annotated[16, 16], image[16, 16])
        self.assertFalse(
            np.array_equal(annotated[16, 48], image[16, 48])
        )


if __name__ == "__main__":
    unittest.main()
