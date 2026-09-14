"""Regression tests for fire-sensor artifact controls."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from utils.fire_sensors.suite import FireSensorSuite


class FireSensorArtifactTests(unittest.TestCase):
    def test_zero_save_interval_disables_all_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "disabled"
            suite = FireSensorSuite(
                dump_dir=str(output_dir),
                save_every=0,
                seed=1,
            )

            suite.save_step({}, episode=0, step=0, agent_id=0)

            self.assertEqual(suite.save_every, 0)
            self.assertFalse(output_dir.exists())

    def test_negative_save_interval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                FireSensorSuite(
                    dump_dir=str(Path(temporary) / "invalid"),
                    save_every=-1,
                )

    def test_visible_human_mask_is_saved_as_png_and_npz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite = FireSensorSuite(
                dump_dir=temporary,
                save_every=1,
                seed=1,
            )
            suite.cfg.save_npz = True
            shape = (4, 5)
            human_mask = np.zeros(shape, dtype=np.float32)
            human_mask[1:3, 2:4] = 1.0
            outputs = {
                "rgb": np.zeros((*shape, 3), dtype=np.uint8),
                "rgb_smoke": np.zeros((*shape, 3), dtype=np.uint8),
                "depth_clean": np.ones(shape, dtype=np.float32),
                "depth_smoke": np.ones(shape, dtype=np.float32),
                "thermal_image": np.zeros((*shape, 3), dtype=np.uint8),
                "thermal_temperature": np.full(shape, 25.0, dtype=np.float32),
                "thermal_flame_mask": np.zeros(shape, dtype=np.float32),
                "thermal_human_mask": human_mask,
                "lidar_image": np.zeros((*shape, 3), dtype=np.uint8),
                "lidar_points": np.zeros((0, 3), dtype=np.float32),
                "radar_image_bev": np.zeros((*shape, 3), dtype=np.uint8),
                "radar_image_az": np.zeros((*shape, 3), dtype=np.uint8),
                "radar_image_el": np.zeros((*shape, 3), dtype=np.uint8),
                "radar_heatmap": np.zeros(shape, dtype=np.float32),
                "radar_points": np.zeros((0, 2), dtype=np.float32),
                "radar_points_3d": np.zeros((0, 3), dtype=np.float32),
            }

            suite.save_step(outputs, episode=2, step=3, agent_id=1)

            saved_dir = Path(temporary) / "ep_0002" / "agent_1"
            mask_path = saved_dir / "step_00003_thermal_human_mask.png"
            arrays_path = saved_dir / "step_00003_arrays.npz"
            saved_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(saved_mask)
            np.testing.assert_array_equal(saved_mask, human_mask * 255)
            with np.load(arrays_path) as arrays:
                self.assertIn("thermal_human_mask", arrays.files)
                np.testing.assert_array_equal(
                    arrays["thermal_human_mask"], human_mask
                )


if __name__ == "__main__":
    unittest.main()
