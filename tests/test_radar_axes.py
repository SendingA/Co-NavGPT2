"""Regressions for physical axes on shared mmWave radar panels."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from utils.fire_sensors.bev import add_metric_axes
from utils.fire_sensors.config import FireSensorConfig
from utils.fire_sensors.sensors.radar import RadarSensor


class MetricAxisRendererTests(unittest.TestCase):
    def test_axis_renderer_uses_image_edge_coordinate_order(self) -> None:
        image = np.full((40, 60, 3), 30, dtype=np.uint8)

        with patch(
            "utils.fire_sensors.bev.cv2.putText", wraps=cv2.putText
        ) as put_text:
            rendered = add_metric_axes(
                image,
                x_label="Lateral Y [m]",
                y_label="Forward X [m]",
                x_limits=(10.0, -10.0),
                y_limits=(10.0, -10.0),
                x_tick_count=5,
                y_tick_count=5,
            )

        texts = [call.args[1] for call in put_text.call_args_list]
        self.assertEqual(texts[:5], ["10", "5", "0", "-5", "-10"])
        self.assertEqual(texts[5:10], ["10", "5", "0", "-5", "-10"])
        self.assertEqual(texts[-2:], ["Lateral Y [m]", "Forward X [m]"])
        self.assertGreater(rendered.shape[0], image.shape[0])
        self.assertGreater(rendered.shape[1], image.shape[1])
        self.assertEqual(rendered.dtype, np.uint8)


class RadarSensorAxisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = FireSensorConfig()
        self.cfg.radar.noise_std = 0.0
        self.cfg.radar.learned_noise_m = 0.0
        self.cfg.radar.learned_target_points = 128
        self.sensor = RadarSensor(self.cfg, np.random.default_rng(0))
        self.rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        self.depth = np.full((48, 64), 2.5, dtype=np.float32)

    def test_sensor_images_have_axes_without_changing_raw_arrays(self) -> None:
        output = self.sensor.process(self.rgb, self.depth)
        rcfg = self.cfg.radar

        self.assertEqual(
            output["heatmap"].shape, (rcfg.range_bins, rcfg.az_bins)
        )
        self.assertEqual(output["points"].shape[1], 2)
        self.assertEqual(output["points_3d"].shape[1], 3)
        self.assertEqual(output["image_bev"].dtype, np.uint8)
        self.assertEqual(output["image_az"].dtype, np.uint8)
        self.assertEqual(output["image_el"].dtype, np.uint8)

        # Axis margins make every preview larger than its underlying plot.
        self.assertGreater(output["image_bev"].shape[0], rcfg.bev_size_px)
        self.assertGreater(output["image_bev"].shape[1], rcfg.bev_size_px)
        self.assertGreater(output["image_az"].shape[0], rcfg.bev_size_px)
        self.assertGreater(output["image_az"].shape[1], rcfg.bev_size_px)
        self.assertGreater(
            output["image_el"].shape[1], rcfg.bev_size_px * 2
        )

    def test_sensor_passes_correct_physical_limits_to_each_panel(self) -> None:
        captured = []

        def record(image, **kwargs):
            captured.append(kwargs)
            return image

        with patch(
            "utils.fire_sensors.sensors.radar.add_metric_axes",
            side_effect=record,
        ):
            output = self.sensor.process(self.rgb, self.depth)

        self.assertEqual(output["heatmap"].shape, (256, 64))
        self.assertEqual(len(captured), 3)
        by_x_label = {item["x_label"]: item for item in captured}
        self.assertEqual(
            by_x_label["Azimuth [deg]"]["x_limits"], (-90.0, 90.0)
        )
        self.assertEqual(
            by_x_label["Elevation [deg]"]["x_limits"], (-25.0, 25.0)
        )
        self.assertEqual(
            by_x_label["Lateral Y [m]  (left +)"]["x_limits"],
            (10.0, -10.0),
        )
        self.assertEqual(
            by_x_label["Lateral Y [m]  (left +)"]["y_limits"],
            (10.0, -10.0),
        )
        self.assertEqual(
            by_x_label["Azimuth [deg]"]["y_limits"], (0.0, 10.0)
        )
        self.assertEqual(
            by_x_label["Elevation [deg]"]["y_limits"], (0.0, 10.0)
        )


if __name__ == "__main__":
    unittest.main()
