"""Headless regressions for Habitat four-slice 360-degree LiDAR."""

from __future__ import annotations

import unittest

import numpy as np
from omegaconf import OmegaConf

from utils.fire_sensors import FireSensorConfig
from utils.fire_sensors.lidar_360 import (
    LIDAR_DEPTH_UUIDS,
    install_lidar_depth_sensors,
    stitch_lidar_360,
)
from utils.fire_sensors.sensors.lidar import LidarSensor


class Lidar360ConfigTests(unittest.TestCase):
    def test_installs_four_yaw_slices_on_every_agent(self) -> None:
        depth_sensor = {
            "type": "HabitatSimDepthSensor",
            "uuid": "depth",
            "width": 640,
            "height": 480,
            "hfov": 79,
            "min_depth": 0.0,
            "max_depth": 5.0,
            "normalize_depth": True,
            "position": [0.0, 0.88, 0.0],
            "orientation": [0.0, 0.0, 0.0],
        }
        config = OmegaConf.create({
            "habitat": {
                "simulator": {
                    "agents_order": ["agent_0", "agent_1"],
                    "agents": {
                        "agent_0": {
                            "sim_sensors": {"depth_sensor": depth_sensor}
                        },
                        "agent_1": {
                            "sim_sensors": {"depth_sensor": depth_sensor}
                        },
                    },
                },
            },
        })

        install_lidar_depth_sensors(config, resolution=96, num_agents=2)

        for agent_name in config.habitat.simulator.agents_order:
            sensors = config.habitat.simulator.agents[
                agent_name
            ].sim_sensors
            self.assertTrue(all(uuid in sensors for uuid in LIDAR_DEPTH_UUIDS))
            for uuid in LIDAR_DEPTH_UUIDS:
                self.assertEqual(int(sensors[uuid].width), 96)
                self.assertEqual(int(sensors[uuid].height), 96)
                self.assertEqual(float(sensors[uuid].hfov), 90.0)


class Lidar360AcquisitionTests(unittest.TestCase):
    @staticmethod
    def _full_observation(value: float = 0.4, side: int = 32):
        return {
            uuid: np.full(
                (side, side, 1), value, dtype=np.float32
            )
            for uuid in LIDAR_DEPTH_UUIDS
        }

    def test_full_scan_covers_all_horizontal_quadrants(self) -> None:
        points = stitch_lidar_360(
            self._full_observation(),
            max_range_m=5.0,
            stride=2,
            normalize_depth=True,
            depth_norm_max_m=5.0,
        )
        self.assertIsNotNone(points)
        quadrants = [
            (points[:, 0] > 0.1) & (points[:, 1] > 0.1),
            (points[:, 0] > 0.1) & (points[:, 1] < -0.1),
            (points[:, 0] < -0.1) & (points[:, 1] > 0.1),
            (points[:, 0] < -0.1) & (points[:, 1] < -0.1),
        ]
        self.assertTrue(all(np.any(mask) for mask in quadrants))

    def test_normalized_no_return_pixels_do_not_form_a_false_ring(self) -> None:
        points = stitch_lidar_360(
            self._full_observation(value=1.0),
            max_range_m=10.0,
            stride=1,
            normalize_depth=True,
            depth_norm_max_m=5.0,
        )
        self.assertIsNotNone(points)
        self.assertEqual(points.shape, (0, 3))

    def test_partial_surround_inputs_use_forward_fallback(self) -> None:
        cfg = FireSensorConfig(
            max_depth_m=5.0,
            smoke_density=0.0,
        )
        sensor = LidarSensor(cfg, np.random.default_rng(0))
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        forward_depth = np.full((32, 32), 2.0, dtype=np.float32)
        partial_obs = {
            LIDAR_DEPTH_UUIDS[0]: np.full(
                (32, 32, 1), 0.4, dtype=np.float32
            )
        }

        partial = sensor.process(rgb, forward_depth, obs=partial_obs)
        complete = sensor.process(
            rgb,
            forward_depth,
            obs=self._full_observation(),
        )

        self.assertFalse(partial["is_360"])
        self.assertTrue(complete["is_360"])


if __name__ == "__main__":
    unittest.main()
