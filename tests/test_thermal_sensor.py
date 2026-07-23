"""Regression tests for localized thermal-camera rendering."""

import unittest
from types import SimpleNamespace

import numpy as np

from utils.fire_sensors.config import VoxelSmokeConfig
from utils.fire_sensors.humans_thermal import (
    HumanThermalTarget,
    add_humans_to_thermal_image,
)
from utils.fire_sensors.voxel_render import (
    VoxelRenderParams,
    compose_thermal,
    volumetric_composite,
)


class ThermalDisplayTest(unittest.TestCase):
    def test_ambient_scene_is_dark_rgb_context_not_yellow(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[:, :32] = (160, 70, 35)
        rgb[:, 32:] = (35, 100, 170)
        ambient = np.full((48, 64), 25.0, dtype=np.float32)

        image, temperature = compose_thermal(
            rgb, ambient, np.zeros_like(ambient), ambient_c=25.0,
            color_blend=1.0,
        )

        np.testing.assert_allclose(temperature, 25.0)
        self.assertLess(float(image.mean()), 35.0)
        self.assertLess(int(image.max()), 55)
        # Cold context retains weak RGB chroma/structure instead of becoming
        # one flat thermal palette color.
        self.assertGreater(
            float(np.mean(np.abs(
                image[:, :32].astype(np.float32)
                - image[:, 32:].astype(np.float32)
            ))),
            4.0,
        )

    def test_local_heat_stays_local_and_reaches_high_contrast(self):
        rgb = np.full((80, 100, 3), (90, 110, 130), dtype=np.uint8)
        apparent = np.full((80, 100), 25.0, dtype=np.float32)
        apparent[28:52, 38:62] = 140.0
        flame = np.zeros((80, 100), dtype=np.float32)
        flame[35:45, 46:54] = 1.0

        image, temperature = compose_thermal(
            rgb, apparent, flame, ambient_c=25.0, color_blend=0.85
        )
        gray = image.mean(axis=-1)
        hot = gray[28:52, 38:62]
        cold_mask = np.ones(gray.shape, dtype=bool)
        cold_mask[28:52, 38:62] = False

        self.assertGreater(float(hot.mean()), 100.0)
        self.assertLess(float(gray[cold_mask].mean()), 35.0)
        self.assertLess(float(np.mean(gray > 100.0)), 0.08)
        self.assertAlmostEqual(float(temperature[40, 50]), 625.0, places=4)
        self.assertAlmostEqual(float(temperature[0, 0]), 25.0, places=4)

    def test_default_sensor_uses_surface_dominant_thermal_model(self):
        cfg = VoxelSmokeConfig()
        self.assertGreater(cfg.thermal_surface_start, 0.5)
        self.assertLess(cfg.thermal_air_coupling, 0.1)
        self.assertGreater(cfg.thermal_color_blend, 0.0)


class ThermalRayModelTest(unittest.TestCase):
    @staticmethod
    def _render(temp_field):
        height, width = 24, 32
        rgb = np.full((height, width, 3), (100, 120, 145), dtype=np.uint8)
        depth = np.full((height, width), 2.5, dtype=np.float32)
        shape = temp_field.shape
        zeros = np.zeros(shape, dtype=np.float32)
        camera_k = SimpleNamespace(
            fx=100.0, fy=100.0, cx=width / 2.0, cy=height / 2.0
        )
        return volumetric_composite(
            rgb_clean=rgb,
            depth_m=depth,
            cam_pos_world=np.array([1.5, 0.9, 2.85], dtype=np.float32),
            R_cam2world=np.eye(3, dtype=np.float32),
            flame_field=zeros,
            smoke_field=zeros,
            temp_field=temp_field,
            origin=np.zeros(3, dtype=np.float32),
            voxel_m=0.15,
            grid_shape=shape,
            ambient_c=25.0,
            camera_K=camera_k,
            params=VoxelRenderParams(
                max_depth_m=3.0,
                n_steps=24,
                render_scale=1.0,
                flame_noise_strength=0.0,
                flame_edge_break=0.0,
                flame_color_jitter=0.0,
                smoke_noise_strength=0.0,
            ),
            t_sim=0.0,
        )

    def test_hot_air_crossing_ray_does_not_heat_background_wall(self):
        temp = np.full((20, 12, 20), 25.0, dtype=np.float32)
        # A 30 cm hot-air curtain halfway to an ambient visible surface.
        temp[:, :, 12:14] = 700.0

        out = self._render(temp)

        self.assertLess(float(out["thermal_temperature"].max()), 35.0)
        self.assertLess(float(out["thermal_image"].mean()), 50.0)

    def test_heated_visible_surface_is_highlighted(self):
        temp = np.full((20, 12, 20), 25.0, dtype=np.float32)
        # The final depth samples land on this hot object/surface.
        temp[:, :, 0:5] = 180.0

        out = self._render(temp)

        center = out["thermal_temperature"][8:16, 12:20]
        center_image = out["thermal_image"][8:16, 12:20]
        self.assertGreater(float(center.mean()), 100.0)
        self.assertGreater(float(center_image.mean()), 90.0)


class HumanThermalHighlightTest(unittest.TestCase):
    def test_person_remains_hot_over_dark_context(self):
        height, width = 100, 120
        rgb = np.full((height, width, 3), (80, 95, 120), dtype=np.uint8)
        ambient = np.full((height, width), 25.0, dtype=np.float32)
        base, temperature = compose_thermal(
            rgb, ambient, np.zeros_like(ambient), ambient_c=25.0,
            color_blend=0.85,
        )
        sensor_state = SimpleNamespace(
            position=np.zeros(3, dtype=np.float64),
            rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        agent_state = SimpleNamespace(sensor_states={"depth": sensor_state})
        camera_k = SimpleNamespace(
            fx=100.0, fy=100.0, cx=width / 2.0, cy=height / 2.0
        )
        person = HumanThermalTarget(
            position=np.array([0.0, 0.0, -2.0], dtype=np.float64),
            excess_c=25.0,
        )

        image, temperature = add_humans_to_thermal_image(
            base,
            temperature,
            [person],
            agent_state,
            camera_k,
            depth_m=None,
            max_depth_m=5.0,
        )

        self.assertGreater(float(temperature.max()), 45.0)
        self.assertGreater(float(image.max()), 220.0)
        self.assertLess(float(image[:10, :10].mean()), 35.0)


if __name__ == "__main__":
    unittest.main()
