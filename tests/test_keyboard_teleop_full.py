"""Focused, headless tests for the all-in-one teleoperation dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from omegaconf import OmegaConf

from scripts.keyboard_teleop_full import (
    compose_view,
    draw_snapshot_button,
    maybe_build_fire,
    point_in_rect,
    save_sensor_snapshot,
    window_point_to_image,
)
from utils.fire_pipeline import step_fire_observation
from utils.fire_sensors.dashboard import render_dashboard


def _solid_image(height: int, width: int, color) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = np.asarray(color, dtype=np.uint8)
    return image


class TeleopDashboardLayoutTests(unittest.TestCase):
    def assert_pixel_color(self, image, y: int, x: int, color) -> None:
        np.testing.assert_array_equal(
            image[y, x], np.asarray(color, dtype=np.uint8)
        )

    def test_render_dashboard_places_header_above_grid_and_range_elevation(
        self,
    ) -> None:
        """The shared renderer keeps its 2x4 grid below a multi-line HUD."""
        panel_colors = {
            "rgb": (10, 20, 30),
            "depth": (20, 30, 40),
            "thermal": (30, 40, 50),
            "lidar": (40, 50, 60),
            "rgb_smoke": (50, 60, 70),
            "depth_smoke": (60, 70, 80),
            "radar": (70, 80, 90),
            "radar_az": (80, 90, 100),
        }
        panels = {
            key: _solid_image(32, 48, color)
            for key, color in panel_colors.items()
        }
        radar_el_color = (90, 100, 110)
        radar_el = _solid_image(32, 48, radar_el_color)
        size = (800, 400)

        first = render_dashboard(
            panels,
            size=size,
            extra_panel=radar_el,
            header_lines=["Active Robot 1/2", "Teleop Step 7"],
        )
        second = render_dashboard(
            panels,
            size=size,
            extra_panel=radar_el,
            header_lines=["Active Robot 2/2", "Teleop Step 8"],
        )

        grid_w, grid_h = size
        cell_w, cell_h = grid_w // 4, grid_h // 2
        aux_h = max(180, grid_h // 4)
        header_h = first.shape[0] - grid_h - aux_h

        self.assertEqual(first.shape[1], grid_w)
        self.assertEqual(first.shape, second.shape)
        self.assertGreater(header_h, 0)

        # Changing only the HUD text must not change any sensor pixel.
        difference = np.any(first != second, axis=2)
        self.assertTrue(np.any(difference[:header_h]))
        self.assertFalse(np.any(difference[header_h:]))

        ordered_keys = [
            "rgb",
            "depth",
            "thermal",
            "lidar",
            "rgb_smoke",
            "depth_smoke",
            "radar",
            "radar_az",
        ]
        for index, key in enumerate(ordered_keys):
            row, col = divmod(index, 4)
            y = header_h + row * cell_h + cell_h // 2
            x = col * cell_w + cell_w // 2
            self.assert_pixel_color(first, y, x, panel_colors[key])

        # Range-elevation is a labelled, full-width auxiliary row below
        # the 2x4 grid, not a replacement for one of the eight main tiles.
        aux_y = header_h + grid_h + aux_h // 2
        for x in (cell_w // 2, grid_w // 2, grid_w - cell_w // 2):
            self.assert_pixel_color(first, aux_y, x, radar_el_color)

    def test_compose_view_keeps_clean_smoke_and_three_radar_products_distinct(
        self,
    ) -> None:
        """The teleop compositor forwards every modality to its own tile."""
        height, width = 160, 200
        clean_rgb_color = (11, 22, 33)  # RGB at the API boundary.
        smoke_rgb_color = (44, 55, 66)  # RGB at the API boundary.
        radar_bev_color = (71, 81, 91)  # Radar products are already BGR.
        radar_az_color = (72, 82, 92)
        radar_el_color = (73, 83, 93)

        kwargs = dict(
            rgb_clean=_solid_image(height, width, clean_rgb_color),
            rgb_smoke=_solid_image(height, width, smoke_rgb_color),
            depth_clean=np.full((height, width, 1), 0.25, dtype=np.float32),
            depth_smoke=np.full((height, width), 1.75, dtype=np.float32),
            thermal=_solid_image(height, width, (20, 30, 40)),
            lidar=_solid_image(height, width, (50, 60, 70)),
            radar_bev=_solid_image(height, width, radar_bev_color),
            radar_az=_solid_image(height, width, radar_az_color),
            radar_el=_solid_image(height, width, radar_el_color),
            max_d=5.0,
            # Keep the synthetic geometry small and explicit.  In production
            # this comes from FireSensorConfig.dashboard_size.
            dashboard_size=(600, 240),
        )
        first = compose_view(
            **kwargs,
            status_lines=["Active Robot 1/2", "Teleop Step 12"],
        )
        second = compose_view(
            **kwargs,
            status_lines=["Active Robot 2/2", "Teleop Step 13"],
        )

        grid_w, grid_h = kwargs["dashboard_size"]
        tile_h = grid_h // 2
        tile_w = grid_w // 4
        aux_h = max(180, grid_h // 4)
        header_h = first.shape[0] - grid_h - aux_h

        self.assertEqual(first.shape[1], grid_w)
        self.assertEqual(first.shape, second.shape)
        self.assertGreater(header_h, 0)

        difference = np.any(first != second, axis=2)
        self.assertTrue(np.any(difference[:header_h]))
        self.assertFalse(np.any(difference[header_h:]))

        top_y = header_h + tile_h // 2
        bottom_y = header_h + tile_h + tile_h // 2
        # RGB inputs are converted to BGR independently; the smoke tile must
        # not alias or repeat the clean tile.
        self.assert_pixel_color(
            first, top_y, tile_w // 2, clean_rgb_color[::-1]
        )
        self.assert_pixel_color(
            first, bottom_y, tile_w // 2, smoke_rgb_color[::-1]
        )

        self.assert_pixel_color(
            first, bottom_y, 2 * tile_w + tile_w // 2, radar_bev_color
        )
        self.assert_pixel_color(
            first, bottom_y, 3 * tile_w + tile_w // 2, radar_az_color
        )
        self.assert_pixel_color(
            first,
            header_h + grid_h + aux_h // 2,
            grid_w // 2,
            radar_el_color,
        )

    def test_snapshot_button_hitbox_and_all_panel_files(self) -> None:
        dashboard = _solid_image(500, 900, (1, 2, 3))
        rendered, rect = draw_snapshot_button(dashboard)
        x1, y1, x2, y2 = rect
        self.assertTrue(point_in_rect((x1 + x2) // 2, (y1 + y2) // 2, rect))
        self.assertFalse(point_in_rect(x1 - 1, y1, rect))
        self.assertFalse(np.array_equal(rendered, dashboard))
        self.assertEqual(
            window_point_to_image(
                450,
                250,
                window_size=(900, 500),
                image_shape=(1000, 1800, 3),
            ),
            (900, 500),
        )

        height, width = 32, 48
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_to = save_sensor_snapshot(
                root_dir=Path(temp_dir),
                scene_id="/dataset/Nfvxx8J5NCo.basis.glb",
                agent_id=1,
                robot_step=17,
                rgb_clean=_solid_image(height, width, (11, 22, 33)),
                rgb_smoke=_solid_image(height, width, (44, 55, 66)),
                depth_clean=np.full(
                    (height, width, 1), 0.5, dtype=np.float32
                ),
                depth_smoke=np.full(
                    (height, width), 1.5, dtype=np.float32
                ),
                thermal=_solid_image(height, width, (20, 30, 40)),
                lidar=_solid_image(height, width, (30, 40, 50)),
                radar_bev=_solid_image(height, width, (40, 50, 60)),
                radar_az=_solid_image(height, width, (50, 60, 70)),
                radar_el=_solid_image(height, width, (60, 70, 80)),
                dashboard=rendered,
                max_depth_m=5.0,
                lidar_is_360=True,
            )

            expected = {
                "rgb_clean.png",
                "depth_clean.png",
                "thermal.png",
                "lidar_bev.png",
                "rgb_smoke.png",
                "depth_smoke.png",
                "radar_bev.png",
                "radar_range_azimuth.png",
                "radar_range_elevation.png",
                "dashboard.png",
            }
            self.assertEqual(
                {path.name for path in saved_to.glob("*.png")},
                expected,
            )
            manifest = json.loads(
                (saved_to / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["agent_id"], 1)
            self.assertEqual(manifest["robot_step"], 17)
            self.assertTrue(manifest["lidar_is_360"])
            self.assertFalse(manifest["missing_files"])
            self.assertEqual(set(manifest["saved_files"]), expected)

            # RGB inputs cross the API boundary as RGB and must be encoded
            # to disk in the BGR order expected by OpenCV.
            saved_rgb = cv2.imread(str(saved_to / "rgb_clean.png"))
            np.testing.assert_array_equal(saved_rgb[0, 0], (33, 22, 11))

    def test_sensor_suite_remains_available_without_a_fire_plan(self) -> None:
        config = OmegaConf.create({
            "habitat": {
                "simulator": {
                    "agents_order": ["agent_0"],
                    "agents": {
                        "agent_0": {
                            "sim_sensors": {
                                "depth_sensor": {
                                    "max_depth": 5.0,
                                    "hfov": 79.0,
                                },
                                "rgb_sensor": {
                                    "width": 64,
                                    "height": 48,
                                    "hfov": 79.0,
                                },
                            },
                        },
                    },
                },
            },
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                plan_id=None,
                smoke_density=0.6,
                n_steps=8,
                fast=1,
                smoke_k_ext=4.0,
                render_scale=0.5,
                flame_smoke_passthrough=0.95,
                save_frames_to=temp_dir,
                seed=7,
            )
            scene, suites = maybe_build_fire(args, config, num_agents=1)

        self.assertIsNone(scene)
        self.assertEqual(len(suites), 1)
        self.assertIsNone(suites[0].scene)
        self.assertEqual(suites[0].cfg.smoke_density, 0.0)


class FirePipelineCleanRgbTests(unittest.TestCase):
    def test_repeated_redraw_uses_pristine_rgb_but_keeps_smoke_in_observation(
        self,
    ) -> None:
        """Idle redraws must not feed a prior smoky frame back as clean RGB."""

        class FakeSuite:
            def __init__(self) -> None:
                self.rgb_inputs = []
                self.smoke_outputs = []

            def process(
                self,
                rgb,
                depth_m,
                obs=None,
                *,
                agent_state=None,
                robot_step=0,
            ):
                del obs, agent_state, robot_step
                self.rgb_inputs.append(np.asarray(rgb).copy())
                smoke_value = 80 + 40 * len(self.rgb_inputs)
                smoke = np.full_like(rgb, smoke_value)
                self.smoke_outputs.append(smoke.copy())
                return {
                    "rgb": np.asarray(rgb).copy(),
                    "rgb_smoke": smoke,
                    "depth_clean": np.asarray(depth_m).copy(),
                    "depth_smoke": np.asarray(depth_m).copy(),
                }

        clean_rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        observations = {
            "rgb": clean_rgb.copy(),
            "depth": np.full((4, 5, 1), 0.5, dtype=np.float32),
        }
        suite = FakeSuite()
        args = SimpleNamespace(
            depth_use_clean=1,
            fire_apply_to_obs=1,
            use_thermal_perception=0,
        )

        for robot_step in (0, 1):
            step_fire_observation(
                observations=observations,
                suite=suite,
                agent_state=object(),
                robot_step=robot_step,
                config=SimpleNamespace(),
                args=args,
            )

        self.assertEqual(len(suite.rgb_inputs), 2)
        np.testing.assert_array_equal(suite.rgb_inputs[0], clean_rgb)
        np.testing.assert_array_equal(suite.rgb_inputs[1], clean_rgb)

        # Navigation/perception still consumes the newest smoke-rendered RGB;
        # only the suite's clean input is cached across idle redraws.
        np.testing.assert_array_equal(
            observations["rgb"], suite.smoke_outputs[-1]
        )
        self.assertFalse(np.array_equal(observations["rgb"], clean_rgb))


if __name__ == "__main__":
    unittest.main()
