"""Focused tests for the optional Torch FireWorld renderer."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from arguments import voxel_smoke_kwargs
from utils.fire_sensors.config import FireSensorConfig, VoxelSmokeConfig
from utils.fire_sensors.sensors.voxel_smoke import VoxelSmokeSensor
from utils.fire_sensors.voxel_render import (
    VoxelRenderParams,
    volumetric_composite,
)
from utils.fire_sensors.voxel_render_torch import (
    TorchVolumeCache,
    shared_scene_cache,
    volumetric_composite_torch,
)


def _render_case():
    height, width = 24, 32
    shape = (20, 12, 20)
    rgb = np.full((height, width, 3), (80, 105, 130), dtype=np.uint8)
    depth = np.full((height, width), 2.5, dtype=np.float32)
    flame = np.zeros(shape, dtype=np.float32)
    flame[8:12, 4:8, 3:7] = 0.8
    smoke = np.zeros(shape, dtype=np.float32)
    smoke[6:14, 2:10, 2:10] = 0.3
    temp = np.full(shape, 25.0, dtype=np.float32)
    temp[8:12, 4:8, 3:7] = 300.0
    camera_k = SimpleNamespace(
        fx=100.0, fy=100.0, cx=width / 2.0, cy=height / 2.0
    )
    params = VoxelRenderParams(
        max_depth_m=3.0,
        n_steps=16,
        render_scale=1.0,
        flame_glow_gain=0.0,
        flame_noise_strength=0.0,
        flame_edge_break=0.0,
        flame_color_jitter=0.0,
        smoke_noise_strength=0.0,
    )
    return {
        "rgb_clean": rgb,
        "depth_m": depth,
        "cam_pos_world": np.array([1.5, 0.9, 2.85], dtype=np.float32),
        "R_cam2world": np.eye(3, dtype=np.float32),
        "flame_field": flame,
        "smoke_field": smoke,
        "temp_field": temp,
        "origin": np.zeros(3, dtype=np.float32),
        "voxel_m": 0.15,
        "grid_shape": shape,
        "ambient_c": 25.0,
        "camera_K": camera_k,
        "params": params,
        "t_sim": 0.0,
    }


class TorchRendererParityTest(unittest.TestCase):
    def test_torch_cpu_matches_numpy_reference_across_ray_tiles(self):
        kwargs = _render_case()
        expected = volumetric_composite(**kwargs)
        actual = volumetric_composite_torch(
            **kwargs,
            device="cpu",
            volume_dtype="float16",  # CPU must safely promote this to FP32.
            max_sample_points=64,
        )

        np.testing.assert_allclose(
            actual["transmittance"],
            expected["transmittance"],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            actual["thermal_temperature"],
            expected["thermal_temperature"],
            rtol=1e-5,
            atol=1e-3,
        )
        np.testing.assert_array_equal(
            actual["flame_mask"], expected["flame_mask"]
        )
        np.testing.assert_array_equal(actual["image"], expected["image"])
        self.assertEqual(actual["image"].dtype, np.uint8)
        self.assertEqual(actual["transmittance"].dtype, np.float32)
        self.assertEqual(
            actual["thermal_temperature"].dtype, np.float32
        )

    def test_active_frame_cache_reuses_volume_and_replaces_old_frame(self):
        kwargs = _render_case()
        cache = TorchVolumeCache()

        volumetric_composite_torch(
            **kwargs,
            device="cpu",
            volume_dtype="float32",
            cache=cache,
            frame_key=("scene", "plan", 4),
        )
        self.assertEqual((cache.hits, cache.uploads), (0, 1))

        volumetric_composite_torch(
            **kwargs,
            device="cpu",
            volume_dtype="float32",
            cache=cache,
            frame_key=("scene", "plan", 4),
        )
        self.assertEqual((cache.hits, cache.uploads), (1, 1))

        volumetric_composite_torch(
            **kwargs,
            device="cpu",
            volume_dtype="float32",
            cache=cache,
            frame_key=("scene", "plan", 5),
        )
        self.assertEqual((cache.hits, cache.uploads), (1, 2))

    def test_procedural_noise_path_produces_finite_visible_outputs(self):
        kwargs = _render_case()
        kwargs["params"] = VoxelRenderParams(
            max_depth_m=3.0,
            n_steps=8,
            render_scale=1.0,
            flame_glow_gain=0.0,
        )

        output = volumetric_composite_torch(
            **kwargs,
            device="cpu",
            volume_dtype="float32",
        )

        self.assertTrue(np.isfinite(output["transmittance"]).all())
        self.assertTrue(np.isfinite(output["thermal_temperature"]).all())
        self.assertGreater(float(output["flame_mask"].sum()), 0.0)
        self.assertGreater(
            float(output["thermal_temperature"].max()), 25.0
        )

    def test_flame_is_chromatic_and_preserves_surface_texture(self):
        kwargs = _render_case()
        height, width = kwargs["rgb_clean"].shape[:2]
        yy, xx = np.mgrid[:height, :width]
        checker = ((xx // 2 + yy // 2) % 2).astype(np.uint8)
        kwargs["rgb_clean"] = np.stack(
            [
                55 + 145 * checker,
                70 + 115 * checker,
                85 + 85 * checker,
            ],
            axis=-1,
        ).astype(np.uint8)
        kwargs["params"] = VoxelRenderParams(
            max_depth_m=3.0,
            n_steps=24,
            render_scale=1.0,
            flame_glow_gain=0.0,
            flame_noise_strength=0.0,
            flame_edge_break=0.0,
            flame_color_jitter=0.0,
            smoke_noise_strength=0.0,
        )

        output = volumetric_composite(**kwargs)
        flame_mask = output["flame_mask"] > 0.0
        self.assertGreater(int(flame_mask.sum()), 20)

        fire_rgb = output["image"][flame_mask].astype(np.float32)
        white_ratio = np.mean(np.min(fire_rgb, axis=1) >= 245.0)
        self.assertLess(float(white_ratio), 0.05)
        self.assertGreater(
            float(np.mean(fire_rgb[:, 0] - fire_rgb[:, 2])),
            15.0,
        )

        clean_luma = kwargs["rgb_clean"].mean(axis=-1)[flame_mask]
        fire_luma = output["image"].mean(axis=-1)[flame_mask]
        texture_correlation = np.corrcoef(clean_luma, fire_luma)[0, 1]
        self.assertGreater(float(texture_correlation), 0.25)

    def test_textured_flame_matches_torch_and_changes_over_time(self):
        kwargs = _render_case()
        kwargs["params"] = VoxelRenderParams(
            max_depth_m=3.0,
            n_steps=12,
            render_scale=1.0,
            flame_glow_gain=0.0,
        )
        expected = volumetric_composite(**kwargs)
        actual = volumetric_composite_torch(
            **kwargs,
            device="cpu",
            volume_dtype="float32",
            max_sample_points=1_000_000,
        )
        np.testing.assert_array_equal(actual["image"], expected["image"])

        later_kwargs = dict(kwargs)
        later_kwargs["t_sim"] = 0.17
        later = volumetric_composite(**later_kwargs)
        flame_region = (
            (expected["flame_mask"] > 0.0)
            | (later["flame_mask"] > 0.0)
        )
        self.assertGreater(int(flame_region.sum()), 20)
        changed = np.any(
            expected["image"][flame_region]
            != later["image"][flame_region],
            axis=1,
        )
        self.assertGreater(float(np.mean(changed)), 0.10)

    def test_scene_cache_is_shared_by_multiple_agent_sensors(self):
        scene = SimpleNamespace()
        self.assertIs(shared_scene_cache(scene), shared_scene_cache(scene))


class TorchRendererIntegrationTest(unittest.TestCase):
    @staticmethod
    def _numpy_output(rgb, depth):
        height, width = depth.shape[:2]
        return {
            "image": rgb.copy(),
            "transmittance": np.ones((height, width), dtype=np.float32),
            "flame_mask": np.zeros((height, width), dtype=np.float32),
            "thermal_image": np.zeros(
                (height, width, 3), dtype=np.uint8
            ),
            "thermal_temperature": np.full(
                (height, width), 25.0, dtype=np.float32
            ),
        }

    def test_invalid_cuda_device_falls_back_to_numpy(self):
        shape = (4, 4, 4)

        class FakeWorld:
            flame = np.zeros((1,) + shape, dtype=np.float16)
            smoke = np.zeros((1,) + shape, dtype=np.float16)
            temp = np.full((1,) + shape, 25.0, dtype=np.float16)

            @staticmethod
            def frame_index(_timestamp):
                return 0

        class FakeScene:
            fw = FakeWorld()
            origin = np.zeros(3, dtype=np.float32)
            voxel_m = 0.25
            shape = (4, 4, 4)
            ambient_c = 25.0
            scene_id = "scene"
            plan_id = "plan"

            @staticmethod
            def camera_pose(_agent_state):
                return np.zeros(3), np.eye(3)

            @staticmethod
            def t_sim(_robot_step):
                return 0.0

            def query(self, _timestamp):
                return (
                    self.fw.flame[0].astype(np.float32),
                    self.fw.smoke[0].astype(np.float32),
                    self.fw.temp[0].astype(np.float32),
                )

        cfg = FireSensorConfig(
            voxel=VoxelSmokeConfig(
                render_backend="torch",
                render_device="cuda:999999",
                flame_noise_strength=0.0,
                flame_edge_break=0.0,
                flame_color_jitter=0.0,
                smoke_noise_strength=0.0,
            )
        )
        sensor = VoxelSmokeSensor(
            cfg,
            camera_K=SimpleNamespace(fx=10.0, fy=10.0, cx=2.0, cy=2.0),
            scene=FakeScene(),
        )
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        depth = np.ones((4, 4), dtype=np.float32)

        with mock.patch(
            "utils.fire_sensors.sensors.voxel_smoke.volumetric_composite",
            side_effect=lambda **kwargs: self._numpy_output(
                kwargs["rgb_clean"], kwargs["depth_m"]
            ),
        ), self.assertWarnsRegex(RuntimeWarning, "falling back to NumPy"):
            output = sensor.process(
                rgb, depth, agent_state=object(), robot_step=0
            )

        self.assertEqual(output["render_backend"], "numpy")
        self.assertEqual(output["render_device"], "cpu")
        self.assertTrue(sensor._torch_failed)

    def test_two_sensors_share_one_native_timeline_frame_upload(self):
        shape = (6, 6, 6)

        class FakeWorld:
            flame = np.zeros((1,) + shape, dtype=np.float16)
            smoke = np.zeros((1,) + shape, dtype=np.float16)
            temp = np.full((1,) + shape, 25.0, dtype=np.float16)
            flame[0, 2:4, 2:4, 1:3] = 0.7
            temp[0, 2:4, 2:4, 1:3] = 250.0

            @staticmethod
            def frame_index(_timestamp):
                return 0

        class FakeScene:
            fw = FakeWorld()
            origin = np.zeros(3, dtype=np.float32)
            voxel_m = 0.2
            shape = (6, 6, 6)
            ambient_c = 25.0
            scene_id = "scene"
            plan_id = "plan"

            @staticmethod
            def camera_pose(_agent_state):
                return (
                    np.array([0.6, 0.6, 1.1], dtype=np.float32),
                    np.eye(3, dtype=np.float32),
                )

            @staticmethod
            def t_sim(_robot_step):
                return 0.0

            @staticmethod
            def query(_timestamp):
                raise AssertionError(
                    "Torch path must not allocate scene.query() FP32 copies"
                )

        scene = FakeScene()
        cfg = FireSensorConfig(
            max_depth_m=1.0,
            voxel=VoxelSmokeConfig(
                n_steps=4,
                render_scale=1.0,
                render_backend="torch",
                render_device="cpu",
                flame_glow_gain=0.0,
                flame_noise_strength=0.0,
                flame_edge_break=0.0,
                flame_color_jitter=0.0,
                smoke_noise_strength=0.0,
            ),
        )
        camera_k = SimpleNamespace(
            fx=20.0, fy=20.0, cx=4.0, cy=4.0
        )
        sensors = [
            VoxelSmokeSensor(cfg, camera_K=camera_k, scene=scene)
            for _ in range(2)
        ]
        rgb = np.full((8, 8, 3), 80, dtype=np.uint8)
        depth = np.full((8, 8), 0.8, dtype=np.float32)

        first = sensors[0].process(
            rgb, depth, agent_state=object(), robot_step=0
        )
        second = sensors[1].process(
            rgb, depth, agent_state=object(), robot_step=0
        )

        self.assertEqual(first["render_backend"], "torch")
        self.assertEqual(second["render_backend"], "torch")
        self.assertEqual(first["render_cache_uploads"], 1)
        self.assertEqual(second["render_cache_uploads"], 1)
        self.assertEqual(first["render_cache_hits"], 0)
        self.assertEqual(second["render_cache_hits"], 1)

    def test_cli_values_flow_into_voxel_config(self):
        args = SimpleNamespace(
            fire_fast=1,
            fire_world_n_steps=24,
            fire_world_render_scale=0.5,
            fire_world_smoke_k_ext=4.0,
            fire_flame_noise=None,
            fire_render_backend="torch",
            fire_render_device="cuda:1",
            fire_render_dtype="float32",
            fire_render_max_sample_points=12345,
        )

        config = VoxelSmokeConfig(**voxel_smoke_kwargs(args))

        self.assertEqual(config.render_backend, "torch")
        self.assertEqual(config.render_device, "cuda:1")
        self.assertEqual(config.render_dtype, "float32")
        self.assertEqual(config.max_sample_points, 12345)
        self.assertEqual(config.n_steps, 10)
        self.assertAlmostEqual(config.render_scale, 0.35)
        self.assertEqual(config.flame_noise_strength, 0.0)

        pretty_args = SimpleNamespace(**vars(args))
        pretty_args.fire_fast = 0
        pretty_config = VoxelSmokeConfig(
            **voxel_smoke_kwargs(pretty_args)
        )
        self.assertAlmostEqual(pretty_config.flame_noise_strength, 0.75)
        self.assertAlmostEqual(pretty_config.flame_edge_break, 1.05)
        self.assertAlmostEqual(pretty_config.flame_color_jitter, 0.32)
        self.assertAlmostEqual(pretty_config.smoke_noise_strength, 0.24)
        self.assertAlmostEqual(pretty_config.flame_k_ext, 0.50)
        self.assertAlmostEqual(
            pretty_config.flame_smoke_displacement, 0.52
        )
        self.assertAlmostEqual(pretty_config.flame_surface_reveal, 0.13)


if __name__ == "__main__":
    unittest.main()
