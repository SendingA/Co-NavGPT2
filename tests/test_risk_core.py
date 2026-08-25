"""Focused tests for the dynamic fire-risk contract and maps."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from utils.risk.config import RiskConfig, RiskWeights
from utils.risk.map import DynamicRiskMap
from utils.risk.model import GridFrame, RiskEvidence
from utils.risk.projection import (
    compute_physical_risk,
    evidence_from_sensor_images,
    hard_unsafe_mask,
    normalize_temperature_c,
    project_fire_fields,
)
from utils.risk.providers import GroundTruthRiskProvider


class RiskCoreTests(unittest.TestCase):
    def test_default_is_backward_compatible_risk_off(self) -> None:
        config = RiskConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.effective_source, "none")

        from_old_args = RiskConfig.from_namespace(SimpleNamespace())
        self.assertEqual(from_old_args, config)

        enabled = RiskConfig.from_namespace(SimpleNamespace(
            risk_enabled=1,
            risk_source=None,
        ))
        self.assertEqual(enabled.effective_source, "sensed")

        temperature_veto = RiskConfig.from_namespace(SimpleNamespace(
            risk_temperature_hard_enabled=1,
        ))
        self.assertTrue(temperature_veto.temperature_hard_enabled)

        inherited_stop = RiskConfig.from_namespace(SimpleNamespace(
            risk_critical_threshold=0.73,
            risk_early_stop_threshold=None,
        ))
        self.assertTrue(inherited_stop.early_stop_enabled)
        self.assertEqual(inherited_stop.early_stop_threshold, 0.73)

        explicit_stop = RiskConfig.from_namespace(SimpleNamespace(
            risk_early_stop_enabled=0,
            risk_early_stop_threshold=0.91,
        ))
        self.assertFalse(explicit_stop.early_stop_enabled)
        self.assertEqual(explicit_stop.early_stop_threshold, 0.91)

        with self.assertRaisesRegex(ValueError, "early_stop_threshold"):
            RiskConfig(early_stop_threshold=1.01)

    def test_weights_and_temperature_have_explicit_units(self) -> None:
        with self.assertRaises(ValueError):
            RiskWeights(temperature=1.0, smoke=1.0)
        config = RiskConfig(weights=RiskWeights(0.6, 0.4))
        np.testing.assert_allclose(
            normalize_temperature_c(
                np.array([25.0, 87.5, 150.0, 300.0]), 25.0, 150.0
            ),
            [0.0, 0.5, 1.0, 1.0],
        )
        result = compute_physical_risk(
            temperature_c=np.array([87.5]),
            smoke=np.array([0.2]),
            config=config,
        )
        self.assertAlmostEqual(float(result[0]), 0.38, places=6)
        bounded = compute_physical_risk(
            temperature_c=np.array([np.nan, np.inf]),
            smoke=np.array([np.nan, np.inf]),
            config=config,
        )
        self.assertTrue(np.all(np.isfinite(bounded)))
        self.assertTrue(np.all((bounded >= 0.0) & (bounded <= 1.0)))

    def test_grid_frame_applies_explicit_world_to_map_affine(self) -> None:
        world_to_map = np.eye(4)
        world_to_map[0, 3] = -10.0
        world_to_map[2, 3] = -20.0
        frame = GridFrame.centered(
            (4, 4), 1.0, world_to_map_matrix=world_to_map
        )
        index = frame.world_to_grid(np.array([10.0, 0.0, 20.0]))
        np.testing.assert_array_equal(index, [2, 2])
        np.testing.assert_allclose(
            frame.grid_to_world(index), [10.5, 0.0, 20.5]
        )

    def test_floor_projection_and_flame_safety_dilation(self) -> None:
        flame = np.zeros((4, 4, 4), dtype=np.float32)
        smoke = np.zeros_like(flame)
        temperature = np.full_like(flame, 25.0)
        flame[1, 0, 2] = 1.0       # inside the selected floor/body band
        flame[3, 3, 3] = 1.0       # another floor; must not leak down
        smoke[0, 0, 0] = 0.8
        temperature[2, 0, 1] = 100.0
        config = RiskConfig(
            weights=RiskWeights(0.6, 0.4),
            floor_min_offset_m=0.0,
            floor_max_offset_m=1.0,
            flame_hard_threshold=0.5,
            flame_safety_distance_m=1.0,
        )
        frame = GridFrame(shape=(4, 4), resolution_m=1.0, origin_xz=(0, 0))
        layers = project_fire_fields(
            flame,
            smoke,
            temperature,
            voxel_origin=np.zeros(3),
            voxel_m=1.0,
            frame=frame,
            floor_y_m=0.0,
            config=config,
            timestamp_s=3.0,
        )
        self.assertEqual(float(layers.flame[1, 2]), 1.0)
        self.assertEqual(float(layers.flame[3, 3]), 0.0)
        self.assertAlmostEqual(float(layers.smoke[0, 0]), 0.8, places=6)
        self.assertAlmostEqual(float(layers.temperature_c[2, 1]), 100.0)
        self.assertTrue(layers.hard_unsafe[1, 2])
        self.assertTrue(layers.hard_unsafe[0, 2])  # one-metre safety ring
        self.assertFalse(layers.unknown.any())
        self.assertTrue(np.all(layers.physical_risk >= 0.0))
        self.assertTrue(np.all(layers.physical_risk <= 1.0))

    def test_default_hard_mask_is_compact_flame_core_with_soft_gradient(
        self,
    ) -> None:
        shape = (7, 7)
        flame = np.zeros(shape, dtype=np.float32)
        flame[3, 3] = 1.0
        flame[3, 4] = 0.79
        temperature = np.full(shape, 25.0, dtype=np.float32)
        temperature[3, 3] = 150.0
        temperature[3, 4] = 105.0
        temperature[3, 5] = 65.0
        # Even an extreme temperature is soft unless the compatibility veto
        # is explicitly enabled.
        temperature[0, 0] = 1000.0
        config = RiskConfig()
        frame = GridFrame(shape, 0.05, (0.0, 0.0))

        hard = hard_unsafe_mask(flame, temperature, frame, config)
        risk = compute_physical_risk(
            temperature,
            np.zeros(shape, dtype=np.float32),
            config,
        )

        self.assertEqual(int(hard.sum()), 1)
        self.assertTrue(hard[3, 3])
        self.assertFalse(hard[3, 4])
        self.assertFalse(hard[0, 0])
        self.assertGreater(float(risk[3, 3]), float(risk[3, 4]))
        self.assertGreater(float(risk[3, 4]), float(risk[3, 5]))
        self.assertGreater(float(risk[3, 5]), 0.0)

    def test_temperature_hard_mask_is_explicit_opt_in(self) -> None:
        shape = (5, 5)
        flame = np.zeros(shape, dtype=np.float32)
        temperature = np.full(shape, 25.0, dtype=np.float32)
        temperature[2, 2] = 300.0
        frame = GridFrame(shape, 0.05, (0.0, 0.0))

        default_hard = hard_unsafe_mask(
            flame, temperature, frame, RiskConfig()
        )
        enabled_hard = hard_unsafe_mask(
            flame,
            temperature,
            frame,
            RiskConfig(
                temperature_hard_enabled=True,
                temperature_hard_c=250.0,
            ),
        )

        self.assertFalse(default_hard.any())
        self.assertTrue(enabled_hard[2, 2])

    def test_temperature_output_preserves_ambient_below_risk_reference(self) -> None:
        shape = (2, 2, 2)
        flame = np.zeros(shape, dtype=np.float32)
        smoke = np.zeros(shape, dtype=np.float32)
        temperature = np.full(shape, 25.0, dtype=np.float32)
        config = RiskConfig(
            temperature_ambient_c=25.0,
            temperature_reference_c=35.0,
            floor_max_offset_m=1.0,
            flame_safety_distance_m=0.0,
        )
        layers = project_fire_fields(
            flame,
            smoke,
            temperature,
            voxel_origin=np.zeros(3),
            voxel_m=1.0,
            frame=GridFrame((2, 2), 1.0, (0.0, 0.0)),
            floor_y_m=0.0,
            config=config,
        )
        np.testing.assert_allclose(layers.temperature_c, 25.0)
        np.testing.assert_allclose(layers.temperature, 0.0)

    def test_sensor_backprojection_has_explicit_privileged_gate(self) -> None:
        depth = np.ones((2, 2), dtype=np.float32)
        temperature = np.full((2, 2), 50.0, dtype=np.float32)
        flame = np.zeros((2, 2), dtype=np.float32)
        k = SimpleNamespace(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        with self.assertRaisesRegex(ValueError, "privileged"):
            evidence_from_sensor_images(
                agent_id=0,
                timestamp_s=0.0,
                depth_m=depth,
                camera_k=k,
                camera_position_world=np.zeros(3),
                rotation_camera_to_world=np.eye(3),
                thermal_temperature_c=temperature,
                thermal_flame=flame,
                privileged_transmittance=np.full((2, 2), 0.5),
            )

        evidence = evidence_from_sensor_images(
            agent_id=0,
            timestamp_s=0.0,
            depth_m=depth,
            camera_k=k,
            camera_position_world=np.zeros(3),
            rotation_camera_to_world=np.eye(3),
            thermal_temperature_c=temperature,
            thermal_flame=flame,
            smoke_estimate=np.full((2, 2), 0.25),
            config=RiskConfig(sensor_stride=1),
        )
        self.assertEqual(evidence.size, 4)
        np.testing.assert_allclose(evidence.points_world[0], [0.0, 0.0, -1.0])
        self.assertFalse(evidence.privileged_transmittance)

        filtered = evidence_from_sensor_images(
            agent_id=0,
            timestamp_s=0.0,
            depth_m=depth,
            camera_k=k,
            camera_position_world=np.array([0.0, 3.0, 0.0]),
            rotation_camera_to_world=np.eye(3),
            thermal_temperature_c=temperature,
            thermal_flame=flame,
            smoke_estimate=np.full((2, 2), 0.25),
            floor_y_m=0.0,
            config=RiskConfig(sensor_stride=1, floor_max_offset_m=1.5),
        )
        self.assertEqual(filtered.size, 0)

    def test_sensed_map_fuses_agents_and_stale_cells_become_unknown(self) -> None:
        frame = GridFrame(shape=(3, 3), resolution_m=1.0, origin_xz=(0, 0))
        config = RiskConfig(
            enabled=True,
            source="sensed",
            decay_tau_s=1.0,
            confidence_decay_tau_s=1.0,
            minimum_known_confidence=0.2,
            flame_safety_distance_m=0.0,
        )
        belief = DynamicRiskMap(frame, config)
        first = RiskEvidence(
            agent_id=0,
            timestamp_s=0.0,
            points_world=np.array([[1.5, 0.5, 1.5]]),
            flame=np.array([0.7]),
            temperature_c=np.array([100.0]),
            smoke=np.array([0.2]),
            confidence=np.array([0.8]),
            uncertainty=np.array([0.2]),
        )
        second = RiskEvidence(
            agent_id=1,
            timestamp_s=0.0,
            points_world=np.array([[1.6, 0.5, 1.6]]),
            flame=np.array([0.9]),
            temperature_c=np.array([80.0]),
            smoke=np.array([0.5]),
            confidence=np.array([0.5]),
            uncertainty=np.array([0.4]),
        )
        layers = belief.update_many([first, second])
        self.assertAlmostEqual(float(layers.flame[1, 1]), 0.9, places=6)
        self.assertAlmostEqual(float(layers.smoke[1, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(layers.temperature_c[1, 1]), 100.0)
        self.assertAlmostEqual(float(layers.uncertainty[1, 1]), 0.4, places=6)
        self.assertFalse(layers.unknown[1, 1])
        self.assertTrue(layers.unknown[0, 0])
        self.assertGreaterEqual(
            float(belief.planning_risk()[0, 0]), config.unknown_risk_prior
        )

        initial_risk = float(layers.physical_risk[1, 1])
        stale = belief.snapshot(timestamp_s=10.0)
        self.assertLess(float(stale.physical_risk[1, 1]), initial_risk)
        self.assertTrue(stale.unknown[1, 1])
        self.assertFalse(hasattr(belief, "fire_world"))

    def test_ground_truth_provider_is_separate_and_cached_defensively(self) -> None:
        class FakeWorld:
            origin = np.zeros(3)
            voxel_m = 1.0
            shape = (2, 2, 2)

            def __init__(self):
                self.calls = 0

            @staticmethod
            def frame_index(_timestamp):
                return 0

            def query(self, _timestamp):
                self.calls += 1
                flame = np.zeros(self.shape, dtype=np.float32)
                flame[0, 0, 0] = 1.0
                smoke = np.zeros_like(flame)
                temperature = np.full_like(flame, 25.0)
                return flame, smoke, temperature

        world = FakeWorld()
        frame = GridFrame((2, 2), 1.0, (0.0, 0.0))
        config = RiskConfig(
            floor_max_offset_m=1.0,
            flame_safety_distance_m=0.0,
        )
        provider = GroundTruthRiskProvider(
            world, frame, config, floor_y_m=0.0
        )
        first = provider.snapshot(0.0)
        first.flame[0, 0] = 0.0
        second = provider.snapshot(0.0)
        self.assertEqual(world.calls, 1)
        self.assertEqual(float(second.flame[0, 0]), 1.0)
        sampled = provider.sample_positions(0.0, [[0.5, 0.0, 0.5]])
        self.assertEqual(float(sampled.physical_risk[0]), 0.0)
        self.assertTrue(bool(sampled.hard_unsafe[0]))


if __name__ == "__main__":
    unittest.main()
