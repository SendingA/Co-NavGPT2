"""Focused regressions for source-centred floor-fire propagation."""

import unittest

import numpy as np

from utils.fire_world.propagation import FirePropagation, run_propagation
from utils.fire_world.templates import (
    TEMPLATE_VERSION,
    _default_propagation_rules,
)
from utils.fire_world.voxel_world import VoxelWorld


class FloorFireSpreadTest(unittest.TestCase):
    def _make_sim(self, duration_s=None, **rule_overrides):
        world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 4.0, 0.6, 4.0], voxel=0.2
        )
        world.floors = np.zeros(world.shape, dtype=bool)
        world.floors[:, 0, :] = True
        source = np.array([2.0, 0.1, 2.0], dtype=np.float64)
        source_radius = 0.2
        source_slice, falloff = world.kindle_ignition(
            source, radius_m=source_radius, temp_c=700.0
        )
        rules = {
            "thermal_diffusivity": 0.0,
            "smoke_diffusivity": 0.0,
            "buoyancy_v_m_per_s": 0.0,
            "ceiling_jet_speed_m_per_s": 0.0,
            "radiative_gain_c": 0.0,
            "spread_speed_m_per_s": 0.0,
            "floor_ignite_radius_cells": 1,
            "floor_fuel_value": 0.7,
            "floor_spread_speed_m_per_s": 0.1,
            "floor_max_spread_radius_m": 0.8,
            "flame_column_cells": 0,
            "inextinguishable_sources": 1,
        }
        rules.update(rule_overrides)
        sim = FirePropagation(
            world,
            rules,
            seed=7,
            duration_s=duration_s,
        )
        sim.add_source(
            source_slice,
            falloff,
            source_temp_c=700.0,
            sustain_s=30.0,
            t_now=0.0,
            position=source,
            source_radius_m=source_radius,
        )
        return world, sim, source

    @staticmethod
    def _burning_floor_distances(world, source):
        burning = world.floors & (world.flame > 0.01)
        indices = np.argwhere(burning)
        centres = world.origin + (indices + 0.5) * world.voxel
        distances = np.linalg.norm(
            centres[:, [0, 2]] - source[[0, 2]], axis=1
        )
        return burning, distances

    def test_floor_fire_grows_slowly_and_stops_at_max_radius(self):
        world, sim, source = self._make_sim()

        sim.step(1.0, t_now=1.0)
        early, early_distances = self._burning_floor_distances(world, source)
        self.assertGreater(int(early.sum()), 0)
        self.assertLessEqual(float(early_distances.max()), 0.3)

        for t_now in range(2, 13):
            sim.step(1.0, t_now=float(t_now))
        mature, mature_distances = self._burning_floor_distances(world, source)

        self.assertGreater(int(mature.sum()), int(early.sum()))
        self.assertGreater(float(mature_distances.max()), 0.6)
        self.assertLessEqual(float(mature_distances.max()), 0.8 + 1e-6)

        # Even a manually introduced out-of-range floor flame is removed
        # on the next step; the restriction is not limited to ignition.
        world.flame[0, 0, 0] = 1.0
        sim.step(1.0, t_now=13.0)
        self.assertEqual(float(world.flame[0, 0, 0]), 0.0)

    def test_hot_floor_outside_source_envelope_does_not_ignite(self):
        world, sim, _ = self._make_sim()
        world.temp[0, 0, 0] = 1000.0

        sim.step(1.0, t_now=1.0)

        self.assertEqual(float(world.fuel[0, 0, 0]), 0.0)
        self.assertEqual(float(world.flame[0, 0, 0]), 0.0)

    def test_explicit_source_scale_can_shrink_floor_envelope(self):
        world, sim, source = self._make_sim()
        sim._floor_sources.clear()
        source_slice, falloff = world.kindle_ignition(
            source, radius_m=0.2, temp_c=700.0
        )
        sim.add_source(
            source_slice,
            falloff,
            source_temp_c=700.0,
            sustain_s=30.0,
            t_now=0.0,
            position=source,
            source_radius_m=0.2,
            floor_spread_scale=0.5,
        )

        allowed = sim._floor_spread_allowed_xz(t_now=100.0)
        indices = np.argwhere(allowed)
        xs = world.origin[0] + (indices[:, 0] + 0.5) * world.voxel
        zs = world.origin[2] + (indices[:, 1] + 0.5) * world.voxel
        distances = np.linalg.norm(
            np.stack([xs, zs], axis=1) - source[[0, 2]],
            axis=1,
        )
        self.assertGreater(float(distances.max()), 0.3)
        self.assertLessEqual(float(distances.max()), 0.4 + 1e-6)

    def test_plans_without_new_fields_receive_compact_defaults(self):
        world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 1.0, 0.4, 1.0], voxel=0.2
        )
        sim = FirePropagation(world, rules={})

        self.assertEqual(sim.spread, 0.12)
        self.assertEqual(sim.radiative_gain_c, 200.0)
        self.assertEqual(sim.radiative_radius_cells, 4)
        self.assertEqual(sim.floor_fuel_value, 0.52)
        self.assertEqual(sim.floor_ignite_radius_cells, 2)
        self.assertEqual(sim.floor_flame_contact_thresh, 0.12)
        self.assertEqual(sim.floor_spread_speed_m_per_s, 0.0044)
        self.assertEqual(sim.floor_max_spread_radius_m, 1.10)
        self.assertEqual(sim.floor_spread_reach_fraction, 0.0)
        self.assertEqual(sim.object_spread_speed_m_per_s, 0.0066)
        self.assertEqual(sim.object_max_spread_radius_m, 1.35)
        self.assertEqual(sim.object_bbox_fill_speed_m_per_s, 0.008)
        self.assertEqual(sim.object_bbox_fill_flame_min, 0.16)
        self.assertEqual(sim.object_bbox_fill_temp_margin_c, 40.0)
        self.assertEqual(sim.object_bbox_fill_front_width_layers, 1.0)
        self.assertEqual(sim.object_bbox_max_vertical_spread_m, 0.0)
        self.assertEqual(sim.floor_seed_flame_min, 0.10)
        self.assertEqual(sim.floor_seed_flame_max, 0.58)
        self.assertEqual(sim.floor_gaussian_sigma_fraction, 0.50)
        self.assertEqual(sim.floor_gaussian_min_influence, 0.08)
        self.assertEqual(sim.floor_front_softness_m, 0.18)
        self.assertEqual(sim.floor_min_visible_flame, 0.06)
        self.assertTrue(sim.limit_flame_to_source_envelope)
        self.assertEqual(sim.flame_column_cells, 3)
        self.assertEqual(sim.max_flame_column_height_m, 0.0)
        self.assertEqual(sim.effective_flame_column_cells(), 3)
        self.assertEqual(sim.flame_column_decay, 0.50)
        self.assertEqual(sim.object_flame_extra_height_cells, 1)

    def test_metric_flame_column_cap_is_resolution_aware(self):
        coarse_world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 1.0, 1.6, 1.0], voxel=0.2
        )
        fine_world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 1.0, 1.6, 1.0], voxel=0.1
        )
        rules = {
            "flame_column_cells": 4,
            "max_flame_column_height_m": 0.35,
        }

        coarse = FirePropagation(coarse_world, rules=rules)
        fine = FirePropagation(fine_world, rules=rules)

        self.assertEqual(coarse.effective_flame_column_cells(), 1)
        self.assertEqual(fine.effective_flame_column_cells(), 3)
        self.assertLessEqual(
            coarse.effective_flame_column_cells() * coarse_world.voxel,
            0.35,
        )
        self.assertLessEqual(
            fine.effective_flame_column_cells() * fine_world.voxel,
            0.35,
        )

        source = np.array([0.5, 0.1, 0.5], dtype=np.float64)
        source_slice, falloff = coarse_world.kindle_ignition(
            source, radius_m=0.2, temp_c=700.0
        )
        coarse.add_source(
            source_slice,
            falloff,
            source_temp_c=700.0,
            sustain_s=10.0,
            t_now=0.0,
            position=source,
            source_radius_m=0.2,
        )
        anchored_max_y = int(
            np.argwhere(coarse_world.fuel > 0.05)[:, 1].max()
        )
        coarse.step(0.5, t_now=0.5)
        visible_max_y = int(
            np.argwhere(coarse_world.flame > 0.01)[:, 1].max()
        )
        self.assertGreater(visible_max_y, anchored_max_y)
        self.assertLessEqual(visible_max_y, anchored_max_y + 1)

    def test_flame_column_height_does_not_stack_across_steps(self):
        world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0], voxel=0.2
        )
        source = np.array([1.0, 0.1, 1.0], dtype=np.float64)
        source_slice, falloff = world.kindle_ignition(
            source, radius_m=0.2, temp_c=700.0
        )
        sim = FirePropagation(
            world,
            rules={
                "thermal_diffusivity": 0.0,
                "smoke_diffusivity": 0.0,
                "buoyancy_v_m_per_s": 0.0,
                "ceiling_jet_speed_m_per_s": 0.0,
                "radiative_gain_c": 0.0,
                "spread_speed_m_per_s": 0.0,
                "floor_fuel_value": 0.0,
                "inextinguishable_sources": 1,
            },
            seed=7,
        )
        sim.add_source(
            source_slice,
            falloff,
            source_temp_c=700.0,
            sustain_s=30.0,
            t_now=0.0,
            position=source,
            source_radius_m=0.2,
        )
        anchored_max_y = int(np.argwhere(world.fuel > 0.05)[:, 1].max())

        for t_now in range(1, 9):
            sim.step(0.5, t_now=0.5 * t_now)

        visible = np.argwhere(world.flame > 0.01)
        self.assertGreater(len(visible), 0)
        self.assertLessEqual(
            int(visible[:, 1].max()),
            anchored_max_y + sim.flame_column_cells,
        )

    def test_non_object_flame_is_clipped_to_source_range(self):
        world, sim, _ = self._make_sim(
            flame_column_cells=2,
            floor_spread_speed_m_per_s=0.0,
            floor_max_spread_radius_m=0.4,
        )

        # Give the distant voxel real fuel so the existing unsupported-column
        # cleanup cannot be what removes it.
        world.fuel[0, 1, 0] = 1.0
        world.flame[0, 1, 0] = 1.0

        sim.step(0.5, t_now=0.5)

        self.assertEqual(float(world.flame[0, 1, 0]), 0.0)

    def test_solver_ignited_object_remains_visible_outside_source_range(self):
        world, sim, source = self._make_sim(
            ignition_temp_c=200.0,
            floor_spread_speed_m_per_s=0.0,
            floor_max_spread_radius_m=0.4,
            object_spread_speed_m_per_s=10.0,
            object_max_spread_radius_m=4.0,
        )
        del source
        object_voxel = (0, 1, 0)
        unsupported_voxel = (0, 1, 1)
        world.object_id_field = np.full(world.shape, -1, dtype=np.int32)
        world.object_id_field[object_voxel] = 99
        world.fuel[object_voxel] = 1.0
        world.temp[object_voxel] = 1000.0
        world.fuel[unsupported_voxel] = 1.0
        world.temp[unsupported_voxel] = 1000.0

        sim.step(0.5, t_now=0.5)

        self.assertGreater(float(world.flame[object_voxel]), 0.0)
        self.assertEqual(float(world.flame[unsupported_voxel]), 0.0)

    def test_ignited_object_slowly_fills_its_own_bounding_box(self):
        world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 1.6, 0.8, 0.8], voxel=0.2
        )
        world.object_id_field = np.full(
            world.shape, -1, dtype=np.int32
        )
        object_a = (slice(1, 6), slice(0, 3), slice(1, 3))
        object_b = (slice(6, 8), slice(0, 3), slice(1, 3))
        world.object_id_field[object_a] = 7
        world.object_id_field[object_b] = 8
        world.fuel[object_a] = 1.0
        world.fuel[object_b] = 1.0
        seed = (1, 1, 1)
        world.temp[seed] = 700.0
        world.flame[seed] = 0.8

        sim = FirePropagation(
            world,
            rules={
                "thermal_diffusivity": 0.0,
                "smoke_diffusivity": 0.0,
                "buoyancy_v_m_per_s": 0.0,
                "ceiling_jet_speed_m_per_s": 0.0,
                "radiative_gain_c": 0.0,
                "spread_speed_m_per_s": 0.0,
                "q_release_c": 0.0,
                "k_burn_per_s": 0.0,
                "floor_fuel_value": 0.0,
                "floor_max_spread_radius_m": 0.3,
                "flame_column_cells": 0,
                "object_flame_extra_height_cells": 1,
                "limit_flame_to_source_envelope": 1,
                "object_spread_speed_m_per_s": 0.0,
                "object_max_spread_radius_m": 0.3,
                "object_bbox_fill_speed_m_per_s": 0.2,
                "object_bbox_fill_flame_min": 0.2,
                "inextinguishable_sources": 0,
            },
            seed=7,
        )
        source_slice = (
            slice(seed[0], seed[0] + 1),
            slice(seed[1], seed[1] + 1),
            slice(seed[2], seed[2] + 1),
        )
        sim.add_source(
            source_slice,
            np.ones((1, 1, 1), dtype=np.float32),
            source_temp_c=700.0,
            sustain_s=0.1,
            t_now=0.0,
            position=world.grid_to_world(np.asarray(seed)),
            source_radius_m=0.2,
        )

        sim.step(0.5, t_now=0.5)
        early = int(np.count_nonzero(world.flame[object_a] > 0.1))
        sim.step(0.5, t_now=1.0)
        partial = int(np.count_nonzero(world.flame[object_a] > 0.0))
        self.assertGreater(partial, early)
        self.assertLess(
            partial, int(np.prod(world.flame[object_a].shape))
        )
        for step in range(1, 9):
            sim.step(1.0, t_now=0.5 + step)
        mature = int(np.count_nonzero(world.flame[object_a] > 0.1))

        self.assertEqual(early, 1)
        self.assertGreater(mature, early)
        self.assertEqual(mature, int(np.prod(world.flame[object_a].shape)))
        self.assertEqual(
            int(np.count_nonzero(world.flame[object_b] > 0.1)),
            0,
        )

    def test_tall_object_bbox_fill_respects_vertical_metric_cap(self):
        world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 1.0, 1.6, 1.0], voxel=0.2
        )
        world.object_id_field = np.full(
            world.shape, -1, dtype=np.int32
        )
        object_slice = (slice(1, 4), slice(0, 7), slice(1, 4))
        world.object_id_field[object_slice] = 11
        world.fuel[object_slice] = 1.0
        seed = (1, 0, 1)
        world.temp[seed] = 700.0
        world.flame[seed] = 0.8
        sim = FirePropagation(
            world,
            rules={
                "ignition_temp_c": 200.0,
                "object_bbox_fill_speed_m_per_s": 1.0,
                "object_bbox_max_vertical_spread_m": 0.4,
                "object_bbox_fill_flame_min": 0.2,
            },
            seed=7,
        )
        allowed_xz = np.ones(
            (world.shape[0], world.shape[2]), dtype=bool
        )

        sim._advance_object_bbox_fill(0.0, allowed_xz)
        sim._advance_object_bbox_fill(10.0, allowed_xz)

        local_flame = world.flame[object_slice]
        self.assertTrue(np.all(local_flame[:, :3, :] >= 0.2))
        self.assertTrue(np.all(local_flame[:, 3:, :] == 0.0))
        self.assertTrue(
            np.all(sim._object_fill_visible[object_slice][:, 3:, :] == 0)
        )

    def test_radial_gaussian_has_bright_core_and_dim_advancing_edge(self):
        world, sim, source = self._make_sim(
            floor_spread_speed_m_per_s=0.1,
            floor_max_spread_radius_m=0.8,
        )

        influence = sim._floor_spread_influence_xz(4.0)
        core = world.world_to_grid(source)
        shoulder = world.world_to_grid(source + np.array([0.4, 0.0, 0.0]))
        outside = world.world_to_grid(source + np.array([1.0, 0.0, 0.0]))
        self.assertGreater(
            float(influence[core[0], core[2]]),
            float(influence[shoulder[0], shoulder[2]]),
        )
        self.assertGreater(float(influence[shoulder[0], shoulder[2]]), 0.0)
        self.assertEqual(float(influence[outside[0], outside[2]]), 0.0)
        # A centred source in a square world must be rotationally symmetric
        # in XZ; propagation is radial rather than axis- or route-directed.
        np.testing.assert_allclose(influence, influence.T, atol=1e-6)

    def test_duration_deadline_uses_one_constant_speed_and_reaches_radius(self):
        _, sim, _ = self._make_sim(
            duration_s=10.0,
            floor_spread_speed_m_per_s=0.01,
            floor_max_spread_radius_m=0.8,
            floor_spread_reach_fraction=0.5,
        )

        speed = sim._floor_spread_effective_speed_m_per_s(
            source_radius_m=0.2,
            ignition_time_s=0.0,
            floor_spread_scale=1.0,
        )
        self.assertAlmostEqual(speed, 0.12)
        self.assertAlmostEqual(
            sim._floor_spread_radius_m(0.2, 0.0, 1.0, 0.0),
            0.2,
        )
        self.assertAlmostEqual(
            sim._floor_spread_radius_m(0.2, 0.0, 1.0, 2.5),
            0.5,
        )
        self.assertAlmostEqual(
            sim._floor_spread_radius_m(0.2, 0.0, 1.0, 5.0),
            0.8,
        )
        self.assertAlmostEqual(
            sim._floor_spread_radius_m(0.2, 0.0, 1.0, 10.0),
            0.8,
        )

    def test_duration_deadline_respects_source_scale_and_remaining_time(self):
        _, sim, _ = self._make_sim(
            duration_s=20.0,
            floor_spread_speed_m_per_s=0.01,
            floor_max_spread_radius_m=1.0,
            floor_spread_reach_fraction=0.5,
        )

        # A delayed half-scale source has 15 s remaining and therefore a
        # 7.5 s deadline to grow from 0.2 m to its 0.5 m target.
        speed = sim._floor_spread_effective_speed_m_per_s(
            source_radius_m=0.2,
            ignition_time_s=5.0,
            floor_spread_scale=0.5,
        )
        self.assertAlmostEqual(speed, 0.04)
        self.assertAlmostEqual(
            sim._floor_spread_radius_m(0.2, 5.0, 0.5, 12.5),
            0.5,
        )

    def test_archived_plan_without_deadline_keeps_fixed_speed(self):
        _, sim, _ = self._make_sim(
            duration_s=10.0,
            floor_spread_speed_m_per_s=0.1,
            floor_max_spread_radius_m=0.8,
        )

        self.assertEqual(sim.floor_spread_reach_fraction, 0.0)
        self.assertAlmostEqual(
            sim._floor_spread_effective_speed_m_per_s(0.2, 0.0, 1.0),
            0.1,
        )
        self.assertAlmostEqual(
            sim._floor_spread_radius_m(0.2, 0.0, 1.0, 4.0),
            0.6,
        )

    def test_configured_speed_remains_minimum_under_duration_deadline(self):
        _, sim, _ = self._make_sim(
            duration_s=100.0,
            floor_spread_speed_m_per_s=0.1,
            floor_max_spread_radius_m=0.8,
            floor_spread_reach_fraction=0.9,
        )

        self.assertAlmostEqual(
            sim._floor_spread_effective_speed_m_per_s(0.2, 0.0, 1.0),
            0.1,
        )

    def test_run_propagation_records_duration_aware_speed(self):
        rules = {
            "thermal_diffusivity": 0.0,
            "smoke_diffusivity": 0.0,
            "buoyancy_v_m_per_s": 0.0,
            "ceiling_jet_speed_m_per_s": 0.0,
            "radiative_gain_c": 0.0,
            "spread_speed_m_per_s": 0.0,
            "floor_fuel_value": 0.0,
            "floor_spread_speed_m_per_s": 0.01,
            "floor_max_spread_radius_m": 0.8,
            "floor_spread_reach_fraction": 0.5,
            "flame_column_cells": 3,
            "max_flame_column_height_m": 0.3,
            "object_bbox_max_vertical_spread_m": 0.7,
            "object_flame_extra_height_cells": 0,
        }
        plan = {
            "scene_id": "synthetic",
            "plan_id": "duration-test",
            "world_aabb": [0.0, 0.0, 0.0, 1.0, 0.4, 1.0],
            "duration_s": 4.0,
            "seed": 3,
            "ignitions": [
                {
                    "position": [0.5, 0.1, 0.5],
                    "ignite_time_s": 0.0,
                    "source_radius_m": 0.2,
                    "source_temp_c": 700.0,
                    "smoke_yield": 0.0,
                }
            ],
            "propagation_rules": rules,
        }
        result = run_propagation(
            inventory={"instances": []},
            plan=plan,
            voxel_m=0.2,
            dt=1.0,
            save_dt=1.0,
            keep_in_memory=True,
            verbose=False,
        )

        self.assertEqual(result["meta"]["floor_spread_reach_fraction"], 0.5)
        self.assertEqual(
            len(result["meta"]["floor_source_effective_speeds_m_per_s"]),
            1,
        )
        self.assertAlmostEqual(
            result["meta"]["floor_source_effective_speeds_m_per_s"][0],
            0.3,
        )
        self.assertEqual(result["meta"]["flame_column_cells"], 3)
        self.assertEqual(result["meta"]["effective_flame_column_cells"], 1)
        self.assertAlmostEqual(
            result["meta"]["effective_flame_column_height_m"], 0.2
        )
        self.assertAlmostEqual(
            result["meta"]["max_flame_column_height_m"], 0.3
        )
        self.assertAlmostEqual(
            result["meta"]["object_bbox_max_vertical_spread_m"], 0.7
        )
        self.assertEqual(
            result["meta"]["object_flame_extra_height_cells"], 0
        )

    def test_floor_front_fades_in_continuously_inside_hard_envelope(self):
        world, sim, source = self._make_sim(
            floor_spread_speed_m_per_s=0.1,
            floor_max_spread_radius_m=0.8,
            floor_front_softness_m=0.18,
        )
        del source

        early = sim._floor_spread_influence_xz(3.0)
        later = sim._floor_spread_influence_xz(4.0)
        allowed = sim._floor_spread_allowed_xz(3.0)
        soft_cells = allowed & (early > 0.0) & (early < 0.4)

        self.assertTrue(soft_cells.any())
        self.assertTrue(np.all(later[soft_cells] > early[soft_cells]))
        self.assertTrue(
            np.all(early[~allowed] == 0.0)
        )

    def test_intensity_presets_keep_moderately_bounded_local_envelopes(self):
        light = _default_propagation_rules("light")
        medium = _default_propagation_rules("medium")
        severe = _default_propagation_rules("severe")

        self.assertEqual(light["flame_column_cells"], 2)
        self.assertEqual(medium["flame_column_cells"], 3)
        self.assertEqual(severe["flame_column_cells"], 3)
        self.assertEqual(TEMPLATE_VERSION, 10)
        self.assertAlmostEqual(light["max_flame_column_height_m"], 0.20)
        self.assertAlmostEqual(medium["max_flame_column_height_m"], 0.30)
        self.assertAlmostEqual(severe["max_flame_column_height_m"], 0.35)
        self.assertAlmostEqual(
            light["object_bbox_max_vertical_spread_m"], 0.55
        )
        self.assertAlmostEqual(
            medium["object_bbox_max_vertical_spread_m"], 0.75
        )
        self.assertAlmostEqual(
            severe["object_bbox_max_vertical_spread_m"], 0.90
        )
        self.assertEqual(light["object_flame_extra_height_cells"], 0)
        self.assertEqual(medium["object_flame_extra_height_cells"], 0)
        self.assertEqual(severe["object_flame_extra_height_cells"], 0)
        self.assertAlmostEqual(light["floor_max_spread_radius_m"], 1.43)
        self.assertAlmostEqual(medium["floor_max_spread_radius_m"], 2.20)
        self.assertAlmostEqual(severe["floor_max_spread_radius_m"], 2.97)
        self.assertAlmostEqual(light["floor_spread_speed_m_per_s"], 0.0033)
        self.assertAlmostEqual(medium["floor_spread_speed_m_per_s"], 0.0044)
        self.assertAlmostEqual(severe["floor_spread_speed_m_per_s"], 0.0055)
        self.assertAlmostEqual(light["object_max_spread_radius_m"], 1.188)
        self.assertAlmostEqual(medium["object_max_spread_radius_m"], 1.35)
        self.assertAlmostEqual(severe["object_max_spread_radius_m"], 1.512)
        self.assertAlmostEqual(light["object_bbox_fill_speed_m_per_s"], 0.006)
        self.assertAlmostEqual(medium["object_bbox_fill_speed_m_per_s"], 0.008)
        self.assertAlmostEqual(severe["object_bbox_fill_speed_m_per_s"], 0.01)
        self.assertAlmostEqual(light["spread_speed_m_per_s"], 0.084)
        self.assertAlmostEqual(medium["spread_speed_m_per_s"], 0.12)
        self.assertAlmostEqual(severe["spread_speed_m_per_s"], 0.18)
        self.assertAlmostEqual(light["radiative_gain_c"], 120.0)
        self.assertAlmostEqual(medium["radiative_gain_c"], 200.0)
        self.assertAlmostEqual(severe["radiative_gain_c"], 300.0)
        self.assertEqual(medium["radiative_radius_cells"], 4)
        self.assertEqual(medium["floor_ignite_radius_cells"], 2)
        self.assertEqual(medium["floor_fuel_value"], 0.52)
        self.assertEqual(medium["floor_spread_reach_fraction"], 0.90)
        self.assertEqual(light["floor_spread_reach_fraction"], 0.90)
        self.assertEqual(severe["floor_spread_reach_fraction"], 0.90)
        self.assertEqual(medium["floor_gaussian_sigma_fraction"], 0.58)
        self.assertEqual(medium["floor_front_softness_m"], 0.18)
        self.assertEqual(
            medium["object_bbox_fill_front_width_layers"], 1.0
        )
        for rules in (light, medium, severe):
            self.assertNotIn("secondary_floor_spread_scale", rules)
            self.assertNotIn("floor_link_sigma_m", rules)


if __name__ == "__main__":
    unittest.main()
