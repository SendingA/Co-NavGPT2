"""Focused regressions for source-centred floor-fire propagation."""

import unittest

import numpy as np

from utils.fire_world.propagation import FirePropagation
from utils.fire_world.voxel_world import VoxelWorld


class FloorFireSpreadTest(unittest.TestCase):
    def _make_sim(self, **rule_overrides):
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
        sim = FirePropagation(world, rules, seed=7)
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

    def test_old_plans_receive_bounded_defaults(self):
        world = VoxelWorld.from_aabb(
            [0.0, 0.0, 0.0, 1.0, 0.4, 1.0], voxel=0.2
        )
        sim = FirePropagation(world, rules={})

        self.assertEqual(sim.floor_spread_speed_m_per_s, 0.015)
        self.assertEqual(sim.floor_max_spread_radius_m, 2.0)


if __name__ == "__main__":
    unittest.main()
