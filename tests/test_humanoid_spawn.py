"""Habitat-Sim smoke test for static-person placement and reuse."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import habitat_sim

    from envs.random_humanoid import RandomHumanoidWalker

    _HAS_HABITAT = True
except (ImportError, ModuleNotFoundError):
    habitat_sim = None
    RandomHumanoidWalker = None
    _HAS_HABITAT = False


@unittest.skipUnless(_HAS_HABITAT, "requires the Habitat 3 environment")
class StaticHumanoidSpawnTests(unittest.TestCase):
    def test_static_spawn_reuses_alive_humanoid_at_new_goal(self) -> None:
        scene = PROJECT_ROOT / (
            "data/scene_datasets/hm3d_v0.2/val/00859-3t8DB4Uzvkt/"
            "3t8DB4Uzvkt.basis.glb"
        )
        humanoid_dir = (
            PROJECT_ROOT / "data/humanoids/humanoid_data/female_2"
        )
        if not scene.is_file() or not humanoid_dir.is_dir():
            self.skipTest("HM3D scene or Habitat humanoid assets unavailable")

        simulator_config = habitat_sim.SimulatorConfiguration()
        simulator_config.scene_id = str(scene)
        simulator_config.create_renderer = False
        simulator_config.enable_physics = True
        sim = habitat_sim.Simulator(
            habitat_sim.Configuration(
                simulator_config,
                [habitat_sim.agent.AgentConfiguration()],
            )
        )
        try:
            walker = RandomHumanoidWalker(
                sim=sim,
                num_humans=1,
                urdf_path=str(humanoid_dir / "female_2.urdf"),
                motion_data_path=str(
                    humanoid_dir / "female_2_motion_data_smplx.pkl"
                ),
            )
            first_goal = np.array([-1.2, 0.0, -7.6], dtype=np.float32)
            walker.reset([first_goal], static=True)
            first_humanoid = walker.humans[0]
            first_sim_obj = first_humanoid.sim_obj
            render_asset = Path(
                first_sim_obj.creation_attributes.render_asset_fullpath
            )

            self.assertTrue(first_sim_obj.is_alive)
            self.assertTrue(render_asset.is_file())
            self.assertEqual(render_asset.suffix.lower(), ".glb")
            self.assertTrue(
                np.allclose(first_humanoid.base_pos, first_goal, atol=1e-5)
            )

            # A live articulated object is reused across episode resets and is
            # moved exactly to the new dataset goal.
            second_goal = np.array([0.5, 0.0, -3.0], dtype=np.float32)
            walker.reset([second_goal], static=True)
            self.assertIs(walker.humans[0], first_humanoid)
            self.assertIs(walker.humans[0].sim_obj, first_sim_obj)
            self.assertTrue(
                np.allclose(
                    walker.humans[0].base_pos, second_goal, atol=1e-5
                )
            )

            # Static walkers remain fixed even if the shared main loop calls
            # step() before each Habitat environment step.
            walker.step()
            self.assertTrue(
                np.allclose(
                    walker.humans[0].base_pos, second_goal, atol=1e-5
                )
            )
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
