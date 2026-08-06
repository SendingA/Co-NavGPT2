"""Regression tests for the Habitat PointNav frontier-policy adapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.local_planners.pointnav import (
    PointNavDecision,
    PointNavPolicyAdapter,
    apply_pointnav_simulator_schema,
    build_pointnav_runtime_spec,
    compute_compass,
    compute_episode_gps,
    compute_pointgoal,
    frontier_grid_to_world,
    load_pointnav_policy_adapter,
    shield_pointnav_action,
)
from utils.visualization import fit_image_to_panel

OFFICIAL_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "ddppo-models"
    / "gibson-2plus-resnet50.pth"
)


def _pointnav_config(
    *,
    height=8,
    width=10,
    hfov=90,
    goal_sensor="integrated",
):
    rgb = SimpleNamespace(
        type="HabitatSimRGBSensor",
        height=height,
        width=width,
        hfov=hfov,
        position=[0.0, 1.25, 0.0],
        orientation=[0.0, 0.0, 0.0],
        sensor_subtype="PINHOLE",
        noise_model="None",
        noise_model_kwargs={},
    )
    depth = SimpleNamespace(
        type="HabitatSimDepthSensor",
        height=height,
        width=width,
        hfov=hfov,
        position=[0.0, 1.25, 0.0],
        orientation=[0.0, 0.0, 0.0],
        sensor_subtype="PINHOLE",
        noise_model="None",
        noise_model_kwargs={},
        min_depth=0.0,
        max_depth=10.0,
        normalize_depth=True,
    )
    if goal_sensor == "integrated":
        lab_sensors = {
            "pointgoal_with_gps_compass_sensor": SimpleNamespace(
                type="PointGoalWithGPSCompassSensor",
                goal_format="POLAR",
                dimensionality=2,
            )
        }
    else:
        lab_sensors = {
            "gps_sensor": SimpleNamespace(
                type="GPSSensor", dimensionality=2
            ),
            "compass_sensor": SimpleNamespace(type="CompassSensor"),
        }
    return SimpleNamespace(
        habitat=SimpleNamespace(
            simulator=SimpleNamespace(
                agents_order=["main_agent"],
                agents={
                    "main_agent": SimpleNamespace(
                        sim_sensors={
                            "rgb_sensor": rgb,
                            "depth_sensor": depth,
                        }
                    )
                },
                forward_step_size=0.25,
                turn_angle=30.0,
            ),
            task=SimpleNamespace(
                lab_sensors=lab_sensors,
                actions={
                    "stop": {},
                    "move_forward": {},
                    "turn_left": {},
                    "turn_right": {},
                },
            ),
            gym=SimpleNamespace(obs_keys=None),
        ),
        habitat_baselines=SimpleNamespace(
            rl=SimpleNamespace(
                policy={
                    "main_agent": SimpleNamespace(
                        name="PointNavResNetPolicy"
                    )
                },
                ddppo=SimpleNamespace(
                    backbone="resnet18",
                    rnn_type="GRU",
                    num_recurrent_layers=1,
                ),
                ppo=SimpleNamespace(hidden_size=128),
            )
        ),
    )


class _ActionData:
    def __init__(self, action, hidden):
        self.actions = torch.tensor([[action]], dtype=torch.long)
        self.rnn_hidden_states = hidden

    @property
    def env_actions(self):
        return self.actions


class _FakePolicy:
    hidden_state_shape = (2, 4)

    def __init__(self, actions):
        self.actions = list(actions)
        self.inputs = []

    def act(
        self,
        observations,
        hidden,
        prev_action,
        mask,
        deterministic=False,
    ):
        self.inputs.append({
            "observations": observations,
            "hidden": hidden.detach().cpu().clone(),
            "prev_action": prev_action.detach().cpu().clone(),
            "mask": mask.detach().cpu().clone(),
            "deterministic": deterministic,
        })
        action = self.actions.pop(0)
        return _ActionData(action, hidden + 1.0)


def _observations(spec):
    output = {}
    for field in spec.observation_fields:
        if field.kind == "rgb":
            output[field.uuid] = np.zeros(field.shape, dtype=np.uint8)
        elif field.kind == "depth":
            output[field.uuid] = np.zeros(field.shape, dtype=np.float32)
    return output


def _agent_state(position=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        position=np.asarray(position, dtype=np.float32),
        rotation=np.eye(3, dtype=np.float64),
    )


class PointGoalConventionTests(unittest.TestCase):
    def test_polar_pointgoal_matches_habitat_forward_and_right_sign(self):
        forward = compute_pointgoal(
            [0, 0, 0], np.eye(3), [0, 0, -2]
        )
        right = compute_pointgoal(
            [0, 0, 0], np.eye(3), [1, 0, 0]
        )
        np.testing.assert_allclose(forward, [2.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(
            right, [1.0, -np.pi / 2.0], atol=1e-6
        )

    def test_gps_and_compass_match_episode_frame(self):
        gps = compute_episode_gps(
            [1, 0, -2], [0, 0, 0], np.eye(3)
        )
        compass = compute_compass(np.eye(3), np.eye(3))
        np.testing.assert_allclose(gps, [2.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(compass, [0.0], atol=1e-6)

    def test_frontier_grid_conversion_uses_existing_world_transform(self):
        result = frontier_grid_to_world(
            [12, 23],
            origins_grid=[10, 20],
            map_resolution_cm=5,
            camera_local_y=1.25,
            initial_agent_position=[1, 2, 3],
            initial_sensor_rotation=np.eye(3),
        )
        rx = np.asarray(
            [[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float32
        )
        expected = rx.T @ np.asarray([0.1, 1.25, 0.15]) + [1, 2, 3]
        np.testing.assert_allclose(result, expected, atol=1e-6)


class PointNavSchemaTests(unittest.TestCase):
    def test_checkpoint_config_defines_camera_goal_and_action_schema(self):
        spec = build_pointnav_runtime_spec(_pointnav_config())
        self.assertEqual(spec.action_names[0], "stop")
        self.assertEqual(spec.forward_step_size, 0.25)
        self.assertEqual(spec.turn_angle, 30.0)
        self.assertEqual(spec.field_map["rgb"].shape, (8, 10, 3))
        self.assertEqual(spec.field_map["depth"].shape, (8, 10, 1))
        self.assertIn("pointgoal_with_gps_compass", spec.field_map)

    def test_official_depth_weights_select_habitat_pretrained_schema(self):
        from utils.local_planners import pointnav as pointnav_module

        config = _pointnav_config()
        with mock.patch.object(
            pointnav_module,
            "_load_checkpoint_payload",
            return_value={"state_dict": {"actor_critic.fake": torch.zeros(1)}},
        ), mock.patch.object(
            pointnav_module,
            "_load_fallback_config",
            return_value=config,
        ):
            spec = pointnav_module.load_pointnav_runtime_spec(
                "gibson-2plus-resnet50.pth"
            )
        self.assertEqual(spec.observation_mode, "depth")
        self.assertNotIn("rgb", spec.field_map)
        self.assertIn("depth", spec.field_map)
        self.assertEqual(
            config.habitat_baselines.rl.ddppo.backbone, "resnet50"
        )
        self.assertEqual(
            config.habitat_baselines.rl.ddppo.rnn_type, "LSTM"
        )
        self.assertEqual(
            config.habitat_baselines.rl.ppo.hidden_size, 512
        )

    def test_official_rgb_profile_uses_registered_backbone_spelling(self):
        from utils.local_planners import pointnav as pointnav_module

        config = _pointnav_config()
        with mock.patch.object(
            pointnav_module,
            "_load_checkpoint_payload",
            return_value={"state_dict": {"actor_critic.fake": torch.zeros(1)}},
        ), mock.patch.object(
            pointnav_module,
            "_load_fallback_config",
            return_value=config,
        ):
            spec = pointnav_module.load_pointnav_runtime_spec(
                "gibson-2plus-mp3d-train-val-test-se-resneXt50-rgb.pth"
            )
        self.assertEqual(spec.observation_mode, "rgb")
        self.assertEqual(
            config.habitat_baselines.rl.ddppo.backbone, "se_resneXt50"
        )

    def test_square_checkpoint_rgb_is_letterboxed_into_legacy_panel(self):
        image = np.full((256, 256, 3), 137, dtype=np.uint8)
        panel = fit_image_to_panel(image, 640, 480)
        self.assertEqual(panel.shape, (480, 640, 3))
        self.assertTrue(np.all(panel[:, :80] == 0))
        self.assertTrue(np.all(panel[:, 80:560] == 137))
        self.assertTrue(np.all(panel[:, 560:] == 0))

    def test_checkpoint_sensors_do_not_override_mapping_cameras(self):
        spec = build_pointnav_runtime_spec(_pointnav_config())
        config = OmegaConf.create({
            "habitat": {
                "simulator": {
                    "forward_step_size": 0.5,
                    "turn_angle": 15,
                    "agents": {
                        "main_agent": {
                            "sim_sensors": {
                                "rgb_sensor": {
                                    "height": 480,
                                    "width": 640,
                                    "hfov": 79,
                                    "position": [0, 0.88, 0],
                                    "orientation": [0, 0, 0],
                                    "sensor_subtype": "PINHOLE",
                                    "noise_model": "None",
                                    "noise_model_kwargs": {},
                                },
                                "depth_sensor": {
                                    "height": 480,
                                    "width": 640,
                                    "hfov": 79,
                                    "position": [0, 0.88, 0],
                                    "orientation": [0, 0, 0],
                                    "sensor_subtype": "PINHOLE",
                                    "noise_model": "None",
                                    "noise_model_kwargs": {},
                                    "min_depth": 0,
                                    "max_depth": 5,
                                    "normalize_depth": True,
                                },
                            }
                        }
                    },
                },
                "task": {
                    "actions": {
                        "stop": {},
                        "move_forward": {},
                        "turn_left": {},
                        "turn_right": {},
                        "look_up": {},
                        "look_down": {},
                    }
                },
            }
        })
        args = SimpleNamespace()
        apply_pointnav_simulator_schema(config, args, spec)
        mapping_rgb = (
            config.habitat.simulator.agents.main_agent
            .sim_sensors.rgb_sensor
        )
        policy_rgb = (
            config.habitat.simulator.agents.main_agent
            .sim_sensors.pointnav_rgb_sensor
        )
        policy_depth = (
            config.habitat.simulator.agents.main_agent
            .sim_sensors.pointnav_depth_sensor
        )
        self.assertEqual(
            (mapping_rgb.height, mapping_rgb.width, mapping_rgb.hfov),
            (480, 640, 79),
        )
        self.assertEqual(
            (policy_rgb.height, policy_rgb.width, policy_rgb.hfov),
            (8, 10, 90),
        )
        self.assertEqual(policy_rgb.position, [0.0, 1.25, 0.0])
        self.assertEqual(policy_rgb.uuid, "pointnav_rgb")
        self.assertEqual(policy_depth.uuid, "pointnav_depth")
        self.assertEqual(
            args.pointnav_observation_source_map,
            {
                "rgb": "pointnav_rgb",
                "depth": "pointnav_depth",
            },
        )
        self.assertEqual(args.pointnav_env_action_map["stop"], 0)
        self.assertEqual(args.pointnav_env_action_map["turn_right"], 3)

    def test_observation_shape_and_dtype_are_strict(self):
        spec = build_pointnav_runtime_spec(_pointnav_config())
        adapter = PointNavPolicyAdapter(_FakePolicy([1]), spec)
        observations = _observations(spec)
        observations["rgb"] = observations["rgb"].astype(np.float32)
        with self.assertRaisesRegex(TypeError, "dtype"):
            adapter.act(
                robot_id=0,
                observations=observations,
                agent_state=_agent_state(),
                goal_world=[0, 0, -1],
                episode_start_position=[0, 0, 0],
                episode_start_rotation=np.eye(3),
            )

    def test_policy_sensor_is_preferred_over_mapping_depth(self):
        spec = build_pointnav_runtime_spec(
            _pointnav_config(), observation_mode="depth"
        )
        policy = _FakePolicy([1])
        adapter = PointNavPolicyAdapter(policy, spec)
        observations = {
            "depth": np.zeros((8, 10, 1), dtype=np.float32),
            "pointnav_depth": np.ones((8, 10, 1), dtype=np.float32),
        }
        adapter.act(
            robot_id=0,
            observations=observations,
            agent_state=_agent_state(),
            goal_world=[0, 0, -1],
            episode_start_position=[0, 0, 0],
            episode_start_rotation=np.eye(3),
        )
        policy_depth = policy.inputs[0]["observations"]["depth"]
        self.assertTrue(torch.all(policy_depth == 1.0))


@unittest.skipUnless(
    OFFICIAL_CHECKPOINT.is_file()
    and importlib.util.find_spec("habitat_baselines") is not None,
    "official PointNav asset or Habitat-Baselines is unavailable",
)
class OfficialPointNavCheckpointTests(unittest.TestCase):
    def test_official_weights_strict_load_and_run_one_cpu_action(self):
        adapter = load_pointnav_policy_adapter(
            str(OFFICIAL_CHECKPOINT),
            device="cpu",
            deterministic=True,
        )
        self.assertEqual(adapter.spec.observation_mode, "depth")
        self.assertEqual(adapter.policy.hidden_state_shape, (4, 512))
        decision = adapter.act(
            robot_id=91,
            observations={
                "pointnav_depth": np.full(
                    (256, 256, 1), 0.5, dtype=np.float32
                )
            },
            agent_state=_agent_state(),
            goal_world=[0, 0, -1],
            episode_start_position=[0, 0, 0],
            episode_start_rotation=np.eye(3),
        )
        self.assertIn(decision.policy_action, range(4))
        self.assertIn(
            decision.action_name,
            ("stop", "move_forward", "turn_left", "turn_right"),
        )


class PointNavRecurrentStateTests(unittest.TestCase):
    def test_robot_states_are_independent_and_goal_change_resets_one(self):
        spec = build_pointnav_runtime_spec(_pointnav_config())
        policy = _FakePolicy([1, 2, 3, 1])
        adapter = PointNavPolicyAdapter(policy, spec)
        observations = _observations(spec)
        common = {
            "observations": observations,
            "agent_state": _agent_state(),
            "episode_start_position": [0, 0, 0],
            "episode_start_rotation": np.eye(3),
        }
        adapter.act(robot_id=0, goal_world=[0, 0, -2], **common)
        adapter.act(robot_id=1, goal_world=[1, 0, -2], **common)
        adapter.act(robot_id=0, goal_world=[0, 0, -2], **common)

        robot0 = adapter.debug_robot_state(0)
        robot1 = adapter.debug_robot_state(1)
        self.assertTrue(torch.all(robot0["hidden"] == 2))
        self.assertTrue(torch.all(robot1["hidden"] == 1))
        self.assertEqual(robot0["goal_generation"], 1)
        self.assertEqual(robot1["goal_generation"], 1)

        adapter.act(robot_id=0, goal_world=[2, 0, -2], **common)
        robot0_changed = adapter.debug_robot_state(0)
        robot1_unchanged = adapter.debug_robot_state(1)
        self.assertTrue(torch.all(robot0_changed["hidden"] == 1))
        self.assertTrue(torch.all(robot1_unchanged["hidden"] == 1))
        self.assertEqual(robot0_changed["goal_generation"], 2)
        self.assertEqual(robot1_unchanged["goal_generation"], 1)
        self.assertFalse(bool(policy.inputs[3]["mask"].item()))

    def test_policy_stop_is_local_and_clears_recurrence(self):
        spec = build_pointnav_runtime_spec(_pointnav_config())
        adapter = PointNavPolicyAdapter(_FakePolicy([0]), spec)
        decision = adapter.act(
            robot_id=3,
            observations=_observations(spec),
            agent_state=_agent_state(),
            goal_world=[0, 0, -1],
            episode_start_position=[0, 0, 0],
            episode_start_rotation=np.eye(3),
        )
        self.assertIsNone(decision.action)
        self.assertTrue(decision.local_goal_reached)
        self.assertTrue(decision.request_global_replan)
        state = adapter.debug_robot_state(3)
        self.assertIsNone(state["local_goal_world"])
        self.assertFalse(bool(state["mask"].item()))
        self.assertTrue(torch.all(state["hidden"] == 0))


class PointNavRiskShieldTests(unittest.TestCase):
    def test_forward_hard_hazard_is_replaced_by_turn(self):
        hard = np.zeros((21, 21), dtype=bool)
        hard[10, 11:16] = True
        action_map = {
            "stop": 0,
            "move_forward": 1,
            "turn_left": 2,
            "turn_right": 3,
        }
        action = shield_pointnav_action(
            1,
            env_action_map=action_map,
            current_cell=[10, 10],
            relative_angle_deg=0.0,
            hard_unsafe_mask=hard,
            risk_map=np.zeros_like(hard, dtype=np.float32),
            map_resolution_cm=5,
            forward_step_size_m=0.25,
            turn_angle_deg=30,
        )
        self.assertIn(action, (2, 3))
        self.assertNotEqual(action, 0)


class AgentStopInterceptionTests(unittest.TestCase):
    class _StopAdapter:
        env_action_map = {
            "stop": 0,
            "move_forward": 1,
            "turn_left": 2,
            "turn_right": 3,
        }

        def act(self, **kwargs):
            return PointNavDecision(
                action=None,
                policy_action=0,
                action_name="stop",
                local_goal_reached=True,
                request_global_replan=True,
                goal_changed=False,
            )

    def test_both_agents_intercept_stop_without_objectnav_stop(self):
        from agents.vlm_agents import VLM_Agent as MainAgent
        from agents.vlm_multi_agents import VLM_Agent as VectorAgent

        for agent_class in (MainAgent, VectorAgent):
            with self.subTest(agent=agent_class.__module__):
                agent = agent_class.__new__(agent_class)
                agent.pointnav_planner = self._StopAdapter()
                agent.agent_id = 0
                agent._latest_pointnav_observations = {
                    "rgb": np.zeros((1, 1, 3), dtype=np.uint8)
                }
                agent._latest_pointnav_agent_state = _agent_state()
                agent.init_agent_position = np.zeros(3)
                agent.init_agent_rotation = np.eye(3)
                agent.risk_navigation_enabled = False
                agent.pointnav_replan_requested = False
                agent.l_step = 0
                action = agent._pointnav_frontier_act([0, 0, -1])
                self.assertEqual(action, 2)
                self.assertNotEqual(action, 0)
                self.assertTrue(agent.pointnav_replan_requested)
                self.assertEqual(agent.l_step, 1)


if __name__ == "__main__":
    unittest.main()
