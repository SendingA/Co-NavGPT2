"""Integration regressions for risk prompting, timing, outputs and fallback."""
from __future__ import annotations

import importlib
import inspect
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

from utils.fire_pipeline import step_fire_observation
from utils.fire_sensors.sensors.voxel_smoke import VoxelSmokeSensor
from utils.fire_sensors.suite import FireSensorSuite
from utils.global_planners import low_risk_fallback_goal
from utils.risk.runtime import RiskRuntime


class RiskPromptRegressionTests(unittest.TestCase):
    @staticmethod
    def _prompt_modules():
        # chat_utils has a legacy import-time argparse call.  Isolate it from
        # unittest's flags without changing production behaviour.
        with mock.patch.object(sys, "argv", ["test_risk_integration"]):
            prompts = importlib.import_module("system_prompt")
            chat_utils = importlib.import_module("utils.chat_utils")
        return prompts, chat_utils

    def test_normal_and_risk_system_prompts_are_independent(self) -> None:
        prompts, _ = self._prompt_modules()
        source = inspect.getsource(prompts)

        for normal_prompt in (
            prompts.system_prompt,
            prompts.obs_system_prompt,
            prompts.full_system_prompt,
        ):
            self.assertNotIn("hard_blocked", normal_prompt)
            self.assertNotIn("hazard_report", normal_prompt)
        self.assertIn("global top-view", prompts.full_system_prompt)
        self.assertIn("global top-view", prompts.risk_prompt)
        self.assertIn("hard_blocked=true", prompts.risk_prompt)
        self.assertIn("route_max_risk", prompts.risk_prompt)
        self.assertIn("Low confidence means uncertain, not safe", prompts.risk_prompt)
        self.assertIn("Return a JSON object only", prompts.risk_prompt)
        self.assertNotIn("def _risk_prompt", source)
        self.assertNotIn("_RISK_SAFETY_GUIDANCE", source)
        self.assertFalse(hasattr(prompts, "risk_system_prompt"))
        self.assertFalse(hasattr(prompts, "risk_obs_system_prompt"))
        self.assertFalse(hasattr(prompts, "risk_full_system_prompt"))

    def test_normal_and_risk_user_messages_are_separate(self) -> None:
        prompts, chat_utils = self._prompt_modules()
        candidate_maps = [BytesIO(b"frontier-0"), BytesIO(b"frontier-1")]
        normal = chat_utils.message_prepare(
            prompts.system_prompt,
            candidate_maps,
            "chair",
            num_agents=3,
        )
        risk = chat_utils.risk_message_prepare(
            prompts.risk_prompt,
            candidate_maps,
            "chair",
            risk_context={
                "hazard_report": [
                    {
                        "frontier_id": 0,
                        "hard_blocked": True,
                        "route_max_risk": 0.9,
                    },
                    {
                        "frontier_id": 1,
                        "hard_blocked": False,
                        "route_max_risk": 0.2,
                    },
                ]
            },
            num_agents=3,
        )
        self.assertEqual(normal[0]["content"], prompts.system_prompt)
        self.assertEqual(
            normal[1]["content"][0]["text"],
            "3 robots need to find a chair",
        )
        self.assertEqual(risk[0]["content"], prompts.risk_prompt)
        risk_text = risk[1]["content"][0]["text"]
        self.assertTrue(risk_text.startswith("Risk-aware frontier assignment"))
        self.assertIn("Robots (3): robot_0, robot_1, robot_2", risk_text)
        self.assertIn("Candidate frontiers: 2 images", risk_text)
        self.assertIn('"hard_blocked": true', risk_text)
        self.assertIn('"route_max_risk": 0.2', risk_text)
        self.assertNotIn("hazard_report", normal[1]["content"][0]["text"])
        # The hazard report stays inside the text block and is not counted as
        # a candidate image/frontier.
        self.assertEqual(len(normal[1]["content"]), 3)
        self.assertEqual(len(risk[1]["content"]), 3)
        with self.assertRaises(TypeError):
            chat_utils.message_prepare(
                prompts.system_prompt,
                candidate_maps,
                "chair",
                risk_context={},
                num_agents=3,
            )
        with self.assertRaisesRegex(ValueError, "risk_context is required"):
            chat_utils.risk_message_prepare(
                prompts.risk_prompt,
                candidate_maps,
                "chair",
                risk_context=None,
                num_agents=3,
            )

    def test_global_planner_selects_only_the_dedicated_risk_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        main_source = (root / "main.py").read_text(encoding="utf-8")
        gpt_source = (
            root / "utils" / "global_planners" / "gpt.py"
        ).read_text(encoding="utf-8")
        self.assertIn("self._chat.risk_message_prepare(", gpt_source)
        self.assertIn("self._prompts.risk_prompt,", gpt_source)
        self.assertNotIn("risk_message_prepare(", main_source)
        self.assertNotIn("risk_system_prompt", gpt_source)


class SharedFireTimeTests(unittest.TestCase):
    @staticmethod
    def _voxel_output(rgb: np.ndarray, depth: np.ndarray, t_sim_s: float) -> dict:
        height, width = depth.shape[:2]
        zeros = np.zeros((height, width), dtype=np.float32)
        return {
            "image": rgb.copy(),
            "transmittance": np.ones((height, width), dtype=np.float32),
            "flame_mask": zeros,
            "thermal_image": np.zeros((height, width, 3), dtype=np.uint8),
            "thermal_temperature": np.full((height, width), 25.0, dtype=np.float32),
            "t_sim_s": float(t_sim_s),
            "robot_step": 3,
        }

    def test_fire_sensor_suite_forwards_shared_time_to_voxel_sensor(self) -> None:
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.ones((2, 3), dtype=np.float32)

        class StubVoxel:
            def __init__(self, owner):
                self.owner = owner

            def process(self, rgb_in, depth_in, **kwargs):
                self.owner.voxel_kwargs = kwargs
                return SharedFireTimeTests._voxel_output(
                    rgb_in, depth_in, kwargs["t_sim_s"]
                )

        class StubDepth:
            @staticmethod
            def process(_rgb, depth_in, transmittance=None):
                del transmittance
                return {"depth": depth_in.copy()}

        class StubRadar:
            @staticmethod
            def process(_rgb, depth_in):
                shape = depth_in.shape[:2]
                image = np.zeros(shape + (3,), dtype=np.uint8)
                return {
                    "heatmap": np.zeros(shape, dtype=np.float32),
                    "image_az": image,
                    "image_el": image,
                    "image_bev": image,
                    "points": np.zeros((0, 2), dtype=np.float32),
                    "points_3d": np.zeros((0, 3), dtype=np.float32),
                }

        class StubLidar:
            @staticmethod
            def process(_rgb, depth_in, obs=None):
                del obs
                return {
                    "image": np.zeros(depth_in.shape[:2] + (3,), dtype=np.uint8),
                    "points": np.zeros((0, 3), dtype=np.float32),
                }

        suite = object.__new__(FireSensorSuite)
        suite.cfg = SimpleNamespace(max_depth_m=5.0, dashboard_size=(320, 240))
        suite.scene = object()
        suite.voxel_sensor = StubVoxel(suite)
        suite.depth_sensor = StubDepth()
        suite.radar_sensor = StubRadar()
        suite.lidar_sensor = StubLidar()
        suite.last_dashboard = None

        with mock.patch(
            "utils.fire_sensors.suite.render_dashboard",
            return_value=np.zeros((12, 12, 3), dtype=np.uint8),
        ):
            output = suite.process(
                rgb,
                depth,
                agent_state=object(),
                robot_step=3,
                t_sim_s=42.25,
            )
        self.assertEqual(suite.voxel_kwargs["t_sim_s"], 42.25)
        self.assertEqual(output["t_sim_s"], 42.25)
        self.assertEqual(output["robot_step"], 3)

    def test_voxel_sensor_uses_override_without_resampling_clock(self) -> None:
        class FakeScene:
            origin = np.zeros(3, dtype=np.float32)
            voxel_m = 1.0
            shape = (1, 1, 1)
            ambient_c = 25.0

            def __init__(self):
                self.clock_calls = 0
                self.query_times = []

            @staticmethod
            def camera_pose(_agent_state):
                return np.zeros(3), np.eye(3)

            def t_sim(self, _robot_step):
                self.clock_calls += 1
                return 99.0

            def query(self, timestamp_s):
                self.query_times.append(float(timestamp_s))
                zero = np.zeros(self.shape, dtype=np.float32)
                return zero, zero, np.full(self.shape, 25.0, dtype=np.float32)

        scene = FakeScene()
        sensor = VoxelSmokeSensor(
            SimpleNamespace(),
            camera_K=SimpleNamespace(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            scene=scene,
        )
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        depth = np.ones((2, 2), dtype=np.float32)

        def fake_render(**kwargs):
            return self._voxel_output(
                kwargs["rgb_clean"], kwargs["depth_m"], kwargs["t_sim"]
            )

        with mock.patch.object(sensor, "_params", return_value=object()), mock.patch(
            "utils.fire_sensors.sensors.voxel_smoke.volumetric_composite",
            side_effect=fake_render,
        ):
            shared = sensor.process(
                rgb, depth, agent_state=object(), robot_step=3, t_sim_s=17.5
            )
            local = sensor.process(rgb, depth, agent_state=object(), robot_step=4)

        self.assertEqual(shared["t_sim_s"], 17.5)
        self.assertEqual(scene.query_times[0], 17.5)
        self.assertEqual(scene.clock_calls, 1)
        self.assertEqual(local["t_sim_s"], 99.0)
        self.assertEqual(scene.query_times[1], 99.0)

    def test_fire_pipeline_passes_the_same_explicit_time(self) -> None:
        class StubSuite:
            scene = object()

            def process(self, rgb, depth, **kwargs):
                self.kwargs = kwargs
                return SharedFireTimeTests._voxel_output(
                    rgb, depth, kwargs["t_sim_s"]
                ) | {"rgb_smoke": rgb.copy(), "depth_smoke": depth.copy(),
                     "thermal_flame_mask": np.zeros(depth.shape, np.float32)}

        suite = StubSuite()
        observations = {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2, 1), dtype=np.float32),
        }
        sensors = step_fire_observation(
            observations=observations,
            suite=suite,
            agent_state=object(),
            robot_step=8,
            t_sim_s=31.75,
            config=SimpleNamespace(),
            args=SimpleNamespace(
                depth_use_clean=1,
                fire_apply_to_obs=1,
                use_thermal_perception=1,
            ),
        )
        self.assertEqual(suite.kwargs["t_sim_s"], 31.75)
        self.assertEqual(sensors["t_sim_s"], 31.75)


class RiskRuntimeArtifactTests(unittest.TestCase):
    def test_runtime_writes_jsonl_png_and_summary(self) -> None:
        class FakeWorld:
            origin = np.array([-2.0, 0.0, -2.0])
            voxel_m = 1.0
            shape = (4, 2, 4)

            @staticmethod
            def frame_index(timestamp_s):
                return int(timestamp_s)

            def query(self, _timestamp_s):
                flame = np.full(self.shape, 0.10, dtype=np.float32)
                smoke = np.full(self.shape, 0.20, dtype=np.float32)
                temperature = np.full(self.shape, 50.0, dtype=np.float32)
                return flame, smoke, temperature

        class FakeScene:
            def __init__(self):
                self.fw = FakeWorld()

            @staticmethod
            def t_sim(robot_step):
                return float(robot_step)

            @staticmethod
            def camera_pose(_agent_state):
                return np.zeros(3), np.eye(3)

        reference = SimpleNamespace(
            init_sim_rotation=np.eye(3),
            init_sim_position=np.zeros(3),
            init_agent_position=np.zeros(3),
            local_w=4,
            local_h=4,
            args=SimpleNamespace(map_resolution=100),
        )
        states = [
            SimpleNamespace(position=np.array([0.0, 0.0, 0.0])),
            SimpleNamespace(position=np.array([1.0, 0.0, 0.0])),
        ]

        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(
                risk_enabled=1,
                risk_source="oracle",
                risk_dump_dir=temporary,
                risk_save_every=1,
                risk_smoke_source="appearance_depth",
            )
            runtime = RiskRuntime(
                fire_scene=FakeScene(),
                reference_agent=reference,
                args=args,
                episode_id=7,
            )
            self.assertEqual(runtime.step_log_path.read_text(), "")
            run_config = json.loads(
                (runtime.run_dir / "risk_config.json").read_text()
            )
            self.assertEqual(run_config["planner_source"], "oracle")
            self.assertNotIn("flame", run_config["risk_config"]["weights"])
            self.assertEqual(
                run_config["risk_config"]["weights"],
                {"temperature": 0.6, "smoke": 0.4},
            )
            runtime.evaluator.prime(
                0.0,
                [state.position for state in states],
                floor_y_m=runtime.floor_y_m,
            )
            runtime.record_exposure(
                1.0,
                states,
                step=1,
                planner_statuses=["unsafe_goal", None],
                actions=[1, 3],
                wall_time_s=0.125,
            )
            layers, planning_risk = runtime.planner_state(1.0)
            runtime.save_step(
                step=1,
                timestamp_s=1.0,
                layers=layers,
                planning_risk=planning_risk,
                obstacle_map=np.zeros(runtime.frame.shape, dtype=np.float32),
                agent_cells=[[2, 2], [2, 3]],
                frontier_points=[[1, 1]],
                frontier_reports=[{"frontier_id": 0, "hard_blocked": False}],
            )

            records = [
                json.loads(line)
                for line in runtime.step_log_path.read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            exposure, snapshot = records
            self.assertEqual(exposure["record_type"], "exposure")
            self.assertEqual(set(exposure["agents"]), {"0", "1"})
            self.assertEqual(exposure["planner_statuses"], ["unsafe_goal", None])
            action_records = [
                json.loads(line)
                for line in runtime.action_log_path.read_text().splitlines()
            ]
            self.assertEqual(len(action_records), 1)
            self.assertEqual(
                [item["action_name"] for item in action_records[0]["actions"]],
                ["move_forward", "turn_right"],
            )
            self.assertEqual(action_records[0]["wall_time_s"], 0.125)
            self.assertEqual(snapshot["record_type"], "planner_snapshot")
            self.assertEqual(snapshot["planner_source"], "oracle")
            self.assertEqual(snapshot["t_sim_s"], 1.0)
            self.assertEqual(snapshot["frontiers"][0]["frontier_id"], 0)

            png_path = runtime.episode_dir / "risk_step_00001.png"
            self.assertTrue(png_path.is_file())
            rendered = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertGreater(rendered.shape[0], 0)
            self.assertGreater(rendered.shape[1], 0)

            summary = runtime.summary(habitat_success=1.0)
            self.assertEqual(summary["planner_source"], "oracle")
            self.assertGreater(summary["team"]["CHE"], 0.0)
            self.assertEqual(summary["safe_success"], 1.0)
            self.assertEqual(summary["team"]["safe_refusal_steps"], 1)
            summary_path = runtime.save_summary(summary)
            self.assertEqual(json.loads(summary_path.read_text()), summary)
            action_list = json.loads(runtime.action_list_path.read_text())
            self.assertEqual(action_list["num_steps"], 1)
            self.assertEqual(action_list["num_agents"], 2)
            self.assertEqual(action_list["total_actions"], 2)
            self.assertEqual(action_list["total_wall_time_s"], 0.125)
            self.assertEqual(
                action_list["per_agent_action_ids"], {"0": [1], "1": [3]}
            )
            self.assertEqual(
                action_list["per_agent_action_names"],
                {"0": ["move_forward"], "1": ["turn_right"]},
            )
            self.assertEqual(action_list["steps"], action_records)

            off_floor = [
                SimpleNamespace(position=np.array([0.0, 2.0, 0.0])),
                states[1],
            ]
            with self.assertRaisesRegex(RuntimeError, "one floor-aware"):
                runtime.record_exposure(1.0, off_floor)

            privileged_args = SimpleNamespace(
                risk_enabled=1,
                risk_source="sensed",
                risk_dump_dir=temporary,
                risk_save_every=0,
                risk_smoke_source="privileged_transmittance",
            )
            privileged_runtime = RiskRuntime(
                fire_scene=FakeScene(),
                reference_agent=reference,
                args=privileged_args,
                episode_id=8,
            )
            sensor = {
                "rgb_smoke": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth_clean": np.ones((2, 2), dtype=np.float32),
                "depth_smoke": np.ones((2, 2), dtype=np.float32),
                "thermal_temperature": np.full((2, 2), 25.0, np.float32),
                "thermal_flame_mask": np.zeros((2, 2), np.float32),
                "t_sim_s": 1.0,
            }
            camera_k = SimpleNamespace(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
            with self.assertRaisesRegex(RuntimeError, "no transmittance"):
                privileged_runtime.update_sensed(
                    timestamp_s=1.0,
                    sensor_outputs=[sensor],
                    agent_states=states[:1],
                    camera_k=camera_k,
                )
            sensor["transmittance"] = np.ones((2, 2), dtype=np.float32)
            sensor["t_sim_s"] = 2.0
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                privileged_runtime.update_sensed(
                    timestamp_s=1.0,
                    sensor_outputs=[sensor],
                    agent_states=states[:1],
                    camera_k=camera_k,
                )


class LowRiskFallbackTests(unittest.TestCase):
    def test_fallback_prefers_low_risk_explored_and_non_hard_cell(self) -> None:
        shape = (9, 9)
        obstacle = np.zeros(shape, dtype=np.float32)
        explored = np.ones(shape, dtype=np.float32)
        planning_risk = np.full(shape, 0.9, dtype=np.float32)
        hard_unsafe = np.zeros(shape, dtype=bool)
        planning_risk[0, 0] = 0.01
        planning_risk[8, 8] = 0.20
        selected = low_risk_fallback_goal(
            [4, 4], obstacle, explored, planning_risk, hard_unsafe
        )
        self.assertEqual(selected, [0, 0])

        hard_unsafe[0, 0] = True
        selected_without_hard = low_risk_fallback_goal(
            [4, 4], obstacle, explored, planning_risk, hard_unsafe
        )
        self.assertEqual(selected_without_hard, [8, 8])

        # An attractive point behind an obstacle partition must not be used
        # as a fallback because the local FMM cannot reach it safely.
        larger = (21, 21)
        obstacle = np.zeros(larger, dtype=np.float32)
        obstacle[:, 10] = 1.0
        explored = np.ones(larger, dtype=np.float32)
        planning_risk = np.full(larger, 0.4, dtype=np.float32)
        planning_risk[10, 18] = 0.0
        planning_risk[2, 2] = 0.1
        selected_connected = low_risk_fallback_goal(
            [10, 3],
            obstacle,
            explored,
            planning_risk,
            np.zeros(larger, dtype=bool),
        )
        self.assertLess(selected_connected[1], 10)


if __name__ == "__main__":
    unittest.main()
