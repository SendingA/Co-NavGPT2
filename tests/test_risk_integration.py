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
from utils.risk.metrics import MultiAgentRiskEvaluator
from utils.risk.model import RiskPointSamples
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
        self.assertIn("not an absolute prohibition", prompts.risk_prompt)
        self.assertIn("route_max_risk", prompts.risk_prompt)
        self.assertIn("diagnostic only", prompts.risk_prompt)
        self.assertIn("only respond in JSON", prompts.risk_prompt)
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

    def test_log_only_suite_skips_radar_lidar_and_dashboard(self) -> None:
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        depth = np.ones((2, 3), dtype=np.float32)

        class ForbiddenDiagnostic:
            @staticmethod
            def process(*_args, **_kwargs):
                raise AssertionError("diagnostic sensor should be skipped")

        suite = object.__new__(FireSensorSuite)
        suite.cfg = SimpleNamespace(max_depth_m=5.0, dashboard_size=(320, 240))
        suite.scene = object()
        suite.voxel_sensor = None
        suite.depth_sensor = SimpleNamespace(
            process=lambda _rgb, depth_in, transmittance=None: {
                "depth": depth_in.copy()
            }
        )
        suite.radar_sensor = ForbiddenDiagnostic()
        suite.lidar_sensor = ForbiddenDiagnostic()
        suite.last_dashboard = object()

        output = suite.process(rgb, depth, diagnostics=False)

        self.assertIsNone(suite.last_dashboard)
        self.assertNotIn("radar_heatmap", output)
        self.assertNotIn("lidar_points", output)
        self.assertIn("thermal_temperature", output)
        self.assertIn("depth_smoke", output)

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
            self.assertGreater(summary["team"]["CHE_per_step"], 0.0)
            self.assertLessEqual(summary["team"]["CHE_per_step"], 1.0)
            self.assertEqual(summary["team"]["exposure_samples"], 2)
            self.assertEqual(summary["team"]["joint_steps"], 1)
            self.assertFalse(summary["early_stop"]["triggered"])
            self.assertAlmostEqual(
                summary["safe_success"],
                1.0 - summary["team"]["CHE_per_step"],
            )
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

            quiet_runtime = RiskRuntime(
                fire_scene=FakeScene(),
                reference_agent=reference,
                args=SimpleNamespace(
                    risk_enabled=1,
                    risk_source="oracle",
                    risk_dump_dir=temporary,
                    risk_save_every=0,
                    risk_save_traces=0,
                    risk_smoke_source="appearance_depth",
                ),
                episode_id=10,
            )
            quiet_runtime.evaluator.prime(
                0.0,
                [state.position for state in states],
                floor_y_m=quiet_runtime.floor_y_m,
            )
            quiet_runtime.record_exposure(
                1.0,
                states,
                step=1,
                actions=[1, 3],
                wall_time_s=0.125,
            )
            quiet_layers, quiet_planning_risk = quiet_runtime.planner_state(1.0)
            quiet_runtime.save_step(
                step=1,
                timestamp_s=1.0,
                layers=quiet_layers,
                planning_risk=quiet_planning_risk,
            )
            quiet_summary = quiet_runtime.summary(habitat_success=1.0)
            quiet_summary_path = quiet_runtime.save_summary(quiet_summary)
            self.assertTrue(quiet_summary_path.is_file())
            self.assertIsNone(quiet_summary["action_trace_file"])
            self.assertFalse(quiet_runtime.step_log_path.exists())
            self.assertFalse(quiet_runtime.action_log_path.exists())
            self.assertFalse(quiet_runtime.action_list_path.exists())
            self.assertEqual(
                list(quiet_runtime.episode_dir.glob("risk_step_*.png")),
                [],
            )

            # Evaluation-side early stop uses GT for every planner source.
            # One agent crossing the threshold fails the whole team while
            # retaining SoftSPL as a terminal-progress diagnostic.
            class OneAgentThresholdProvider:
                is_privileged = True

                def __init__(self, risk_config):
                    self.config = risk_config

                @staticmethod
                def sample_positions(
                    _timestamp_s,
                    positions,
                    floor_y_m=None,
                ):
                    del floor_y_m
                    count = len(positions)
                    zeros = np.zeros(count, dtype=np.float32)
                    risks = np.full(count, 0.05, dtype=np.float32)
                    risks[0] = 0.20
                    return RiskPointSamples(
                        flame=zeros,
                        temperature_c=np.full(
                            count, 25.0, dtype=np.float32
                        ),
                        temperature=zeros,
                        smoke=zeros,
                        physical_risk=risks,
                        hard_unsafe=np.zeros(count, dtype=bool),
                        confidence=np.ones(count, dtype=np.float32),
                    )

            for source in ("none", "oracle", "sensed"):
                stop_runtime = RiskRuntime(
                    fire_scene=FakeScene(),
                    reference_agent=reference,
                    args=SimpleNamespace(
                        risk_enabled=1,
                        risk_source=source,
                        risk_dump_dir=temporary,
                        risk_run_id=f"early-stop-{source}",
                        risk_save_every=0,
                        risk_save_traces=0,
                        risk_smoke_source="appearance_depth",
                        risk_early_stop_enabled=1,
                        risk_early_stop_threshold=0.10,
                    ),
                    episode_id=11,
                )
                stop_runtime.evaluator = MultiAgentRiskEvaluator(
                    OneAgentThresholdProvider(stop_runtime.config),
                    stop_runtime.config,
                )
                stop_runtime.evaluator.prime(
                    0.0,
                    [state.position for state in states],
                    floor_y_m=stop_runtime.floor_y_m,
                )
                stop_runtime.record_exposure(
                    1.0,
                    states,
                    step=7,
                    actions=[1, 3],
                )
                self.assertTrue(stop_runtime.early_stop_triggered)
                event = stop_runtime.early_stop_event
                self.assertIsNotNone(event)
                self.assertEqual(event["step"], 7)
                self.assertEqual(event["threshold"], 0.10)
                self.assertEqual(event["agent_ids"], ["0"])
                overridden = stop_runtime.apply_early_stop_metric_overrides({
                    "success": 1.0,
                    "spl": 0.75,
                    "soft_spl": 0.80,
                    "distance_to_goal": 0.1,
                })
                self.assertEqual(overridden["success"], 0.0)
                self.assertEqual(overridden["spl"], 0.0)
                self.assertEqual(overridden["soft_spl"], 0.80)
                self.assertEqual(overridden["distance_to_goal"], 0.1)
                stopped_summary = stop_runtime.summary(
                    habitat_success=overridden["success"]
                )
                self.assertEqual(stopped_summary["safe_success"], 0.0)
                self.assertEqual(stopped_summary["team"]["early_stop"], 1)
                self.assertEqual(
                    stopped_summary["team"]["exposure_samples"], 2
                )

            disabled_stop_runtime = RiskRuntime(
                fire_scene=FakeScene(),
                reference_agent=reference,
                args=SimpleNamespace(
                    risk_enabled=1,
                    risk_source="none",
                    risk_dump_dir=temporary,
                    risk_run_id="early-stop-disabled",
                    risk_save_every=0,
                    risk_save_traces=0,
                    risk_smoke_source="appearance_depth",
                    risk_danger_threshold=0.05,
                    risk_critical_threshold=0.10,
                    risk_early_stop_enabled=0,
                    risk_early_stop_threshold=0.0,
                ),
                episode_id=12,
            )
            disabled_stop_runtime.evaluator = MultiAgentRiskEvaluator(
                OneAgentThresholdProvider(disabled_stop_runtime.config),
                disabled_stop_runtime.config,
            )
            disabled_stop_runtime.record_exposure(1.0, states, step=1)
            self.assertFalse(disabled_stop_runtime.early_stop_triggered)
            unchanged = disabled_stop_runtime.apply_early_stop_metric_overrides({
                "success": 1.0,
                "spl": 0.75,
            })
            self.assertEqual(unchanged, {"success": 1.0, "spl": 0.75})
            relaxed_summary = disabled_stop_runtime.summary(
                habitat_success=unchanged["success"]
            )
            self.assertEqual(relaxed_summary["team"]["critical_steps"], 1)
            self.assertAlmostEqual(
                relaxed_summary["team"]["CHE_per_step"], 0.125
            )
            self.assertAlmostEqual(relaxed_summary["safe_success"], 0.875)

            off_floor = [
                SimpleNamespace(position=np.array([0.0, 2.0, 0.0])),
                states[1],
            ]
            # Oracle now owns one current-floor projection per agent.
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
            with self.assertRaisesRegex(RuntimeError, "one floor-aware"):
                privileged_runtime.record_exposure(1.0, off_floor)

    def test_none_mode_evaluates_each_agent_on_its_current_floor(self) -> None:
        class LayeredWorld:
            origin = np.array([-2.0, 0.0, -2.0])
            voxel_m = 1.0
            shape = (4, 4, 4)

            @staticmethod
            def frame_index(timestamp_s):
                return int(timestamp_s)

            def query(self, _timestamp_s):
                flame = np.zeros(self.shape, dtype=np.float32)
                smoke = np.zeros(self.shape, dtype=np.float32)
                temperature = np.full(self.shape, 25.0, dtype=np.float32)
                # Fire occupies only the lower storey's body-height band.
                flame[:, 0, :] = 1.0
                smoke[:, 0, :] = 1.0
                temperature[:, 0, :] = 500.0
                return flame, smoke, temperature

        class LayeredScene:
            def __init__(self):
                self.fw = LayeredWorld()

            @staticmethod
            def t_sim(robot_step):
                return float(robot_step)

        reference = SimpleNamespace(
            init_sim_rotation=np.eye(3),
            init_sim_position=np.zeros(3),
            init_agent_position=np.zeros(3),
            local_w=4,
            local_h=4,
            args=SimpleNamespace(map_resolution=100),
        )
        initial_states = [
            SimpleNamespace(position=np.array([0.0, 0.0, 0.0])),
            SimpleNamespace(position=np.array([1.0, 0.0, 0.0])),
        ]
        split_floor_states = [
            SimpleNamespace(position=np.array([0.0, 2.0, 0.0])),
            initial_states[1],
        ]

        with tempfile.TemporaryDirectory() as temporary:
            runtime = RiskRuntime(
                fire_scene=LayeredScene(),
                reference_agent=reference,
                args=SimpleNamespace(
                    risk_enabled=1,
                    risk_source="none",
                    risk_dump_dir=temporary,
                    risk_save_every=0,
                    risk_smoke_source="appearance_depth",
                ),
                episode_id=9,
            )
            run_config = json.loads(
                (runtime.run_dir / "risk_config.json").read_text()
            )
            self.assertEqual(
                run_config["evaluator_floor_mode"], "per_agent_current"
            )
            runtime.prime_exposure(0.0, initial_states)
            report = runtime.record_exposure(1.0, split_floor_states)

            self.assertEqual(report["0"]["risk"], 0.0)
            self.assertGreater(report["1"]["risk"], 0.0)
            self.assertFalse(bool(report["0"]["hard_unsafe"]))
            self.assertTrue(bool(report["1"]["hard_unsafe"]))

            with self.assertRaisesRegex(ValueError, "one value per position"):
                runtime.gt_provider.sample_positions(
                    1.0,
                    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    floor_y_m=[0.0],
                )

    def test_oracle_plans_and_evaluates_each_agent_on_current_floor(self) -> None:
        class LayeredWorld:
            origin = np.array([-2.0, 0.0, -2.0])
            voxel_m = 1.0
            shape = (4, 4, 4)

            @staticmethod
            def frame_index(timestamp_s):
                return int(timestamp_s)

            def query(self, _timestamp_s):
                flame = np.zeros(self.shape, dtype=np.float32)
                smoke = np.zeros(self.shape, dtype=np.float32)
                temperature = np.full(self.shape, 25.0, dtype=np.float32)
                flame[:, 0, :] = 1.0
                smoke[:, 0, :] = 1.0
                temperature[:, 0, :] = 500.0
                return flame, smoke, temperature

        class LayeredScene:
            def __init__(self):
                self.fw = LayeredWorld()

            @staticmethod
            def t_sim(robot_step):
                return float(robot_step)

        reference = SimpleNamespace(
            init_sim_rotation=np.eye(3),
            init_sim_position=np.zeros(3),
            init_agent_position=np.zeros(3),
            local_w=4,
            local_h=4,
            args=SimpleNamespace(map_resolution=100),
        )
        initial_states = [
            SimpleNamespace(position=np.array([0.0, 0.0, 0.0])),
            SimpleNamespace(position=np.array([1.0, 0.0, 0.0])),
        ]
        split_floor_states = [
            SimpleNamespace(position=np.array([0.0, 2.0, 0.0])),
            initial_states[1],
        ]

        with tempfile.TemporaryDirectory() as temporary:
            runtime = RiskRuntime(
                fire_scene=LayeredScene(),
                reference_agent=reference,
                args=SimpleNamespace(
                    risk_enabled=1,
                    risk_source="oracle",
                    risk_dump_dir=temporary,
                    risk_save_every=0,
                    risk_smoke_source="appearance_depth",
                ),
                episode_id=11,
            )
            run_config = json.loads(
                (runtime.run_dir / "risk_config.json").read_text()
            )
            self.assertEqual(
                run_config["planner_floor_mode"], "per_agent_current"
            )
            self.assertEqual(
                run_config["evaluator_floor_mode"], "per_agent_current"
            )

            layers, planning_risks = runtime.planner_states_for_agents(
                1.0,
                split_floor_states,
            )
            self.assertEqual(len(layers), 2)
            self.assertEqual(float(layers[0].physical_risk.max()), 0.0)
            self.assertGreater(float(layers[1].physical_risk.max()), 0.0)
            self.assertEqual(float(planning_risks[0].max()), 0.0)
            self.assertGreater(float(planning_risks[1].max()), 0.0)
            self.assertFalse(bool(layers[0].hard_unsafe.any()))
            self.assertTrue(bool(layers[1].hard_unsafe.any()))

            runtime.prime_exposure(0.0, initial_states)
            report = runtime.record_exposure(1.0, split_floor_states)
            self.assertEqual(report["0"]["risk"], 0.0)
            self.assertGreater(report["1"]["risk"], 0.0)


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
