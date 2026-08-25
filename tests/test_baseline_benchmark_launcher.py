"""Headless tests for the normal/person baseline benchmark launcher."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.run_baseline_benchmarks import (
    Baseline,
    BenchmarkConfigurationError,
    RunSpec,
    build_baselines,
    build_command,
    build_run_specs,
    count_dataset_episodes,
    create_parser,
    execute_run,
    parse_final_aggregate,
    parse_run_id,
    preflight,
    run_launcher,
    validate_extra_main_args,
    _existing_study_runs,
)


class BaselineMatrixTests(unittest.TestCase):
    def test_run_id_round_trip(self) -> None:
        run = RunSpec("objectnav", "random", "pointnav", 7, 200)
        self.assertEqual(parse_run_id(run.run_id), run)

    def test_fire_condition_run_id_round_trip(self) -> None:
        run = RunSpec(
            "person", "nearest", "fmm", 1, 200, condition="fire-none"
        )
        self.assertEqual(parse_run_id(run.run_id), run)
        self.assertIn("person__fire-none", run.run_id)

    def test_default_global_planners_include_random_without_greedy_alias(self) -> None:
        args = create_parser().parse_args([])

        self.assertIn("random", args.global_planners)
        self.assertNotIn("greedy", args.global_planners)

    def test_controlled_matrix_deduplicates_reference_baseline(self) -> None:
        baselines = build_baselines(
            ["nearest", "co_ut", "fill", "gpt"],
            ["fmm", "astar", "pointnav"],
            matrix="controlled",
            reference_global="co_ut",
            reference_local="fmm",
        )

        self.assertEqual(
            [(item.global_planner, item.local_planner) for item in baselines],
            [
                ("nearest", "fmm"),
                ("co_ut", "fmm"),
                ("fill", "fmm"),
                ("gpt", "fmm"),
                ("co_ut", "astar"),
                ("co_ut", "pointnav"),
            ],
        )

    def test_cartesian_matrix_contains_every_selected_pair(self) -> None:
        baselines = build_baselines(
            ["nearest", "fill"],
            ["fmm", "astar", "pointnav"],
            matrix="cartesian",
            reference_global="co_ut",
            reference_local="fmm",
        )
        self.assertEqual(len(baselines), 6)
        self.assertEqual(
            (baselines[-1].global_planner, baselines[-1].local_planner),
            ("fill", "pointnav"),
        )

    def test_two_datasets_receive_separate_200_episode_runs(self) -> None:
        runs = build_run_specs(
            ["objectnav", "person"],
            [Baseline("co_ut", "fmm")],
            [1],
            200,
        )

        self.assertEqual(len(runs), 2)
        self.assertEqual({run.episodes for run in runs}, {200})
        self.assertNotEqual(runs[0].run_id, runs[1].run_id)


class DatasetAndCommandTests(unittest.TestCase):
    def test_sharded_dataset_episode_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "val"
            content = root / "content"
            content.mkdir(parents=True)
            with gzip.open(root / "val.json.gz", "wt", encoding="utf-8") as stream:
                json.dump({"episodes": []}, stream)
            for index, count in enumerate((2, 3)):
                with gzip.open(
                    content / f"scene-{index}.json.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    json.dump({"episodes": [{}] * count}, stream)

            self.assertEqual(count_dataset_episodes(root / "val.json.gz"), 5)

    def test_commands_select_dataset_episode_cap_and_planner_assets(self) -> None:
        run = RunSpec("person", "co_ut", "pointnav", 7, 200)
        with tempfile.TemporaryDirectory() as temporary:
            command = build_command(
                run,
                Path(temporary),
                python_executable=sys.executable,
                pointnav_checkpoint="pointnav.pth",
                pointnav_device="cuda:1",
                pointnav_deterministic=1,
                rl_checkpoint=None,
                rl_device="cpu",
                rl_deterministic=1,
                num_agents=2,
                extra_main_args=["--sem_threshold", "0.9"],
            )

        self.assertIn("person_objectnav_hm3d.yaml", command)
        self.assertEqual(command[command.index("--max_episodes") + 1], "200")
        self.assertEqual(command[command.index("--nav_mode") + 1], "co_ut")
        self.assertEqual(
            command[command.index("--local_planner") + 1],
            "pointnav",
        )
        self.assertEqual(
            command[command.index("--pointnav_device") + 1],
            "cuda:1",
        )
        self.assertEqual(command[-2:], ["--sem_threshold", "0.9"])

    def test_fire_none_command_uses_step_clock_and_torch_cuda(self) -> None:
        run = RunSpec(
            "person", "nearest", "fmm", 1, 200, condition="fire-none"
        )
        with tempfile.TemporaryDirectory() as temporary:
            command = build_command(
                run,
                Path(temporary),
                python_executable=sys.executable,
                pointnav_checkpoint="pointnav.pth",
                pointnav_device="cuda:0",
                pointnav_deterministic=1,
                rl_checkpoint=None,
                rl_device="cuda:0",
                rl_deterministic=1,
                num_agents=2,
                extra_main_args=[],
            )

        def value(flag):
            return command[command.index(flag) + 1]

        self.assertEqual(value("--fire_world"), "1")
        self.assertEqual(value("--fire_world_plan_id"), "auto")
        self.assertEqual(value("--fire_world_fire_type"), "multi_origin")
        self.assertEqual(value("--fire_world_intensity"), "medium")
        self.assertEqual(value("--fire_clock_mode"), "step")
        self.assertEqual(value("--risk_enabled"), "1")
        self.assertEqual(value("--risk_source"), "none")
        self.assertEqual(value("--fmm_backend"), "grid")
        self.assertEqual(value("--fire_render_backend"), "torch")
        self.assertEqual(value("--fire_render_device"), "cuda:0")
        self.assertEqual(value("--visualize"), "0")
        self.assertEqual(value("--print_images"), "0")
        self.assertEqual(value("--fire_save_every"), "0")
        self.assertEqual(value("--fire_save_npz"), "0")
        self.assertEqual(value("--fire_show_window"), "0")
        self.assertEqual(value("--risk_save_every"), "0")
        self.assertEqual(value("--risk_save_traces"), "0")

    def test_main_args_cannot_reenable_benchmark_artifacts(self) -> None:
        for flag in (
            "--visualize",
            "--print_images",
            "--fire_save_every",
            "--fire_save_npz",
            "--fire_show_window",
            "--risk_save_every",
            "--risk_save_traces",
        ):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(
                    BenchmarkConfigurationError,
                    "launcher-owned flags",
                ):
                    validate_extra_main_args([flag, "1"])

    def test_rl_requires_an_explicit_checkpoint(self) -> None:
        run = RunSpec("objectnav", "co_ut", "rl", 1, 200)
        with self.assertRaisesRegex(
            BenchmarkConfigurationError,
            "--rl-checkpoint",
        ):
            preflight(
                [run],
                python_executable=sys.executable,
                pointnav_checkpoint="unused.pth",
                rl_checkpoint=None,
                dry_run=True,
            )

    def test_extra_args_cannot_override_owned_experiment_fields(self) -> None:
        with self.assertRaisesRegex(
            BenchmarkConfigurationError,
            "--max_episodes",
        ):
            validate_extra_main_args(["--max_episodes=5"])
        with self.assertRaisesRegex(
            BenchmarkConfigurationError,
            "--start_episode",
        ):
            validate_extra_main_args(["--start_episode", "118"])
        with self.assertRaisesRegex(
            BenchmarkConfigurationError,
            "--risk_source",
        ):
            validate_extra_main_args(["--risk_source", "oracle"])


class ExecutionAndResumeTests(unittest.TestCase):
    def test_existing_study_subset_preserves_master_manifest(self) -> None:
        fill_fmm = RunSpec("objectnav", "fill", "fmm", 1, 200)
        fill_astar = RunSpec("objectnav", "fill", "astar", 1, 200)
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            study_dir = output_root / "existing"
            study_dir.mkdir()
            manifest_path = study_dir / "study_manifest.json"
            manifest_payload = {
                "schema_version": 1,
                "study_id": "existing",
                "num_agents": 2,
                "run_ids": [fill_fmm.run_id, fill_astar.run_id],
            }
            manifest_text = json.dumps(manifest_payload, indent=2) + "\n"
            manifest_path.write_text(manifest_text, encoding="utf-8")

            master, selected, loaded = _existing_study_runs(
                study_dir,
                [fill_astar.run_id],
            )
            self.assertEqual(master, [fill_fmm, fill_astar])
            self.assertEqual(selected, [fill_astar])
            self.assertEqual(loaded["study_id"], "existing")

            parser = create_parser()
            args = parser.parse_args(
                [
                    "--output-root",
                    str(output_root),
                    "--study-id",
                    "existing",
                    "--only-run-ids",
                    fill_astar.run_id,
                    "--dry-run",
                ]
            )
            self.assertEqual(run_launcher(args), 0)
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"),
                manifest_text,
            )

    def test_existing_study_subset_rejects_unknown_run(self) -> None:
        known = RunSpec("objectnav", "fill", "fmm", 1, 200)
        unknown = RunSpec("person", "fill", "fmm", 1, 200)
        with tempfile.TemporaryDirectory() as temporary:
            study_dir = Path(temporary)
            (study_dir / "study_manifest.json").write_text(
                json.dumps(
                    {
                        "num_agents": 2,
                        "run_ids": [known.run_id],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                BenchmarkConfigurationError,
                "not part of the existing study",
            ):
                _existing_study_runs(study_dir, [unknown.run_id])

    def test_successful_run_writes_aggregate_and_resumes(self) -> None:
        run = RunSpec("objectnav", "nearest", "fmm", 1, 2)
        command = [
            sys.executable,
            "-c",
            (
                "print('success: 0.500, spl: 0.250, "
                "num_steps: 10.000 ---(2/2)')"
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / run.run_id
            first = execute_run(run, run_dir, command, force=False)
            second = execute_run(run, run_dir, command, force=False)
            aggregate = parse_final_aggregate(run_dir / "stdout.log")

            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["observed_episodes"], 2)
            self.assertEqual(second["launcher_result"], "resumed")
            self.assertEqual(aggregate["metrics"]["success"], 0.5)
            self.assertTrue((run_dir / "metrics" / "aggregate.json").is_file())

    def test_incomplete_run_can_adopt_log_only_artifact_flags(self) -> None:
        run = RunSpec("objectnav", "nearest", "fmm", 1, 2)
        code = (
            "import sys; "
            "done = 2 if '--risk_save_traces' in sys.argv else 1; "
            "print(f'success: 0.500 ---({done}/2)')"
        )
        old_command = [sys.executable, "-c", code]
        log_only_command = [
            *old_command,
            "--fire_save_every",
            "0",
            "--risk_save_every",
            "0",
            "--risk_save_traces",
            "0",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / run.run_id
            first = execute_run(run, run_dir, old_command, force=False)
            resumed = execute_run(
                run,
                run_dir,
                log_only_command,
                force=False,
            )
            manifest = json.loads((run_dir / "manifest.json").read_text())

            self.assertEqual(first["status"], "failed")
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(resumed["resume"]["start_episode"], 2)
            self.assertEqual(manifest["command"], log_only_command)

    def test_zero_exit_with_wrong_episode_count_is_failed(self) -> None:
        run = RunSpec("objectnav", "nearest", "fmm", 1, 2)
        command = [
            sys.executable,
            "-c",
            "print('success: 1.000 ---(1/2)')",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            result = execute_run(
                run,
                Path(temporary) / run.run_id,
                command,
                force=False,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["return_code"], 0)
        self.assertIn("exact requested episode count", result["error"])

    def test_incomplete_run_resumes_from_legacy_aggregate(self) -> None:
        run = RunSpec("objectnav", "gpt", "fmm", 1, 4)
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "assert sys.argv[sys.argv.index('--start_episode') + 1] == '3'; "
                "print('success: 0.750, spl: 0.500 ---(4/4)')"
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / run.run_id
            (run_dir / "metrics").mkdir(parents=True)
            (run_dir / "metrics" / "aggregate.json").write_text(
                json.dumps(
                    {
                        "episodes_completed": 2,
                        "episodes_planned": 4,
                        "metrics": {"success": 0.5, "spl": 0.25},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"status": "interrupted", "attempt": 1}),
                encoding="utf-8",
            )
            (run_dir / "stdout.log").write_text(
                "success: 0.500 ---(2/4)\n",
                encoding="utf-8",
            )

            result = execute_run(run, run_dir, command, force=False)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["resume"]["start_episode"], 3)
            self.assertEqual(
                result["resume"]["precision"],
                "legacy_3_decimal_average",
            )
            command_text = (run_dir / "command.txt").read_text()
            self.assertIn("--start_episode 3", command_text)
            self.assertIn("--resume_metrics_path", command_text)
            self.assertEqual(
                (
                    run_dir
                    / "attempts"
                    / "attempt_01_stdout.log"
                ).read_text(),
                "success: 0.500 ---(2/4)\n",
            )

    def test_less_advanced_retry_does_not_replace_resume_aggregate(
        self,
    ) -> None:
        run = RunSpec("objectnav", "nearest", "fmm", 1, 4)
        command = [
            sys.executable,
            "-c",
            "print('success: 0.000 ---(1/4)')",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / run.run_id
            aggregate_path = run_dir / "metrics" / "aggregate.json"
            aggregate_path.parent.mkdir(parents=True)
            aggregate_path.write_text(
                json.dumps(
                    {
                        "episodes_completed": 2,
                        "episodes_planned": 4,
                        "metrics": {"success": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            result = execute_run(
                run,
                run_dir,
                command,
                force=False,
                resume_incomplete=False,
            )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(
                json.loads(aggregate_path.read_text())["episodes_completed"],
                2,
            )

    def test_dry_run_writes_no_study_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "not-created"
            parser = create_parser()
            args = parser.parse_args(
                [
                    "--datasets",
                    "objectnav",
                    "--global-planners",
                    "nearest",
                    "--local-planners",
                    "fmm",
                    "--matrix",
                    "cartesian",
                    "--episodes",
                    "200",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ]
            )
            return_code = run_launcher(args)

            self.assertEqual(return_code, 0)
            self.assertFalse(output_root.exists())


class MainEpisodeLimitBoundaryTests(unittest.TestCase):
    def test_main_exposes_and_applies_episode_limit(self) -> None:
        root = Path(__file__).resolve().parents[1]
        argument_source = (root / "arguments.py").read_text(encoding="utf-8")
        main_source = (root / "main.py").read_text(encoding="utf-8")

        self.assertIn('"--max_episodes"', argument_source)
        self.assertIn('"--start_episode"', argument_source)
        self.assertIn('"--resume_metrics_path"', argument_source)
        self.assertIn("min(available_episodes, episode_limit)", main_source)
        self.assertIn('"--max_episodes must be non-negative"', main_source)
        self.assertIn("advance_episode_iterator(", main_source)
        self.assertIn("write_metric_resume(", main_source)
        self.assertIn("_dataset_content_scene_ids(config)", main_source)


if __name__ == "__main__":
    unittest.main()
