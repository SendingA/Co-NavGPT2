import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_fireworld_costs import (
    distribution_stats,
    summarize_bake_manifest,
    summarize_observation_samples,
)


class BenchmarkFireWorldCostsTests(unittest.TestCase):
    def test_distribution_stats(self):
        result = distribution_stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["median"], 2.5)
        self.assertAlmostEqual(result["p95"], 3.85)
        self.assertAlmostEqual(result["std_population"], 1.11803398875)

    def test_bake_manifest_keeps_batch_and_task_time_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            log_a = root / "a.log"
            log_b = root / "b.log"
            log_a.write_text("[propagation] a: wall=1.5s\n", encoding="utf-8")
            log_b.write_text("[propagation] b: wall=3.0s\n", encoding="utf-8")
            records = [
                {
                    "stage": "timeline",
                    "status": "generated",
                    "scene_id": "a",
                    "plan_id": "a_plan",
                    "elapsed_s": 2.0,
                    "log_path": str(log_a),
                    "estimate": {
                        "voxels": 10,
                        "timeline_uncompressed_bytes": 100,
                    },
                    "timeline": {
                        "duration_s": 6.0,
                        "shape": [7, 1, 2, 5],
                        "size_bytes": 40,
                    },
                },
                {
                    "stage": "timeline",
                    "status": "generated",
                    "scene_id": "b",
                    "plan_id": "b_plan",
                    "elapsed_s": 4.0,
                    "log_path": str(log_b),
                    "estimate": {
                        "voxels": 20,
                        "timeline_uncompressed_bytes": 200,
                    },
                    "timeline": {
                        "duration_s": 6.0,
                        "shape": [7, 1, 4, 5],
                        "size_bytes": 80,
                    },
                },
                {"stage": "plan", "status": "generated", "elapsed_s": 99.0},
            ]
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            summary = root / "summary.json"
            summary.write_text(json.dumps({"wall_time_s": 6.5}), encoding="utf-8")

            result = summarize_bake_manifest(manifest, summary)

        self.assertEqual(result["generated_timeline_count"], 2)
        self.assertEqual(result["per_plan_elapsed_s_sum"], 6.0)
        self.assertEqual(result["batch_wall_time_s"], 6.5)
        self.assertEqual(result["batch_overhead_s"], 0.5)
        self.assertEqual(result["compressed_size_bytes"]["total"], 120)
        self.assertEqual(result["propagation_log_count"], 2)
        self.assertEqual(result["propagation_only_elapsed_s_sum"], 4.5)
        self.assertEqual(result["pipeline_non_propagation_s_sum"], 1.5)
        self.assertAlmostEqual(result["pipeline_non_propagation_fraction"], 0.25)
        self.assertAlmostEqual(
            result["simulated_seconds_per_batch_wall_second"], 12.0 / 6.5
        )
        self.assertEqual(result["fastest"]["scene_id"], "a")
        self.assertEqual(result["slowest"]["scene_id"], "b")

    def test_observation_summary_derives_joint_end_to_end(self):
        result = summarize_observation_samples(
            habitat_joint_ms=[10.0, 20.0],
            fireworld_agent_ms=[[2.0, 4.0], [3.0, 5.0]],
        )
        self.assertEqual(result["sample_count_joint_steps"], 2)
        self.assertEqual(result["sample_count_agent_fire_observations"], 4)
        self.assertAlmostEqual(result["fireworld_only_joint_ms"]["mean"], 7.0)
        self.assertAlmostEqual(result["fire_end_to_end_joint_ms"]["mean"], 22.0)
        self.assertAlmostEqual(result["mean_slowdown_ratio"], 22.0 / 15.0)


if __name__ == "__main__":
    unittest.main()
