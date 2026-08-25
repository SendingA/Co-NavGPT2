"""Unit tests for exact and legacy-compatible evaluation resume state."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from utils.evaluation_resume import (
    advance_episode_iterator,
    load_metric_resume,
    write_metric_resume,
)


class _FakeEnv:
    def __init__(self, count=5):
        episodes = [
            SimpleNamespace(
                episode_id=str(index),
                scene_id=f"scene-{index // 2}",
            )
            for index in range(1, count + 1)
        ]
        self.current_episode = episodes[0]
        self.episode_iterator = iter(episodes[1:])


class EvaluationResumeTests(unittest.TestCase):
    def test_legacy_aggregate_reconstructs_metric_sums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "aggregate.json"
            path.write_text(
                json.dumps(
                    {
                        "episodes_completed": 117,
                        "episodes_planned": 200,
                        "metrics": {
                            "success": 0.581,
                            "spl": 0.287,
                        },
                    }
                ),
                encoding="utf-8",
            )
            state = load_metric_resume(path)

        self.assertEqual(state.episodes_completed, 117)
        self.assertEqual(state.precision, "legacy_3_decimal_average")
        self.assertAlmostEqual(state.metric_sums["success"], 67.977)
        self.assertAlmostEqual(state.metric_sums["spl"], 33.579)

    def test_exact_state_round_trip_preserves_sums_and_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics" / "resume_state.json"
            write_metric_resume(
                path,
                episodes_completed=2,
                episodes_planned=5,
                metric_sums={"success": 1.0, "spl": 0.625},
                precision="exact",
                metric_contract="fireworld-risk-v4",
                last_episode_id="ep-2",
                last_scene_id="scene-a",
            )
            state = load_metric_resume(path)

        self.assertEqual(state.metric_sums, {"spl": 0.625, "success": 1.0})
        self.assertEqual(state.last_episode_id, "ep-2")
        self.assertEqual(state.last_scene_id, "scene-a")
        self.assertEqual(state.precision, "exact")
        self.assertEqual(state.metric_contract, "fireworld-risk-v4")

    def test_iterator_advances_to_first_unfinished_episode(self) -> None:
        env = _FakeEnv()

        last_completed = advance_episode_iterator(env, 3)

        self.assertEqual(last_completed.episode_id, "3")
        self.assertEqual(env.current_episode.episode_id, "4")


if __name__ == "__main__":
    unittest.main()
