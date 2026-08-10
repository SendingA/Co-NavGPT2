"""Tests for optional curated multi-agent episode starts."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from utils.multi_agent_start import (
    GOAL_POSITIONS_KEY,
    START_STATES_KEY,
    TARGET_AGENT_IDS_KEY,
    apply_episode_agent_starts,
    episode_agent_start_states,
    episode_agent_goal_positions,
    episode_target_agent_ids,
)


class _Simulator:
    def __init__(self):
        self.calls = []

    def set_agent_state(self, position, rotation, agent_id=0):
        self.calls.append((agent_id, position, rotation))
        return True


def _episode(states=None):
    info = {} if states is None else {START_STATES_KEY: states}
    return SimpleNamespace(
        start_position=[1.0, 0.0, 2.0],
        info=info,
    )


class MultiAgentStartTests(unittest.TestCase):
    def test_ordinary_episode_keeps_native_reset(self):
        sim = _Simulator()
        self.assertEqual(apply_episode_agent_starts(sim, _episode(), 2), [])
        self.assertEqual(sim.calls, [])

    def test_curated_starts_are_normalized_and_applied(self):
        states = [
            {
                "position": [1.0, 0.0, 2.0],
                "rotation": [0.0, 0.0, 0.0, 2.0],
            },
            {
                "position": [8.0, 0.0, -3.0],
                "rotation": [0.0, 2.0, 0.0, 0.0],
            },
        ]
        sim = _Simulator()
        result = apply_episode_agent_starts(sim, _episode(states), 2)

        self.assertEqual(len(result), 2)
        self.assertEqual([call[0] for call in sim.calls], [0, 1])
        self.assertEqual(sim.calls[0][2], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(sim.calls[1][2], [0.0, 1.0, 0.0, 0.0])

    def test_agent_zero_cannot_move_away_from_native_start(self):
        states = [
            {
                "position": [2.0, 0.0, 2.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "position": [8.0, 0.0, -3.0],
                "rotation": [0.0, 1.0, 0.0, 0.0],
            },
        ]
        with self.assertRaisesRegex(ValueError, "agent 0"):
            episode_agent_start_states(_episode(states), 2)

    def test_state_count_must_match_runtime_agents(self):
        states = [{
            "position": [1.0, 0.0, 2.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }]
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            episode_agent_start_states(_episode(states), 2)

    def test_curated_goal_can_target_only_primary_agent(self):
        episode = _episode()
        episode.info[GOAL_POSITIONS_KEY] = [[3.0, 0.0, 4.0], None]
        self.assertEqual(
            episode_agent_goal_positions(episode, 2),
            [[3.0, 0.0, 4.0], None],
        )

    def test_controlled_target_agent_ids_are_opt_in_and_validated(self):
        episode = _episode()
        self.assertEqual(episode_target_agent_ids(episode, 2), [])
        episode.info[TARGET_AGENT_IDS_KEY] = [0]
        self.assertEqual(episode_target_agent_ids(episode, 2), [0])
        episode.info[TARGET_AGENT_IDS_KEY] = [2]
        with self.assertRaisesRegex(ValueError, "outside"):
            episode_target_agent_ids(episode, 2)


if __name__ == "__main__":
    unittest.main()
