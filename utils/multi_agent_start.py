"""Optional episode-defined start states for curated multi-agent scenes.

Habitat's native multi-agent NavigationTask intentionally places every agent
at the single ObjectNav episode start.  Curated demonstrations can opt into
separate starts through ``episode.info['multi_agent_start_states']``.  Agent 0
must remain at the native episode start so Habitat's Success/SPL contract is
not silently changed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np


START_STATES_KEY = "multi_agent_start_states"
GOAL_POSITIONS_KEY = "curated_agent_goal_positions"
TARGET_AGENT_IDS_KEY = "controlled_target_agent_ids"


def episode_agent_start_states(
    episode: Any,
    num_agents: int,
) -> List[Dict[str, List[float]]]:
    """Validate and return optional per-agent start states."""

    info = getattr(episode, "info", None) or {}
    raw_states = info.get(START_STATES_KEY)
    if raw_states is None:
        return []
    if not isinstance(raw_states, list) or len(raw_states) != int(num_agents):
        raise ValueError(
            f"episode.info[{START_STATES_KEY!r}] must contain exactly "
            f"{int(num_agents)} states"
        )

    states: List[Dict[str, List[float]]] = []
    for agent_id, raw in enumerate(raw_states):
        if not isinstance(raw, Mapping):
            raise ValueError(f"agent {agent_id} start state must be a mapping")
        position = np.asarray(raw.get("position"), dtype=np.float64)
        rotation = np.asarray(raw.get("rotation"), dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(
                f"agent {agent_id} start position must contain 3 finite values"
            )
        if rotation.shape != (4,) or not np.all(np.isfinite(rotation)):
            raise ValueError(
                f"agent {agent_id} start rotation must contain 4 finite values"
            )
        norm = float(np.linalg.norm(rotation))
        if norm <= 1e-8:
            raise ValueError(f"agent {agent_id} start rotation is degenerate")
        states.append({
            "position": [float(value) for value in position],
            "rotation": [float(value) for value in rotation / norm],
        })

    native_start = np.asarray(
        getattr(episode, "start_position", None), dtype=np.float64
    )
    if native_start.shape != (3,) or not np.allclose(
        states[0]["position"], native_start, rtol=0.0, atol=1e-4
    ):
        raise ValueError(
            "curated agent 0 must remain at episode.start_position so native "
            "ObjectNav metrics keep their original start-state semantics"
        )
    return states


def apply_episode_agent_starts(
    sim: Any,
    episode: Any,
    num_agents: int,
) -> List[Dict[str, List[float]]]:
    """Apply validated curated starts and return the states that were used."""

    states = episode_agent_start_states(episode, num_agents)
    for agent_id, state in enumerate(states):
        success = sim.set_agent_state(
            state["position"], state["rotation"], agent_id=agent_id
        )
        if success is False:
            raise RuntimeError(
                f"simulator rejected curated start for agent {agent_id}: "
                f"{state['position']}"
            )
    return states


def episode_agent_goal_positions(
    episode: Any,
    num_agents: int,
) -> List[Any]:
    """Return optional known route goals for controlled scenario agents."""

    info = getattr(episode, "info", None) or {}
    raw_goals = info.get(GOAL_POSITIONS_KEY)
    if raw_goals is None:
        return []
    if not isinstance(raw_goals, list) or len(raw_goals) != int(num_agents):
        raise ValueError(
            f"episode.info[{GOAL_POSITIONS_KEY!r}] must contain exactly "
            f"{int(num_agents)} entries"
        )
    goals: List[Any] = []
    for agent_id, raw in enumerate(raw_goals):
        if raw is None:
            goals.append(None)
            continue
        position = np.asarray(raw, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(
                f"agent {agent_id} curated goal must contain 3 finite values"
            )
        goals.append([float(value) for value in position])
    if not any(goal is not None for goal in goals):
        raise ValueError("curated goal list must contain at least one position")
    return goals


def episode_target_agent_ids(episode: Any, num_agents: int) -> List[int]:
    """Return agents allowed to terminate a controlled target-search episode."""

    info = getattr(episode, "info", None) or {}
    raw_ids = info.get(TARGET_AGENT_IDS_KEY)
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError(
            f"episode.info[{TARGET_AGENT_IDS_KEY!r}] must be a non-empty list"
        )
    agent_ids = [int(value) for value in raw_ids]
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("controlled target agent ids must be unique")
    if any(agent_id < 0 or agent_id >= int(num_agents) for agent_id in agent_ids):
        raise ValueError("controlled target agent id is outside the runtime team")
    return agent_ids


__all__ = [
    "GOAL_POSITIONS_KEY",
    "START_STATES_KEY",
    "TARGET_AGENT_IDS_KEY",
    "apply_episode_agent_starts",
    "episode_agent_goal_positions",
    "episode_agent_start_states",
    "episode_target_agent_ids",
]
