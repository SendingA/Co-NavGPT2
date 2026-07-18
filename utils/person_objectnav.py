"""Shared helpers for the native static-person ObjectNav benchmark."""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, List, Mapping, Sequence

import numpy as np


PERSON_CATEGORY = "person"
PERSON_CATEGORY_ID = 6
PERSON_CLOSE_VIEW_RADIUS_M = 0.65


def episode_is_person_goal(episode: Any) -> bool:
    """Return whether a Habitat episode targets the person category."""
    return getattr(episode, "object_category", None) == PERSON_CATEGORY


def person_goal_positions(episode: Any) -> List[np.ndarray]:
    """Extract finite 3-D person goal positions from a Habitat episode."""
    if not episode_is_person_goal(episode):
        return []

    positions: List[np.ndarray] = []
    for goal in getattr(episode, "goals", ()):
        position = np.asarray(getattr(goal, "position", ()), dtype=np.float32)
        if position.shape == (3,) and np.all(np.isfinite(position)):
            positions.append(position)
    return positions


def refresh_simulator_observations(
    sim: Any,
    task_observations: Any,
    num_agents: int,
) -> List[dict]:
    """Re-render simulator sensors without advancing the Habitat task.

    Habitat Env.reset returns RGB/depth before runtime articulated objects are
    spawned. As in the original humanoid demo, ``sim.step(None)`` reads fresh
    simulator frames after placement without applying an action. The frames
    are merged into the reset observations so ObjectNav task sensors such as
    objectgoal, GPS, and compass remain intact.
    """
    originals = (
        list(task_observations)
        if isinstance(task_observations, list)
        else [task_observations]
    )
    if len(originals) != num_agents:
        raise ValueError(
            f"received {len(originals)} observations for {num_agents} agents"
        )

    rendered_observations = sim.step(None)
    rendered = (
        list(rendered_observations)
        if isinstance(rendered_observations, list)
        else [rendered_observations]
    )
    if len(rendered) != num_agents:
        raise ValueError(
            f"sim.step(None) returned {len(rendered)} observations for "
            f"{num_agents} agents"
        )

    refreshed: List[dict] = []
    for original, visual_observation in zip(originals, rendered):
        merged = dict(original)
        merged.update(visual_observation)
        refreshed.append(merged)
    return refreshed


def objectnav_goal_debug_info(
    sim: Any,
    episode: Any,
    num_agents: int,
) -> Mapping[str, Any]:
    """Return both agent positions and the closest agent/goal-center pair."""
    agent_positions = [
        np.asarray(sim.get_agent_state(agent_id).position, dtype=np.float64)
        for agent_id in range(num_agents)
    ]
    if any(
        position.shape != (3,) or not np.all(np.isfinite(position))
        for position in agent_positions
    ):
        raise ValueError("all agent positions must contain exactly three values")

    best: dict = {}
    best_distance = math.inf
    for goal_index, goal in enumerate(getattr(episode, "goals", ())):
        goal_position = np.asarray(
            getattr(goal, "position", ()), dtype=np.float64
        )
        if goal_position.shape != (3,) or not np.all(np.isfinite(goal_position)):
            continue

        for agent_id, agent_position in enumerate(agent_positions):
            distance = float(np.linalg.norm(agent_position - goal_position))
            if distance >= best_distance:
                continue
            best_distance = distance
            best = {
                "nearest_agent_id": agent_id,
                "goal_index": goal_index,
                "goal_position": goal_position,
                "nearest_goal_l2": distance,
            }

    if not best:
        raise ValueError("episode has no finite goal positions")

    return {
        "agent_positions": agent_positions,
        **best,
    }


def person_goals_key(scene_id: str) -> str:
    """Match ObjectGoalNavEpisode.goals_key for the person category."""
    return f"{os.path.basename(scene_id)}_{PERSON_CATEGORY}"


def add_person_category_mappings(dataset: dict) -> None:
    """Add and validate the canonical person id in an ObjectNav JSON dict."""
    mapping_names = (
        "category_to_task_category_id",
        "category_to_scene_annotation_category_id",
    )
    for name in mapping_names:
        mapping = dataset.setdefault(name, {})
        existing = mapping.get(PERSON_CATEGORY)
        if existing is not None and int(existing) != PERSON_CATEGORY_ID:
            raise ValueError(
                f"{name} maps person to {existing}, expected {PERSON_CATEGORY_ID}"
            )
        used_by = [key for key, value in mapping.items()
                   if int(value) == PERSON_CATEGORY_ID and key != PERSON_CATEGORY]
        if used_by:
            raise ValueError(
                f"{name} id {PERSON_CATEGORY_ID} is already used by {used_by}"
            )
        mapping[PERSON_CATEGORY] = PERSON_CATEGORY_ID


def yaw_quaternion_facing(source: Sequence[float],
                          target: Sequence[float]) -> List[float]:
    """Return Habitat's [x,y,z,w] yaw quaternion facing target from source."""
    source_arr = np.asarray(source, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    dx = float(target_arr[0] - source_arr[0])
    dz = float(target_arr[2] - source_arr[2])
    yaw = math.atan2(-dx, -dz)
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def validate_person_dataset_dict(dataset: Mapping[str, Any],
                                 *, require_episodes: bool = True) -> None:
    """Validate the structural invariants needed by Habitat ObjectNav."""
    for name in (
        "category_to_task_category_id",
        "category_to_scene_annotation_category_id",
    ):
        mapping = dataset.get(name, {})
        if mapping.get(PERSON_CATEGORY) != PERSON_CATEGORY_ID:
            raise ValueError(f"{name} must contain person: {PERSON_CATEGORY_ID}")

    goals_by_category = dataset.get("goals_by_category", {})
    episodes = dataset.get("episodes", [])
    if require_episodes and not episodes:
        raise ValueError("person dataset contains no episodes")

    for episode in episodes:
        if episode.get("object_category") != PERSON_CATEGORY:
            raise ValueError("all generated episodes must target person")
        key = person_goals_key(episode["scene_id"])
        goals = goals_by_category.get(key)
        if not goals:
            raise ValueError(f"missing person goals for {key}")
        for goal in goals:
            position = goal.get("position")
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(f"invalid person goal position in {key}")
            goal_position = np.asarray(position, dtype=np.float64)
            if not np.all(np.isfinite(goal_position)):
                raise ValueError(f"non-finite person goal position in {key}")
            view_points = goal.get("view_points")
            if not view_points:
                raise ValueError(f"person goal in {key} has no view_points")
            view_positions: List[np.ndarray] = []
            for view in view_points:
                state = view.get("agent_state", {})
                if len(state.get("position", [])) != 3:
                    raise ValueError(f"invalid view point position in {key}")
                if len(state.get("rotation", [])) != 4:
                    raise ValueError(f"invalid view point rotation in {key}")
                view_position = np.asarray(
                    state["position"], dtype=np.float64
                )
                if not np.all(np.isfinite(view_position)):
                    raise ValueError(f"non-finite view point position in {key}")
                view_positions.append(view_position)
            closest_horizontal = min(
                float(np.linalg.norm((view - goal_position)[[0, 2]]))
                for view in view_positions
            )
            if closest_horizontal >= PERSON_CLOSE_VIEW_RADIUS_M:
                raise ValueError(
                    f"person goal in {key} has no close view point below "
                    f"{PERSON_CLOSE_VIEW_RADIUS_M:.2f}m"
                )


def unique_positions(positions: Iterable[Sequence[float]],
                     *, tolerance: float = 0.05) -> List[np.ndarray]:
    """Deduplicate positions by horizontal distance while preserving order."""
    result: List[np.ndarray] = []
    for value in positions:
        point = np.asarray(value, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            continue
        if all(np.linalg.norm((point - old)[[0, 2]]) > tolerance
               for old in result):
            result.append(point)
    return result
