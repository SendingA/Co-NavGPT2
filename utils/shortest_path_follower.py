"""Habitat-3 compatible ShortestPathFollower wrapper for Co-NavGPT2.

Preserves the tiny public surface consumed by :class:`agents.vlm_agents.VLM_Agent`::

    follower.get_next_action(goal_pos, current_grid_pose,
                             grid_angle, next_stg_x, next_stg_y) -> int
    follower.get_path_points(goal_pos) -> list of xyz waypoints

so the migration from Habitat-Lab 0.2.1 to 0.3.3 only touches the config
schema (uppercase FORWARD_STEP_SIZE / TURN_ANGLE → lowercase) and drops
the ``ShortestPathFollowerCompat`` name.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np

from habitat.sims.habitat_simulator.actions import HabitatSimActions

EPSILON = 1e-6


def action_to_one_hot(action: int) -> np.ndarray:
    one_hot = np.zeros(len(HabitatSimActions), dtype=np.float32)
    one_hot[action] = 1
    return one_hot


class ShortestPathFollowerCompat:
    """H3.3-flavoured re-implementation of the original compat follower.

    Args:
        sim: HabitatSim instance (H3.3).
        goal_radius: geodesic distance under which STOP is emitted.
        return_one_hot: keep the H2 signature, but callers currently
            always pass ``False`` so we just return the action integer.
        agent_id: index into ``sim.habitat_config.agents_order`` — one
            follower per navigation agent.
    """

    def __init__(
        self,
        sim,
        goal_radius: float,
        return_one_hot: bool = True,
        agent_id: int = 0,
    ) -> None:
        assert (
            getattr(sim, "geodesic_distance", None) is not None
        ), f"{type(sim).__name__} must expose geodesic_distance()"

        self._sim = sim
        self._agent_id = agent_id
        self._goal_radius = goal_radius
        self._return_one_hot = return_one_hot

        # H3.3 lowercased the config keys.  Fall back to the legacy
        # uppercase names to stay compatible if somebody hands us a
        # patched YACS config during unit tests.
        habitat_config = sim.habitat_config
        step_size = getattr(
            habitat_config,
            "forward_step_size",
            getattr(habitat_config, "FORWARD_STEP_SIZE", 0.25),
        )
        self._step_size = float(step_size)
        self._max_delta = self._step_size - EPSILON
        self._turn_angle = float(
            getattr(
                habitat_config,
                "turn_angle",
                getattr(habitat_config, "TURN_ANGLE", 30.0),
            )
        )

    # ------------------------------------------------------------------
    # Public API kept identical to the H2 follower
    # ------------------------------------------------------------------
    def get_next_action(
        self,
        goal_pos,
        current_grid_pose,
        grid_angle: float,
        next_stg_x: int,
        next_stg_y: int,
    ) -> Optional[Union[int, np.ndarray]]:
        """Return the next discrete action along the shortest path."""

        agent_pos = self._sim.get_agent_state(self._agent_id).position
        if self._sim.geodesic_distance(agent_pos, goal_pos) <= self._goal_radius:
            return self._wrap(int(HabitatSimActions.stop))

        angle_st_goal = math.degrees(
            math.atan2(
                next_stg_x - current_grid_pose[0],
                next_stg_y - current_grid_pose[1],
            )
        )
        angle_agent = (360 - float(grid_angle)) % 360.0
        if angle_agent > 180:
            angle_agent -= 360
        relative_angle = angle_agent - angle_st_goal
        if relative_angle > 180:
            relative_angle -= 360
        if relative_angle < -180:
            relative_angle += 360

        if relative_angle > self._turn_angle:
            return self._wrap(int(HabitatSimActions.turn_right))
        if relative_angle < -self._turn_angle:
            return self._wrap(int(HabitatSimActions.turn_left))
        return self._wrap(int(HabitatSimActions.move_forward))

    def get_closet_navigable_point(self, target_point):
        return self._sim.pathfinder.snap_point(target_point)

    def get_path_points(self, goal_pos):
        return self._sim.get_straight_shortest_path_points(
            self._sim.get_agent_state(self._agent_id).position,
            self.get_closet_navigable_point(goal_pos),
        )

    # ------------------------------------------------------------------
    def _wrap(self, action: int):
        return action_to_one_hot(action) if self._return_one_hot else action
