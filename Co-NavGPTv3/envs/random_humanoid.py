"""Random walking humanoids for classic Habitat ObjectNav scenes.

This module intentionally does not use Habitat's SocialNav or Rearrange task
stack.  The humanoids are plain kinematic articulated objects inserted into the
active Habitat-Sim scene and advanced once per ObjectNav step.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Sequence, Union

import habitat_sim
import magnum as mn
import numpy as np

from habitat.articulated_agent_controllers import HumanoidRearrangeController
from habitat.articulated_agents.humanoids.kinematic_humanoid import (
    KinematicHumanoid,
)


@dataclass
class _WalkerState:
    humanoid: KinematicHumanoid
    controller: HumanoidRearrangeController
    urdf_path: str
    motion_data_path: str
    target: Optional[mn.Vector3] = None


class RandomHumanoidWalker:
    """Spawn and animate humanoids toward random navigable points."""

    def __init__(
        self,
        sim,
        num_humans: int,
        urdf_path: Union[str, Sequence[str]],
        motion_data_path: Union[str, Sequence[str]],
        seed: int = 0,
        walk_speed: float = 0.7,
        turn_speed: float = 1.5,
        goal_radius: float = 0.6,
        target_radius: float = 4.0,
        min_spawn_distance: float = 1.2,
        replan_attempts: int = 20,
        motion_dt: float = 1.0 / 30.0,
        use_controller_root_motion: bool = True,
    ) -> None:
        self.sim = sim
        self.num_humans = max(0, int(num_humans))
        self.urdf_paths = self._as_path_list(urdf_path)
        self.motion_data_paths = self._as_path_list(motion_data_path)
        if len(self.motion_data_paths) not in (1, len(self.urdf_paths)):
            raise ValueError(
                "motion_data_path must be a single path or have the same "
                "length as urdf_path."
            )
        self.walk_speed = walk_speed
        self.turn_speed = turn_speed
        self.goal_radius = goal_radius
        self.target_radius = target_radius
        self.min_spawn_distance = min_spawn_distance
        self.replan_attempts = replan_attempts
        self.motion_dt = motion_dt
        self.use_controller_root_motion = use_controller_root_motion
        self.rng = random.Random(seed)
        self._states: List[_WalkerState] = []

    @property
    def humans(self) -> List[KinematicHumanoid]:
        return [state.humanoid for state in self._states]

    def reset(self) -> None:
        """Create missing humanoids and place all of them in the current scene."""

        while len(self._states) < self.num_humans:
            self._states.append(self._make_state(len(self._states)))

        for state in self._states:
            if state.humanoid.sim_obj is None or not state.humanoid.sim_obj.is_alive:
                state.humanoid.reconfigure()
            self._place_humanoid(state)

    def step(self) -> None:
        """Advance each humanoid one controller frame along a random path."""

        for state in self._states:
            humanoid = state.humanoid
            if humanoid.sim_obj is None or not humanoid.sim_obj.is_alive:
                continue

            start = mn.Vector3(humanoid.base_pos)
            if state.target is None or self._xz_distance(
                start, state.target
            ) < self.goal_radius:
                state.target = self._sample_reachable_target(start)

            rel_target = self._next_relative_waypoint(
                start=start,
                target=state.target,
            )
            for _ in range(self.replan_attempts):
                if rel_target is not None:
                    break
                state.target = self._sample_reachable_target(start)
                rel_target = self._next_relative_waypoint(
                    start=start,
                    target=state.target,
                )

            controller = state.controller
            controller.obj_transform_base = humanoid.base_transformation

            if rel_target is None:
                controller.calculate_stop_pose()
            else:
                controller.calculate_walk_pose(rel_target)
                if self.use_controller_root_motion:
                    self._filter_controller_to_navmesh(humanoid, controller)
                else:
                    self._advance_base_transform(humanoid, controller, rel_target)

            self._apply_controller_pose(humanoid, controller)
            humanoid.update()

    def _make_state(self, human_index: int) -> _WalkerState:
        urdf_path = self.urdf_paths[human_index % len(self.urdf_paths)]
        motion_data_path = self.motion_data_paths[
            human_index % len(self.motion_data_paths)
        ]
        cfg = SimpleNamespace(
            articulated_agent_urdf=urdf_path,
            motion_data_path=motion_data_path,
            auto_update_sensor_transform=False,
        )
        humanoid = KinematicHumanoid(cfg, self.sim, fixed_base=False)
        humanoid.reconfigure()

        controller = HumanoidRearrangeController(motion_data_path)
        ctrl_freq = 1.0 / self.motion_dt if self.motion_dt > 0 else float(
            getattr(self.sim.habitat_config, "ctrl_freq", 120.0)
        )
        controller.set_framerate_for_linspeed(
            self.walk_speed, self.turn_speed, ctrl_freq
        )
        return _WalkerState(
            humanoid=humanoid,
            controller=controller,
            urdf_path=urdf_path,
            motion_data_path=motion_data_path,
        )

    @staticmethod
    def _as_path_list(value: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(value, str):
            return [value]
        paths = [str(item) for item in value]
        if not paths:
            raise ValueError("At least one humanoid path must be provided.")
        return paths

    def _place_humanoid(self, state: _WalkerState) -> None:
        human_pos = self._sample_spawn_point()
        yaw = self.rng.uniform(-math.pi, math.pi)

        state.humanoid.base_pos = human_pos
        state.humanoid.base_rot = yaw
        state.humanoid.set_rest_position()
        state.controller.reset(state.humanoid.base_transformation)
        state.target = self._sample_reachable_target(state.humanoid.base_pos)

    def _sample_spawn_point(self) -> mn.Vector3:
        robot_positions = [
            np.array(self.sim.get_agent_state(i).position, dtype=np.float32)
            for i in range(len(self.sim.habitat_config.agents_order))
        ]

        fallback = None
        for _ in range(100):
            point = self._sample_target()
            fallback = point
            point_np = np.array(point, dtype=np.float32)
            if all(
                np.linalg.norm((point_np - robot_pos)[[0, 2]])
                >= self.min_spawn_distance
                for robot_pos in robot_positions
            ):
                return point

        return fallback if fallback is not None else mn.Vector3.zero_init()

    def _sample_target(self) -> mn.Vector3:
        point = self.sim.pathfinder.get_random_navigable_point()
        return mn.Vector3(point)

    def _sample_reachable_target(self, start: mn.Vector3) -> mn.Vector3:
        nav_start = mn.Vector3(self.sim.pathfinder.snap_point(start))
        island = self.sim.pathfinder.get_island(nav_start)
        fallback = nav_start
        for _ in range(self.replan_attempts):
            target = mn.Vector3(
                self.sim.pathfinder.get_random_navigable_point_near(
                    nav_start,
                    self.target_radius,
                    100,
                    island,
                )
            )
            if self._has_path(nav_start, target):
                return target
            fallback = target
        return fallback

    def _next_relative_waypoint(
        self, start: mn.Vector3, target: mn.Vector3
    ) -> Optional[mn.Vector3]:
        path_start = mn.Vector3(self.sim.pathfinder.snap_point(start))
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(path_start, dtype=np.float32)
        path.requested_end = np.array(target, dtype=np.float32)

        if not self.sim.pathfinder.find_path(path) or len(path.points) == 0:
            return None

        waypoint = mn.Vector3(path.points[min(1, len(path.points) - 1)])
        next_pos = mn.Vector3(self.sim.step_filter(path_start, waypoint))
        rel = next_pos - path_start
        rel.y = 0.0
        if rel.length() < 1e-4:
            return None
        return rel

    def _has_path(self, start: mn.Vector3, target: mn.Vector3) -> bool:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.array(start, dtype=np.float32)
        path.requested_end = np.array(target, dtype=np.float32)
        return bool(self.sim.pathfinder.find_path(path)) and len(path.points) > 1

    def _filter_controller_to_navmesh(
        self,
        humanoid: KinematicHumanoid,
        controller: HumanoidRearrangeController,
    ) -> None:
        base_offset = humanoid.params.base_offset
        prev_query_pos = humanoid.base_pos
        target_query_pos = controller.obj_transform_base.translation + base_offset
        filtered_query_pos = self.sim.step_filter(
            prev_query_pos, target_query_pos
        )
        fixup = filtered_query_pos - target_query_pos
        controller.obj_transform_base.translation += fixup

    def _advance_base_transform(
        self,
        humanoid: KinematicHumanoid,
        controller: HumanoidRearrangeController,
        rel_target: mn.Vector3,
    ) -> None:
        direction = mn.Vector3(rel_target)
        direction.y = 0.0
        if direction.length() < 1e-6:
            return

        direction = direction.normalized()
        step_distance = min(self.walk_speed * self.motion_dt, rel_target.length())
        desired = mn.Vector3(humanoid.base_pos) + direction * step_distance
        filtered = mn.Vector3(self.sim.step_filter(humanoid.base_pos, desired))
        delta = filtered - humanoid.base_pos
        delta.y = 0.0
        if delta.length() < 1e-6:
            return

        base_translation = humanoid.base_transformation.translation + delta
        look_direction = mn.Vector3([direction.z, 0.0, -direction.x])
        look_at_path = mn.Matrix4.look_at(
            base_translation,
            base_translation + look_direction.normalized(),
            mn.Vector3.y_axis(),
        )
        rot_offset = mn.Matrix4.rotation(
            mn.Rad(-np.pi / 2), mn.Vector3.x_axis()
        )
        controller.obj_transform_base = look_at_path @ rot_offset

    @staticmethod
    def _apply_controller_pose(
        humanoid: KinematicHumanoid,
        controller: HumanoidRearrangeController,
    ) -> None:
        pose = controller.get_pose()
        joints = pose[:-32]
        offset = RandomHumanoidWalker._matrix_from_flat_row_major(pose[-32:-16])
        base = RandomHumanoidWalker._matrix_from_flat_row_major(pose[-16:])
        humanoid.set_joint_transform(joints, offset, base)

    @staticmethod
    def _matrix_from_flat_row_major(values: List[float]) -> mn.Matrix4:
        # Habitat's HumanoidJointAction stores transposed matrices in the action.
        columns = [mn.Vector4(values[i * 4 : (i + 1) * 4]) for i in range(4)]
        return mn.Matrix4(*columns)

    @staticmethod
    def _xz_distance(a: mn.Vector3, b: mn.Vector3) -> float:
        return float(np.linalg.norm([a.x - b.x, a.z - b.z]))
