"""Visible Habitat3 robot models for the classic Co-NavGPT2 ObjectNav agents.

Ported from ``Co-NavGPTv3/envs/robot_models.py`` (reference implementation).

The ObjectNav agents themselves remain plain navigation agents.  This
helper loads Habitat3 articulated robot URDFs as kinematic visual models
and synchronizes them to each nav agent after reset/step so users can
still see a Fetch / Spot / Stretch body in the scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Type, Union

import magnum as mn
import numpy as np
from habitat_sim.physics import MotionType

from habitat.articulated_agents.robots.fetch_robot import (
    FetchRobot,
    FetchRobotNoWheels,
)
from habitat.articulated_agents.robots.fetch_suction import FetchSuctionRobot
from habitat.articulated_agents.robots.spot_robot import SpotRobot
from habitat.articulated_agents.robots.stretch_robot import StretchRobot
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)


RobotInstance = Union[
    FetchRobot,
    FetchRobotNoWheels,
    FetchSuctionRobot,
    SpotRobot,
    StretchRobot,
]
RobotClass = Type[RobotInstance]


@dataclass(frozen=True)
class RobotProfile:
    robot_class: RobotClass
    urdf_path: str
    description: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBOT_PROFILES: Dict[str, RobotProfile] = {
    "fetch": RobotProfile(
        FetchRobot,
        "data/robots/hab_fetch/robots/hab_fetch.urdf",
        "Fetch mobile manipulator.",
    ),
    "fetch_no_wheels": RobotProfile(
        FetchRobotNoWheels,
        "data/robots/hab_fetch/robots/fetch_no_base.urdf",
        "Fetch upper body without wheel links.",
    ),
    "fetch_suction": RobotProfile(
        FetchSuctionRobot,
        "data/robots/hab_fetch/robots/hab_suction.urdf",
        "Fetch suction-gripper variant.",
    ),
    "spot": RobotProfile(
        SpotRobot,
        "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
        "Boston Dynamics Spot arm model.",
    ),
    "stretch": RobotProfile(
        StretchRobot,
        "data/robots/hab_stretch/urdf/hab_stretch.urdf",
        "Hello Robot Stretch model.",
    ),
}


@dataclass
class _RobotModelState:
    robot: RobotInstance
    agent_id: int
    profile_name: str
    urdf_path: str


class RobotModelManager:
    """Load and synchronize Habitat3 robot URDFs with ObjectNav agents."""

    def __init__(
        self,
        sim,
        num_robots: int,
        profiles: Union[str, Sequence[str]] = "fetch",
        urdf_paths: Optional[Union[str, Sequence[str]]] = None,
        enabled: bool = True,
    ) -> None:
        self.sim = sim
        self.num_robots = max(0, int(num_robots))
        self.enabled = bool(enabled)
        self.profile_names: List[str] = self._as_list(profiles)
        self.custom_urdf_paths = (
            self._as_list(urdf_paths) if urdf_paths is not None else None
        )
        self._states: List[_RobotModelState] = []

    @property
    def robots(self) -> List[RobotInstance]:
        return [state.robot for state in self._states]

    @staticmethod
    def available_profiles() -> Dict[str, RobotProfile]:
        return ROBOT_PROFILES

    def reset(self) -> None:
        if not self.enabled:
            return

        while len(self._states) < self.num_robots:
            self._states.append(self._make_state(len(self._states)))

        for state in self._states[: self.num_robots]:
            robot = state.robot
            if robot.sim_obj is None or not robot.sim_obj.is_alive:
                robot.reconfigure()
                self._make_kinematic(robot)
            robot.reset()

        self.step()

    def step(self) -> None:
        if not self.enabled:
            return

        for state in self._states[: self.num_robots]:
            robot = state.robot
            if robot.sim_obj is None or not robot.sim_obj.is_alive:
                continue

            agent_state = self.sim.get_agent_state(state.agent_id)
            robot.base_pos = mn.Vector3(agent_state.position)
            robot.base_rot = self._agent_yaw(agent_state.rotation)
            robot.update()
            robot.sim_obj.awake = True

    def _make_state(self, agent_id: int) -> _RobotModelState:
        profile_name = self.profile_names[agent_id % len(self.profile_names)]
        if profile_name not in ROBOT_PROFILES:
            known = ", ".join(sorted(ROBOT_PROFILES))
            raise ValueError(f"Unknown robot profile '{profile_name}'. Known: {known}")

        profile = ROBOT_PROFILES[profile_name]
        urdf_path = self._urdf_path_for(agent_id, profile)
        self._ensure_urdf_exists(profile_name, urdf_path)

        cfg = SimpleNamespace(articulated_agent_urdf=urdf_path)
        robot = profile.robot_class(cfg, self.sim, fixed_base=False)
        robot.reconfigure()
        self._make_kinematic(robot)
        return _RobotModelState(
            robot=robot,
            agent_id=agent_id,
            profile_name=profile_name,
            urdf_path=urdf_path,
        )

    def _urdf_path_for(self, agent_id: int, profile: RobotProfile) -> str:
        if self.custom_urdf_paths:
            value = self.custom_urdf_paths[agent_id % len(self.custom_urdf_paths)]
        else:
            value = profile.urdf_path

        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((PROJECT_ROOT / path).resolve())

    @staticmethod
    def _make_kinematic(robot) -> None:
        try:
            robot.sim_obj.motion_type = MotionType.KINEMATIC
        except (AttributeError, RuntimeError, ValueError):
            pass

    @staticmethod
    def _ensure_urdf_exists(profile_name: str, urdf_path: str) -> None:
        if Path(urdf_path).exists():
            return
        raise FileNotFoundError(
            f"Robot profile '{profile_name}' needs URDF asset: {urdf_path}\n"
            "Place Habitat robot assets under data/robots, or set "
            "conav.robot_model_urdfs (or --robot_urdfs) to an existing URDF path."
        )

    @staticmethod
    def _agent_yaw(rotation) -> float:
        try:
            forward = quaternion_rotate_vector(
                rotation, np.array([0.0, 0.0, -1.0], dtype=np.float32)
            )
        except AttributeError:
            forward = quaternion_rotate_vector(
                quaternion_from_coeff(rotation),
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
            )

        x = float(forward[0])
        z = float(forward[2])
        if abs(x) + abs(z) < 1e-6:
            return 0.0
        return math.atan2(-z, x)

    @staticmethod
    def _as_list(value: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",")]
        else:
            items = [str(item).strip() for item in value]
        items = [item for item in items if item]
        if not items:
            raise ValueError("At least one robot profile or URDF path is required.")
        return items
