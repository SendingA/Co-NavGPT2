"""Environment-side helpers for Co-NavGPT2 on Habitat-Lab 0.3.3.

* :class:`RandomHumanoidWalker` spawns kinematic humanoids as random
  pedestrians alongside the classic ObjectNav navigation agents.
* :class:`RobotModelManager` loads Habitat3 articulated robot URDFs
  (Fetch / Spot / Stretch / ...) as visible kinematic models that follow
  each ObjectNav agent's pose.
"""

from envs.random_humanoid import RandomHumanoidWalker
from envs.robot_models import RobotModelManager

__all__ = ["RandomHumanoidWalker", "RobotModelManager"]
