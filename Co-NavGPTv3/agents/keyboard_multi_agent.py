"""Minimal keyboard controller for synchronized multi-robot navigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


@dataclass
class KeyboardCommand:
    """A decoded keyboard command."""

    action: Optional[int]
    quit: bool = False


class KeyboardMultiAgent:
    """Maps one keyboard command to the same discrete action for all robots."""

    def __init__(self, num_robots: int) -> None:
        if num_robots <= 0:
            raise ValueError("num_robots must be positive.")
        self.num_robots = num_robots

    def decode_key(self, key: int) -> KeyboardCommand:
        """Decode a cv2.waitKey result into a navigation command."""

        if key < 0:
            return KeyboardCommand(action=None)

        key &= 0xFF
        if key in (ord("q"), 27):
            return KeyboardCommand(action=None, quit=True)
        if key in (ord("w"), ord("W")):
            return KeyboardCommand(action=MOVE_FORWARD)
        if key in (ord("a"), ord("A")):
            return KeyboardCommand(action=TURN_LEFT)
        if key in (ord("d"), ord("D")):
            return KeyboardCommand(action=TURN_RIGHT)
        if key in (ord("f"), ord("F"), ord(" ")):
            return KeyboardCommand(action=STOP)

        return KeyboardCommand(action=None)

    def synchronized_actions(self, action: int) -> List[int]:
        """Return the same action for each Habitat-Sim robot agent."""

        return [int(action)] * self.num_robots


class KeyboardAgent:
    """Single robot wrapper so main.py can use the v2-style agent[i].act loop."""

    def __init__(self, agent_id: int) -> None:
        self.agent_id = int(agent_id)
        self.observation = None
        self.agent_state = None

    def reset(self, observation, agent_state) -> None:
        self.observation = observation
        self.agent_state = agent_state

    def act(self, observation, agent_state, keyboard_action: int) -> int:
        self.observation = observation
        self.agent_state = agent_state
        return int(keyboard_action)
