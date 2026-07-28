"""Checkpointed PPO policy used as a learned grid local planner."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .base import GRID_ACTIONS, GridCell, GridPlannerBase


RL_CHECKPOINT_VERSION = 1
_POLICY_CACHE: Dict[Tuple[str, str, bool, int], "RLGridPolicy"] = {}


def _crop_grid(
    array: np.ndarray,
    centre: GridCell,
    crop_size: int,
    fill: float,
) -> np.ndarray:
    radius = int(crop_size) // 2
    padded = np.pad(
        np.asarray(array),
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=fill,
    )
    row, col = centre
    return padded[row : row + crop_size, col : col + crop_size]


def build_rl_observation(
    traversible: np.ndarray,
    goal_map: np.ndarray,
    visited: np.ndarray,
    cell: GridCell,
    *,
    crop_size: int,
    risk_aware: bool,
    risk_map: Optional[np.ndarray] = None,
    hard_unsafe_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the checkpoint-stable map channels and relative-goal vector."""
    traversible = np.asarray(traversible)
    goal_map = np.asarray(goal_map, dtype=bool)
    visited = np.asarray(visited, dtype=bool)
    channels = [
        _crop_grid(traversible > 0, cell, crop_size, 0.0),
        _crop_grid(goal_map, cell, crop_size, 0.0),
        _crop_grid(visited, cell, crop_size, 0.0),
    ]
    if risk_aware:
        risk = (
            np.zeros(traversible.shape, dtype=np.float32)
            if risk_map is None
            else np.asarray(risk_map, dtype=np.float32)
        )
        hard = (
            np.zeros(traversible.shape, dtype=bool)
            if hard_unsafe_mask is None
            else np.asarray(hard_unsafe_mask, dtype=bool)
        )
        channels.extend(
            [
                _crop_grid(risk, cell, crop_size, 1.0),
                _crop_grid(hard, cell, crop_size, 1.0),
            ]
        )

    goals = np.argwhere(goal_map)
    if goals.size:
        delta = goals - np.asarray(cell, dtype=np.int64)[None, :]
        nearest = delta[int(np.argmin(np.sum(delta * delta, axis=1)))]
        scale = float(max(traversible.shape))
        goal_vector = np.clip(
            nearest.astype(np.float32) / max(scale, 1.0), -1.0, 1.0
        )
    else:
        goal_vector = np.zeros(2, dtype=np.float32)
    return np.stack(channels, axis=0).astype(np.float32), goal_vector


class RLGridPolicy(nn.Module):
    """Small map policy that selects one of eight neighboring grid moves."""

    def __init__(
        self,
        *,
        risk_aware: bool,
        crop_size: int = 31,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if int(crop_size) < 5 or int(crop_size) % 2 == 0:
            raise ValueError("crop_size must be an odd integer >= 5")
        self.risk_aware = bool(risk_aware)
        self.crop_size = int(crop_size)
        self.hidden_size = int(hidden_size)
        self.input_channels = 5 if self.risk_aware else 3
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 * 4 * 4 + 2, self.hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(self.hidden_size, len(GRID_ACTIONS))
        self.value_head = nn.Linear(self.hidden_size, 1)

    def forward(self, map_tensor, goal_vector):
        encoded = self.encoder(map_tensor)
        fused = self.fusion(torch.cat([encoded, goal_vector], dim=-1))
        return self.policy_head(fused), self.value_head(fused).squeeze(-1)

    def checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "format_version": RL_CHECKPOINT_VERSION,
            "risk_aware": self.risk_aware,
            "crop_size": self.crop_size,
            "hidden_size": self.hidden_size,
            "action_count": len(GRID_ACTIONS),
            "action_deltas": [list(delta) for delta in GRID_ACTIONS],
            "input_channels": self.input_channels,
        }


def save_rl_checkpoint(
    path,
    policy: RLGridPolicy,
    *,
    optimizer=None,
    training_state: Optional[Dict[str, object]] = None,
) -> Path:
    """Save weights plus the schema required for safe benchmark loading."""
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": RL_CHECKPOINT_VERSION,
        "metadata": policy.checkpoint_metadata(),
        "model_state_dict": policy.state_dict(),
        "training_state": dict(training_state or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, output)
    return output


def load_rl_policy(
    checkpoint_path,
    *,
    device="cpu",
    expected_risk_aware: Optional[bool] = None,
    expected_crop_size: Optional[int] = None,
) -> RLGridPolicy:
    """Load and validate a learned planner; random fallback is forbidden."""
    if not checkpoint_path:
        raise ValueError(
            "--rl_local_checkpoint is required when --local_planner=rl"
        )
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "RL local-planner checkpoint does not exist: {}".format(path)
        )
    cache_key = (
        str(path),
        str(device),
        bool(expected_risk_aware),
        int(expected_crop_size or -1),
    )
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]

    payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("RL checkpoint must contain a dictionary payload")
    if int(payload.get("format_version", -1)) != RL_CHECKPOINT_VERSION:
        raise ValueError(
            "unsupported RL checkpoint format_version {!r}".format(
                payload.get("format_version")
            )
        )
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("RL checkpoint is missing metadata")
    risk_aware = bool(metadata.get("risk_aware", False))
    crop_size = int(metadata.get("crop_size", -1))
    hidden_size = int(metadata.get("hidden_size", -1))
    if expected_risk_aware is not None and risk_aware != bool(
        expected_risk_aware
    ):
        raise ValueError(
            "RL checkpoint risk_aware={} does not match requested {}".format(
                risk_aware, bool(expected_risk_aware)
            )
        )
    if expected_crop_size is not None and crop_size != int(expected_crop_size):
        raise ValueError(
            "RL checkpoint crop_size={} does not match requested {}".format(
                crop_size, int(expected_crop_size)
            )
        )
    if metadata.get("action_deltas") != [
        list(delta) for delta in GRID_ACTIONS
    ]:
        raise ValueError("RL checkpoint action ordering is incompatible")

    policy = RLGridPolicy(
        risk_aware=risk_aware,
        crop_size=crop_size,
        hidden_size=hidden_size,
    ).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("RL checkpoint is missing model_state_dict")
    policy.load_state_dict(state, strict=True)
    policy.eval()
    _POLICY_CACHE[cache_key] = policy
    return policy


class RLGridPlanner(GridPlannerBase):
    """Learned short-horizon waypoint planner with deterministic inference."""

    def __init__(
        self,
        traversible,
        *,
        checkpoint_path,
        device="cpu",
        deterministic: bool = True,
        crop_size: int = 31,
        rollout_steps: int = 5,
        risk_aware: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(traversible, **kwargs)
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.crop_size = int(crop_size)
        self.rollout_steps = max(1, int(rollout_steps))
        self.policy = load_rl_policy(
            checkpoint_path,
            device=self.device,
            expected_risk_aware=bool(risk_aware),
            expected_crop_size=self.crop_size,
        )
        self.policy_risk_aware = bool(risk_aware)

    def policy_observation(
        self, cell: GridCell, visited: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.goal_map is None:
            raise RuntimeError("set_multi_goal must be called before planning")
        return build_rl_observation(
            self.traversible,
            self.goal_map,
            visited,
            cell,
            crop_size=self.crop_size,
            risk_aware=self.policy_risk_aware,
            risk_map=self.risk_map,
            hard_unsafe_mask=self.hard_unsafe_mask,
        )

    def _select_action(
        self, cell: GridCell, visited: np.ndarray
    ) -> Optional[int]:
        valid = self.valid_action_mask(cell)
        if not np.any(valid):
            return None

        unvisited = valid.copy()
        for index, (drow, dcol) in enumerate(GRID_ACTIONS):
            if valid[index] and visited[cell[0] + drow, cell[1] + dcol]:
                unvisited[index] = False
        if np.any(unvisited):
            valid = unvisited

        maps, goal = self.policy_observation(cell, visited)
        with torch.no_grad():
            logits, _ = self.policy(
                torch.from_numpy(maps).unsqueeze(0).to(self.device),
                torch.from_numpy(goal).unsqueeze(0).to(self.device),
            )
            valid_tensor = torch.from_numpy(valid).to(self.device).unsqueeze(0)
            logits = logits.masked_fill(~valid_tensor, -torch.inf)
            if self.deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                action = int(
                    torch.distributions.Categorical(logits=logits)
                    .sample()
                    .item()
                )
        return action

    def get_short_term_goal(self, state):
        start = self._clip_cell(state)
        goal_distance = self.goal_distance()
        stop = bool(goal_distance[start] < float(self.step_size))
        if stop:
            self.last_path = [start]
            return float(start[0]), float(start[1]), False, True

        visited = np.zeros(self.traversible.shape, dtype=bool)
        visited[start] = True
        path = [start]
        current = start
        replan = False
        for _ in range(self.rollout_steps):
            action = self._select_action(current, visited)
            if action is None:
                replan = True
                break
            drow, dcol = GRID_ACTIONS[action]
            current = (current[0] + drow, current[1] + dcol)
            path.append(current)
            visited[current] = True
            if self.goal_map is not None and self.goal_map[current]:
                break
        self.last_path = path
        return float(current[0]), float(current[1]), replan, False
