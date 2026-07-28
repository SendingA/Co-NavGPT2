#!/usr/bin/env python3
"""Train the map-based RL local planner with PPO on randomized grids.

This trainer validates the learned-planner mechanics without coupling PPO to
the VLM/global-frontier stack.  Benchmark checkpoints should use a substantial
training budget and only train-split-derived FireWorld/map distributions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch import nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.local_planners.base import GRID_ACTIONS, GridPlannerBase
from utils.local_planners.rl import (
    RLGridPolicy,
    build_rl_observation,
    save_rl_checkpoint,
)


class SyntheticGridEnv:
    """Small randomized navigation MDP for PPO policy training."""

    def __init__(
        self,
        *,
        grid_size: int,
        crop_size: int,
        risk_aware: bool,
        max_episode_steps: int,
        seed: int,
    ) -> None:
        self.grid_size = int(grid_size)
        self.crop_size = int(crop_size)
        self.risk_aware = bool(risk_aware)
        self.max_episode_steps = int(max_episode_steps)
        self.rng = np.random.default_rng(int(seed))
        self.reset()

    def _sample_world(self) -> None:
        for _ in range(100):
            free = np.ones((self.grid_size, self.grid_size), dtype=np.uint8)
            free[[0, -1], :] = 0
            free[:, [0, -1]] = 0
            for _ in range(max(2, self.grid_size // 5)):
                height = int(self.rng.integers(1, max(2, self.grid_size // 5)))
                width = int(self.rng.integers(1, max(2, self.grid_size // 5)))
                row = int(self.rng.integers(1, self.grid_size - height))
                col = int(self.rng.integers(1, self.grid_size - width))
                free[row : row + height, col : col + width] = 0

            impulse = np.zeros_like(free, dtype=np.float32)
            for _ in range(max(1, self.grid_size // 10)):
                row = int(self.rng.integers(1, self.grid_size - 1))
                col = int(self.rng.integers(1, self.grid_size - 1))
                impulse[row, col] = float(self.rng.uniform(0.65, 1.0))
            risk = cv2.GaussianBlur(impulse, (0, 0), sigmaX=2.0)
            if float(risk.max()) > 0:
                risk /= float(risk.max())
            hard = (risk >= 0.82) & (free > 0) if self.risk_aware else np.zeros_like(
                free, dtype=bool
            )
            domain = (free > 0) & ~hard
            count, labels = cv2.connectedComponents(
                domain.astype(np.uint8), connectivity=4
            )
            if count <= 1:
                continue
            component_sizes = np.bincount(labels.ravel())
            component_sizes[0] = 0
            component = int(np.argmax(component_sizes))
            cells = np.argwhere(labels == component)
            if len(cells) < max(20, self.grid_size):
                continue
            first = int(self.rng.integers(0, len(cells)))
            distances = np.linalg.norm(cells - cells[first], axis=1)
            far = np.flatnonzero(distances >= self.grid_size * 0.35)
            if not len(far):
                continue
            second = int(self.rng.choice(far))
            self.free = free.astype(bool)
            self.risk = risk.astype(np.float32)
            self.hard = hard.astype(bool)
            self.traversible = domain.astype(np.float32)
            self.cell = tuple(map(int, cells[first]))
            self.goal = tuple(map(int, cells[second]))
            self.goal_map = np.zeros_like(free, dtype=bool)
            self.goal_map[self.goal] = True
            self.visited = np.zeros_like(free, dtype=bool)
            self.visited[self.cell] = True
            self.grid = GridPlannerBase(self.traversible)
            return
        raise RuntimeError("failed to sample a connected synthetic grid")

    def reset(self):
        self._sample_world()
        self.steps = 0
        return self.observation()

    def observation(self):
        maps, goal = build_rl_observation(
            self.traversible,
            self.goal_map,
            self.visited,
            self.cell,
            crop_size=self.crop_size,
            risk_aware=self.risk_aware,
            risk_map=self.risk,
            hard_unsafe_mask=self.hard,
        )
        return maps, goal, self.grid.valid_action_mask(self.cell)

    def step(self, action: int):
        self.steps += 1
        valid = self.grid.valid_action_mask(self.cell)
        previous_distance = float(np.linalg.norm(np.subtract(self.goal, self.cell)))
        collision = not (0 <= int(action) < len(GRID_ACTIONS) and valid[int(action)])
        if not collision:
            drow, dcol = GRID_ACTIONS[int(action)]
            self.cell = (self.cell[0] + drow, self.cell[1] + dcol)
        self.visited[self.cell] = True
        distance = float(np.linalg.norm(np.subtract(self.goal, self.cell)))
        reward = previous_distance - distance - 0.01
        if collision:
            reward -= 0.15
        if self.risk_aware:
            reward -= 0.30 * float(self.risk[self.cell])
        reached = self.cell == self.goal
        if reached:
            reward += 2.0
        done = reached or self.steps >= self.max_episode_steps
        return self.observation(), float(reward), bool(done)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/rl_local_planner_ppo.yaml"
    )
    parser.add_argument("--output", default="outputs/local_planner_rl/policy.pth")
    parser.add_argument("--risk-aware", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--crop-size", type=int, default=31)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--max-episode-steps", type=int, default=160)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-param", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", default="configs/rl_local_planner_ppo.yaml"
    )
    known, _ = config_parser.parse_known_args(argv)
    parser = _parser()
    config_path = Path(known.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("RL training config must contain a mapping")
        parser.set_defaults(**{
            str(key).replace("-", "_"): value for key, value in loaded.items()
        })
    return parser.parse_args(argv)


def _stack_observations(envs: List[SyntheticGridEnv], device):
    items = [env.observation() for env in envs]
    maps = torch.from_numpy(np.stack([item[0] for item in items])).to(device)
    goals = torch.from_numpy(np.stack([item[1] for item in items])).to(device)
    masks = torch.from_numpy(np.stack([item[2] for item in items])).to(device)
    return maps, goals, masks


def train(args) -> Dict[str, float]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(1)
    device = torch.device(args.device)
    risk_aware = bool(int(args.risk_aware))
    policy = RLGridPolicy(
        risk_aware=risk_aware,
        crop_size=int(args.crop_size),
        hidden_size=int(args.hidden_size),
    ).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(args.learning_rate)
    )
    envs = [
        SyntheticGridEnv(
            grid_size=int(args.grid_size),
            crop_size=int(args.crop_size),
            risk_aware=risk_aware,
            max_episode_steps=int(args.max_episode_steps),
            seed=int(args.seed) + index * 1009,
        )
        for index in range(int(args.num_envs))
    ]
    recent_rewards: List[float] = []
    completed = 0

    for update in range(1, int(args.updates) + 1):
        storage = {
            key: [] for key in (
                "maps", "goals", "masks", "actions", "log_probs",
                "values", "rewards", "dones",
            )
        }
        for _ in range(int(args.rollout_steps)):
            maps, goals, masks = _stack_observations(envs, device)
            with torch.no_grad():
                logits, values = policy(maps, goals)
                masked_logits = logits.masked_fill(~masks.bool(), -torch.inf)
                distribution = torch.distributions.Categorical(
                    logits=masked_logits
                )
                actions = distribution.sample()
                log_probs = distribution.log_prob(actions)
            rewards = []
            dones = []
            for index, env in enumerate(envs):
                _, reward, done = env.step(int(actions[index].item()))
                rewards.append(reward)
                dones.append(done)
                recent_rewards.append(reward)
                if done:
                    completed += 1
                    env.reset()
            storage["maps"].append(maps.cpu())
            storage["goals"].append(goals.cpu())
            storage["masks"].append(masks.cpu())
            storage["actions"].append(actions.cpu())
            storage["log_probs"].append(log_probs.cpu())
            storage["values"].append(values.cpu())
            storage["rewards"].append(torch.tensor(rewards, dtype=torch.float32))
            storage["dones"].append(torch.tensor(dones, dtype=torch.float32))

        with torch.no_grad():
            next_maps, next_goals, _ = _stack_observations(envs, device)
            _, next_value = policy(next_maps, next_goals)
            next_value = next_value.cpu()

        values = torch.stack(storage["values"])
        rewards = torch.stack(storage["rewards"])
        dones = torch.stack(storage["dones"])
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(int(args.num_envs))
        for step in reversed(range(int(args.rollout_steps))):
            following = next_value if step == int(args.rollout_steps) - 1 else values[step + 1]
            active = 1.0 - dones[step]
            delta = (
                rewards[step]
                + float(args.gamma) * following * active
                - values[step]
            )
            gae = (
                delta
                + float(args.gamma)
                * float(args.gae_lambda)
                * active
                * gae
            )
            advantages[step] = gae
        returns = advantages + values

        flat = {
            "maps": torch.stack(storage["maps"]).flatten(0, 1),
            "goals": torch.stack(storage["goals"]).flatten(0, 1),
            "masks": torch.stack(storage["masks"]).flatten(0, 1),
            "actions": torch.stack(storage["actions"]).flatten(),
            "log_probs": torch.stack(storage["log_probs"]).flatten(),
            "returns": returns.flatten(),
            "advantages": advantages.flatten(),
        }
        flat["advantages"] = (
            flat["advantages"] - flat["advantages"].mean()
        ) / (flat["advantages"].std() + 1e-8)
        sample_count = flat["actions"].shape[0]

        policy.train()
        for _ in range(int(args.ppo_epochs)):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, int(args.mini_batch_size)):
                indices = permutation[start : start + int(args.mini_batch_size)]
                maps = flat["maps"][indices].to(device)
                goals = flat["goals"][indices].to(device)
                masks = flat["masks"][indices].to(device).bool()
                actions = flat["actions"][indices].to(device)
                old_log_probs = flat["log_probs"][indices].to(device)
                target_returns = flat["returns"][indices].to(device)
                target_advantages = flat["advantages"][indices].to(device)

                logits, predicted_values = policy(maps, goals)
                logits = logits.masked_fill(~masks, -torch.inf)
                distribution = torch.distributions.Categorical(logits=logits)
                log_probs = distribution.log_prob(actions)
                ratio = torch.exp(log_probs - old_log_probs)
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(args.clip_param),
                    1.0 + float(args.clip_param),
                )
                policy_loss = -torch.min(
                    ratio * target_advantages,
                    clipped * target_advantages,
                ).mean()
                value_loss = nn.functional.mse_loss(
                    predicted_values, target_returns
                )
                loss = (
                    policy_loss
                    + float(args.value_loss_coef) * value_loss
                    - float(args.entropy_coef) * distribution.entropy().mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), float(args.max_grad_norm)
                )
                optimizer.step()
        policy.eval()

        if update % max(1, int(args.log_interval)) == 0 or update == 1:
            mean_reward = float(np.mean(recent_rewards[-1000:]))
            print(
                f"update={update}/{args.updates} "
                f"mean_step_reward={mean_reward:.4f} "
                f"completed_episodes={completed}"
            )

    output = save_rl_checkpoint(
        PROJECT_ROOT / args.output
        if not Path(args.output).is_absolute()
        else args.output,
        policy,
        optimizer=optimizer,
        training_state={
            "updates": int(args.updates),
            "num_envs": int(args.num_envs),
            "rollout_steps": int(args.rollout_steps),
            "seed": int(args.seed),
            "completed_episodes": int(completed),
            "synthetic_grid_training": True,
        },
    )
    return {
        "checkpoint": str(output),
        "completed_episodes": float(completed),
        "mean_step_reward": float(np.mean(recent_rewards[-1000:])),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    result = train(args)
    print(
        "saved checkpoint={} completed_episodes={} mean_step_reward={:.4f}".format(
            result["checkpoint"],
            int(result["completed_episodes"]),
            result["mean_step_reward"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
