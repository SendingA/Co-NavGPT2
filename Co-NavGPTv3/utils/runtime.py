"""Small runtime helpers used by the Co-NavGPTv3 demo entrypoint."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import cv2
import numpy as np


def as_obs_list(observations) -> List[dict]:
    if isinstance(observations, list):
        return observations
    return [observations]


def agent_observation(observations, agent_id: int) -> dict:
    obs_list = as_obs_list(observations)
    return obs_list[min(agent_id, len(obs_list) - 1)]


def merge_visual_observations(observations, visual_observations):
    obs_list = as_obs_list(observations)
    visual_list = as_obs_list(visual_observations)
    for obs, visual_obs in zip(obs_list, visual_list):
        for key in ("rgb", "depth", "semantic"):
            if key in visual_obs:
                obs[key] = visual_obs[key]
    return obs_list if isinstance(observations, list) else obs_list[0]


def tile_rgb_observations(observations: Sequence[dict]) -> np.ndarray:
    frames = []
    for idx, obs in enumerate(observations):
        rgb = obs.get("rgb")
        if rgb is None:
            continue
        frame = np.ascontiguousarray(rgb[:, :, :3])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            f"agent_{idx}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(frame)

    if not frames:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    min_h = min(frame.shape[0] for frame in frames)
    resized = [
        cv2.resize(frame, (int(frame.shape[1] * min_h / frame.shape[0]), min_h))
        for frame in frames
    ]
    return np.concatenate(resized, axis=1)


def print_episode(env, num_robots: int, num_humans: int) -> None:
    episode = env.current_episode
    print(
        f"episode={episode.episode_id} scene={episode.scene_id} "
        f"robots={num_robots} humans={num_humans}"
    )
    print("keys: w=forward, a=left, d=right, f/space=stop, q/esc=quit")


def format_metrics(metrics: dict, names: Iterable[str]) -> str:
    parts = []
    for name in names:
        if name in metrics:
            value = metrics[name]
            if isinstance(value, float):
                parts.append(f"{name}={value:.3f}")
            else:
                parts.append(f"{name}={value}")
    return " ".join(parts)
