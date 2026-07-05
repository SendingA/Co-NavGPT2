#!/usr/bin/env python3
"""Minimal Habitat3 multi-robot ObjectNav demo with random pedestrians."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import habitat  # noqa: E402

from agents import KeyboardAgent, KeyboardMultiAgent  # noqa: E402
from arguments import (  # noqa: E402
    get_args,
    humanoid_kwargs,
    load_config,
    robot_model_kwargs,
)
from envs import RandomHumanoidWalker, RobotModelManager  # noqa: E402
from utils.runtime import (  # noqa: E402
    agent_observation,
    as_obs_list,
    format_metrics,
    merge_visual_observations,
    print_episode,
    tile_rgb_observations,
)


def main(args) -> None:
    config = load_config(args)
    env = habitat.Env(config=config)
    num_episodes = env.number_of_episodes
    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = int(config.conav.num_robots)
    keyboard = KeyboardMultiAgent(num_robots=num_agents)
    agent = []
    for i in range(num_agents):
        agent.append(KeyboardAgent(i))

    walker = RandomHumanoidWalker(
        sim=env.sim,
        **humanoid_kwargs(config, args.seed),
    )
    robot_models = RobotModelManager(
        sim=env.sim,
        **robot_model_kwargs(config, num_agents),
    )

    try:
        count_episodes = 0
        while count_episodes < num_episodes:
            observations = env.reset()
            walker.reset()
            robot_models.reset()
            observations = env.sim.step(None)

            for i in range(num_agents):
                agent_state = env.sim.get_agent_state(i)
                agent[i].reset(agent_observation(observations, i), agent_state)

            print_episode(env, num_agents, int(config.conav.num_humans))
            if robot_models.enabled:
                profiles = ",".join(robot_models.profile_names)
                print(f"robot_models=True profiles={profiles}")

            count_step = 0
            while not env.episode_over:
                obs_list = as_obs_list(observations)
                if not args.no_display:
                    cv2.imshow(
                        "Co-NavGPTv3 Habitat3 ObjectNav",
                        tile_rgb_observations(obs_list),
                    )
                    key = cv2.waitKey(30)
                else:
                    key = ord(input("action [w/a/d/f/q]: ").strip()[:1] or "w")

                command = keyboard.decode_key(key)
                if command.quit:
                    return
                if command.action is None:
                    if not args.no_display:
                        walker.step()
                        robot_models.step()
                        observations = env.sim.step(None)
                    continue

                walker.step()
                actions = []
                for i in range(num_agents):
                    agent_state = env.sim.get_agent_state(i)
                    action = agent[i].act(
                        agent_observation(observations, i),
                        agent_state,
                        command.action,
                    )
                    actions.append(action)

                observations = env.step(actions)
                robot_models.step()
                if robot_models.enabled:
                    observations = merge_visual_observations(
                        observations, env.sim.step(None)
                    )
                count_step += 1

                metrics = env.get_metrics()
                metric_line = format_metrics(
                    metrics, ("distance_to_goal", "success", "spl")
                )
                print(
                    f"episode={count_episodes} step={count_step} "
                    f"action={command.action} {metric_line}"
                )

            count_episodes += 1
    finally:
        env.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main(get_args())
