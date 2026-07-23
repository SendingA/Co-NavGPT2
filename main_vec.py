from collections import defaultdict
from typing import Dict
import os
import logging
import time
import random
import multiprocessing as mp

import numpy as np
import torch
import cv2  # noqa: F401  (imported for downstream visualization helpers)
import open3d as o3d

import habitat
from habitat import Env, make_dataset
from habitat.config.read_write import read_write

from arguments import get_args, load_config, humanoid_kwargs, robot_model_kwargs
from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils import chat_utils
import system_prompt
import utils.visualization as vu

from agents.vlm_multi_agents import VLM_Agent
from envs import RandomHumanoidWalker, RobotModelManager
from utils.explored_map_utils import Global_Map_Proc
from utils.fire_sensors import FireSensorSuite, FireSensorConfig


def CoNav_env(args, config, rank, dataset, send_queue, receive_queue):
    if int(getattr(args, "risk_enabled", 0)):
        raise RuntimeError(
            "dynamic risk assessment is wired through main.py only; "
            "main_vec.py currently refuses --risk_enabled=1 so a run cannot "
            "be mislabeled without synchronized risk maps/evaluation"
        )
    args.rank = rank
    seed = int(config.habitat.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    env = Env(config=config, dataset=dataset)

    num_episodes = len(env.episodes)
    print("num_episodes: ", num_episodes)
    receive_queue.put(num_episodes)
    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = int(config.conav.num_robots)
    agent = []
    for i in range(num_agents):
        follower = ShortestPathFollowerCompat(env.sim, 0.1, False, i)
        agent.append(VLM_Agent(args, i, follower))

    map_process = Global_Map_Proc(args)

    walker = RandomHumanoidWalker(
        sim=env.sim,
        **humanoid_kwargs(config, args.seed),
    )
    robot_models = RobotModelManager(
        sim=env.sim,
        **robot_model_kwargs(config, num_agents),
    )

    # ------------------------------------------------------------------
    # Fire scene + sensor suite (per rank)
    # ------------------------------------------------------------------
    fire_scene = None
    if int(getattr(args, "fire_world", 0)):
        from utils.fire_world.scene import FireScene
        fire_scene = FireScene.from_args(args, config)
        print(f"[fire_world rank={rank}] {fire_scene.describe()}")

    fire_suites = None
    if getattr(args, "fire_sensors", 0) or fire_scene is not None:
        from utils.general_utils import get_camera_K
        from utils.fire_sensors.config import VoxelSmokeConfig

        main_agent_name = config.habitat.simulator.agents_order[0]
        depth_cfg = (
            config.habitat.simulator.agents[main_agent_name]
            .sim_sensors.depth_sensor
        )

        rgb_source = "voxel" if fire_scene is not None else "beer_lambert"
        thermal_source = "voxel" if fire_scene is not None else "hsv"
        from arguments import voxel_smoke_kwargs
        fire_cfg = FireSensorConfig(
            max_depth_m=float(depth_cfg.max_depth),
            hfov_deg=float(depth_cfg.hfov),
            smoke_density=float(args.smoke_density),
            save_npz=bool(args.fire_save_npz),
            rgb_source=rgb_source,
            thermal_source=thermal_source,
            compound_rgb=bool(int(getattr(args, "fire_world_compound_rgb", 0))),
            voxel=VoxelSmokeConfig(**voxel_smoke_kwargs(args)),
        )
        K = get_camera_K(args.frame_width, args.frame_height, args.hfov)
        fire_suites = [
            FireSensorSuite(
                cfg=fire_cfg,
                dump_dir=os.path.join(
                    args.fire_dump_dir, f"rank_{rank}", f"agent_{i}"
                ),
                save_every=int(args.fire_save_every),
                seed=args.seed + rank * 100 + i,
                scene=fire_scene,
                camera_K=K,
            )
            for i in range(num_agents)
        ]

    from utils.fire_pipeline import step_fire_observation

    start_signal = send_queue.get()
    print(start_signal)

    count_episodes = 0
    goal_points = []
    target_edge_map = None
    target_score = None
    goal_frontiers = None
    max_episode_steps = int(config.habitat.environment.max_episode_steps)

    while count_episodes < num_episodes:
        observations = env.reset()
        # walker/robot_models.reset() only spawn/repose articulated
        # objects; do NOT re-issue env.sim.step(None) — that would
        # bypass the ObjectNav task sensors (objectgoal/gps/compass).
        walker.reset()
        robot_models.reset()
        if not isinstance(observations, list):
            observations = [observations]

        actions = []
        map_process.reset()
        if fire_scene is not None:
            fire_scene.clock.start()

        agent_state = env.sim.get_agent_state(0)
        for i in range(num_agents):
            agent[i].reset(observations[i], agent_state)
            actions.append(0)

        count_steps = 0
        episode_start_time = time.time()
        point_sum = o3d.geometry.PointCloud()

        while not env.episode_over:
            visited_vis = []
            pose_pred = []
            point_sum.clear()
            found_goal = False
            clean_diff = True

            if fire_suites is not None:
                for i in range(num_agents):
                    a_state = env.sim.get_agent_state(i)
                    sensors = step_fire_observation(
                        observations=observations[i],
                        suite=fire_suites[i],
                        agent_state=a_state,
                        robot_step=int(getattr(agent[i], "l_step", 0)),
                        config=config,
                        args=args,
                        walker=walker,
                    )
                    if sensors is not None:
                        fire_suites[i].save_step(
                            sensors,
                            episode=count_episodes,
                            step=int(getattr(agent[i], "l_step", 0)),
                            agent_id=i,
                        )

            for i in range(num_agents):
                agent_state = env.sim.get_agent_state(i)
                agent[i].mapping(observations[i], agent_state)
                point_sum += agent[i].point_sum
                visited_vis.append(agent[i].visited_vis)
                pose_pred.append([
                    agent[i].current_grid_pose[1],
                    int(agent[i].map_size) - agent[i].current_grid_pose[0],
                    np.deg2rad(agent[i].relative_angle),
                ])
                if agent[i].found_goal:
                    found_goal = True
                if agent[i].clean_diff:
                    clean_diff = False

            obstacle_map, explored_map, top_view_map = map_process.Map_Extraction(
                point_sum, agent[0].camera_position[1], clean_diff
            )

            if (
                agent[0].l_step % args.num_local_steps == args.num_local_steps - 1
                or agent[0].l_step == 0
            ) and not found_goal:
                goal_points.clear()
                target_score, target_edge_map, target_point_list = (
                    map_process.Frontier_Det(threshold_point=8)
                )

                if args.fill_mode and len(target_point_list) > 0:
                    for i in range(num_agents):
                        if agent[i].curr_frontier_count > 2 * args.num_local_steps + 1:
                            if goal_frontiers is not None and (
                                "robot_" + str(i) in goal_frontiers
                            ):
                                idx = int(
                                    goal_frontiers["robot_" + str(i)].split("_")[1]
                                )
                                if idx < len(target_point_list):
                                    map_process.obstacle_map[
                                        target_edge_map == idx + 1
                                    ] = 1
                                    obstacle_map[target_edge_map == idx + 1] = 1
                            agent[i].curr_frontier_count = 0
                    target_score, target_edge_map, target_point_list = (
                        map_process.Frontier_Det(threshold_point=8)
                    )

                if args.nav_mode == "gpt":
                    if len(target_point_list) > 0 and agent[0].l_step > 0:
                        candidate_map_list = chat_utils.get_all_candidate_maps(
                            target_edge_map, top_view_map, pose_pred
                        )
                        message = chat_utils.message_prepare(
                            system_prompt.system_prompt,
                            candidate_map_list,
                            agent[i].goal_name,
                        )
                        goal_frontiers = chat_utils.chat_with_gpt4v(message)
                        for i in range(num_agents):
                            goal_points.append(
                                target_point_list[
                                    int(
                                        goal_frontiers["robot_" + str(i)].split("_")[1]
                                    )
                                ]
                            )
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(action[0]), int(action[1])])

                elif args.nav_mode == "nearest":
                    if len(target_point_list) > 0:
                        for i in range(num_agents):
                            distances = [
                                np.linalg.norm(
                                    np.array(target_point_list[j])
                                    - np.array(pose_pred[i][:2])
                                )
                                for j in range(len(target_point_list))
                            ]
                            goal_points.append(
                                target_point_list[np.argmin(distances)]
                            )
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(action[0]), int(action[1])])

                elif args.nav_mode == "co_ut":
                    if len(target_point_list) > 0:
                        assigned_frontiers = set()
                        for i in range(num_agents):
                            best_idx = -1
                            best_dist = float("inf")
                            for j, frontier in enumerate(target_point_list):
                                if j not in assigned_frontiers:
                                    dist = np.linalg.norm(
                                        np.array(frontier)
                                        - np.array(pose_pred[i][:2])
                                    )
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_idx = j
                            if best_idx != -1:
                                goal_points.append(target_point_list[best_idx])
                                assigned_frontiers.add(best_idx)
                            else:
                                distances = [
                                    np.linalg.norm(
                                        np.array(target_point_list[j])
                                        - np.array(pose_pred[i][:2])
                                    )
                                    for j in range(len(target_point_list))
                                ]
                                goal_points.append(
                                    target_point_list[np.argmin(distances)]
                                )
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(action[0]), int(action[1])])

                elif args.nav_mode == "fill":
                    if len(target_point_list) > 0:
                        for i in range(num_agents):
                            best_idx = 0
                            best_score = -1
                            for j, frontier in enumerate(target_point_list):
                                if target_score is not None and j < len(target_score):
                                    score = target_score[j]
                                else:
                                    score = 1.0 / (
                                        1.0
                                        + np.linalg.norm(
                                            np.array(frontier)
                                            - np.array(pose_pred[i][:2])
                                        )
                                    )
                                if score > best_score:
                                    best_score = score
                                    best_idx = j
                            goal_points.append(target_point_list[best_idx])
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(action[0]), int(action[1])])

                else:
                    for i in range(num_agents):
                        if len(target_point_list) > 0:
                            goal_points.append(
                                target_point_list[
                                    np.random.randint(0, len(target_point_list))
                                ]
                            )
                        else:
                            action = np.random.rand(1, 2).squeeze() * (
                                obstacle_map.shape[0] - 1
                            )
                            goal_points.append([int(action[0]), int(action[1])])

            goal_map = []
            for i in range(num_agents):
                agent[i].obstacle_map = obstacle_map
                agent[i].explored_map = explored_map
                actions[i] = agent[i].act(goal_points[i])
                goal_map.append(agent[i].goal_map)

            if args.visualize or args.print_images:
                vu.Visualize(
                    args, agent[0].l_step,
                    pose_pred,
                    obstacle_map,
                    explored_map,
                    agent[0].goal_id,
                    visited_vis,
                    target_edge_map,
                    goal_map,
                    top_view_map,
                    agent[0].episode_n,
                    rank,
                )

            walker.step()
            observations = env.step(actions)
            robot_models.step()
            if not isinstance(observations, list):
                observations = [observations]
            count_steps += 1

        infos = 0
        if 0 in actions and env.get_metrics()["spl"]:
            infos = 1  # success
        else:
            if count_steps >= max_episode_steps - 1:
                infos = 2  # exploration
            else:
                infos = 3  # detection

        count_episodes += 1

        episode_runtime = time.time() - episode_start_time
        fps = count_steps / max(episode_runtime, 1e-6)
        metrics = env.get_metrics()
        metrics["episode_runtime"] = episode_runtime
        metrics["episode_steps"] = count_steps
        metrics["fps"] = fps
        metrics["avg_step_time"] = episode_runtime / max(count_steps, 1)

        receive_queue.put([metrics, infos, count_steps, episode_runtime])


def _split_scenes_across_processes(scenes, num_processes):
    sizes = [int(np.floor(len(scenes) / num_processes)) for _ in range(num_processes)]
    for i in range(len(scenes) % num_processes):
        sizes[i] += 1
    return sizes


def main():
    args = get_args()
    if int(getattr(args, "risk_enabled", 0)):
        raise RuntimeError(
            "main_vec.py does not yet support RiskRuntime; run main.py for "
            "dynamic risk assessment"
        )

    log_dir = "{}/logs/{}/".format(args.dump_location, args.nav_mode)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=log_dir + "multi-agent.log",
        level=logging.INFO,
    )
    print("Dumping at {}".format(log_dir))
    logging.info(args)

    mp_ctx = mp.get_context("forkserver")
    receive_queue = mp_ctx.Queue()
    send_queue = mp_ctx.Queue()

    # Base config (Habitat 3.3 DictConfig).
    config_env = load_config(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if int(getattr(args, "lidar_360", 0)) and getattr(args, "fire_sensors", 0):
        from utils.fire_sensors.lidar_360 import (
            install_lidar_depth_sensors,
            LIDAR_DEPTH_UUIDS,
        )
        with habitat.config.read_write(config_env):
            install_lidar_depth_sensors(
                config_env,
                resolution=int(getattr(args, "lidar_resolution", 320)),
                num_agents=args.num_agents,
            )
        print(f"[lidar_360] installed sensors: {LIDAR_DEPTH_UUIDS}")

    # Discover scenes and split them across workers.
    dataset_cfg = config_env.habitat.dataset
    scenes = list(dataset_cfg.get("content_scenes", ["*"]))
    dataset = make_dataset(dataset_cfg.type, config=dataset_cfg)
    if "*" in scenes:
        scenes = dataset.get_scenes_to_load(dataset_cfg)

    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there aren't "
            "enough number of scenes"
        )
        scene_split_sizes = _split_scenes_across_processes(scenes, args.num_processes)

    num_episode = []
    processes = []
    for i in range(args.num_processes):
        proc_config = config_env.copy() if hasattr(config_env, "copy") else config_env
        with read_write(proc_config):
            if len(scenes) > 0:
                proc_config.habitat.dataset.content_scenes = scenes[
                    sum(scene_split_sizes[:i]) : sum(scene_split_sizes[: i + 1])
                ]
                print(f"Thread {i}: {list(proc_config.habitat.dataset.content_scenes)}")

        proc_dataset = make_dataset(
            proc_config.habitat.dataset.type, config=proc_config.habitat.dataset
        )
        with read_write(proc_config):
            proc_config.habitat.simulator.scene = proc_dataset.episodes[0].scene_id

        proc = mp_ctx.Process(
            target=CoNav_env,
            args=(args, proc_config, i, proc_dataset, send_queue, receive_queue),
        )
        processes.append(proc)
        proc.start()

        num_episode.append(receive_queue.get())

    num_episodes = sum(num_episode)
    print("total num_episodes: ", num_episodes)
    logging.info(num_episodes)

    for _ in range(args.num_processes):
        send_queue.put("start!")

    count_episodes = 0
    agg_metrics: Dict = defaultdict(float)
    total_fail = []
    total_steps = 0
    total_runtime = 0.0
    start = time.time()

    while count_episodes < num_episodes:
        if not receive_queue.empty():
            count_episodes += 1
            metrics, infos, count_steps, episode_runtime = receive_queue.get()

            total_steps += count_steps
            total_runtime += episode_runtime

            for m, v in metrics.items():
                agg_metrics[m] += v

            if infos > 0:
                total_fail.append(infos)

            end = time.time()
            time_elapsed = time.gmtime(end - start)
            overall_fps = total_steps / max(end - start, 1e-6)

            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "total_timesteps {},".format(total_steps),
                "overall_FPS {:.2f},".format(overall_fps),
                "avg_runtime {:.2f}s".format(
                    total_runtime / max(count_episodes, 1)
                ),
            ]) + "\n"

            log += "Failed Case: exploration/detection/success/total:"
            log += " {:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
                total_fail.count(2),
                total_fail.count(3),
                total_fail.count(1),
                len(total_fail),
            ) + "\n"

            log += "Metrics: "
            log += (
                ", ".join(
                    k + ": {:.3f}".format(v / count_episodes)
                    for k, v in agg_metrics.items()
                )
                + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)
            )

            print(log)
            logging.info(log)

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
