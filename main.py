from collections import deque, defaultdict
from typing import Dict
from itertools import count
import os
import logging
import time
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from habitat import Env, logger
from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils import chat_utils
import system_prompt
from utils.explored_map_utils import Global_Map_Proc, detect_frontier


from agents.vlm_agents import VLM_Agent
import utils.visualization as vu
from arguments import get_args
from utils.fire_sensors import FireSensorSuite, FireSensorConfig, FireSensorViewer


import cv2
import open3d as o3d


from habitat.config.default import get_config

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

import threading
from multiprocessing import Process, Queue
import multiprocessing as mp

# Gui
import open3d.visualization.gui as gui

from utils.vis_gui import ReconstructionWindow


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]
    
def main(args, send_queue, receive_queue):

    # ------------------------------------------------------------------
    ##### Setup Logging
    # ------------------------------------------------------------------
    log_dir = "{}/logs/{}/".format(args.dump_location, args.nav_mode)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.nav_mode)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'output.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    # print(args)
    logging.info(args)
    
    agg_metrics: Dict = defaultdict(float)
    # ------------------------------------------------------------------
    
    # ------------------------------------------------------------------
    ##### Setup Configuration
    # ------------------------------------------------------------------
    config = get_config(config_paths=["configs/"+ args.task_config])
    args.turn_angle = config.SIMULATOR.TURN_ANGLE
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config.defrost()
    config.SIMULATOR.NUM_AGENTS = args.num_agents
    config.SIMULATOR.AGENTS = ["AGENT_"+str(i) for i in range(args.num_agents)]
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    # ------------------------------------------------------------------
    # 360° LIDAR: install 4 yaw-rotated depth sensors on every agent so
    # the fire-scene LIDAR simulator can stitch a true 360° point cloud.
    # Each slice covers HFOV=90°, oriented at yaw {0, +π/2, π, -π/2}.
    # The 'rgb' / 'depth' UUIDs used by the navigation stack stay
    # untouched - we only add four extra UUIDs.
    # ------------------------------------------------------------------
    if int(getattr(args, "lidar_360", 0)) and getattr(args, "fire_sensors", 0):
        from utils.fire_sensors.lidar_360 import (
            install_lidar_depth_sensors,
            LIDAR_DEPTH_UUIDS,
        )
        install_lidar_depth_sensors(
            config,
            base_depth_cfg=config.SIMULATOR.DEPTH_SENSOR,
            resolution=int(args.lidar_resolution),
            num_agents=args.num_agents,
        )
        print(f"[lidar_360] installed sensors: {LIDAR_DEPTH_UUIDS}")
    config.freeze()
    # ------------------------------------------------------------------
    
    
    # ------------------------------------------------------------------
    ##### Setup Environment and Agents
    # ------------------------------------------------------------------
    env = Env(config=config)
    
    num_episodes = env.number_of_episodes

    assert num_episodes > 0, "num_episodes should be greater than 0"

    num_agents = config.SIMULATOR.NUM_AGENTS
    agent = []
    for i in range(num_agents):
        follower = ShortestPathFollowerCompat(
            env._sim, 0.1, False, i
        )
        agent.append(VLM_Agent(args, i, follower, receive_queue))
        
    map_process = Global_Map_Proc(args)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    ##### Fire scene + sensor suite
    # The fire-world side owns the *what* (3D voxel timeline of flames
    # / smoke / temperature). The sensor suite owns the *how the agent
    # sees it* (Beer-Lambert RGB + noisy depth + radar/lidar/thermal +
    # voxel observer). Scene is optional; suite handles "no scene"
    # gracefully and falls back to Beer-Lambert / HSV thermal.
    # ------------------------------------------------------------------
    fire_scene = None
    if int(getattr(args, "fire_world", 0)):
        from utils.fire_world.scene import FireScene
        fire_scene = FireScene.from_args(args, config)
        print(f"[fire_world] {fire_scene.describe()}")

    fire_suites = None
    fire_viewers = None
    if getattr(args, "fire_sensors", 0) or fire_scene is not None:
        from utils.general_utils import get_camera_K
        from utils.fire_sensors.config import VoxelSmokeConfig

        rgb_source = "voxel" if fire_scene is not None else "beer_lambert"
        thermal_source = "voxel" if fire_scene is not None else "hsv"
        fire_cfg = FireSensorConfig(
            max_depth_m=float(config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH),
            hfov_deg=float(config.SIMULATOR.DEPTH_SENSOR.HFOV),
            smoke_density=float(args.smoke_density),
            save_npz=bool(args.fire_save_npz),
            rgb_source=rgb_source,
            thermal_source=thermal_source,
            compound_rgb=bool(int(getattr(args, "fire_world_compound_rgb", 0))),
            voxel=VoxelSmokeConfig(
                n_steps=int(args.fire_world_n_steps),
                smoke_k_ext=float(args.fire_world_smoke_k_ext),
                render_scale=float(getattr(args, "fire_world_render_scale", 0.5)),
                # Match the keyboard teleop window so flame regions in
                # the dashboard always render as warm INFERNO instead
                # of a near-grey blob, regardless of exposure.
                thermal_color_blend=1.0,
            ),
        )
        K = get_camera_K(args.frame_width, args.frame_height, args.hfov)
        fire_suites = [
            FireSensorSuite(
                cfg=fire_cfg,
                dump_dir=os.path.join(args.fire_dump_dir, f"agent_{i}"),
                save_every=int(args.fire_save_every),
                seed=args.seed + i,
                scene=fire_scene,
                camera_K=K,
            )
            for i in range(num_agents)
        ]
        if int(getattr(args, "fire_show_window", 0)):
            fire_viewers = [
                FireSensorViewer.start(
                    window_name=f"Fire Sensors - agent {i}",
                    fps=10.0,
                    fallback_path=os.path.join(
                        args.fire_dump_dir, f"agent_{i}", "live.png"
                    ),
                )
                for i in range(num_agents)
            ]
        print(f"[fire_sensors] enabled, density={args.smoke_density}, "
              f"rgb_source={rgb_source}, thermal_source={thermal_source}, "
              f"dump_dir={args.fire_dump_dir}")

    from utils.fire_pipeline import step_fire_observation

    count_episodes = 0
    goal_points = []
    log_start = time.time()
    total_usage = []
    
    while count_episodes < num_episodes:
        observations = env.reset()
        actions = []
        map_process.reset()

        # Reset the fire-time origin so each episode starts at t_sim=base_t0.
        # In wallclock mode this captures "now" so the timeline advances
        # continuously while the agent navigates; in step mode it's a no-op.
        if fire_scene is not None:
            fire_scene.clock.start()
        
        agent_state = env.sim.get_agent_state(0)
        for i in range(num_agents):
            agent[i].reset(observations[i], agent_state)
            actions.append(0)
            
        count_step = 0
        point_sum = o3d.geometry.PointCloud()
        while not env.episode_over:
            start = time.time()
            visited_vis = []
            pose_pred = []
            point_sum.clear()
            found_goal = False
            # ----------------------------------------------------------
            # Fire-scene perception: run the sensor suite on the agent
            # observations. The suite owns whether RGB/thermal come
            # from FireWorld voxels or Beer-Lambert/HSV; the world
            # model is bound once at construction time.
            # ----------------------------------------------------------
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
                    )
                    if sensors is not None:
                        fire_suites[i].save_step(
                            sensors,
                            episode=count_episodes,
                            step=int(getattr(agent[i], "l_step", 0)),
                            agent_id=i,
                        )
                        if fire_viewers is not None:
                            fire_viewers[i].update(sensors.get("dashboard"))

            for i in range(num_agents):
                agent_state = env.sim.get_agent_state(i)
                agent[i].mapping(observations[i], agent_state)
                point_sum += agent[i].point_sum
                visited_vis.append(agent[i].visited_vis)
                pose_pred.append([agent[i].current_grid_pose[1], int(agent[i].map_size)-agent[i].current_grid_pose[0], np.deg2rad(agent[i].relative_angle)])
                if agent[i].found_goal:
                    found_goal = True 
                
            obstacle_map, explored_map, top_view_map = map_process.Map_Extraction(point_sum, agent[0].camera_position[1])
            # target_score, target_edge_map, target_point_list = map_process.Frontier_Det(threshold_point=8)
            
            if (agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0) and not found_goal:
                goal_points.clear()
                target_score, target_edge_map, target_point_list = map_process.Frontier_Det(threshold_point=8)
                
                if args.nav_mode == "gpt":
                    # ===== GPT 模式：使用 GPT 选择全局目标 =====
                    if len(target_point_list) > 0 and agent[0].l_step > 0:
                        candidate_map_list = chat_utils.get_all_candidate_maps(target_edge_map, top_view_map, pose_pred)
                        message = chat_utils.message_prepare(system_prompt.system_prompt, candidate_map_list, agent[i].goal_name)
                        goal_frontiers = chat_utils.chat_with_gpt4v(message)
                        for i in range(num_agents):
                            goal_points.append(target_point_list[int(goal_frontiers["robot_"+ str(i)].split('_')[1])])
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                            goal_points.append([int(action[0]), int(action[1])])
                
                elif args.nav_mode == "nearest":
                    # ===== 最近距离模式：每个 agent 选择离自己最近的前沿点 =====
                    if len(target_point_list) > 0:
                        for i in range(num_agents):
                            distances = [np.linalg.norm(np.array(target_point_list[j]) - np.array(pose_pred[i][:2])) for j in range(len(target_point_list))]
                            closest_idx = np.argmin(distances)
                            goal_points.append(target_point_list[closest_idx])
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                            goal_points.append([int(action[0]), int(action[1])])
                
                elif args.nav_mode == "co_ut":
                    # ===== 合作模式：为每个 agent 分配不同的前沿点 =====
                    if len(target_point_list) > 0:
                        assigned_frontiers = set()
                        for i in range(num_agents):
                            # 优先选择未被分配的距离最近的前沿点
                            best_idx = -1
                            best_dist = float('inf')
                            for j, frontier in enumerate(target_point_list):
                                if j not in assigned_frontiers:
                                    dist = np.linalg.norm(np.array(frontier) - np.array(pose_pred[i][:2]))
                                    if dist < best_dist:
                                        best_dist = dist
                                        best_idx = j
                            
                            if best_idx != -1:
                                goal_points.append(target_point_list[best_idx])
                                assigned_frontiers.add(best_idx)
                            else:
                                # 如果所有前沿点都被分配，选择最近的
                                distances = [np.linalg.norm(np.array(target_point_list[j]) - np.array(pose_pred[i][:2])) for j in range(len(target_point_list))]
                                goal_points.append(target_point_list[np.argmin(distances)])
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                            goal_points.append([int(action[0]), int(action[1])])
                
                elif args.nav_mode == "fill":
                    # ===== 覆盖模式：优先探索高价值区域（基于前沿得分） =====
                    if len(target_point_list) > 0:
                        for i in range(num_agents):
                            # 选择前沿点中得分最高的（探索价值最大）
                            best_idx = 0
                            best_score = -1
                            for j, frontier in enumerate(target_point_list):
                                if target_score is not None and j < len(target_score):
                                    score = target_score[j]
                                else:
                                    # 如果没有得分，用距离倒数作为替代
                                    score = 1.0 / (1.0 + np.linalg.norm(np.array(frontier) - np.array(pose_pred[i][:2])))
                                
                                if score > best_score:
                                    best_score = score
                                    best_idx = j
                            
                            goal_points.append(target_point_list[best_idx])
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                            goal_points.append([int(action[0]), int(action[1])])
                
                else:
                    # 默认：随机选择前沿点
                    for i in range(num_agents):
                        if len(target_point_list) > 0:
                            goal_points.append(target_point_list[np.random.randint(0, len(target_point_list))])
                        else:
                            action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                            goal_points.append([int(action[0]), int(action[1])])
                            
            goal_map = []
            for i in range(num_agents):
                agent[i].obstacle_map = obstacle_map
                agent[i].explored_map = explored_map
                actions[i] = agent[i].act(goal_points[i])
                goal_map.append(agent[i].goal_map)
            # print(actions)
            
            if args.visualize or args.print_images:
                vis_image = vu.Visualize(
                    args, agent[0].l_step, 
                    pose_pred, 
                    obstacle_map, 
                    explored_map, 
                    agent[0].goal_id, 
                    visited_vis, 
                    target_edge_map, 
                    goal_map, 
                    transform_rgb_bgr(top_view_map),
                    agent[0].episode_n)
        
            observations = env.step(actions)
            
            step_end = time.time()
            step_time = step_end - start
            # print('step_time: %.3f秒'%step_time)

       
        count_episodes += 1
        count_step += agent[0].l_step

        # ------------------------------------------------------------------
        ##### Logging
        # ------------------------------------------------------------------
        log_end = time.time()
        time_elapsed = time.gmtime(log_end - log_start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_step),
            "FPS {},".format(int(count_step / (log_end - log_start)))
        ]) + '\n'

        metrics = env.get_metrics()
        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v

        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

        # log += "Total usage: " + str(sum(total_usage)) + ", average usage: " + str(np.mean(total_usage))
        print(log)
        logging.info(log)
        # ------------------------------------------------------------------


    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    if fire_viewers is not None:
        for v in fire_viewers:
            v.stop()

    return avg_metrics

def visualization_thread(send_queue, receive_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app_win = ReconstructionWindow(args, mono, send_queue, receive_queue)
    app.run()


if __name__ == "__main__":
    args = get_args()

    send_queue = Queue()
    receive_queue = Queue()

    if args.visualize:
        # Create a thread for the Open3D visualization
        visualization = threading.Thread(target=visualization_thread, args=(send_queue, receive_queue,))
        visualization.start()

    # Run ROS code in the main thread
    main(args, send_queue, receive_queue)
