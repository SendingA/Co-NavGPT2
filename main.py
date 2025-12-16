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

# def simulate_fire(observations, fire_mask_prev=None, spread_prob=0.3):
#     """
#     给 observations 添加火焰和蔓延效果，同时更新 RGB 和 Depth。
    
#     Args:
#         observations: list of dict，每个 dict 包含 'rgb' 和 'depth'
#         fire_mask_prev: 上一步火焰位置 mask，shape = (H, W)
#         spread_prob: 火焰蔓延概率
    
#     Returns:
#         new_fire_mask: 当前火焰 mask
#         observations: 修改后的 RGB 和 Depth
#     """
#     H, W, _ = observations[0]["rgb"].shape
    
#     # 初始化火焰 mask
#     if fire_mask_prev is None:
#         fire_mask = np.zeros((H, W), dtype=bool)
#         # 随机点火源，10 个点
#         for _ in range(10):
#             x = np.random.randint(0, W)
#             y = np.random.randint(0, H)
#             fire_mask[y, x] = True
#     else:
#         fire_mask = fire_mask_prev.copy()
    
#     # 火焰蔓延：卷积扩散 + 随机概率
#     kernel = np.ones((3, 3), dtype=np.uint8)
#     dilated = cv2.dilate(fire_mask.astype(np.uint8), kernel)
#     new_cells = (dilated.astype(bool)) & (~fire_mask)
    
#     random_mask = np.random.rand(H, W) < spread_prob
#     fire_mask |= new_cells & random_mask
    
#     # RGB 渲染火焰
#     fire_color = np.array([255, 100, 0], dtype=np.uint8)  # 火焰橙色
#     for obs in observations:
#         rgb = obs["rgb"]
#         rgb[fire_mask] = np.clip(rgb[fire_mask] + fire_color, 0, 255)
#         # Depth 前景遮挡
#         depth = obs["depth"]
#         depth[fire_mask] = 0
#         obs["rgb"] = rgb
#         obs["depth"] = depth
    
#     return fire_mask, observations

def simulate_fire_with_blindspots(observations, fire_mask_prev=None, spread_prob=0.3, blind_prob=0.7):
    """
    火灾蔓延 + 感知盲区（局部深度/RGB遮挡）
    
    blind_prob: 机器人在火焰区域丢失感知的概率
    """
    H, W, _ = observations[0]["rgb"].shape
    
    # 初始化火焰 mask
    if fire_mask_prev is None:
        fire_mask = np.zeros((H, W), dtype=bool)
        for _ in range(10):
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            fire_mask[y, x] = True
    else:
        fire_mask = fire_mask_prev.copy()
    
    # 火焰蔓延
    kernel = np.ones((3,3), dtype=np.uint8)
    dilated = cv2.dilate(fire_mask.astype(np.uint8), kernel)
    new_cells = (dilated.astype(bool)) & (~fire_mask)
    random_mask = np.random.rand(H, W) < spread_prob
    fire_mask |= new_cells & random_mask

    # 感知盲区：部分 fire_mask 区域被随机遮挡
    blind_mask = (np.random.rand(H, W) < blind_prob) & fire_mask

    # 修改 observations
    for obs in observations:
        rgb = obs["rgb"].copy()
        depth = obs["depth"].copy()
        # RGB 遮挡: 用黑色覆盖
        rgb[blind_mask] = 0
        # Depth 遮挡: 用 0 覆盖
        depth[blind_mask] = 0
        obs["rgb"] = rgb
        obs["depth"] = depth

    return fire_mask, observations


def create_fire_texture(height, width):
    """
    创建火焰纹理，使用 Perlin 噪声模拟火焰形状
    返回火焰强度图 (0-1) 和火焰颜色图
    """
    # 使用多层噪声创建火焰效果
    y_coords = np.linspace(0, 4, height)
    x_coords = np.linspace(0, 4, width)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # 基础噪声
    noise1 = np.sin(xx * 3 + np.random.rand() * 10) * np.cos(yy * 2 + np.random.rand() * 10)
    noise2 = np.sin(xx * 7 + np.random.rand() * 10) * np.cos(yy * 5 + np.random.rand() * 10) * 0.5
    noise3 = np.sin(xx * 13 + np.random.rand() * 10) * np.cos(yy * 11 + np.random.rand() * 10) * 0.25
    
    noise = (noise1 + noise2 + noise3 + 1.75) / 3.5  # 归一化到 0-1
    
    # 火焰向上变细（底部宽，顶部窄）
    y_factor = np.linspace(1, 0, height).reshape(-1, 1)
    fire_shape = noise * (0.3 + 0.7 * y_factor)
    
    return np.clip(fire_shape, 0, 1)


def get_fire_color(intensity):
    """
    根据火焰强度返回颜色 (BGR格式)
    强度高 -> 黄白色（中心）
    强度中 -> 橙黄色
    强度低 -> 红橙色（边缘）
    """
    # 火焰颜色渐变 (BGR)
    # 红色 -> 橙色 -> 黄色 -> 白色
    if intensity > 0.8:
        # 黄白色核心
        return np.array([200, 255, 255], dtype=np.float32)
    elif intensity > 0.6:
        # 亮黄色
        return np.array([50, 255, 255], dtype=np.float32)
    elif intensity > 0.4:
        # 橙黄色
        return np.array([0, 180, 255], dtype=np.float32)
    elif intensity > 0.2:
        # 橙红色
        return np.array([0, 100, 255], dtype=np.float32)
    else:
        # 暗红色边缘
        return np.array([0, 50, 200], dtype=np.float32)


def simulate_fire_with_blindspots_and_visual(observations, fire_mask_prev=None, spread_prob=0.05, blind_prob=0.5, 
                                              smoke_intensity=0.4, fire_depth_threshold=3.5):
    """
    基于深度的火灾模拟：逼真的火焰和烟雾效果
    
    Args:
        observations: 观测数据列表
        fire_mask_prev: 上一帧的火焰/烟雾 mask
        spread_prob: 烟雾扩散概率
        blind_prob: 烟雾遮挡概率（深度失效）
        smoke_intensity: 烟雾强度 (0-1)
        fire_depth_threshold: 火源距离阈值（米）
    
    Returns:
        fire_mask: 当前火焰 mask
        observations: 修改后的观测数据
    """
    H, W, _ = observations[0]["rgb"].shape
    
    modified_observations = []
    
    for obs_idx, obs in enumerate(observations):
        rgb = obs["rgb"].copy()
        # 处理 RGBA 图像
        has_alpha = rgb.shape[-1] == 4
        if has_alpha:
            alpha_channel = rgb[:, :, 3:4].copy()
            rgb = rgb[:, :, :3]
        
        rgb = rgb.astype(np.float32)
        depth = obs["depth"].copy().squeeze()  # (H, W)
        
        # ====== 1. 基于深度生成火焰区域 ======
        valid_depth = (depth > 0.3) & (depth < fire_depth_threshold)
        
        if fire_mask_prev is None:
            # 第一帧：初始化火源位置
            fire_mask = np.zeros((H, W), dtype=np.float32)
            fire_region = valid_depth.copy()
            fire_region[:int(H*0.4), :] = False  # 上部不生成火源
            
            if np.any(fire_region):
                fire_points = np.where(fire_region)
                if len(fire_points[0]) > 0:
                    num_sources = min(3, len(fire_points[0]) // 100 + 1)
                    indices = np.random.choice(len(fire_points[0]), num_sources, replace=False)
                    for idx in indices:
                        cy, cx = fire_points[0][idx], fire_points[1][idx]
                        # 创建椭圆形火焰区域
                        fire_h, fire_w = np.random.randint(40, 80), np.random.randint(30, 60)
                        y_min, y_max = max(0, cy - fire_h), min(H, cy + fire_h // 3)
                        x_min, x_max = max(0, cx - fire_w // 2), min(W, cx + fire_w // 2)
                        
                        # 创建火焰形状的 mask（椭圆 + 噪声）
                        local_h, local_w = y_max - y_min, x_max - x_min
                        if local_h > 5 and local_w > 5:
                            fire_texture = create_fire_texture(local_h, local_w)
                            fire_mask[y_min:y_max, x_min:x_max] = np.maximum(
                                fire_mask[y_min:y_max, x_min:x_max], fire_texture
                            )
        else:
            fire_mask = fire_mask_prev.copy()
            # 火焰轻微晃动
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-1, 2)
            fire_mask = np.roll(np.roll(fire_mask, shift_x, axis=1), shift_y, axis=0)
            # 添加随机波动
            fire_mask = fire_mask * (0.85 + 0.3 * np.random.rand(H, W))
            fire_mask = np.clip(fire_mask, 0, 1)
        
        # ====== 2. 烟雾生成 ======
        # 烟雾从火焰上方产生并向上扩散
        smoke_mask = np.zeros((H, W), dtype=np.float32)
        
        # 烟雾基于火焰位置向上扩散
        fire_binary = (fire_mask > 0.1).astype(np.uint8)
        if np.any(fire_binary):
            # 向上扩散烟雾
            kernel_smoke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25))
            smoke_dilated = cv2.dilate(fire_binary, kernel_smoke, iterations=3)
            
            # 烟雾主要在火焰上方
            for i in range(1, 8):
                rolled = np.roll(fire_binary, -i * 15, axis=0)
                smoke_mask += rolled.astype(np.float32) * (0.8 ** i)
            
            smoke_mask = cv2.GaussianBlur(smoke_mask, (21, 21), 0)
            smoke_mask = np.clip(smoke_mask, 0, 1)
            
            # 烟雾随机扩散
            random_smoke = (np.random.rand(H, W) < spread_prob * 3).astype(np.float32) * 0.3
            smoke_mask = np.maximum(smoke_mask, random_smoke * smoke_dilated)
        
        # ====== 3. 应用火焰视觉效果 ======
        fire_visible = (fire_mask > 0.05) & valid_depth
        
        if np.any(fire_visible):
            # 为每个像素计算火焰颜色（基于强度的渐变）
            fire_intensity = fire_mask[fire_visible]
            
            # 创建火焰颜色数组
            fire_colors = np.zeros((np.sum(fire_visible), 3), dtype=np.float32)
            
            # 根据强度分配颜色
            # 核心：黄白色
            mask_core = fire_intensity > 0.7
            fire_colors[mask_core] = [150, 255, 255]  # BGR: 亮黄白
            
            # 中间：橙黄色
            mask_mid = (fire_intensity > 0.4) & (fire_intensity <= 0.7)
            fire_colors[mask_mid] = [0, 200, 255]  # BGR: 橙黄
            
            # 边缘：橙红色
            mask_edge = (fire_intensity > 0.15) & (fire_intensity <= 0.4)
            fire_colors[mask_edge] = [0, 80, 255]  # BGR: 橙红
            
            # 外围：暗红色
            mask_outer = fire_intensity <= 0.15
            fire_colors[mask_outer] = [0, 30, 180]  # BGR: 暗红
            
            # 添加闪烁效果
            flicker = 0.7 + 0.3 * np.random.rand(np.sum(fire_visible), 1)
            fire_colors = fire_colors * flicker
            
            # 混合原图和火焰
            alpha = (fire_intensity * 0.8 + 0.2).reshape(-1, 1)  # 火焰越强越不透明
            rgb[fire_visible] = np.clip(
                rgb[fire_visible] * (1 - alpha) + fire_colors * alpha, 0, 255
            )
            
            # 添加火焰光晕效果
            glow_mask = cv2.GaussianBlur(fire_mask, (15, 15), 0)
            glow_visible = (glow_mask > 0.02) & valid_depth & (~fire_visible)
            if np.any(glow_visible):
                glow_intensity = glow_mask[glow_visible].reshape(-1, 1)
                glow_color = np.array([0, 100, 255], dtype=np.float32)  # 橙红色光晕
                rgb[glow_visible] = np.clip(
                    rgb[glow_visible] + glow_color * glow_intensity * 0.5, 0, 255
                )
        
        # ====== 4. 应用烟雾视觉效果 ======
        smoke_visible = (smoke_mask > 0.05) & (~fire_visible)
        
        if np.any(smoke_visible):
            smoke_intensity_local = smoke_mask[smoke_visible].reshape(-1, 1)
            
            # 烟雾颜色：灰色带一点暖色调
            smoke_color = np.array([90, 85, 80], dtype=np.float32)  # BGR: 暖灰色
            
            # 混合
            alpha_smoke = smoke_intensity_local * smoke_intensity
            rgb[smoke_visible] = np.clip(
                rgb[smoke_visible] * (1 - alpha_smoke) + smoke_color * alpha_smoke, 0, 255
            )
        
        # ====== 5. 深度传感器干扰 ======
        # 烟雾和火焰区域深度不可靠
        depth_affected = (smoke_mask > 0.1) | (fire_mask > 0.1)
        depth_noise_mask = depth_affected & (np.random.rand(H, W) < blind_prob)
        
        if np.any(depth_noise_mask):
            # 烟雾区域添加深度噪声或置零
            noise_type = np.random.rand(H, W)
            # 50% 置零，50% 添加噪声
            zero_mask = depth_noise_mask & (noise_type < 0.5)
            noise_mask = depth_noise_mask & (noise_type >= 0.5)
            
            depth[zero_mask] = 0
            if np.any(noise_mask):
                depth[noise_mask] += np.random.randn(np.sum(noise_mask)) * 0.3
                depth = np.clip(depth, 0, 10)
        
        # ====== 6. 全局烟雾效果 ======
        # 整体画面添加轻微烟雾感
        if np.any(smoke_mask > 0.1):
            # 全局对比度降低
            rgb = rgb * 0.92 + 20
            
            # 轻微模糊
            if np.random.rand() < 0.5:
                rgb = cv2.GaussianBlur(rgb.astype(np.uint8), (3, 3), 0).astype(np.float32)
        
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        # 恢复 alpha 通道
        if has_alpha:
            rgb = np.concatenate([rgb, alpha_channel], axis=-1)
        
        # 更新观测
        obs_modified = obs.copy()
        obs_modified["rgb"] = rgb
        obs_modified["depth"] = depth.reshape(H, W, 1) if len(obs["depth"].shape) == 3 else depth
        modified_observations.append(obs_modified)

    return fire_mask, modified_observations

    return smoke_mask, modified_observations


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

    count_episodes = 0
    goal_points = []
    log_start = time.time()
    total_usage = []
    
    while count_episodes < num_episodes:
        observations = env.reset()
        actions = []
        map_process.reset()
        
        agent_state = env.sim.get_agent_state(0)
        
        # 初始化火焰 mask（每个 episode 开始时重置）
        fire_mask = None
        
        # 对初始观测应用火焰效果
        fire_mask, observations = simulate_fire_with_blindspots_and_visual(
            observations, fire_mask_prev=fire_mask, spread_prob=0.05, blind_prob=0.5,
            smoke_intensity=0.3, fire_depth_threshold=3.0
        )
        
        for i in range(num_agents):
            agent[i].reset(observations[i], agent_state)  # 使用修改后的 observations
            actions.append(0)
            
        count_step = 0
        point_sum = o3d.geometry.PointCloud()

        while not env.episode_over:
            start = time.time()
            visited_vis = []
            pose_pred = []
            point_sum.clear()
            found_goal = False
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
                # if args.nav_mode == "gpt":
                target_score, target_edge_map, target_point_list = map_process.Frontier_Det(threshold_point=8)
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
            
            # 对新观测应用火焰效果（火焰会随时间蔓延）
            fire_mask, observations = simulate_fire_with_blindspots_and_visual(
                observations, fire_mask_prev=fire_mask, spread_prob=0.05, blind_prob=0.5,
                smoke_intensity=0.3, fire_depth_threshold=3.0
            )
            
            step_end = time.time()
            step_time = step_end - start
            # print('step_time: %.3f秒'%step_time)

       
        count_episodes += 1
        count_step += agent[0].l_step

        # ------------------------------------------------------------------
        ##### 生成 Episode 视频
        # ------------------------------------------------------------------
        if args.print_images and args.save_video:
            video_paths = vu.create_episode_video(args, agent[0].episode_n, rank=0)
            if video_paths:
                for vp in video_paths:
                    logging.info(f"Episode {agent[0].episode_n} video saved: {vp}")

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
