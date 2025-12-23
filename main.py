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
from utils.pointcloud_vis import export_episode_point_cloud

# 导入 Obstacle & Hazard Map 可视化工具
from scripts.visualize_obstacle_hazard_maps import ObstacleHazardMapVisualizer

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


def create_fire_texture(height, width, seed=None):
    """
    创建火焰纹理，使用 Perlin 噪声模拟火焰形状
    返回火焰强度图 (0-1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成多层傅里叶基础噪声，模拟火焰的卷曲边界
    y_coords = np.linspace(0, 4, height)
    x_coords = np.linspace(0, 4, width)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # 多层噪声组合：不同频率的正弦/余弦波
    noise1 = np.sin(xx * 2.5 + np.random.rand() * 6.28) * np.cos(yy * 1.8 + np.random.rand() * 6.28)
    noise2 = np.sin(xx * 5.2 + np.random.rand() * 6.28) * np.cos(yy * 4.1 + np.random.rand() * 6.28) * 0.6
    noise3 = np.sin(xx * 9.7 + np.random.rand() * 6.28) * np.cos(yy * 8.3 + np.random.rand() * 6.28) * 0.35
    noise4 = np.sin(xx * 15 + np.random.rand() * 6.28) * np.cos(yy * 13 + np.random.rand() * 6.28) * 0.15
    
    # 组合所有噪声层
    combined = (noise1 + noise2 * 0.8 + noise3 * 0.6 + noise4 * 0.4) / 2.5
    noise = (combined + 1.0) / 2.0  # 归一化到 0-1
    
    # 火焰应该向上变细，底部宽，顶部尖端
    # 使用二次函数形状因子
    y_factor = np.linspace(1.2, 0.05, height).reshape(-1, 1)
    fire_shape = noise * y_factor
    
    # 添加随机局部毛刺，增加真实感
    roughness = (np.random.rand(height, width) * 0.3 - 0.15)
    fire_shape = fire_shape + roughness * (y_factor ** 2)
    
    return np.clip(fire_shape, 0, 1)


def get_fire_color(intensity):
    """
    根据火焰强度返回 RGB 颜色（注意：Habitat 使用 RGB，不是 BGR）
    强度高 -> 白黄色（中心）
    强度中 -> 黄橙色
    强度低 -> 红橙色（边缘）
    """
    # 火焰颜色渐变 (RGB格式，不是BGR!)
    # 暗红 -> 红 -> 橙 -> 黄 -> 白
    if intensity > 0.85:
        # 亮白黄色核心
        return np.array([255, 255, 200], dtype=np.uint8)  # RGB: 白黄
    elif intensity > 0.7:
        # 亮黄色
        return np.array([255, 255, 80], dtype=np.uint8)  # RGB: 黄
    elif intensity > 0.5:
        # 黄橙色
        return np.array([255, 200, 50], dtype=np.uint8)  # RGB: 黄橙
    elif intensity > 0.3:
        # 橙色（主要火焰区）
        return np.array([255, 140, 20], dtype=np.uint8)  # RGB: 橙
    elif intensity > 0.15:
        # 红橙色
        return np.array([220, 80, 10], dtype=np.uint8)  # RGB: 红橙
    else:
        # 深红色边缘
        return np.array([180, 40, 5], dtype=np.uint8)  # RGB: 深红


class GlobalFireSimulator:
    """
    全局火灾模拟器：
    - 在世界坐标系中跟踪多个火焰簇
    - 每帧进行物理蔓延（扩散 + 消耗燃料）
    - 根据视角投影到相机视锥体中渲染
    - 智能选择火源位置：贴近地面/家具
    """
    def __init__(self, map_bounds=100.0, num_fire_sources=2, observations=None):
        self.fire_clusters = []
        self.map_bounds = map_bounds
        self.time_step = 0
        self.observations = observations  # 用于智能检测火源位置
        
        # 基于深度图智能初始化火源位置
        if observations is not None:
            self._initialize_fire_sources_smart(num_fire_sources)
        else:
            # 回退方案：随机位置
            for _ in range(num_fire_sources):
                x = np.random.uniform(-map_bounds / 4, map_bounds / 4)
                y = 0.3 + np.random.uniform(0, 0.3)  # 地面附近 0.3-0.6m
                z = np.random.uniform(-map_bounds / 4, map_bounds / 4)
                self.fire_clusters.append({
                    'center': np.array([x, y, z], dtype=np.float32),
                    'radius': np.random.uniform(1.5, 2.5),  # 增大初始半径
                    'intensity': 1.0,
                    'age': 0,
                    'texture_seed': np.random.randint(0, 10000)
                })
    
    def _initialize_fire_sources_smart(self, num_sources):
        """基于深度图智能选择火源位置"""
        if self.observations is None or len(self.observations) == 0:
            return
        
        # 使用第一个观测的深度图
        depth = self.observations[0]["depth"].squeeze()
        H, W = depth.shape
        
        # 找到有效的深度点（地面附近）
        valid_depth = (depth > 0.3) & (depth < 4.0)
        
        # 优先在下半部分找火源（地面/家具附近）
        valid_depth[:H//3, :] = False
        
        fire_points = np.where(valid_depth)
        
        if len(fire_points[0]) == 0:
            # 没有找到合适位置，使用随机
            for _ in range(num_sources):
                x = np.random.uniform(-self.map_bounds / 4, self.map_bounds / 4)
                y = 0.3 + np.random.uniform(0, 0.3)
                z = np.random.uniform(-self.map_bounds / 4, self.map_bounds / 4)
                self.fire_clusters.append({
                    'center': np.array([x, y, z], dtype=np.float32),
                    'radius': np.random.uniform(1.5, 2.5),
                    'intensity': 1.0,
                    'age': 0,
                    'texture_seed': np.random.randint(0, 10000)
                })
        else:
            # 从有效点中随机采样火源
            num_to_sample = min(num_sources, len(fire_points[0]))
            sampled_idx = np.random.choice(len(fire_points[0]), num_to_sample, replace=False)
            
            for idx in sampled_idx:
                py, px = fire_points[0][idx], fire_points[1][idx]
                fire_depth = depth[py, px]
                
                # 深度转世界坐标的近似（假设简单投影）
                # 这里使用像素坐标作为估计（真实应用需要相机内参反投影）
                x_world = (px / W - 0.5) * fire_depth
                z_world = (py / H - 0.5) * fire_depth + 1.0  # 加偏移
                y_world = 0.2 + np.random.uniform(0, 0.3)  # 地面附近
                
                self.fire_clusters.append({
                    'center': np.array([x_world, y_world, z_world], dtype=np.float32),
                    'radius': np.random.uniform(1.8, 2.8),  # 增大初始半径
                    'intensity': 1.0,
                    'age': 0,
                    'texture_seed': np.random.randint(0, 10000)
                })
    
    def update(self, dt=0.1):
        """更新火焰状态：蔓延、衰减、消耗"""
        self.time_step += 1
        
        for cluster in self.fire_clusters:
            cluster['age'] += dt
            
            # 蔓延：随时间增大半径（更明显）
            growth_rate = 0.12 + 0.05 * np.sin(self.time_step * 0.08)
            cluster['radius'] += growth_rate * dt
            cluster['radius'] = min(cluster['radius'], 4.5)
            
            # 衰减：年龄越大强度越低（更快衰减）
            cluster['intensity'] = max(0.1, 1.0 - cluster['age'] / 60.0)
    
    def get_fire_strength_at_world_pos(self, world_pos, max_distance=4.0):
        """计算世界坐标某点的火焰强度 (0-1)"""
        max_strength = 0.0
        for cluster in self.fire_clusters:
            dist = np.linalg.norm(world_pos - cluster['center'])
            if dist < cluster['radius'] + max_distance:
                # 高斯衰减
                strength = cluster['intensity'] * np.exp(-(dist / (cluster['radius'] + 0.3)) ** 2)
                max_strength = max(max_strength, strength)
        return max_strength

    def is_position_in_fire(self, world_pos, margin: float = 0.0) -> bool:
        """
        判断一个世界坐标是否进入火焰区域（严格接触判定）。
        当位置落在任意火焰簇半径范围内（考虑 margin）则视为接触。
        """
        for cluster in self.fire_clusters:
            dist = np.linalg.norm(world_pos - cluster['center'])
            if dist <= max(0.0, cluster['radius'] - margin):
                return True
        return False
    
    def get_fire_intensity_at_position(self, world_pos) -> float:
        """
        获取在世界坐标位置的火焰强度（0-1）。
        结合所有活跃火焰簇的贡献度。
        """
        max_intensity = 0.0
        for cluster in self.fire_clusters:
            dist = np.linalg.norm(world_pos - cluster['center'])
            # 当进入火焰簇范围内时计算强度
            if dist <= cluster['radius']:
                # 从中心到边界的强度从 cluster['intensity'] 衰减到 0
                intensity = cluster['intensity'] * (1.0 - dist / max(cluster['radius'], 0.01))
                max_intensity = max(max_intensity, intensity)
        return max_intensity
    
    def is_position_in_severe_fire(self, world_pos, intensity_threshold: float = 0.7) -> bool:
        """
        严格判定：位置是否在高强度火焰区域内。
        必须满足：(1) 在火焰簇范围内 (2) 该位置的火焰强度 >= intensity_threshold
        """
        fire_intensity = self.get_fire_intensity_at_position(world_pos)
        return fire_intensity >= intensity_threshold
    
    def project_to_camera(self, observations, agent_positions, camera_extrinsics, K):
        """
        将全局火焰投影到相机视图中，返回每个观测的火焰 mask 和颜色
        """
        H, W = observations[0]["rgb"].shape[:2]
        fire_masks = []
        fire_colors = []  # 新增：火焰颜色
        
        for obs_idx, obs in enumerate(observations):
            fire_mask = np.zeros((H, W), dtype=np.float32)
            fire_color_map = np.zeros((H, W, 3), dtype=np.float32)
            
            for cluster in self.fire_clusters:
                # 世界坐标转相机坐标
                world_pos_h = np.append(cluster['center'], 1.0)
                cam_pos = camera_extrinsics[obs_idx] @ world_pos_h
                cam_pos = cam_pos[:3]
                
                # 检查是否在相机前方
                if cam_pos[2] <= 0.1:
                    continue
                
                # 投影到像素坐标
                pixel_pos = K @ cam_pos
                pixel_pos = pixel_pos[:2] / pixel_pos[2]
                px, py = int(pixel_pos[0]), int(pixel_pos[1])
                
                if not (0 <= px < W and 0 <= py < H):
                    continue
                
                # 深度用来计算火焰的视觉大小（增大投影大小）
                depth_cam = cam_pos[2]
                projected_radius = max(40, min(150, int((cluster['radius'] / depth_cam) * K[0, 0] * 1.5)))
                
                # 绘制火焰区域：旋转的椭圆高斯 + 分形噪声，让形状更不规则
                size = projected_radius * 2 + 1
                y, x = np.ogrid[-projected_radius:projected_radius+1, -projected_radius:projected_radius+1]

                # 椭圆参数与随机旋转
                rs = np.random.RandomState(cluster['texture_seed'] + self.time_step)
                angle = rs.uniform(0, np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                # 椭圆长短轴，纵向更长，避免圆形单调
                sigma_y = projected_radius / rs.uniform(1.8, 2.4)
                sigma_x = projected_radius / rs.uniform(1.2, 1.8)
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                base_gaussian = np.exp(-(x_rot**2) / (2 * sigma_x**2) - (y_rot**2) / (2 * sigma_y**2))

                # 垂直方向收窄，让顶部更尖
                y_factor = np.linspace(1.1, 0.15, size).reshape(-1, 1)
                tapered = base_gaussian * y_factor

                # 分形/涡流感：多尺度噪声叠加扰动边缘
                noise1 = rs.rand(size, size)
                noise2 = cv2.GaussianBlur(rs.rand(size, size), (9, 9), 0)
                noise3 = cv2.GaussianBlur(rs.rand(size, size), (17, 17), 0)
                turbulence = (noise1 * 0.4 + noise2 * 0.4 + noise3 * 0.2)
                turbulence = 0.65 + 0.45 * (turbulence - turbulence.min()) / (turbulence.ptp() + 1e-6)

                # 中空效果：减去一层更小的高斯，形成枝桠状边缘
                inner = np.exp(-(x_rot**2 + y_rot**2) / (2 * (sigma_x * 0.55)**2)) * 0.25
                gaussian = np.clip((tapered * turbulence) - inner, 0, 1)
                
                # 裁剪到画面内
                y_min = max(0, py - projected_radius)
                y_max = min(H, py + projected_radius + 1)
                x_min = max(0, px - projected_radius)
                x_max = min(W, px + projected_radius + 1)
                
                gy_min = max(0, -(py - projected_radius))
                gy_max = gy_min + (y_max - y_min)
                gx_min = max(0, -(px - projected_radius))
                gx_max = gx_min + (x_max - x_min)
                
                # 火焰强度混合
                fire_mask[y_min:y_max, x_min:x_max] = np.maximum(
                    fire_mask[y_min:y_max, x_min:x_max],
                    gaussian[gy_min:gy_max, gx_min:gx_max] * cluster['intensity']
                )
            
            fire_masks.append(fire_mask)
            fire_colors.append(fire_color_map)
        
        return fire_masks, fire_colors


def simulate_fire_with_blindspots_and_visual(
    observations,
    fire_simulator,
    agent_positions=None,
    camera_matrices=None,
    fire_depth_threshold=4.0,
    severe_intensity_threshold=0.6,
    severe_depth_threshold=2.0,
):
    """
    逼真的火灾仿真：动态蔓延 + 视角相关渲染
    
    Args:
        observations: list of dict，每个包含 'rgb' 和 'depth'
        fire_simulator: GlobalFireSimulator 实例
        agent_positions: list of (x, y, z) 世界坐标
        camera_matrices: dict {'extrinsic': list of 4x4, 'intrinsic': 3x3}
        fire_depth_threshold: 深度阈值
    
    Returns:
        observations: 修改后的观测列表
    """
    if fire_simulator is None:
        return observations
    
    # 更新全局火灾状态
    fire_simulator.update(dt=0.1)
    
    H, W, _ = observations[0]["rgb"].shape
    
    # 获取火焰强度 mask
    use_camera_projection = (agent_positions is not None and camera_matrices is not None)
    
    if use_camera_projection:
        fire_masks_projected, _ = fire_simulator.project_to_camera(
            observations, agent_positions, 
            camera_matrices['extrinsic'], camera_matrices['intrinsic']
        )
    else:
        fire_masks_projected = None
    
    modified_observations = []
    fire_severe_flags = []  # 每个观测对应是否进入严重火灾区域
    
    for obs_idx, obs in enumerate(observations):
        rgb = obs["rgb"].copy().astype(np.float32)
        depth = obs["depth"].copy().squeeze()
        
        # 处理 RGBA
        has_alpha = obs["rgb"].shape[-1] == 4
        if has_alpha:
            alpha_channel = rgb[:, :, 3:4].copy()
            rgb = rgb[:, :, :3]
        
        # 获取本帧的火焰强度图
        if use_camera_projection:
            fire_intensity_map = fire_masks_projected[obs_idx]
        else:
            # 简化模式：基于深度的火焰（地面/家具附近）
            fire_intensity_map = np.zeros((H, W), dtype=np.float32)
            valid_depth = (depth > 0.2) & (depth < fire_depth_threshold)
            
            # 在下半部分检测可能的火源位置
            fire_region = valid_depth.copy()
            fire_region[:int(H*0.25), :] = False  # 上部不生成
            fire_region[int(H*0.85):, :] = False  # 太下方也不生成
            
            if np.any(fire_region):
                # 基于深度分布选择火源点
                fire_y, fire_x = np.where(fire_region)
                
                # 选择 2-3 个点作为火源
                num_sources = np.random.randint(2, 4)
                if len(fire_y) > num_sources:
                    sampled_indices = np.random.choice(len(fire_y), num_sources, replace=False)
                else:
                    sampled_indices = np.arange(len(fire_y))
                
                for idx in sampled_indices:
                    cy, cx = fire_y[idx], fire_x[idx]
                    # 火焰半径更大（50-100 像素）
                    radius = np.random.randint(60, 120)
                    
                    # 创建不规则的火焰形状（顶部尖、底部宽）
                    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                    dist_sq = x**2 + y**2
                    
                    # 垂直方向上火焰向上变尖
                    y_factor = np.exp(-(y.astype(np.float32) / (radius / 2.5)) ** 2)
                    gaussian = np.exp(-dist_sq / (2 * (radius / 2.5) ** 2)) * y_factor * 0.9
                    
                    # 裁剪到画面内
                    y_min = max(0, cy - radius)
                    y_max = min(H, cy + radius + 1)
                    x_min = max(0, cx - radius)
                    x_max = min(W, cx + radius + 1)
                    
                    gy_min = max(0, -(cy - radius))
                    gy_max = gy_min + (y_max - y_min)
                    gx_min = max(0, -(cx - radius))
                    gx_max = gx_min + (x_max - x_min)
                    
                    fire_intensity_map[y_min:y_max, x_min:x_max] = np.maximum(
                        fire_intensity_map[y_min:y_max, x_min:x_max],
                        gaussian[gy_min:gy_max, gx_min:gx_max]
                    )
        
        # ====== 应用火焰视觉效果 ======
        valid_depth = (depth > 0.2) & (depth < fire_depth_threshold)
        fire_visible = fire_intensity_map > 0.08
        fire_visible = fire_visible & valid_depth

        # 严重火情判定：强度高且距离近
        severe_mask = (fire_intensity_map > severe_intensity_threshold) & (depth < severe_depth_threshold)
        fire_severe_flags.append(bool(np.any(severe_mask)))
        
        if np.any(fire_visible):
            fire_pixels = np.where(fire_visible)
            intensities = fire_intensity_map[fire_pixels]
            
            # 基于强度分配颜色和混合方式（RGB 格式）
            for i, intensity in enumerate(intensities):
                py, px = fire_pixels[0][i], fire_pixels[1][i]
                
                # 计算火焰颜色（RGB，不是BGR！）
                if intensity > 0.85:
                    color = np.array([255, 255, 200], dtype=np.float32)  # 白黄
                    alpha = 0.9
                elif intensity > 0.7:
                    color = np.array([255, 255, 80], dtype=np.float32)  # 黄
                    alpha = 0.85
                elif intensity > 0.5:
                    color = np.array([255, 200, 50], dtype=np.float32)  # 黄橙
                    alpha = 0.8
                elif intensity > 0.3:
                    color = np.array([255, 140, 20], dtype=np.float32)  # 橙
                    alpha = 0.75
                elif intensity > 0.15:
                    color = np.array([220, 80, 10], dtype=np.float32)  # 红橙
                    alpha = 0.65
                else:
                    color = np.array([180, 40, 5], dtype=np.float32)  # 深红
                    alpha = 0.55
                
                # 混合：火焰颜色 + 原图
                rgb[py, px] = np.clip(
                    rgb[py, px] * (1 - alpha) + color * alpha, 0, 255
                )
            
            # 火焰光晕（增强层次感）
            glow = cv2.GaussianBlur(fire_intensity_map, (35, 35), 0)
            glow_visible = (glow > 0.05) & (~fire_visible) & valid_depth
            
            if np.any(glow_visible):
                glow_pixels = np.where(glow_visible)
                glow_strength = glow[glow_pixels]
                # 光晕颜色：偏向红橙（RGB）
                glow_color = np.array([220, 100, 50], dtype=np.float32)
                
                rgb[glow_pixels] = np.clip(
                    rgb[glow_pixels] + glow_color * (glow_strength.reshape(-1, 1) * 0.5), 0, 255
                )
        
        # ====== 烟雾效果 ======
        smoke_map = cv2.GaussianBlur(fire_intensity_map, (45, 45), 0)
        smoke_visible = (smoke_map > 0.02) & (~fire_visible) & valid_depth
        
        if np.any(smoke_visible):
            smoke_pixels = np.where(smoke_visible)
            smoke_strength = smoke_map[smoke_pixels]
            # 烟雾颜色：暖灰色（RGB）
            smoke_color = np.array([120, 110, 100], dtype=np.float32)
            
            rgb[smoke_pixels] = np.clip(
                rgb[smoke_pixels] * (1 - smoke_strength.reshape(-1, 1) * 0.4) + 
                smoke_color * (smoke_strength.reshape(-1, 1) * 0.25),
                0, 255
            )
        
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        if has_alpha:
            rgb = np.concatenate([rgb, alpha_channel], axis=-1)
        
        obs_modified = obs.copy()
        obs_modified["rgb"] = rgb
        obs_modified["depth"] = depth.reshape(H, W, 1) if len(obs["depth"].shape) == 3 else depth
        modified_observations.append(obs_modified)
    
    return modified_observations, fire_severe_flags


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
    
    # 初始化全局火灾模拟器
    fire_simulator = GlobalFireSimulator(map_bounds=100.0, num_fire_sources=3)
    # ------------------------------------------------------------------

    count_episodes = 0
    goal_points = []
    log_start = time.time()
    total_usage = []
    
    while count_episodes < num_episodes:
        observations = env.reset()
        actions = []
        map_process.reset()
        
        # 每个 episode 开始时重置火灾状态，基于初始观测智能选择火源位置
        fire_simulator = GlobalFireSimulator(map_bounds=100.0, num_fire_sources=2, observations=observations)
        
        agent_state = env.sim.get_agent_state(0)
        
        # 对初始观测应用火焰效果
        observations, _ = simulate_fire_with_blindspots_and_visual(
            observations, fire_simulator=fire_simulator
        )
        
        for i in range(num_agents):
            agent[i].reset(observations[i], agent_state)
            actions.append(0)
            
            # 初始位置的火灾危害计算
            init_pos = np.array(env.sim.get_agent_state(i).position, dtype=np.float32)
            init_fire_intensity = fire_simulator.get_fire_intensity_at_position(init_pos)
            
            # 初始化危害暴露指标
            agent[i].cumulative_hazard_exposure = init_fire_intensity
            agent[i].step_hazard_exposure = init_fire_intensity
            agent[i].max_hazard_intensity = init_fire_intensity
            agent[i].total_steps = 1
            
            if init_fire_intensity > 0.1:
                agent[i].hazard_contact_steps = 1
            
            # 严格接触判定：初始位置是否进入高强度火焰区域
            if fire_simulator.is_position_in_severe_fire(init_pos, intensity_threshold=0.7):
                agent[i].unsafe_fire_event = True
                logging.info(f"Agent {i} started in severe fire (intensity={init_fire_intensity:.3f})")

            
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
                
                # ===== 火灾危害暴露计算 =====
                curr_pos = np.array(agent_state.position, dtype=np.float32)
                fire_intensity = fire_simulator.get_fire_intensity_at_position(curr_pos)
                
                # 累积危害暴露（火焰强度 * 步数权重）
                agent[i].step_hazard_exposure = fire_intensity
                agent[i].cumulative_hazard_exposure += fire_intensity
                agent[i].max_hazard_intensity = max(agent[i].max_hazard_intensity, fire_intensity)
                
                # 统计与火焰接触的步数（强度 > 0.1 视为接触）
                if fire_intensity > 0.1:
                    agent[i].hazard_contact_steps += 1
                
                agent[i].total_steps += 1
                
                # 严格接触判定：当前位置是否进入高强度火焰区域
                if fire_simulator.is_position_in_severe_fire(curr_pos, intensity_threshold=0.7):
                    agent[i].unsafe_fire_event = True
                    logging.warning(f"Agent {i} entered severe fire region (intensity={fire_intensity:.3f})")
                
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
                    # ===== 合作模式：选择不同的前沿点避免重复 =====
                    if len(target_point_list) > 0:
                        # 为每个 agent 分配不同的前沿点
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
                                # 如果没有未分配的前沿点，使用最近的
                                distances = [np.linalg.norm(np.array(target_point_list[j]) - np.array(pose_pred[i][:2])) for j in range(len(target_point_list))]
                                goal_points.append(target_point_list[np.argmin(distances)])
                    else:
                        for i in range(num_agents):
                            action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
                            goal_points.append([int(action[0]), int(action[1])])
                
                elif args.nav_mode == "fill":
                    # ===== 覆盖模式：优先探索未探索区域（基于前沿得分） =====
                    if len(target_point_list) > 0:
                        for i in range(num_agents):
                            # 选择前沿点中得分最高的（探索价值最大）
                            best_idx = 0
                            best_score = -1
                            for j, frontier in enumerate(target_point_list):
                                if target_score is not None and j < len(target_score):
                                    score = target_score[j]
                                else:
                                    # 如果没有得分，用距离作为替代
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
                    # 默认随机模式
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
            
            # 对新观测应用火焰效果（火焰随时间蔓延）；安全判定使用世界坐标接触
            observations, _ = simulate_fire_with_blindspots_and_visual(
                observations, fire_simulator=fire_simulator
            )
            
            step_end = time.time()
            step_time = step_end - start
            # print('step_time: %.3f秒'%step_time)

       
        count_episodes += 1
        count_step += agent[0].l_step

        # ------------------------------------------------------------------
        ##### 导出本 episode 的点云（先保存各 Agent 局部图，再生成 merged）
        # ------------------------------------------------------------------
        # try:
        #     export_episode_point_cloud(
        #         agent,
        #         count_episodes,
        #         output_dir=os.path.join(args.dump_location, "pointclouds")
        #     )
        # except Exception as e:
        #     logging.error(f"Export point cloud failed: {e}")

        # ------------------------------------------------------------------
        ##### 生成 Obstacle 和 Hazard Map 可视化
        # ------------------------------------------------------------------
        try:
            # 初始化地图可视化器
            map_visualizer = ObstacleHazardMapVisualizer(
                map_size_cm=args.map_size_cm,
                map_resolution=args.map_resolution,
                output_dir=os.path.join(args.dump_location, "maps")
            )
            
            # 设置已提取的地图数据
            map_visualizer.obstacle_map = obstacle_map.astype(np.float32)
            map_visualizer.explored_map = explored_map.astype(np.float32)
            map_visualizer.top_view_rgb = top_view_map.copy()  # 保持原始格式
            
            # 从 top_view_rgb 中检测火焰颜色生成 hazard map
            # 这样 hazard map 与 obstacle map 使用完全相同的坐标系
            hazard_map = map_visualizer.generate_hazard_map_from_topview_rgb(top_view_map)
            
            # 获取 agent 世界坐标位置
            agent_positions = []
            for i in range(num_agents):
                pos = env.sim.get_agent_state(i).position
                agent_positions.append((pos[0], pos[2]))  # XZ 平面
            
            # 生成可视化
            fire_clusters = fire_simulator.fire_clusters if fire_simulator else []
            map_visualizer.visualize_maps(
                agent_positions=agent_positions,
                fire_clusters=fire_clusters,
                filename=f"episode_{count_episodes:04d}_obstacle_hazard"
            )
            
            # 保存单独的地图文件
            map_visualizer.save_individual_maps(f"episode_{count_episodes:04d}")
            
            logging.info(f"[Episode {count_episodes}] Obstacle & Hazard maps saved")
        except Exception as e:
            logging.warning(f"Failed to generate obstacle/hazard maps: {e}")
            import traceback
            traceback.print_exc()

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
        
        # ===== 添加火灾危害暴露指标（Cumulative Hazard Exposure）=====
        # 保持原有 success/spl 不变，基于现有数据添加新的安全指标
        
        # 计算多智能体的平均指标
        if num_agents > 0:
            avg_cumulative_hazard = np.mean([getattr(a, 'cumulative_hazard_exposure', 0.0) for a in agent])
            avg_max_hazard_intensity = np.mean([getattr(a, 'max_hazard_intensity', 0.0) for a in agent])
            avg_hazard_contact_ratio = np.mean([
                getattr(a, 'hazard_contact_steps', 0) / max(getattr(a, 'total_steps', 1), 1) 
                for a in agent
            ])
        else:
            avg_cumulative_hazard = 0.0
            avg_max_hazard_intensity = 0.0
            avg_hazard_contact_ratio = 0.0
        
        # 添加到 metrics（不覆盖原有指标）
        metrics['cumulative_hazard_exposure'] = avg_cumulative_hazard
        metrics['max_hazard_intensity'] = avg_max_hazard_intensity
        metrics['hazard_contact_ratio'] = avg_hazard_contact_ratio
        
        # 记录单个 agent 的详细 hazard exposure
        for i, ag in enumerate(agent):
            metrics[f'agent_{i}/cumulative_hazard'] = getattr(ag, 'cumulative_hazard_exposure', 0.0)
            metrics[f'agent_{i}/max_hazard_intensity'] = getattr(ag, 'max_hazard_intensity', 0.0)
            metrics[f'agent_{i}/hazard_contact_ratio'] = (
                getattr(ag, 'hazard_contact_steps', 0) / max(getattr(ag, 'total_steps', 1), 1)
            ) 
        
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
