#!/usr/bin/env python3
"""
从 merged 3D Point Cloud 提取并可视化 Obstacle Map 和 Hazard (火灾) Map

功能：
1. 从点云数据提取 2D obstacle map (障碍物地图)
2. 基于 GlobalFireSimulator 的火焰簇生成 hazard map (与 obstacle map 对齐)
3. 生成可视化图像（俯视图）

使用方法:
    # 从已有的 PLY 文件加载点云并可视化
    python scripts/visualize_obstacle_hazard_maps.py --ply_file tmp/pointclouds/episode_0001_merged.ply
    
    # 使用模拟数据测试
    python scripts/visualize_obstacle_hazard_maps.py --demo

    # 在导航过程中使用（需要结合 main.py）
    # 见下方 integration 示例
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def world_to_grid(x_world, z_world, map_size, map_resolution):
    """
    将世界坐标 (x, z) 转换为栅格坐标 (i, j)
    与 main.py / explored_map_utils.py 中的转换完全一致
    
    Args:
        x_world: 世界坐标 X（米）
        z_world: 世界坐标 Z（米）
        map_size: 地图像素大小
        map_resolution: 每像素厘米数
    
    Returns:
        grid_i, grid_j: 栅格坐标
    """
    grid_i = int(np.floor(x_world * 100 / map_resolution) + map_size // 2)
    grid_j = int(np.floor(z_world * 100 / map_resolution) + map_size // 2)
    return grid_i, grid_j


def grid_to_world(grid_i, grid_j, map_size, map_resolution):
    """
    将栅格坐标 (i, j) 转换为世界坐标 (x, z)
    
    Args:
        grid_i, grid_j: 栅格坐标
        map_size: 地图像素大小
        map_resolution: 每像素厘米数
    
    Returns:
        x_world, z_world: 世界坐标（米）
    """
    x_world = (grid_i - map_size // 2) * map_resolution / 100.0
    z_world = (grid_j - map_size // 2) * map_resolution / 100.0
    return x_world, z_world


class ObstacleHazardMapVisualizer:
    """
    从 3D 点云生成并可视化 Obstacle Map 和 Hazard Map
    坐标系统与 main.py 中的 Global_Map_Proc 完全对齐
    """
    
    def __init__(self, 
                 map_size_cm=2400,       # 地图大小（厘米）
                 map_resolution=5,        # 每个像素的厘米数
                 map_height_cm=200,       # 障碍物高度范围（厘米）
                 output_dir="tmp/maps"):
        """
        Args:
            map_size_cm: 地图总大小（厘米），默认 24m x 24m
            map_resolution: 分辨率，每像素代表的厘米数
            map_height_cm: 用于障碍物检测的高度范围
            output_dir: 输出目录
        """
        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_height_cm = map_height_cm
        self.output_dir = output_dir
        
        self.map_size = map_size_cm // map_resolution  # 像素数
        self.map_real_halfsize = map_size_cm / 100.0 / 2.0  # 真实坐标半宽（米）
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化地图
        self.reset()
        
    def reset(self):
        """重置所有地图"""
        self.obstacle_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.explored_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.hazard_map = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        self.top_view_rgb = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.z_buffer = np.full((self.map_size, self.map_size), -np.inf)
        
    def extract_obstacle_map(self, point_cloud, camera_height_z=0.0):
        """
        从 3D 点云提取 2D 障碍物地图
        
        Args:
            point_cloud: Open3D PointCloud 或 dict {'points': (N,3), 'colors': (N,3)}
            camera_height_z: 相机高度（米），用于判断障碍物的高度范围
        
        Returns:
            obstacle_map: (H, W) 二值地图，1 表示障碍物
            explored_map: (H, W) 二值地图，1 表示已探索区域
            top_view_rgb: (H, W, 3) 俯视图 RGB
        """
        # 获取点云数据
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
        elif isinstance(point_cloud, dict):
            points = point_cloud['points']
            colors = point_cloud.get('colors', None)
        else:
            raise ValueError("point_cloud 必须是 Open3D PointCloud 或 dict")
        
        if len(points) == 0:
            print("[Warning] 点云为空")
            return self.obstacle_map, self.explored_map, self.top_view_rgb
        
        if colors is None:
            colors = np.ones((len(points), 3)) * 0.5  # 默认灰色
        
        # 高度范围
        z_min = camera_height_z - self.map_height_cm / 100.0 / 2.0
        z_max = camera_height_z + self.map_height_cm / 100.0 / 2.0
        
        # 空间范围过滤
        common_mask = (
            (points[:, 0] >= -self.map_real_halfsize) &
            (points[:, 0] <= self.map_real_halfsize) &
            (points[:, 2] >= -self.map_real_halfsize) &
            (points[:, 2] <= self.map_real_halfsize)
        )
        
        # 障碍物：在高度范围内的点
        mask_obstacle = common_mask & ((points[:, 1] >= z_min) & (points[:, 1] <= z_max))
        # 已探索：高度不超过上限的所有点
        mask_explored = common_mask & (points[:, 1] <= z_max)
        
        points_obstacle = points[mask_obstacle]
        points_explored = points[mask_explored]
        colors_explored = colors[mask_explored]
        
        # 转换为地图坐标（与 explored_map_utils.py 完全一致）
        obs_i = np.floor(points_obstacle[:, 0] * 100 / self.map_resolution).astype(int) + self.map_size // 2
        obs_j = np.floor(points_obstacle[:, 2] * 100 / self.map_resolution).astype(int) + self.map_size // 2
        
        exp_i = np.floor(points_explored[:, 0] * 100 / self.map_resolution).astype(int) + self.map_size // 2
        exp_j = np.floor(points_explored[:, 2] * 100 / self.map_resolution).astype(int) + self.map_size // 2
        
        # 边界检查
        valid_obs = (obs_i >= 0) & (obs_i < self.map_size) & (obs_j >= 0) & (obs_j < self.map_size)
        valid_exp = (exp_i >= 0) & (exp_i < self.map_size) & (exp_j >= 0) & (exp_j < self.map_size)
        
        obs_i, obs_j = obs_i[valid_obs], obs_j[valid_obs]
        exp_i, exp_j = exp_i[valid_exp], exp_j[valid_exp]
        points_explored_valid = points_explored[valid_exp]
        colors_explored_valid = colors_explored[valid_exp]
        
        # 填充障碍物地图
        self.obstacle_map[obs_i, obs_j] = 1.0
        
        # 填充已探索区域
        self.explored_map[exp_i, exp_j] = 1.0
        
        # 生成俯视图 RGB（保留最高点的颜色）
        for i in range(len(points_explored_valid)):
            if points_explored_valid[i, 1] > self.z_buffer[exp_i[i], exp_j[i]]:
                self.z_buffer[exp_i[i], exp_j[i]] = points_explored_valid[i, 1]
                self.top_view_rgb[exp_i[i], exp_j[i]] = (colors_explored_valid[i] * 255).astype(np.uint8)
        
        # 清除已探索但非障碍物的区域
        diff = self.explored_map - self.obstacle_map
        self.obstacle_map[diff == 1] = 0
        
        return self.obstacle_map.copy(), self.explored_map.copy(), self.top_view_rgb.copy()
    
    def generate_hazard_map_from_topview_rgb(self, top_view_rgb=None):
        """
        从 Top View RGB 图像中检测火焰颜色，生成 Hazard Map
        与 obstacle map 使用完全相同的坐标系
        
        火焰颜色特征（在 RGB 空间）：
        - 高强度火焰：白黄色 [255, 255, 200] - [255, 255, 80]
        - 中强度火焰：黄橙色 [255, 200, 50] - [255, 140, 20]  
        - 低强度火焰：红橙色 [220, 80, 10] - [180, 40, 5]
        
        注意：如果 RGB 和 BGR 混淆，火焰可能显示为蓝色/青色
        
        Args:
            top_view_rgb: (H, W, 3) RGB 图像，如果为 None 则使用 self.top_view_rgb
        
        Returns:
            hazard_map: (H, W) 危害强度地图，值域 [0, 1]
        """
        if top_view_rgb is None:
            top_view_rgb = self.top_view_rgb
        
        if top_view_rgb is None or top_view_rgb.size == 0:
            print("[Warning] Top view RGB 为空")
            return self.hazard_map.copy()
        
        self.hazard_map.fill(0)
        
        # 转换为 float 方便计算
        rgb = top_view_rgb.astype(np.float32)
        
        # 提取各通道
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        
        # ===== 检测火焰颜色 =====
        # 方法1: 检测暖色调（红/橙/黄）- 正常 RGB
        # 火焰特征: R > G > B, R 很高
        warm_fire_mask = (
            (r > 150) &  # R 通道较高
            (r > g) &     # R > G
            (r > b + 50) &  # R 明显大于 B
            (g > b)       # G > B (排除纯红)
        )
        
        # 方法2: 检测冷色调（蓝/青）- BGR 混淆情况
        # 如果火焰显示为蓝色，说明 BGR/RGB 反了
        cold_fire_mask = (
            (b > 150) &   # B 通道较高
            (b > r + 50) &  # B 明显大于 R
            ((g > 100) | (b > 200))  # 有一定的 G 或很高的 B
        )
        
        # 合并两种情况
        fire_mask = warm_fire_mask | cold_fire_mask
        
        # 计算火焰强度
        # 对于暖色火焰：基于 R 通道强度
        warm_intensity = np.where(
            warm_fire_mask,
            np.clip((r - 150) / 105.0, 0, 1) * np.clip((r - b) / 200.0, 0.3, 1),
            0
        )
        
        # 对于冷色火焰（BGR混淆）：基于 B 通道强度
        cold_intensity = np.where(
            cold_fire_mask,
            np.clip((b - 150) / 105.0, 0, 1) * np.clip((b - r) / 200.0, 0.3, 1),
            0
        )
        
        # 取最大值
        self.hazard_map = np.maximum(warm_intensity, cold_intensity).astype(np.float32)
        
        # 轻微膨胀以连接相邻的火焰像素
        if np.any(self.hazard_map > 0.1):
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate((self.hazard_map * 255).astype(np.uint8), kernel)
            # 高斯平滑使火焰边缘更自然
            smoothed = cv2.GaussianBlur(dilated.astype(np.float32), (5, 5), 0)
            self.hazard_map = np.maximum(self.hazard_map, smoothed / 255.0)
        
        # 统计
        fire_pixel_count = np.sum(self.hazard_map > 0.1)
        max_intensity = self.hazard_map.max()
        print(f"[Info] 从 Top View RGB 检测火焰:")
        print(f"       暖色火焰像素: {np.sum(warm_fire_mask)}")
        print(f"       冷色火焰像素: {np.sum(cold_fire_mask)}")
        print(f"       总火焰像素 (>0.1): {fire_pixel_count}")
        print(f"       最大强度: {max_intensity:.3f}")
        
        return self.hazard_map.copy()
    
    def generate_hazard_map_from_fire_simulator(self, fire_simulator):
        """
        从 GlobalFireSimulator 实例生成 Hazard Map
        使用与 obstacle map 相同的坐标转换，确保完全对齐
        
        Args:
            fire_simulator: main.py 中的 GlobalFireSimulator 实例
        
        Returns:
            hazard_map: (H, W) 危害强度地图，值域 [0, 1]
        """
        if fire_simulator is None:
            return self.hazard_map.copy()
        
        return self.generate_hazard_map_from_fire_clusters(fire_simulator.fire_clusters)
    
    def generate_hazard_map_from_fire_clusters(self, fire_clusters):
        """
        从火灾簇列表生成 2D Hazard Map
        使用与 obstacle map 相同的坐标转换，确保完全对齐
        
        火灾簇来自 main.py 中的 GlobalFireSimulator.fire_clusters，格式为：
            - 'center': np.array([x, y, z]) 世界坐标（米）
            - 'radius': 火焰半径（米）
            - 'intensity': 火焰强度 (0-1)
        
        Args:
            fire_clusters: list of dict，火灾簇列表
        
        Returns:
            hazard_map: (H, W) 危害强度地图，值域 [0, 1]
        """
        self.hazard_map.fill(0)
        
        if not fire_clusters:
            print("[Warning] 没有火灾簇数据")
            return self.hazard_map.copy()
        
        print(f"[Info] 处理 {len(fire_clusters)} 个火灾簇")
        
        # 生成坐标网格（与 obstacle map 完全一致）
        i_coords = np.arange(self.map_size)
        j_coords = np.arange(self.map_size)
        ii, jj = np.meshgrid(i_coords, j_coords, indexing='ij')
        
        # 转换为世界坐标（与 explored_map_utils.py 的逆操作）
        # grid_i = floor(x * 100 / resolution) + map_size/2
        # => x = (grid_i - map_size/2) * resolution / 100
        x_world = (ii - self.map_size // 2) * self.map_resolution / 100.0
        z_world = (jj - self.map_size // 2) * self.map_resolution / 100.0
        
        for idx, cluster in enumerate(fire_clusters):
            # 获取火灾中心的世界坐标
            # GlobalFireSimulator 使用 [x, y, z]，其中 y 是高度
            center = cluster['center']
            cx = center[0]  # X 坐标
            cz = center[2]  # Z 坐标（注意：不是 Y，Y 是高度）
            radius = cluster['radius']
            intensity = cluster.get('intensity', 1.0)
            
            # 计算火灾中心对应的栅格位置（用于调试）
            center_i, center_j = world_to_grid(cx, cz, self.map_size, self.map_resolution)
            print(f"  Fire {idx}: world=({cx:.2f}, {cz:.2f}), grid=({center_i}, {center_j}), "
                  f"radius={radius:.2f}m, intensity={intensity:.2f}")
            
            # 计算每个栅格到火灾中心的距离（在 XZ 平面上）
            dist = np.sqrt((x_world - cx) ** 2 + (z_world - cz) ** 2)
            
            # 计算火焰强度分布
            local_intensity = np.zeros_like(dist)
            
            # 核心区域（dist <= radius）：高强度，从中心向外略微衰减
            core_mask = dist <= radius
            # 使用与 GlobalFireSimulator.get_fire_intensity_at_position 一致的衰减
            local_intensity[core_mask] = intensity * (1.0 - dist[core_mask] / max(radius, 0.01))
            
            # 外围区域（radius < dist <= radius*2）：渐变衰减
            outer_mask = (dist > radius) & (dist <= radius * 2.0)
            outer_dist_norm = (dist[outer_mask] - radius) / radius
            local_intensity[outer_mask] = intensity * 0.3 * np.exp(-2.0 * outer_dist_norm ** 2)
            
            # 合并（取最大值）
            self.hazard_map = np.maximum(self.hazard_map, local_intensity)
        
        # 统计
        nonzero_count = np.sum(self.hazard_map > 0.01)
        max_val = self.hazard_map.max()
        print(f"[Info] Hazard map: {nonzero_count} 个非零像素, 最大值={max_val:.3f}")
        
        return self.hazard_map.copy()
    
    def generate_hazard_map_fast(self, fire_clusters):
        """
        使用向量化操作快速生成 Hazard Map（别名函数）
        
        Args:
            fire_clusters: 火灾簇列表
        
        Returns:
            hazard_map: (H, W) 危害强度地图
        """
        return self.generate_hazard_map_from_fire_clusters(fire_clusters)
    
    def visualize_maps(self, 
                       obstacle_map=None, 
                       explored_map=None, 
                       hazard_map=None,
                       top_view_rgb=None,
                       agent_positions=None,
                       fire_clusters=None,
                       filename="obstacle_hazard_maps",
                       show=False):
        """
        可视化所有地图
        
        Args:
            obstacle_map: 障碍物地图
            explored_map: 已探索区域
            hazard_map: 危害区域地图
            top_view_rgb: 俯视 RGB 图
            agent_positions: agent 位置列表 [(x, z), ...]
            fire_clusters: 火灾簇列表
            filename: 输出文件名
            show: 是否显示（需要 GUI）
        """
        if obstacle_map is None:
            obstacle_map = self.obstacle_map
        if explored_map is None:
            explored_map = self.explored_map
        if hazard_map is None:
            hazard_map = self.hazard_map
        if top_view_rgb is None:
            top_view_rgb = self.top_view_rgb
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Obstacle Map
        ax = axes[0, 0]
        im = ax.imshow(obstacle_map.T, cmap='binary', origin='lower')
        ax.set_title('Obstacle Map', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        plt.colorbar(im, ax=ax, label='Obstacle (1=blocked)')
        
        # 2. Explored Map
        ax = axes[0, 1]
        im = ax.imshow(explored_map.T, cmap='Blues', origin='lower')
        ax.set_title('Explored Map', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        plt.colorbar(im, ax=ax, label='Explored (1=visited)')
        
        # 3. Hazard Map (火灾)
        ax = axes[0, 2]
        # 使用自定义火焰颜色映射
        fire_cmap = LinearSegmentedColormap.from_list(
            'fire', 
            ['#000000', '#330000', '#660000', '#990000', '#CC3300', '#FF6600', '#FFCC00', '#FFFF66']
        )
        im = ax.imshow(hazard_map.T, cmap=fire_cmap, origin='lower', vmin=0, vmax=1)
        ax.set_title('Hazard Map (Fire)', fontsize=14, fontweight='bold', color='red')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        cbar = plt.colorbar(im, ax=ax, label='Fire Intensity')
        
        # 标记火灾中心
        if fire_clusters:
            for cluster in fire_clusters:
                cx, cy, cz = cluster['center']
                # 世界坐标转地图坐标
                gi = int(cx * 100 / self.map_resolution) + self.map_size // 2
                gj = int(cz * 100 / self.map_resolution) + self.map_size // 2
                if 0 <= gi < self.map_size and 0 <= gj < self.map_size:
                    ax.scatter(gi, gj, c='yellow', s=100, marker='*', edgecolors='red', linewidths=2, zorder=5)
        
        # 4. Top View RGB
        ax = axes[1, 0]
        ax.imshow(np.transpose(top_view_rgb, (1, 0, 2)), origin='lower')
        ax.set_title('Top View RGB', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        
        # 5. Combined Map (Obstacle + Hazard)
        ax = axes[1, 1]
        combined = np.zeros((self.map_size, self.map_size, 3), dtype=np.float32)
        # 障碍物：灰色
        combined[obstacle_map > 0.5] = [0.3, 0.3, 0.3]
        # 已探索但无障碍：浅蓝
        combined[(explored_map > 0.5) & (obstacle_map < 0.5)] = [0.7, 0.85, 1.0]
        # 火灾：红橙色叠加
        fire_overlay = np.zeros_like(combined)
        fire_overlay[:, :, 0] = hazard_map * 1.0  # R
        fire_overlay[:, :, 1] = hazard_map * 0.3  # G
        fire_overlay[:, :, 2] = 0                  # B
        combined = np.clip(combined + fire_overlay * 0.7, 0, 1)
        
        ax.imshow(np.transpose(combined, (1, 0, 2)), origin='lower')
        ax.set_title('Combined (Obstacle + Hazard)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        
        # 标记 agent 位置
        if agent_positions:
            for i, (ax_pos, az_pos) in enumerate(agent_positions):
                gi = int(ax_pos * 100 / self.map_resolution) + self.map_size // 2
                gj = int(az_pos * 100 / self.map_resolution) + self.map_size // 2
                ax.scatter(gi, gj, c='lime', s=150, marker='o', edgecolors='black', linewidths=2, zorder=10)
                ax.annotate(f'A{i}', (gi, gj), fontsize=10, ha='center', va='bottom', color='white', fontweight='bold')
        
        # 6. 图例和说明
        ax = axes[1, 2]
        ax.axis('off')
        legend_text = """
        Map Legend
        ══════════════════════════════
        
        Obstacle Map:
          • White = Free space
          • Black = Obstacle
        
        Explored Map:
          • Blue = Explored area
          • White = Unexplored
        
        Hazard Map (Fire):
          • Black = Safe zone
          • Red/Orange = Fire zone
          • Yellow star = Fire center
        
        Combined Map:
          • Gray = Obstacle
          • Light Blue = Safe explored
          • Red/Orange = Fire hazard
          • Green dot = Agent position
        
        ══════════════════════════════
        Map Size: {0}x{0} pixels
        Resolution: {1} cm/pixel
        Real Size: {2}m x {2}m
        """.format(self.map_size, self.map_resolution, self.map_size_cm / 100)
        
        ax.text(0.1, 0.5, legend_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] 地图已保存: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def save_individual_maps(self, filename_prefix="map"):
        """
        单独保存每个地图为图像文件
        
        Args:
            filename_prefix: 文件名前缀
        """
        # Obstacle Map (灰度) - 单独大图
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.obstacle_map.T, cmap='binary', origin='lower')
        ax.set_title('Obstacle Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        plt.colorbar(im, ax=ax, label='Obstacle', shrink=0.8)
        plt.tight_layout()
        obstacle_path = os.path.join(self.output_dir, f"{filename_prefix}_obstacle.png")
        plt.savefig(obstacle_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Obstacle Map 已保存: {obstacle_path}")
        
        # Explored Map (灰度) - 单独大图
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.explored_map.T, cmap='Blues', origin='lower')
        ax.set_title('Explored Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        plt.colorbar(im, ax=ax, label='Explored', shrink=0.8)
        plt.tight_layout()
        explored_path = os.path.join(self.output_dir, f"{filename_prefix}_explored.png")
        plt.savefig(explored_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Explored Map 已保存: {explored_path}")
        
        # Hazard Map (火灾热力图) - 单独大图，使用自定义火焰色彩
        fig, ax = plt.subplots(figsize=(12, 10))
        fire_cmap = LinearSegmentedColormap.from_list(
            'fire', 
            ['#000000', '#1a0000', '#330000', '#660000', '#990000', 
             '#CC3300', '#FF6600', '#FF9900', '#FFCC00', '#FFFF66', '#FFFFFF']
        )
        im = ax.imshow(self.hazard_map.T, cmap=fire_cmap, origin='lower', vmin=0, vmax=1)
        ax.set_title('Hazard Map (Fire Intensity)', fontsize=16, fontweight='bold', color='darkred')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        cbar = plt.colorbar(im, ax=ax, label='Fire Intensity (0-1)', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)
        
        # 添加等高线增强可读性
        if self.hazard_map.max() > 0.1:
            contour_levels = [0.2, 0.4, 0.6, 0.8]
            ax.contour(self.hazard_map.T, levels=contour_levels, colors='white', 
                      linewidths=0.5, alpha=0.7, origin='lower')
        
        plt.tight_layout()
        hazard_path = os.path.join(self.output_dir, f"{filename_prefix}_hazard.png")
        plt.savefig(hazard_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Hazard Map 已保存: {hazard_path}")
        
        # Top View RGB - 单独大图
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np.transpose(self.top_view_rgb, (1, 0, 2)), origin='lower')
        ax.set_title('Top View RGB', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        plt.tight_layout()
        topview_path = os.path.join(self.output_dir, f"{filename_prefix}_topview.png")
        plt.savefig(topview_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Top View RGB 已保存: {topview_path}")
        
        # Obstacle + Hazard 叠加图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 创建 RGB 叠加图
        combined = np.zeros((self.map_size, self.map_size, 3), dtype=np.float32)
        
        # 背景：已探索区域为浅灰
        combined[self.explored_map > 0.5] = [0.85, 0.85, 0.85]
        
        # 障碍物：深灰色
        combined[self.obstacle_map > 0.5] = [0.3, 0.3, 0.3]
        
        # 火灾危害：红-黄渐变叠加
        for i in range(self.map_size):
            for j in range(self.map_size):
                h = self.hazard_map[i, j]
                if h > 0.01:
                    # 火焰颜色：从暗红到亮黄
                    fire_r = min(1.0, 0.5 + h * 0.5)
                    fire_g = h * 0.6
                    fire_b = 0
                    # Alpha 混合
                    alpha = min(0.9, h + 0.2)
                    combined[i, j, 0] = combined[i, j, 0] * (1 - alpha) + fire_r * alpha
                    combined[i, j, 1] = combined[i, j, 1] * (1 - alpha) + fire_g * alpha
                    combined[i, j, 2] = combined[i, j, 2] * (1 - alpha) + fire_b * alpha
        
        combined = np.clip(combined, 0, 1)
        
        ax.imshow(np.transpose(combined, (1, 0, 2)), origin='lower')
        ax.set_title('Obstacle + Hazard Map Overlay', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Z (grid)')
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#D9D9D9', edgecolor='black', label='Explored'),
            Patch(facecolor='#4D4D4D', edgecolor='black', label='Obstacle'),
            Patch(facecolor='#FF6600', edgecolor='black', label='Fire Hazard'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        overlay_path = os.path.join(self.output_dir, f"{filename_prefix}_obstacle_hazard_overlay.png")
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Obstacle+Hazard Overlay 已保存: {overlay_path}")
        
        print(f"\n[OK] 所有单独地图已保存到: {self.output_dir}")


def demo_visualization():
    """
    演示功能：生成模拟数据并可视化
    """
    print("=" * 60)
    print("Obstacle & Hazard Map Visualization Demo")
    print("=" * 60)
    
    # 初始化可视化器
    visualizer = ObstacleHazardMapVisualizer(
        map_size_cm=1200,  # 12m x 12m
        map_resolution=5,
        output_dir="tmp/maps_demo"
    )
    
    # 生成模拟点云
    np.random.seed(42)
    n_points = 50000
    
    # 地面点（y≈0）
    ground_x = np.random.uniform(-5, 5, n_points // 2)
    ground_y = np.random.uniform(-0.1, 0.1, n_points // 2)
    ground_z = np.random.uniform(-5, 5, n_points // 2)
    ground_colors = np.tile([0.4, 0.35, 0.3], (n_points // 2, 1))  # 棕色地面
    
    # 墙壁点（y=0~2m）
    wall_points = []
    wall_colors = []
    
    # 四面墙
    for wall_x in [-4.5, 4.5]:
        n_wall = 3000
        wall_points.append(np.column_stack([
            np.full(n_wall, wall_x),
            np.random.uniform(0, 2, n_wall),
            np.random.uniform(-4, 4, n_wall)
        ]))
        wall_colors.append(np.tile([0.8, 0.8, 0.75], (n_wall, 1)))
    
    for wall_z in [-4.5, 4.5]:
        n_wall = 3000
        wall_points.append(np.column_stack([
            np.random.uniform(-4, 4, n_wall),
            np.random.uniform(0, 2, n_wall),
            np.full(n_wall, wall_z)
        ]))
        wall_colors.append(np.tile([0.75, 0.75, 0.8], (n_wall, 1)))
    
    # 障碍物（家具等）
    furniture_centers = [(-2, 0.5, 1), (1.5, 0.4, -2), (0, 0.6, 3)]
    for cx, cy, cz in furniture_centers:
        n_furn = 2000
        furn_points = np.random.randn(n_furn, 3) * 0.4 + [cx, cy, cz]
        wall_points.append(furn_points)
        wall_colors.append(np.tile([0.5, 0.3, 0.2], (n_furn, 1)))
    
    # 合并所有点
    all_points = np.vstack([
        np.column_stack([ground_x, ground_y, ground_z]),
        *wall_points
    ])
    all_colors = np.vstack([ground_colors, *wall_colors])
    
    print(f"生成模拟点云: {len(all_points)} 点")
    
    # 提取障碍物地图
    obstacle_map, explored_map, top_view = visualizer.extract_obstacle_map(
        {'points': all_points, 'colors': all_colors},
        camera_height_z=1.0
    )
    
    # 生成火灾簇 - 使用更大的半径以便在地图上更明显
    fire_clusters = [
        {'center': np.array([2.0, 0.3, 1.5]), 'radius': 2.5, 'intensity': 0.95},
        {'center': np.array([-2.0, 0.2, -1.5]), 'radius': 2.0, 'intensity': 0.85},
        {'center': np.array([0.0, 0.4, -3.0]), 'radius': 1.8, 'intensity': 0.75},
        {'center': np.array([-3.0, 0.3, 2.0]), 'radius': 1.5, 'intensity': 0.65},
    ]
    
    # 生成危害地图
    hazard_map = visualizer.generate_hazard_map_fast(fire_clusters)
    
    # Agent 位置
    agent_positions = [(-3.0, -3.0), (3.0, 3.0)]
    
    # 可视化
    visualizer.visualize_maps(
        obstacle_map=obstacle_map,
        explored_map=explored_map,
        hazard_map=hazard_map,
        top_view_rgb=top_view,
        agent_positions=agent_positions,
        fire_clusters=fire_clusters,
        filename="demo_obstacle_hazard",
        show=False
    )
    
    # 保存单独地图
    visualizer.save_individual_maps("demo")
    
    print("\n✓ 演示完成！查看 tmp/maps_demo/ 目录")


def visualize_from_ply(ply_path, fire_clusters=None, output_dir="tmp/maps"):
    """
    从 PLY 文件加载点云并可视化地图
    
    Args:
        ply_path: PLY 文件路径
        fire_clusters: 火灾簇列表（可选）
        output_dir: 输出目录
    """
    print(f"加载点云: {ply_path}")
    
    if not os.path.exists(ply_path):
        print(f"[Error] 文件不存在: {ply_path}")
        return
    
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"点云点数: {len(pcd.points)}")
    
    if len(pcd.points) == 0:
        print("[Error] 点云为空")
        return
    
    # 初始化可视化器
    visualizer = ObstacleHazardMapVisualizer(output_dir=output_dir)
    
    # 提取障碍物地图
    obstacle_map, explored_map, top_view = visualizer.extract_obstacle_map(
        pcd, camera_height_z=1.0
    )
    
    # 如果没有提供火灾簇，自动生成一些演示用的
    if fire_clusters is None:
        points = np.asarray(pcd.points)
        center = points.mean(axis=0)
        fire_clusters = [
            {'center': center + np.array([2, 0, 1]), 'radius': 1.5, 'intensity': 0.8},
            {'center': center + np.array([-1, 0, -2]), 'radius': 1.2, 'intensity': 0.6},
        ]
        print(f"自动生成 {len(fire_clusters)} 个火灾点用于演示")
    
    # 生成危害地图
    hazard_map = visualizer.generate_hazard_map_fast(fire_clusters)
    
    # 提取文件名
    base_name = os.path.splitext(os.path.basename(ply_path))[0]
    
    # 可视化
    visualizer.visualize_maps(
        fire_clusters=fire_clusters,
        filename=f"{base_name}_maps"
    )
    
    visualizer.save_individual_maps(base_name)


def integrate_with_main_loop():
    """
    展示如何在 main.py 的导航循环中集成此可视化工具
    
    （这是一个代码示例，不会实际执行）
    """
    example_code = '''
# ============================================================
# 在 main.py 中集成 Obstacle & Hazard Map 可视化
# ============================================================

from scripts.visualize_obstacle_hazard_maps import ObstacleHazardMapVisualizer

# 初始化（在 episode 开始时）
map_visualizer = ObstacleHazardMapVisualizer(
    map_size_cm=args.map_size_cm,
    map_resolution=args.map_resolution,
    output_dir=os.path.join(args.dump_location, "maps")
)

# 在导航循环中（每隔 N 步或 episode 结束时）
while not env.episode_over:
    # ... 原有的导航逻辑 ...
    
    # 汇总点云
    point_sum = o3d.geometry.PointCloud()
    for i in range(num_agents):
        point_sum += agent[i].point_sum
    
    # 提取地图
    map_visualizer.reset()  # 如果需要累积，可以去掉这行
    obstacle_map, explored_map, top_view = map_visualizer.extract_obstacle_map(
        point_sum, 
        camera_height_z=agent[0].camera_position[1]
    )
    
    # 从 GlobalFireSimulator 获取火灾簇
    fire_clusters = fire_simulator.fire_clusters
    hazard_map = map_visualizer.generate_hazard_map_fast(fire_clusters)
    
    # 获取 agent 位置
    agent_positions = []
    for i in range(num_agents):
        pos = env.sim.get_agent_state(i).position
        agent_positions.append((pos[0], pos[2]))  # XZ 平面
    
    # 每隔 50 步可视化一次
    if count_step % 50 == 0:
        map_visualizer.visualize_maps(
            agent_positions=agent_positions,
            fire_clusters=fire_clusters,
            filename=f"episode_{count_episodes:04d}_step_{count_step:04d}"
        )

# Episode 结束时保存最终地图
map_visualizer.visualize_maps(
    agent_positions=agent_positions,
    fire_clusters=fire_clusters,
    filename=f"episode_{count_episodes:04d}_final"
)
map_visualizer.save_individual_maps(f"episode_{count_episodes:04d}")
'''
    print(example_code)


def main():
    parser = argparse.ArgumentParser(description="Obstacle & Hazard Map Visualization")
    parser.add_argument('--ply_file', type=str, default=None, help="从 PLY 文件加载点云")
    parser.add_argument('--demo', action='store_true', help="运行演示模式")
    parser.add_argument('--show_integration', action='store_true', help="显示集成代码示例")
    parser.add_argument('--output_dir', type=str, default="tmp/maps", help="输出目录")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_visualization()
    elif args.ply_file:
        visualize_from_ply(args.ply_file, output_dir=args.output_dir)
    elif args.show_integration:
        integrate_with_main_loop()
    else:
        print("用法示例:")
        print("  python scripts/visualize_obstacle_hazard_maps.py --demo")
        print("  python scripts/visualize_obstacle_hazard_maps.py --ply_file tmp/pointclouds/episode_0001_merged.ply")
        print("  python scripts/visualize_obstacle_hazard_maps.py --show_integration")


if __name__ == "__main__":
    main()
