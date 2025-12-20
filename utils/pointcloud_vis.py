#!/usr/bin/env python3
"""
点云可视化工具集

提供多种方式将 Agent 观测到的点云可视化出来：
1. 实时 Open3D GUI（需要 --visualize）
2. 导出 PLY 文件（离线查看）
3. 多 Agent 点云对比
4. 距离/置信度热力图着色
"""

import open3d as o3d
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from utils.mmwave_radar_simulator import MMWaveRadarSimulator


class PointCloudVisualizer:
    """点云可视化管理器"""
    
    def __init__(self, output_dir="logs/pointclouds", auto_color_agents=True):
        """
        Args:
            output_dir: PLY 文件保存目录
            auto_color_agents: 是否自动为不同 Agent 着色
        """
        self.output_dir = output_dir
        self.auto_color_agents = auto_color_agents
        self.agent_colors = self._generate_agent_colors(10)
        
        os.makedirs(output_dir, exist_ok=True)
        
    @staticmethod
    def _generate_agent_colors(num_agents):
        """为多个 Agent 生成不同的颜色"""
        colors = []
        for i in range(num_agents):
            # 使用 HSV 色彩空间生成均匀分布的颜色
            h = i / num_agents
            s = 0.8
            v = 0.9
            rgb = plt.cm.hsv(h)[:3]  # (R, G, B) in [0, 1]
            colors.append(rgb)
        return colors
    
    def export_point_cloud(self, points, colors, filename, verbose=True):
        """
        导出点云为 PLY 文件
        
        Args:
            points: (N, 3) float32 数组，单位米
            colors: (N, 3) float32 数组，范围 [0, 1]
            filename: 输出文件名（不需要扩展名）
            verbose: 是否打印信息
        """
        if len(points) == 0:
            if verbose:
                print(f"[Warning] 点云为空，跳过保存: {filename}")
            return None
        
        # 创建 Open3D 点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
        # 确保颜色在 [0, 1] 范围内
        colors_clipped = np.clip(colors, 0, 1).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors_clipped)
        
        # 完整路径
        filepath = os.path.join(self.output_dir, f"{filename}.ply")
        o3d.io.write_point_cloud(filepath, pcd)
        
        if verbose:
            print(f"[OK] 点云已保存: {filepath}")
            print(f"     点数: {len(points)}, 大小: {len(points) * 3 * 4 / 1024 / 1024:.1f} MB")
        
        return filepath
    
    def color_by_distance(self, points, reference_point=None, colormap='viridis'):
        """
        根据到参考点的距离为点云着色（热力图）
        
        Args:
            points: (N, 3) 点坐标
            reference_point: (3,) 参考点，默认为原点
            colormap: matplotlib colormap 名称
        
        Returns:
            colors: (N, 3) RGB 颜色数组 [0, 1]
        """
        if reference_point is None:
            reference_point = np.array([0, 0, 0])
        
        # 计算距离
        distances = np.linalg.norm(points - reference_point, axis=1)
        
        # 归一化到 [0, 1]
        if distances.max() > distances.min():
            norm_dist = (distances - distances.min()) / (distances.max() - distances.min())
        else:
            norm_dist = np.ones_like(distances)
        
        # 映射到颜色
        cmap = cm.get_cmap(colormap)
        colors = cmap(norm_dist)[:, :3]  # 取前 3 个通道（RGB）
        
        return colors
    
    def color_by_height(self, points, colormap='viridis'):
        """
        根据高度（Z 坐标）为点云着色
        
        Args:
            points: (N, 3) 点坐标
            colormap: matplotlib colormap 名称
        
        Returns:
            colors: (N, 3) RGB 颜色数组 [0, 1]
        """
        heights = points[:, 2]  # Z 坐标
        
        # 归一化到 [0, 1]
        if heights.max() > heights.min():
            norm_height = (heights - heights.min()) / (heights.max() - heights.min())
        else:
            norm_height = np.ones_like(heights)
        
        # 映射到颜色
        cmap = cm.get_cmap(colormap)
        colors = cmap(norm_height)[:, :3]
        
        return colors
    
    def filter_points_by_distance(self, points, colors, max_distance=5.0, reference_point=None):
        """
        根据距离过滤点云（去除太远的点）
        
        Args:
            points: (N, 3) 点坐标
            colors: (N, 3) 颜色
            max_distance: 最大距离阈值
            reference_point: 参考点
        
        Returns:
            filtered_points, filtered_colors
        """
        if reference_point is None:
            reference_point = np.array([0, 0, 0])
        
        distances = np.linalg.norm(points - reference_point, axis=1)
        mask = distances <= max_distance
        
        return points[mask], colors[mask]
    
    def merge_agent_point_clouds(self, agent_point_clouds_dict):
        """
        合并多个 Agent 的点云，并自动着色
        
        Args:
            agent_point_clouds_dict: {agent_id: {'points': (N,3), 'colors': (N,3)}}
        
        Returns:
            merged_points, merged_colors
        """
        all_points = []
        all_colors = []
        
        for agent_id, pcd_data in agent_point_clouds_dict.items():
            points = pcd_data['points']
            colors = pcd_data.get('colors', None)
            
            if colors is None:
                # 如果没有颜色，使用 Agent 预设颜色
                colors = np.full_like(points, self.agent_colors[agent_id % len(self.agent_colors)])
            
            all_points.append(points)
            all_colors.append(colors)
        
        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        
        return merged_points, merged_colors
    
    def visualize_with_matplotlib(self, points, colors, title="Point Cloud"):
        """
        用 matplotlib 显示点云
        
        Args:
            points: (N, 3) 点坐标
            colors: (N, 3) RGB 颜色
            title: 图标题
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 剪裁颜色到 [0, 1]
        colors_clipped = np.clip(colors, 0, 1)
        
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors_clipped,
            s=1,
            marker='.'
        )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        
        # 设置坐标轴相等
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
    
    def visualize_with_open3d(self, point_clouds_list, geometry_list=None, window_name="Point Clouds"):
        """
        用 Open3D 显示点云（支持多个）
        
        Args:
            point_clouds_list: [{'points': (N,3), 'colors': (N,3)}, ...]
                               或 [o3d.geometry.PointCloud, ...]
            geometry_list: 其他几何体列表（LineSet, TriangleMesh 等）
            window_name: 窗口名
        """
        geometries = []
        
        # 处理点云列表
        for pcd_data in point_clouds_list:
            if isinstance(pcd_data, o3d.geometry.PointCloud):
                geometries.append(pcd_data)
            elif isinstance(pcd_data, dict):
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_data['points'].astype(np.float64))
                colors = np.clip(pcd_data['colors'], 0, 1).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(pcd)
        
        # 添加其他几何体
        if geometry_list:
            geometries.extend(geometry_list)
        
        # 添加坐标轴
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        
        # 显示
        o3d.visualization.draw_geometries(geometries, window_name=window_name)


def export_episode_point_cloud(agent_list, episode_id, output_dir="logs/pointclouds"):
    """
    便利函数：导出一个 episode 中所有 Agent 的点云
    
    Args:
        agent_list: [VLM_Agent, ...]
        episode_id: episode 编号
        output_dir: 输出目录
    
    使用方法：
        在 main.py 的 episode 循环中调用：
        export_episode_point_cloud(agent, count_episodes)
    """
    vis = PointCloudVisualizer(output_dir)
    
    # 先为每个 Agent 导出局部点云的俯视/侧视图，满足“先保存局部，再生成 merged”
    for ag in agent_list:
        if len(ag.point_sum.points) == 0:
            continue
        pts = np.asarray(ag.point_sum.points)
        cols = np.asarray(ag.point_sum.colors)

        # 生成俯视 + 侧视 快速截图（使用 Agg 后端，无需 GUI）
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

        # 俯视图：XY 平面，颜色用高度或自身颜色均可
        axes[0].scatter(pts[:, 0], pts[:, 1], c=cols, s=0.5)
        axes[0].set_title(f"Agent {ag.agent_id} Top View")
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].axis('equal')
        axes[0].grid(alpha=0.2)

        # 侧视图：XZ 平面，颜色沿用原色
        axes[1].scatter(pts[:, 0], pts[:, 2], c=cols, s=0.5)
        axes[1].set_title(f"Agent {ag.agent_id} Side View")
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Z (m)')
        axes[1].axis('equal')
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, f"episode_{episode_id:04d}_agent_{ag.agent_id}_local.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] 已保存 Agent {ag.agent_id} 的局部点云图像: {img_path}")

    # 合并所有 Agent 的点云
    agent_dict = {}
    for agent in agent_list:
        if len(agent.point_sum.points) > 0:
            agent_dict[agent.agent_id] = {
                'points': np.asarray(agent.point_sum.points),
                'colors': np.asarray(agent.point_sum.colors)
            }
    
    if agent_dict:
        merged_points, merged_colors = vis.merge_agent_point_clouds(agent_dict)
        vis.export_point_cloud(
            merged_points, merged_colors,
            f"episode_{episode_id:04d}_merged"
        )
        
        # 导出每个 Agent 的点云
        for agent_id, pcd_data in agent_dict.items():
            vis.export_point_cloud(
                pcd_data['points'], pcd_data['colors'],
                f"episode_{episode_id:04d}_agent_{agent_id}"
            )


def batch_export_height_colored_pointclouds(input_dir, output_dir):
    """
    批量处理 PLY 文件：按高度重新着色
    
    Usage:
        python -c "from utils.pointcloud_vis import batch_export_height_colored_pointclouds; \
                   batch_export_height_colored_pointclouds('logs/pointclouds', 'logs/pointclouds_colored')"
    """
    vis = PointCloudVisualizer(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.ply'):
            filepath = os.path.join(input_dir, filename)
            pcd = o3d.io.read_point_cloud(filepath)
            
            points = np.asarray(pcd.points)
            colors = vis.color_by_height(points)
            
            output_name = filename.replace('.ply', '_height_colored')
            vis.export_point_cloud(points, colors, output_name)


if __name__ == "__main__":
    # 示例：生成和显示测试点云
    print("Point Cloud Visualization Tools")
    print("=" * 50)
    
    # 生成测试数据
    np.random.seed(42)
    n_points = 10000
    
    # 球体点云
    radius = 1.0
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    points = np.column_stack([x, y, z])
    
    # 初始化可视化器
    vis = PointCloudVisualizer()
    
    # 导出原始点云
    colors_original = np.random.rand(n_points, 3)
    vis.export_point_cloud(points, colors_original, "test_original")
    
    # 按距离着色
    colors_distance = vis.color_by_distance(points)
    vis.export_point_cloud(points, colors_distance, "test_distance_colored")
    
    # 按高度着色
    colors_height = vis.color_by_height(points)
    vis.export_point_cloud(points, colors_height, "test_height_colored")
    
    print("\n✓ 测试点云已生成，可用 CloudCompare/Meshlab 打开查看")
