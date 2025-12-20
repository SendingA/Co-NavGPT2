#!/usr/bin/env python3
"""
毫米波雷达仿真器 (mmWave Radar Simulator)

模拟现实的毫米波雷达特性：
1. 稀疏采样：仅选择部分点（模拟有限的射线数）
2. 距离测量噪声：高斯噪声
3. 角度分辨率限制：量化角度
4. 运动补偿：考虑agent运动
5. 多径效应：部分点的多次反射
6. 材料相关反射率：不同材料的反射强度不同
"""

import numpy as np
import open3d as o3d
from typing import Tuple, List, Dict
import logging


class MMWaveRadarSimulator:
    """
    毫米波雷达仿真器
    
    主要参数（可配置）：
    - n_beams_h: 水平射线数（通常 64-256）
    - n_beams_v: 垂直射线数（通常 8-16）
    - range_min/max: 测距范围（通常 0.1-200m）
    - range_std: 距离测量标准差（通常 1-5cm）
    - azimuth_resolution: 方位角分辨率（°）
    - elevation_resolution: 仰角分辨率（°）
    """
    
    def __init__(
        self,
        n_beams_h: int = 128,          # 水平射线数
        n_beams_v: int = 8,            # 垂直射线数
        range_min: float = 0.1,        # 最小测距 (m)
        range_max: float = 100.0,      # 最大测距 (m)
        range_std: float = 0.05,       # 距离噪声标准差 (m)
        fov_h: float = 360.0,          # 水平视场 (°)
        fov_v: float = 30.0,           # 垂直视场 (°)
        reflection_threshold: float = 0.1,  # 最小反射强度阈值
        multipath_ratio: float = 0.1,  # 多径效应比例
    ):
        self.n_beams_h = n_beams_h
        self.n_beams_v = n_beams_v
        self.range_min = range_min
        self.range_max = range_max
        self.range_std = range_std
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.reflection_threshold = reflection_threshold
        self.multipath_ratio = multipath_ratio
        
        # 生成射线方向（极坐标）
        self.azimuth_angles = np.linspace(-fov_h / 2, fov_h / 2, n_beams_h)  # 度
        self.elevation_angles = np.linspace(-fov_v / 2, fov_v / 2, n_beams_v)  # 度
        
        # 材料反射率字典（RGB 色值映射到反射率）
        self.material_reflectance = {
            'metal': 0.9,
            'concrete': 0.6,
            'brick': 0.5,
            'wood': 0.4,
            'fabric': 0.2,
            'glass': 0.3,
            'default': 0.5
        }
        
        logging.info(f"[mmWave] Initialized radar simulator:")
        logging.info(f"  Beams: {n_beams_h}×{n_beams_v} (horizontal × vertical)")
        logging.info(f"  Range: {range_min}m ~ {range_max}m")
        logging.info(f"  FOV: {fov_h}° × {fov_v}°")
    
    def _rgb_to_material(self, rgb_color: np.ndarray) -> str:
        """
        根据RGB颜色估计材料类型（用于反射率估计）
        
        Args:
            rgb_color: (3,) RGB 值 [0-255]
        
        Returns:
            material: 材料名称
        """
        r, g, b = rgb_color[:3]
        
        # 简单的色彩启发式分类
        gray = (r + g + b) / 3
        
        if gray > 200:  # 浅色 → 可能是混凝土/墙面
            return 'concrete'
        elif gray > 150:  # 中灰 → 砖或木材
            if abs(r - b) > 30:  # 偏黄/棕 → 木材
                return 'wood'
            else:
                return 'brick'
        elif gray > 100:  # 深灰 → 织物/地毯
            return 'fabric'
        else:  # 很深 → 金属或特殊材料
            return 'metal'
    
    def _estimate_reflectance(self, rgb_color: np.ndarray) -> float:
        """根据RGB颜色估计反射率"""
        material = self._rgb_to_material(rgb_color)
        return self.material_reflectance.get(material, self.material_reflectance['default'])
    
    def _point_in_view(self, point_3d: np.ndarray) -> Tuple[bool, float, float]:
        """
        判断一个3D点是否在雷达视场内，返回是否可见、方位角、仰角
        
        Args:
            point_3d: (3,) 3D 点坐标 (x, y, z)
        
        Returns:
            in_view: 是否在视场内
            azimuth: 方位角 (°)
            elevation: 仰角 (°)
        """
        x, y, z = point_3d
        
        # 计算距离
        dist = np.sqrt(x**2 + y**2 + z**2)
        if dist < 1e-6 or dist < self.range_min or dist > self.range_max:
            return False, None, None
        
        # 计算方位角（相对于 x 轴）
        azimuth = np.degrees(np.arctan2(y, x))
        
        # 计算仰角
        elevation = np.degrees(np.arcsin(np.clip(z / dist, -1, 1)))
        
        # 检查是否在视场内
        if abs(azimuth) > self.fov_h / 2 + 1:  # 加 1° 容差
            return False, None, None
        if abs(elevation) > self.fov_v / 2 + 1:
            return False, None, None
        
        return True, azimuth, elevation
    
    def simulate_radar_pointcloud(
        self,
        rgbd_pointcloud: o3d.geometry.PointCloud,
        rgbd_colors: np.ndarray = None,
        agent_pose: np.ndarray = None,
        add_multipath: bool = True,
        add_noise: bool = True,
    ) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """
        根据 RGBD 点云仿真毫米波雷达点云
        
        Args:
            rgbd_pointcloud: Open3D 点云对象（RGBD 原始数据）
            rgbd_colors: (N, 3) RGBD 原色数据，用于材料反射率估计
            agent_pose: (6,) agent 位姿 [x, y, z, roll, pitch, yaw]
            add_multipath: 是否添加多径效应
            add_noise: 是否添加测量噪声
        
        Returns:
            radar_pcd: 仿真的雷达点云 (Open3D PointCloud)
            radar_info: 字典，包含统计信息
        """
        if len(rgbd_pointcloud.points) == 0:
            logging.warning("[mmWave] Empty input point cloud")
            return o3d.geometry.PointCloud(), {'n_points': 0}
        
        points = np.asarray(rgbd_pointcloud.points)
        colors = np.asarray(rgbd_pointcloud.colors) if rgbd_colors is None else rgbd_colors
        
        if len(colors) != len(points):
            colors = np.ones((len(points), 3)) * 0.5
        
        # 模拟射线追踪：为每条射线找到最近的有效点
        radar_points = []
        radar_colors = []
        reflectances = []
        
        for az in self.azimuth_angles:
            for el in self.elevation_angles:
                # 射线方向（笛卡尔坐标）
                az_rad = np.radians(az)
                el_rad = np.radians(el)
                ray_dir = np.array([
                    np.cos(el_rad) * np.cos(az_rad),
                    np.cos(el_rad) * np.sin(az_rad),
                    np.sin(el_rad)
                ])
                
                # 在该射线方向上找最近的点（射线追踪）
                distances = np.sum(points * ray_dir, axis=1)  # 投影距离
                
                # 过滤有效距离范围
                valid_mask = (distances >= self.range_min) & (distances <= self.range_max)
                
                if np.any(valid_mask):
                    valid_dist = distances[valid_mask]
                    valid_idx = np.where(valid_mask)[0]
                    
                    # 取最近的点
                    closest_idx = valid_idx[np.argmin(valid_dist)]
                    closest_point = points[closest_idx]
                    closest_dist = np.linalg.norm(closest_point)
                    closest_color = colors[closest_idx]
                    
                    # 检查反射强度（根据颜色和角度估计）
                    reflectance = self._estimate_reflectance(closest_color * 255)
                    
                    # 角度相关的反射衰减（掠射角效应）
                    point_to_ray = closest_point / (closest_dist + 1e-6)
                    cosine_angle = np.abs(np.dot(point_to_ray, ray_dir))
                    reflectance *= (cosine_angle ** 0.5)  # 掠射角衰减
                    
                    # 反射强度阈值
                    if reflectance >= self.reflection_threshold:
                        # 添加测量噪声
                        if add_noise:
                            noise = np.random.normal(0, self.range_std)
                            noisy_dist = closest_dist + noise
                        else:
                            noisy_dist = closest_dist
                        
                        # 转换回笛卡尔坐标
                        radar_point = ray_dir * noisy_dist
                        radar_points.append(radar_point)
                        radar_colors.append(closest_color)
                        reflectances.append(reflectance)
                        
                        # 多径效应：以较小概率添加虚假回波
                        if add_multipath and np.random.rand() < self.multipath_ratio:
                            # 模拟二次反射
                            multipath_dist = noisy_dist * (0.9 + 0.2 * np.random.rand())
                            multipath_point = ray_dir * multipath_dist
                            radar_points.append(multipath_point)
                            radar_colors.append(closest_color * 0.7)  # 二次反射更暗
                            reflectances.append(reflectance * 0.5)
        
        # 创建输出点云
        if len(radar_points) == 0:
            logging.warning("[mmWave] No valid radar points generated")
            radar_pcd = o3d.geometry.PointCloud()
        else:
            radar_points = np.array(radar_points)
            radar_colors = np.clip(np.array(radar_colors), 0, 1)
            
            radar_pcd = o3d.geometry.PointCloud()
            radar_pcd.points = o3d.utility.Vector3dVector(radar_points)
            radar_pcd.colors = o3d.utility.Vector3dVector(radar_colors)
        
        # 统计信息
        radar_info = {
            'n_points': len(radar_points),
            'n_rays': len(self.azimuth_angles) * len(self.elevation_angles),
            'point_density': len(radar_points) / (len(self.azimuth_angles) * len(self.elevation_angles)),
            'avg_reflectance': np.mean(reflectances) if reflectances else 0.0,
            'rgbd_points': len(points),
        }
        
        return radar_pcd, radar_info
    
    def fuse_pointclouds(
        self,
        rgbd_pcd: o3d.geometry.PointCloud,
        radar_pcd: o3d.geometry.PointCloud,
        rgbd_color: Tuple[float, float, float] = (0.5, 0.5, 1.0),  # 蓝色
        radar_color: Tuple[float, float, float] = (1.0, 0.5, 0.0),  # 橙色
    ) -> o3d.geometry.PointCloud:
        """
        融合 RGBD 和雷达点云，用不同颜色区分来源
        
        Args:
            rgbd_pcd: RGBD 点云
            radar_pcd: 雷达点云
            rgbd_color: RGBD 点的颜色
            radar_color: 雷达点的颜色
        
        Returns:
            fused_pcd: 融合点云
        """
        fused_pcd = o3d.geometry.PointCloud()
        
        # 合并点坐标
        points_list = []
        colors_list = []
        
        if len(rgbd_pcd.points) > 0:
            points_list.append(np.asarray(rgbd_pcd.points))
            n_rgbd = len(rgbd_pcd.points)
            colors_list.append(np.tile(rgbd_color, (n_rgbd, 1)))
        
        if len(radar_pcd.points) > 0:
            points_list.append(np.asarray(radar_pcd.points))
            n_radar = len(radar_pcd.points)
            colors_list.append(np.tile(radar_color, (n_radar, 1)))
        
        if points_list:
            all_points = np.vstack(points_list)
            all_colors = np.vstack(colors_list)
            
            fused_pcd.points = o3d.utility.Vector3dVector(all_points)
            fused_pcd.colors = o3d.utility.Vector3dVector(
                np.clip(all_colors, 0, 1).astype(np.float64)
            )
        
        return fused_pcd


def visualize_radar_comparison(rgbd_pcd, radar_pcd, fused_pcd=None):
    """
    并排可视化 RGBD、雷达和融合点云
    
    Args:
        rgbd_pcd: RGBD 点云
        radar_pcd: 雷达点云
        fused_pcd: 融合点云（可选）
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # RGBD 和雷达统计
    n_rgbd = len(rgbd_pcd.points)
    n_radar = len(radar_pcd.points)
    
    fig = plt.figure(figsize=(15, 5))
    
    # RGBD 点云
    if n_rgbd > 0:
        ax1 = fig.add_subplot(131, projection='3d')
        pts_rgbd = np.asarray(rgbd_pcd.points)
        cols_rgbd = np.asarray(rgbd_pcd.colors)
        ax1.scatter(pts_rgbd[:, 0], pts_rgbd[:, 1], pts_rgbd[:, 2],
                   c=cols_rgbd, s=1, alpha=0.6)
        ax1.set_title(f'RGBD Point Cloud\n({n_rgbd} points)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
    
    # 雷达点云
    if n_radar > 0:
        ax2 = fig.add_subplot(132, projection='3d')
        pts_radar = np.asarray(radar_pcd.points)
        cols_radar = np.asarray(radar_pcd.colors)
        ax2.scatter(pts_radar[:, 0], pts_radar[:, 1], pts_radar[:, 2],
                   c=cols_radar, s=1, alpha=0.6)
        ax2.set_title(f'mmWave Radar Point Cloud\n({n_radar} points)')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
    
    # 融合点云
    if fused_pcd is not None and len(fused_pcd.points) > 0:
        ax3 = fig.add_subplot(133, projection='3d')
        pts_fused = np.asarray(fused_pcd.points)
        cols_fused = np.asarray(fused_pcd.colors)
        ax3.scatter(pts_fused[:, 0], pts_fused[:, 1], pts_fused[:, 2],
                   c=cols_fused, s=1, alpha=0.6)
        ax3.set_title(f'Fused Point Cloud\n({len(fused_pcd.points)} points)')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Z (m)')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("mmWave Radar Simulator")
    print("=" * 50)
    
    # 测试：生成随机 RGBD 点云
    np.random.seed(42)
    n_points = 5000
    
    # 生成随机点云（盒子内）
    rgbd_pts = np.random.uniform(-5, 5, (n_points, 3))
    rgbd_pts[:, 2] = np.abs(rgbd_pts[:, 2])  # 仅上半空间
    rgbd_colors = np.random.rand(n_points, 3)
    
    rgbd_pcd = o3d.geometry.PointCloud()
    rgbd_pcd.points = o3d.utility.Vector3dVector(rgbd_pts)
    rgbd_pcd.colors = o3d.utility.Vector3dVector(rgbd_colors)
    
    # 初始化雷达仿真器
    radar_sim = MMWaveRadarSimulator(
        n_beams_h=128,
        n_beams_v=8,
        range_max=50.0
    )
    
    # 仿真雷达点云
    radar_pcd, radar_info = radar_sim.simulate_radar_pointcloud(rgbd_pcd)
    
    print(f"Radar simulation result:")
    for k, v in radar_info.items():
        print(f"  {k}: {v}")
    
    # 融合点云
    fused = radar_sim.fuse_pointclouds(rgbd_pcd, radar_pcd)
    
    # 可视化
    fig = visualize_radar_comparison(rgbd_pcd, radar_pcd, fused)
    fig.savefig('/tmp/radar_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to /tmp/radar_comparison.png")
