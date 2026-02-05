#!/usr/bin/env python3
"""
毫米波雷达仿真器 (mmWave Radar Simulator)

模拟现实的毫米波雷达特性，重点在于提取点云的边缘/轮廓：
1. 边缘检测：提取物体边界点
2. 深度不连续检测：识别深度跳变位置
3. 稀疏轮廓：只保留场景轮廓点
4. 距离测量噪声：高斯噪声
"""

import numpy as np
import open3d as o3d
from typing import Tuple, List, Dict
import logging
from scipy.spatial import KDTree
from scipy import ndimage


class MMWaveRadarSimulator:
    """
    毫米波雷达仿真器 - 边缘/轮廓提取模式
    
    主要参数（可配置）：
    - edge_threshold: 边缘检测阈值（深度差异）
    - neighbor_radius: 邻域搜索半径
    - downsample_voxel: 体素下采样大小
    - boundary_ratio: 边界点采样比例
    """
    
    def __init__(
        self,
        n_beams_h: int = 64,           # 保留用于兼容性
        n_beams_v: int = 8,            # 保留用于兼容性
        range_min: float = 0.1,        # 最小测距 (m)
        range_max: float = 50.0,       # 最大测距 (m)
        range_std: float = 0.08,       # 距离噪声标准差 (m)（增大噪声）
        fov_h: float = 360.0,          # 水平视场 (°)
        fov_v: float = 60.0,           # 垂直视场 (°)
        reflection_threshold: float = 0.1,
        multipath_ratio: float = 0.02,
        # 边缘检测专用参数 - 调整为更稀疏
        edge_threshold: float = 0.5,   # 深度不连续阈值 (m)（增大，减少检测）
        neighbor_radius: float = 0.25, # 邻域搜索半径 (m)（增大，减少边界点）
        boundary_ratio: float = 0.03,  # 边界点采样比例（减小）
        downsample_voxel: float = 0.12, # 体素下采样大小 (m)（增大，更稀疏）
        curvature_threshold: float = 0.05,  # 曲率阈值（增大，减少检测）
        random_dropout: float = 0.5,   # 随机丢弃比例（新增）
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
        
        # 边缘检测参数
        self.edge_threshold = edge_threshold
        self.neighbor_radius = neighbor_radius
        self.boundary_ratio = boundary_ratio
        self.downsample_voxel = downsample_voxel
        self.curvature_threshold = curvature_threshold
        self.random_dropout = random_dropout
        
        logging.info(f"[mmWave] Initialized radar simulator (edge/contour mode):")
        logging.info(f"  Edge threshold: {edge_threshold}m, Neighbor radius: {neighbor_radius}m")
        logging.info(f"  Boundary ratio: {boundary_ratio}, Voxel size: {downsample_voxel}m, Dropout: {random_dropout}")
    
    def _extract_boundary_points(self, points: np.ndarray, colors: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取点云的边界/轮廓点
        
        使用多种方法检测边界：
        1. 深度不连续检测
        2. 法向量变化检测
        3. 点密度变化检测
        
        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 点云颜色（可选）
        
        Returns:
            boundary_points: 边界点坐标
            boundary_colors: 边界点颜色
        """
        if len(points) < 10:
            return points, colors if colors is not None else np.ones((len(points), 3)) * 0.5
        
        # 创建点云对象用于处理
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 1. 先进行体素下采样
        if self.downsample_voxel > 0:
            pcd_down = pcd.voxel_down_sample(voxel_size=self.downsample_voxel)
        else:
            pcd_down = pcd
        
        points_down = np.asarray(pcd_down.points)
        
        if len(points_down) < 10:
            return points_down, np.ones((len(points_down), 3)) * 0.5
        
        # 2. 计算法向量
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.neighbor_radius * 2, max_nn=30
            )
        )
        
        # 3. 构建 KDTree 用于邻域搜索
        kdtree = KDTree(points_down)
        
        boundary_mask = np.zeros(len(points_down), dtype=bool)
        
        # 4. 多种边界检测方法
        for i, point in enumerate(points_down):
            # 查找邻域点
            neighbors_idx = kdtree.query_ball_point(point, self.neighbor_radius)
            
            if len(neighbors_idx) < 3:
                # 孤立点 -> 边界
                boundary_mask[i] = True
                continue
            
            neighbor_points = points_down[neighbors_idx]
            
            # 方法1: 深度不连续检测
            # 计算到原点的距离（模拟雷达视角）
            dist_center = np.linalg.norm(point)
            dist_neighbors = np.linalg.norm(neighbor_points, axis=1)
            depth_diff = np.max(np.abs(dist_neighbors - dist_center))
            
            if depth_diff > self.edge_threshold:
                boundary_mask[i] = True
                continue
            
            # 方法2: 点密度变化检测
            # 边界处通常只有半边有点
            centroid = np.mean(neighbor_points, axis=0)
            dist_to_centroid = np.linalg.norm(point - centroid)
            avg_neighbor_dist = np.mean(np.linalg.norm(neighbor_points - point, axis=1))
            
            # 如果点偏离邻域质心，说明可能是边界
            if dist_to_centroid > avg_neighbor_dist * 0.5:
                boundary_mask[i] = True
                continue
            
            # 方法3: 法向量散度检测
            if pcd_down.has_normals():
                normals = np.asarray(pcd_down.normals)
                center_normal = normals[i]
                neighbor_normals = normals[neighbors_idx]
                
                # 计算法向量一致性
                normal_dots = np.abs(np.dot(neighbor_normals, center_normal))
                normal_variance = 1 - np.mean(normal_dots)
                
                if normal_variance > 0.3:  # 法向量变化大 -> 边界
                    boundary_mask[i] = True
                    continue
            
            # 方法4: 局部平面拟合检测
            if len(neighbors_idx) >= 4:
                # PCA 分析
                centered = neighbor_points - centroid
                cov = np.cov(centered.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                
                # 计算局部曲率（最小特征值 / 特征值之和）
                curvature = eigenvalues[2] / (np.sum(eigenvalues) + 1e-8)
                
                if curvature > self.curvature_threshold:
                    boundary_mask[i] = True
        
        # 5. 提取边界点
        boundary_points = points_down[boundary_mask]
        
        # 6. 如果边界点太少，随机采样一些补充
        min_points = int(len(points_down) * self.boundary_ratio)
        if len(boundary_points) < min_points:
            # 随机采样补充
            non_boundary_idx = np.where(~boundary_mask)[0]
            n_extra = min(min_points - len(boundary_points), len(non_boundary_idx))
            if n_extra > 0:
                extra_idx = np.random.choice(non_boundary_idx, n_extra, replace=False)
                extra_points = points_down[extra_idx]
                boundary_points = np.vstack([boundary_points, extra_points])
        
        # 7. 如果边界点太多，下采样（更激进的限制）
        max_points = int(len(points_down) * 0.10)  # 最多保留10%（从25%降低）
        if len(boundary_points) > max_points:
            idx = np.random.choice(len(boundary_points), max_points, replace=False)
            boundary_points = boundary_points[idx]
        
        # 7.5 额外随机丢弃
        if hasattr(self, 'random_dropout') and self.random_dropout > 0 and len(boundary_points) > 0:
            keep_mask = np.random.rand(len(boundary_points)) > self.random_dropout
            if np.any(keep_mask):
                boundary_points = boundary_points[keep_mask]
        
        # 8. 为边界点生成颜色
        if colors is not None and len(colors) > 0:
            # 为每个边界点找到最近的原始点，获取颜色
            original_kdtree = KDTree(points)
            _, nearest_idx = original_kdtree.query(boundary_points)
            boundary_colors = colors[nearest_idx]
        else:
            # 使用橙红色标识雷达点
            boundary_colors = np.tile([1.0, 0.4, 0.1], (len(boundary_points), 1))
        
        return boundary_points, boundary_colors
    
    def _extract_silhouette_points(self, points: np.ndarray, colors: np.ndarray = None, 
                                   sensor_origin: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从传感器视角提取轮廓点（silhouette edges）
        
        基于深度不连续性检测，类似真实雷达的边缘检测
        
        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 点云颜色
            sensor_origin: (3,) 传感器位置
        
        Returns:
            silhouette_points: 轮廓点坐标
            silhouette_colors: 轮廓点颜色
        """
        if sensor_origin is None:
            sensor_origin = np.array([0, 0, 0])
        
        if len(points) < 10:
            return points, colors if colors is not None else np.ones((len(points), 3)) * 0.5
        
        # 计算每个点相对于传感器的方向和距离
        rel_points = points - sensor_origin
        distances = np.linalg.norm(rel_points, axis=1)
        
        # 转换为球坐标
        azimuth = np.arctan2(rel_points[:, 1], rel_points[:, 0])
        elevation = np.arcsin(np.clip(rel_points[:, 2] / (distances + 1e-8), -1, 1))
        
        # 将球坐标量化到网格
        az_bins = 180  # 水平分辨率
        el_bins = 60   # 垂直分辨率
        
        az_idx = ((azimuth + np.pi) / (2 * np.pi) * az_bins).astype(int) % az_bins
        el_idx = ((elevation + np.pi/2) / np.pi * el_bins).astype(int) % el_bins
        
        # 创建深度图
        depth_image = np.full((el_bins, az_bins), np.inf)
        point_idx_map = np.full((el_bins, az_bins), -1, dtype=int)
        
        for i, (ai, ei, d) in enumerate(zip(az_idx, el_idx, distances)):
            if d < depth_image[ei, ai]:
                depth_image[ei, ai] = d
                point_idx_map[ei, ai] = i
        
        # 在深度图上检测边缘（深度不连续）
        # 使用Sobel算子或简单差分
        valid_mask = depth_image < np.inf
        depth_image_filled = np.where(valid_mask, depth_image, 0)
        
        # 计算深度梯度
        grad_x = np.abs(np.diff(depth_image_filled, axis=1, prepend=0))
        grad_y = np.abs(np.diff(depth_image_filled, axis=0, prepend=0))
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 边缘检测：梯度大于阈值
        edge_mask = (gradient_mag > self.edge_threshold) & valid_mask
        
        # 提取边缘点的索引
        edge_point_indices = point_idx_map[edge_mask]
        edge_point_indices = edge_point_indices[edge_point_indices >= 0]
        edge_point_indices = np.unique(edge_point_indices)
        
        if len(edge_point_indices) == 0:
            # 回退：随机采样
            n_sample = max(int(len(points) * self.boundary_ratio), 100)
            edge_point_indices = np.random.choice(len(points), min(n_sample, len(points)), replace=False)
        
        silhouette_points = points[edge_point_indices]
        
        if colors is not None and len(colors) > 0:
            silhouette_colors = colors[edge_point_indices]
        else:
            silhouette_colors = np.tile([1.0, 0.4, 0.1], (len(silhouette_points), 1))
        
        return silhouette_points, silhouette_colors
    
    def simulate_radar_pointcloud(
        self,
        rgbd_pointcloud: o3d.geometry.PointCloud,
        rgbd_colors: np.ndarray = None,
        agent_pose: np.ndarray = None,
        add_multipath: bool = False,
        add_noise: bool = True,
        method: str = "combined",  # "boundary", "silhouette", "combined"
    ) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """
        根据 RGBD 点云仿真毫米波雷达点云（边缘/轮廓模式）
        
        Args:
            rgbd_pointcloud: Open3D 点云对象（RGBD 原始数据）
            rgbd_colors: (N, 3) RGBD 原色数据
            agent_pose: (6,) agent 位姿 [x, y, z, roll, pitch, yaw]
            add_multipath: 是否添加多径效应
            add_noise: 是否添加测量噪声
            method: 边缘检测方法 - "boundary"（几何边界）, "silhouette"（视角轮廓）, "combined"（两者结合）
        
        Returns:
            radar_pcd: 仿真的雷达点云 (Open3D PointCloud)
            radar_info: 字典，包含统计信息
        """
        if len(rgbd_pointcloud.points) == 0:
            logging.warning("[mmWave] Empty input point cloud")
            return o3d.geometry.PointCloud(), {'n_points': 0}
        
        points = np.asarray(rgbd_pointcloud.points)
        
        if rgbd_colors is None and rgbd_pointcloud.has_colors():
            colors = np.asarray(rgbd_pointcloud.colors)
        elif rgbd_colors is not None:
            colors = rgbd_colors
        else:
            colors = np.ones((len(points), 3)) * 0.5
        
        # 确保颜色数组与点数匹配
        if len(colors) != len(points):
            colors = np.ones((len(points), 3)) * 0.5
        
        # 传感器位置
        sensor_origin = agent_pose[:3] if agent_pose is not None else np.array([0, 0, 0])
        
        # 根据方法提取边缘点
        if method == "boundary":
            radar_points, radar_colors = self._extract_boundary_points(points, colors)
        elif method == "silhouette":
            radar_points, radar_colors = self._extract_silhouette_points(points, colors, sensor_origin)
        else:  # combined
            # 结合两种方法
            boundary_pts, boundary_cols = self._extract_boundary_points(points, colors)
            silhouette_pts, silhouette_cols = self._extract_silhouette_points(points, colors, sensor_origin)
            
            if len(boundary_pts) > 0 and len(silhouette_pts) > 0:
                radar_points = np.vstack([boundary_pts, silhouette_pts])
                radar_colors = np.vstack([boundary_cols, silhouette_cols])
                
                # 去重（使用体素网格）
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(radar_points)
                temp_pcd.colors = o3d.utility.Vector3dVector(radar_colors)
                temp_pcd = temp_pcd.voxel_down_sample(voxel_size=self.downsample_voxel)
                
                radar_points = np.asarray(temp_pcd.points)
                radar_colors = np.asarray(temp_pcd.colors)
            elif len(boundary_pts) > 0:
                radar_points, radar_colors = boundary_pts, boundary_cols
            else:
                radar_points, radar_colors = silhouette_pts, silhouette_cols
        
        # 添加测量噪声
        if add_noise and len(radar_points) > 0:
            noise = np.random.normal(0, self.range_std, radar_points.shape)
            radar_points = radar_points + noise
        
        # 距离过滤
        if len(radar_points) > 0:
            distances = np.linalg.norm(radar_points - sensor_origin, axis=1)
            valid_mask = (distances >= self.range_min) & (distances <= self.range_max)
            radar_points = radar_points[valid_mask]
            radar_colors = radar_colors[valid_mask]
        
        # 创建输出点云
        radar_pcd = o3d.geometry.PointCloud()
        if len(radar_points) > 0:
            radar_pcd.points = o3d.utility.Vector3dVector(radar_points)
            radar_pcd.colors = o3d.utility.Vector3dVector(np.clip(radar_colors, 0, 1))
        
        # 统计信息
        radar_info = {
            'n_points': len(radar_points),
            'rgbd_points': len(points),
            'sparsity_ratio': len(radar_points) / max(len(points), 1),
            'method': method,
        }
        
        logging.info(f"[mmWave] Generated {len(radar_points)} edge/contour points from {len(points)} RGBD points")
        
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
        """
        fused_pcd = o3d.geometry.PointCloud()
        
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
    """
    import matplotlib.pyplot as plt
    
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
                   c=cols_radar, s=2, alpha=0.8)
        ax2.set_title(f'mmWave Radar Point Cloud\n({n_radar} points - edge/contour)')
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
    print("mmWave Radar Simulator - Edge/Contour Mode")
    print("=" * 50)
    
    # 测试：生成一个简单的房间点云
    np.random.seed(42)
    
    # 创建一个简单的房间（墙壁 + 地板）
    points_list = []
    
    # 地板
    floor_x = np.random.uniform(-5, 5, 2000)
    floor_y = np.random.uniform(-5, 5, 2000)
    floor_z = np.zeros(2000)
    points_list.append(np.stack([floor_x, floor_y, floor_z], axis=1))
    
    # 墙壁
    for wall_x in [-5, 5]:
        wall_y = np.random.uniform(-5, 5, 500)
        wall_z = np.random.uniform(0, 3, 500)
        points_list.append(np.stack([np.full(500, wall_x), wall_y, wall_z], axis=1))
    
    for wall_y in [-5, 5]:
        wall_x = np.random.uniform(-5, 5, 500)
        wall_z = np.random.uniform(0, 3, 500)
        points_list.append(np.stack([wall_x, np.full(500, wall_y), wall_z], axis=1))
    
    points = np.vstack(points_list)
    colors = np.random.rand(len(points), 3) * 0.3 + 0.5  # 灰白色
    
    rgbd_pcd = o3d.geometry.PointCloud()
    rgbd_pcd.points = o3d.utility.Vector3dVector(points)
    rgbd_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 初始化雷达仿真器
    radar_sim = MMWaveRadarSimulator(
        edge_threshold=0.3,
        neighbor_radius=0.2,
        boundary_ratio=0.1,
        downsample_voxel=0.05,
    )
    
    # 仿真雷达点云
    radar_pcd, radar_info = radar_sim.simulate_radar_pointcloud(rgbd_pcd, method="combined")
    
    print(f"Radar simulation result:")
    for k, v in radar_info.items():
        print(f"  {k}: {v}")
    
    # 融合点云
    fused = radar_sim.fuse_pointclouds(rgbd_pcd, radar_pcd)
    
    # 可视化
    fig = visualize_radar_comparison(rgbd_pcd, radar_pcd, fused)
    fig.savefig('/tmp/radar_edge_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to /tmp/radar_edge_comparison.png")
