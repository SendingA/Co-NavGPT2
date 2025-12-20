#!/usr/bin/env python3
"""
将点云可视化集成到 main.py 的示例代码

在 main.py 中添加这些代码，可以实现自动保存和可视化点云
"""

# ============================================================================
# 在 main.py 顶部添加导入
# ============================================================================
# from utils.pointcloud_vis import PointCloudVisualizer, export_episode_point_cloud


# ============================================================================
# 在 main() 函数中，初始化部分添加这些代码
# ============================================================================

def main_with_pointcloud_viz(args, send_queue, receive_queue):
    """
    改进的 main 函数，集成了点云可视化功能
    
    Usage:
        在 main.py 中导入 PointCloudVisualizer，然后：
        1. 初始化可视化器
        2. 在循环中定期导出点云
    """
    
    # ... 现有的初始化代码 ...
    
    from utils.pointcloud_vis import PointCloudVisualizer
    
    # 初始化点云可视化器
    pointcloud_dir = os.path.join(args.dump_location, "pointclouds")
    pcvis = PointCloudVisualizer(output_dir=pointcloud_dir)
    
    # ... 其他初始化代码 ...
    
    count_episodes = 0
    
    while count_episodes < num_episodes:
        observations = env.reset()
        
        # 其他 episode 初始化代码...
        
        count_step = 0
        point_sum = o3d.geometry.PointCloud()
        
        while not env.episode_over:
            # ... 现有的 Agent 映射代码 ...
            
            for i in range(num_agents):
                agent_state = env.sim.get_agent_state(i)
                agent[i].mapping(observations[i], agent_state)
                point_sum += agent[i].point_sum
            
            # ================================================================
            # ★ 定期导出点云 ★
            # ================================================================
            if count_step % args.pointcloud_export_interval == 0:
                if len(point_sum.points) > 0:
                    points = np.asarray(point_sum.points)
                    colors = np.asarray(point_sum.colors)
                    
                    # 方式 1: 导出原始点云
                    pcvis.export_point_cloud(
                        points, colors,
                        f"episode_{count_episodes:04d}_step_{count_step:06d}",
                        verbose=False
                    )
                    
                    # 方式 2: 导出按距离着色的点云
                    if len(agent) > 0:
                        camera_pos = agent[0].camera_position
                        colors_distance = pcvis.color_by_distance(
                            points, reference_point=camera_pos, colormap='hot'
                        )
                        pcvis.export_point_cloud(
                            points, colors_distance,
                            f"episode_{count_episodes:04d}_step_{count_step:06d}_distance",
                            verbose=False
                        )
                    
                    # 方式 3: 导出按高度着色的点云
                    colors_height = pcvis.color_by_height(points, colormap='viridis')
                    pcvis.export_point_cloud(
                        points, colors_height,
                        f"episode_{count_episodes:04d}_step_{count_step:06d}_height",
                        verbose=False
                    )
            
            # ... 其他循环代码 ...
            
            count_step += 1
        
        # ====================================================================
        # ★ Episode 结束时导出完整点云 ★
        # ====================================================================
        logging.info(f"[Episode {count_episodes}] 导出最终点云...")
        
        # 合并所有 Agent 的点云
        agent_dict = {}
        for i, ag in enumerate(agent):
            if len(ag.point_sum.points) > 0:
                agent_dict[i] = {
                    'points': np.asarray(ag.point_sum.points),
                    'colors': np.asarray(ag.point_sum.colors)
                }
        
        if agent_dict:
            # 导出原始颜色
            merged_points, merged_colors = pcvis.merge_agent_point_clouds(agent_dict)
            pcvis.export_point_cloud(
                merged_points, merged_colors,
                f"episode_{count_episodes:04d}_final_merged",
                verbose=True
            )
            
            # 导出按高度着色
            colors_height = pcvis.color_by_height(merged_points)
            pcvis.export_point_cloud(
                merged_points, colors_height,
                f"episode_{count_episodes:04d}_final_merged_height",
                verbose=False
            )
            
            # 也导出每个 Agent 的单独点云
            for agent_id, pcd_data in agent_dict.items():
                pcvis.export_point_cloud(
                    pcd_data['points'], pcd_data['colors'],
                    f"episode_{count_episodes:04d}_agent_{agent_id}_final"
                )
        
        count_episodes += 1


# ============================================================================
# 在 arguments.py 中添加这些参数
# ============================================================================

"""
# 在 get_args() 中添加：

    parser.add_argument('--pointcloud-export-interval', type=int, default=10,
                       help='每 N 步导出一次点云 (default: 10)')
    
    parser.add_argument('--pointcloud-export-format', 
                       choices=['ply', 'pcd', 'xyz'],
                       default='ply',
                       help='点云导出格式 (default: ply)')
    
    parser.add_argument('--pointcloud-height-colored', action='store_true',
                       help='是否导出按高度着色的点云')
    
    parser.add_argument('--pointcloud-distance-colored', action='store_true',
                       help='是否导出按距离着色的点云')
    
    parser.add_argument('--pointcloud-max-distance', type=float, default=10.0,
                       help='点云最大显示距离 (default: 10.0 m)')
"""


# ============================================================================
# 一个独立的脚本示例：在 main.py 运行后可视化保存的点云
# ============================================================================

def visualize_saved_pointclouds(pointcloud_dir, episode_id=None, colormap_mode='original'):
    """
    可视化已保存的点云文件
    
    Usage:
        python -c "from scripts.pointcloud_viewer import visualize_saved_pointclouds; \
                   visualize_saved_pointclouds('logs/gpt/pointclouds', episode_id=0)"
    
    Args:
        pointcloud_dir: 点云目录路径
        episode_id: 要查看的 episode ID（None 则查看所有）
        colormap_mode: 'original' 或 'height' 或 'distance'
    """
    from utils.pointcloud_vis import PointCloudVisualizer
    
    vis = PointCloudVisualizer(output_dir=pointcloud_dir)
    
    pcds_to_load = []
    
    for filename in sorted(os.listdir(pointcloud_dir)):
        if not filename.endswith('.ply'):
            continue
        
        # 过滤 episode
        if episode_id is not None:
            if f"episode_{episode_id:04d}" not in filename:
                continue
        
        # 过滤最终文件
        if 'final_merged' in filename:
            filepath = os.path.join(pointcloud_dir, filename)
            pcd = o3d.io.read_point_cloud(filepath)
            pcds_to_load.append((filename, pcd))
            print(f"[Loaded] {filename} - {len(np.asarray(pcd.points))} points")
    
    if pcds_to_load:
        # 显示所有加载的点云
        print(f"\n共加载 {len(pcds_to_load)} 个点云，按任意键进行下一个...")
        
        for name, pcd in pcds_to_load:
            print(f"\n显示: {name}")
            o3d.visualization.draw_geometries(
                [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)],
                window_name=name
            )
    else:
        print(f"未找到点云文件在 {pointcloud_dir}")


# ============================================================================
# 实用工具：点云统计分析
# ============================================================================

def analyze_pointcloud_statistics(pointcloud_dir, output_csv=None):
    """
    分析保存的点云统计信息
    
    输出：点数、坐标范围、颜色分布等
    """
    import csv
    
    stats_list = []
    
    for filename in sorted(os.listdir(pointcloud_dir)):
        if not filename.endswith('.ply'):
            continue
        
        filepath = os.path.join(pointcloud_dir, filename)
        pcd = o3d.io.read_point_cloud(filepath)
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        stats = {
            'filename': filename,
            'num_points': len(points),
            'x_min': points[:, 0].min(),
            'x_max': points[:, 0].max(),
            'y_min': points[:, 1].min(),
            'y_max': points[:, 1].max(),
            'z_min': points[:, 2].min(),
            'z_max': points[:, 2].max(),
            'mean_r': colors[:, 0].mean(),
            'mean_g': colors[:, 1].mean(),
            'mean_b': colors[:, 2].mean(),
        }
        
        stats_list.append(stats)
        
        print(f"{filename}: {len(points)} points, "
              f"X:[{stats['x_min']:.2f}, {stats['x_max']:.2f}], "
              f"Y:[{stats['y_min']:.2f}, {stats['y_max']:.2f}], "
              f"Z:[{stats['z_min']:.2f}, {stats['z_max']:.2f}]")
    
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats_list[0].keys())
            writer.writeheader()
            writer.writerows(stats_list)
        print(f"\n统计信息已保存至: {output_csv}")


# ============================================================================
# 快速集成模板：复制到 main.py 的合适位置
# ============================================================================

"""
# 在 main() 函数中的导入部分添加：
from utils.pointcloud_vis import PointCloudVisualizer

# 在变量初始化部分添加：
pcvis = PointCloudVisualizer(
    output_dir=os.path.join(args.dump_location, "pointclouds")
)
pointcloud_export_counter = 0

# 在 while not env.episode_over 循环中，observations = env.step(actions) 后添加：
pointcloud_export_counter += 1
if pointcloud_export_counter % args.pointcloud_export_interval == 0:
    points = np.asarray(point_sum.points)
    colors = np.asarray(point_sum.colors)
    if len(points) > 0:
        pcvis.export_point_cloud(
            points, colors,
            f"ep{count_episodes:03d}_step{count_step:06d}"
        )

# 在 episode 结束后（count_episodes += 1 之前）添加：
if args.save_final_pointcloud:
    # 导出最终点云
    agent_dict = {
        i: {'points': np.asarray(ag.point_sum.points), 
            'colors': np.asarray(ag.point_sum.colors)}
        for i, ag in enumerate(agent)
        if len(ag.point_sum.points) > 0
    }
    if agent_dict:
        merged_points, merged_colors = pcvis.merge_agent_point_clouds(agent_dict)
        pcvis.export_point_cloud(
            merged_points, merged_colors,
            f"episode_{count_episodes:04d}_final"
        )
"""
