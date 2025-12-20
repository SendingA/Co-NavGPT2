#!/usr/bin/env python3
"""
集成脚本：为每个 episode 导出 RGBD + mmWave 雷达融合数据
"""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

# 添加项目路径
sys.path.insert(0, '/home/liushe10/Co-NavGPT2')

from utils.pointcloud_vis import PointCloudVisualizer
from utils.mmwave_radar_simulator import MMWaveRadarSimulator


def export_episode_rgbd_and_radar(agent_list, episode_id, output_dir="logs/pointclouds"):
    """
    为每个 episode 导出 Agent 的 RGBD 点云和仿真的毫米波雷达点云
    """
    vis = PointCloudVisualizer(output_dir)
    
    # 初始化毫米波雷达仿真器
    radar_sim = MMWaveRadarSimulator(
        n_beams_h=128,
        n_beams_v=8,
        range_max=50.0,
        range_std=0.03
    )
    
    print(f"\n{'='*60}")
    print(f"Episode {episode_id}: Processing RGBD + Radar data")
    print(f"{'='*60}")
    
    for ag in agent_list:
        if len(ag.point_sum.points) == 0:
            print(f"Agent {ag.agent_id}: Empty point cloud, skipped")
            continue
        
        pts = np.asarray(ag.point_sum.points)
        cols = np.asarray(ag.point_sum.colors)
        
        print(f"\nAgent {ag.agent_id}:")
        print(f"  RGBD points: {len(pts)}")
        
        # 仿真毫米波雷达点云
        radar_pcd, radar_info = radar_sim.simulate_radar_pointcloud(ag.point_sum)
        print(f"  Radar points: {radar_info['n_points']}")
        print(f"  Radar density: {radar_info['point_density']:.1%}")
        print(f"  Avg reflectance: {radar_info['avg_reflectance']:.3f}")
        
        # 融合点云
        fused_pcd = radar_sim.fuse_pointclouds(
            ag.point_sum, radar_pcd,
            rgbd_color=(0.2, 0.8, 1.0),   # 青色
            radar_color=(1.0, 0.2, 0.0),  # 红色
        )
        print(f"  Fused points: {len(fused_pcd.points)}")
        
        # 生成可视化图
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
        fig.suptitle(f'Episode {episode_id} - Agent {ag.agent_id}', fontsize=16)
        
        # RGBD 俯视图
        axes[0, 0].scatter(pts[:, 0], pts[:, 1], c=cols, s=0.5, alpha=0.7)
        axes[0, 0].set_title(f'RGBD Top View ({len(pts)} points)', fontsize=12)
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].axis('equal')
        axes[0, 0].grid(alpha=0.2)
        
        # RGBD 侧视图
        axes[0, 1].scatter(pts[:, 0], pts[:, 2], c=cols, s=0.5, alpha=0.7)
        axes[0, 1].set_title(f'RGBD Side View', fontsize=12)
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Z (m)')
        axes[0, 1].axis('equal')
        axes[0, 1].grid(alpha=0.2)
        
        # 雷达俯视图
        if len(radar_pcd.points) > 0:
            radar_pts = np.asarray(radar_pcd.points)
            radar_cols = np.asarray(radar_pcd.colors)
            axes[1, 0].scatter(radar_pts[:, 0], radar_pts[:, 1], c=radar_cols, s=3, alpha=0.8)
            axes[1, 0].set_title(f'mmWave Radar Top View ({len(radar_pts)} points)', fontsize=12)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Radar Points', ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('mmWave Radar Top View (0 points)', fontsize=12)
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        axes[1, 0].axis('equal')
        axes[1, 0].grid(alpha=0.2)
        
        # 融合俯视图（RGBD 青色 + 雷达 红色）
        if len(fused_pcd.points) > 0:
            fused_pts = np.asarray(fused_pcd.points)
            fused_cols = np.asarray(fused_pcd.colors)
            axes[1, 1].scatter(fused_pts[:, 0], fused_pts[:, 1], c=fused_cols, s=1.5, alpha=0.8)
            axes[1, 1].set_title(f'Fused (RGBD+Radar) Top View ({len(fused_pts)} points)', fontsize=12)
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        axes[1, 1].axis('equal')
        axes[1, 1].grid(alpha=0.2)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0.2, 0.8, 1.0), label='RGBD'),
            Patch(facecolor=(1.0, 0.2, 0.0), label='Radar')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.02))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # 保存图像
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, f"ep{episode_id:04d}_ag{ag.agent_id}_comparison.png")
        plt.savefig(img_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved comparison image: {img_path}")
        
        # 导出 PLY 文件
        try:
            vis.export_point_cloud(pts, cols, f"ep{episode_id:04d}_ag{ag.agent_id}_rgbd")
            o3d.io.write_point_cloud(
                os.path.join(output_dir, f"ep{episode_id:04d}_ag{ag.agent_id}_radar.ply"),
                radar_pcd
            )
            o3d.io.write_point_cloud(
                os.path.join(output_dir, f"ep{episode_id:04d}_ag{ag.agent_id}_fused.ply"),
                fused_pcd
            )
            print(f"  ✓ Exported PLY files (RGBD, Radar, Fused)")
        except Exception as e:
            print(f"  ✗ PLY export error: {e}")


if __name__ == "__main__":
    print("Test mmWave Radar + RGBD Integration")
