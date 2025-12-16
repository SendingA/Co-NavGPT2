#!/usr/bin/env python3
"""
图片转视频脚本

将保存的导航可视化图片转换为视频文件。
可以处理单个 episode 或批量处理所有 episode。

使用方法:
    # 处理所有 episode
    python scripts/images_to_video.py
    
    # 指定参数
    python scripts/images_to_video.py --dump_location ./tmp --nav_mode gpt --fps 15
    
    # 只处理特定 episode
    python scripts/images_to_video.py --episode 0
    
    # 只处理特定 rank
    python scripts/images_to_video.py --rank 0
"""

import argparse
import os
import sys
import glob
import cv2

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def images_to_video(image_dir, output_path, fps=10, pattern="Merged_Vis-*.png"):
    """
    将指定目录下的图片序列合成视频
    
    Args:
        image_dir: 图片所在目录
        output_path: 输出视频路径
        fps: 视频帧率
        pattern: 图片文件名匹配模式
    
    Returns:
        bool: 是否成功生成视频
    """
    # 获取所有图片文件
    image_files = glob.glob(os.path.join(image_dir, pattern))
    
    if len(image_files) == 0:
        return False
    
    # 按数字排序 (提取文件名中的最后一个数字)
    def extract_number(filepath):
        filename = os.path.basename(filepath)
        parts = filename.replace('.png', '').split('-')
        try:
            return int(parts[-1])
        except ValueError:
            return 0
    
    image_files.sort(key=extract_number)
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"  错误: 无法读取图片 {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"  错误: 无法创建视频文件 {output_path}")
        return False
    
    # 写入所有帧
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is not None:
            video_writer.write(frame)
    
    video_writer.release()
    return True


def process_episode(ep_dir, video_dir, episode_n, rank, fps=10, num_agents=2):
    """
    处理单个 episode，生成多个视频
    
    Returns:
        list: 生成的视频路径列表
    """
    video_paths = []
    
    # 1. 为每个 agent 生成视频
    for agent_id in range(num_agents):
        pattern = f"agent-{agent_id}-Vis-*.png"
        video_path = os.path.join(video_dir, f'episode_{episode_n}_rank_{rank}_agent_{agent_id}.mp4')
        if images_to_video(ep_dir, video_path, fps=fps, pattern=pattern):
            video_paths.append(video_path)
            print(f"  ✓ Agent {agent_id} 视频: {os.path.basename(video_path)}")
    
    # 2. 生成 Merged 视频
    pattern = "Merged_Vis-*.png"
    video_path = os.path.join(video_dir, f'episode_{episode_n}_rank_{rank}_merged.mp4')
    if images_to_video(ep_dir, video_path, fps=fps, pattern=pattern):
        video_paths.append(video_path)
        print(f"  ✓ Merged 视频: {os.path.basename(video_path)}")
    
    return video_paths


def find_all_episodes(dump_dir, nav_mode):
    """
    查找所有 episode 目录
    
    Returns:
        list: [(rank, episode_n, ep_dir), ...]
    """
    episodes = []
    base_dir = os.path.join(dump_dir, "dump", nav_mode, "episodes_multi")
    
    if not os.path.exists(base_dir):
        return episodes
    
    # 遍历 rank 目录
    for rank_dir in os.listdir(base_dir):
        rank_path = os.path.join(base_dir, rank_dir)
        if not os.path.isdir(rank_path):
            continue
        
        try:
            rank = int(rank_dir)
        except ValueError:
            continue
        
        # 遍历 episode 目录
        for ep_dir_name in os.listdir(rank_path):
            if not ep_dir_name.startswith("eps_"):
                continue
            
            ep_path = os.path.join(rank_path, ep_dir_name)
            if not os.path.isdir(ep_path):
                continue
            
            try:
                episode_n = int(ep_dir_name.replace("eps_", ""))
                episodes.append((rank, episode_n, ep_path))
            except ValueError:
                continue
    
    # 按 rank 和 episode 排序
    episodes.sort(key=lambda x: (x[0], x[1]))
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description='将导航可视化图片转换为视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python scripts/images_to_video.py
    python scripts/images_to_video.py --fps 15
    python scripts/images_to_video.py --episode 0 --rank 0
    python scripts/images_to_video.py --dump_location ./tmp --nav_mode gpt
        """
    )
    
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp",
                        help='图片保存的根目录 (default: ./tmp)')
    parser.add_argument('--nav_mode', type=str, default="gpt",
                        choices=['nearest', 'co_ut', 'fill', 'gpt'],
                        help='导航模式 (default: gpt)')
    parser.add_argument('--fps', type=int, default=10,
                        help='视频帧率 (default: 10)')
    parser.add_argument('--num_agents', type=int, default=2,
                        help='agent 数量 (default: 2)')
    parser.add_argument('--episode', type=int, default=None,
                        help='只处理指定的 episode (default: 处理所有)')
    parser.add_argument('--rank', type=int, default=None,
                        help='只处理指定的 rank (default: 处理所有)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("图片转视频工具")
    print("="*60)
    print(f"图片目录: {args.dump_location}/dump/{args.nav_mode}/episodes_multi/")
    print(f"帧率: {args.fps} FPS")
    print(f"Agent 数量: {args.num_agents}")
    print("="*60)
    
    # 查找所有 episode
    episodes = find_all_episodes(args.dump_location, args.nav_mode)
    
    if not episodes:
        print(f"\n未找到任何 episode 图片目录!")
        print(f"请检查路径: {args.dump_location}/dump/{args.nav_mode}/episodes_multi/")
        return
    
    # 过滤指定的 episode 和 rank
    if args.episode is not None:
        episodes = [(r, e, p) for r, e, p in episodes if e == args.episode]
    if args.rank is not None:
        episodes = [(r, e, p) for r, e, p in episodes if r == args.rank]
    
    if not episodes:
        print(f"\n未找到匹配的 episode!")
        return
    
    print(f"\n找到 {len(episodes)} 个 episode 待处理\n")
    
    # 创建视频输出目录
    video_dir = os.path.join(args.dump_location, "dump", args.nav_mode, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    # 处理每个 episode
    total_videos = 0
    for rank, episode_n, ep_dir in episodes:
        print(f"处理 Episode {episode_n} (Rank {rank})...")
        
        videos = process_episode(
            ep_dir, video_dir, episode_n, rank,
            fps=args.fps, num_agents=args.num_agents
        )
        
        if not videos:
            print(f"  ⚠ 未生成任何视频 (可能没有找到图片)")
        
        total_videos += len(videos)
    
    print("\n" + "="*60)
    print(f"完成! 共生成 {total_videos} 个视频")
    print(f"视频保存在: {video_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
