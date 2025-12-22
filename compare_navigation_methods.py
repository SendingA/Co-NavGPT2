#!/usr/bin/env python3
"""
导航方法对比脚本

快速对比不同的导航方法性能，无需手动输入多个命令
"""

import subprocess
import argparse
import os
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """运行命令并输出日志"""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 运行: {description}")
    print(f"{'='*70}")
    print(f"命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠️  命令失败 (exit code: {result.returncode})")
    else:
        print(f"✓ 命令成功")
    
    return result.returncode


def compare_methods(methods, num_episodes=2, num_agents=2):
    """
    对比多个导航方法
    
    Args:
        methods: 列表 ['nearest', 'fill', 'gpt']
        num_episodes: 每个方法跑多少个 episode
        num_agents: 机器人数量
    """
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║           Co-NavGPT 导航方法对比脚本                              ║
╚════════════════════════════════════════════════════════════════════╝

配置:
  - 方法: {', '.join(methods)}
  - 每个方法 episodes: {num_episodes}
  - 机器人数: {num_agents}
  
开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
    
    results = {}
    
    for method in methods:
        cmd = f"python main.py --nav_mode {method} --num_episodes {num_episodes} --num_agents {num_agents} -v 0"
        
        # 特殊处理 GPT 方法的参数
        if method == 'gpt':
            cmd += " --gpt_type 2"
        
        success = run_command(cmd, f"{method.upper()} 方法测试")
        results[method] = "✓" if success == 0 else "✗"
    
    print(f"\n{'='*70}")
    print("对比结果总结")
    print(f"{'='*70}")
    print(f"\n运行结果:")
    for method, status in results.items():
        print(f"  {status} {method.upper():15} - 检查 logs/{method}/ 和 dump/{method}/ 查看详细结果")
    
    print(f"\n后续步骤:")
    print(f"  1. 查看日志: ls -la logs/*/")
    print(f"  2. 对比指标: 查看 metrics 或 success_rate")
    print(f"  3. 分析视频: 查看 dump/*/ 中的图像/视频")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def benchmark_speed():
    """性能基准测试 - 对比各方法的速度"""
    import time
    
    methods = ['nearest', 'fill', 'co_ut', 'gpt']
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║           导航方法速度基准测试 (1 episode each)                    ║
╚════════════════════════════════════════════════════════════════════╝
""")
    
    speed_results = {}
    
    for method in methods:
        cmd = f"python main.py --nav_mode {method} --num_episodes 1 --num_agents 2 -v 0"
        if method == 'gpt':
            cmd += " --gpt_type 2"
        
        print(f"\n测试 {method.upper()}...", end=' ', flush=True)
        start = time.time()
        subprocess.run(cmd, shell=True, capture_output=True)
        elapsed = time.time() - start
        speed_results[method] = elapsed
        print(f"✓ {elapsed:.1f}s")
    
    print(f"\n{'='*70}")
    print("速度对比结果 (越低越快)")
    print(f"{'='*70}\n")
    
    # 排序
    sorted_results = sorted(speed_results.items(), key=lambda x: x[1])
    
    max_time = max(speed_results.values())
    for rank, (method, time_taken) in enumerate(sorted_results, 1):
        bar_length = int(20 * time_taken / max_time)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        print(f"  {rank}. {method.upper():10} {bar} {time_taken:6.1f}s")
    
    fastest = sorted_results[0][0]
    print(f"\n🏆 最快: {fastest.upper()}")


def quick_test():
    """快速测试 - 用 nearest 跑 1 episode"""
    print(f"""
╔════════════════════════════════════════════════════════════════════╗
║           快速测试 (Greedy/Nearest 方法)                          ║
╚════════════════════════════════════════════════════════════════════╝

这将使用最快的方法 (nearest) 运行 1 个 episode，
用时通常不超过 2 分钟。

开始测试...
""")
    
    cmd = "python main.py --nav_mode nearest --num_episodes 1 --num_agents 1 -v 0"
    run_command(cmd, "快速测试")
    
    print("""
✓ 快速测试完成!

下一步:
  1. 查看生成的日志: logs/nearest/
  2. 查看结果: dump/nearest/
  3. 尝试其他方法，进行对比
""")


def main():
    parser = argparse.ArgumentParser(
        description="Co-NavGPT 导航方法对比脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 对比所有方法，每个 3 个 episode
  python compare_navigation_methods.py --all --episodes 3
  
  # 仅对比 GPT 和 Greedy
  python compare_navigation_methods.py --methods gpt nearest --episodes 2
  
  # 速度基准测试
  python compare_navigation_methods.py --benchmark
  
  # 快速测试
  python compare_navigation_methods.py --quick
""")
    
    parser.add_argument('--all', action='store_true',
                       help='对比所有方法 (nearest, fill, co_ut, gpt)')
    parser.add_argument('--methods', nargs='+', default=['nearest'],
                       help='指定要对比的方法 (default: nearest)')
    parser.add_argument('--episodes', type=int, default=2,
                       help='每个方法运行的 episode 数 (default: 2)')
    parser.add_argument('--agents', type=int, default=2,
                       help='机器人数量 (default: 2)')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行速度基准测试')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试 (1 episode, nearest 方法)')
    
    args = parser.parse_args()
    
    # 确定要运行的方法
    if args.all:
        methods = ['nearest', 'fill', 'co_ut', 'gpt']
    else:
        methods = args.methods
    
    # 检查是否配置了 OpenAI API（如果要用 GPT）
    if 'gpt' in methods and not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  警告: 使用 GPT 方法但未设置 OPENAI_API_KEY")
        print("   请先运行: export OPENAI_API_KEY='your_key'")
        print("   或使用 --methods 排除 gpt 方法\n")
    
    # 运行选定的模式
    if args.quick:
        quick_test()
    elif args.benchmark:
        benchmark_speed()
    else:
        compare_methods(methods, num_episodes=args.episodes, num_agents=args.agents)


if __name__ == "__main__":
    main()
