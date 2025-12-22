#!/usr/bin/env python3
"""
验证脚本：展示新增的 Cumulative Hazard Exposure 指标

用法:
  python verify_hazard_metrics.py
"""

import numpy as np

def demonstrate_hazard_metrics():
    """
    演示和解释新增的火灾危害暴露指标
    """
    print("=" * 80)
    print("Cumulative Hazard Exposure Metrics 演示")
    print("=" * 80)
    
    # 场景 1：安全导航（避开火灾）
    print("\n[场景 1] 安全导航 - VLM 方法")
    print("-" * 80)
    
    vlm_agent = {
        'success': 1.0,
        'spl': 0.95,
        'cumulative_hazard_exposure': 5.2,      # 很低
        'max_hazard_intensity': 0.25,           # 避开高强度区域
        'hazard_contact_ratio': 0.12,           # 很少接触火焰
        'unsafe_fire_event': 0.0                # 未进入严重区域
    }
    
    print(f"原有导航指标:")
    print(f"  success:              {vlm_agent['success']:.3f} ✓ (成功到达目标)")
    print(f"  spl:                  {vlm_agent['spl']:.3f} ✓ (路径效率高)")
    
    print(f"\n新增安全指标:")
    print(f"  cumulative_hazard:    {vlm_agent['cumulative_hazard_exposure']:.3f} ✓ (低暴露)")
    print(f"  max_hazard_intensity: {vlm_agent['max_hazard_intensity']:.3f} ✓ (避开强火)")
    print(f"  hazard_contact_ratio: {vlm_agent['hazard_contact_ratio']:.3f} ✓ (接触少)")
    print(f"  unsafe_fire_event:    {vlm_agent['unsafe_fire_event']:.1f} ✓ (安全)")
    
    print(f"\n[评价] 导航效果好且安全性高 - 理想方案")
    
    # 场景 2：高效导航但穿过火灾
    print("\n" + "=" * 80)
    print("[场景 2] 高效但危险 - Greedy 方法")
    print("-" * 80)
    
    greedy_agent = {
        'success': 1.0,
        'spl': 0.92,
        'cumulative_hazard_exposure': 35.8,     # 很高
        'max_hazard_intensity': 0.78,           # 遭遇强火焰
        'hazard_contact_ratio': 0.65,           # 大部分时间在火焰中
        'unsafe_fire_event': 1.0                # 进入严重区域
    }
    
    print(f"原有导航指标:")
    print(f"  success:              {greedy_agent['success']:.3f} ✓ (成功到达目标)")
    print(f"  spl:                  {greedy_agent['spl']:.3f} ~ (路径可以更优)")
    
    print(f"\n新增安全指标:")
    print(f"  cumulative_hazard:    {greedy_agent['cumulative_hazard_exposure']:.3f} ✗ (高暴露)")
    print(f"  max_hazard_intensity: {greedy_agent['max_hazard_intensity']:.3f} ✗ (强火焰)")
    print(f"  hazard_contact_ratio: {greedy_agent['hazard_contact_ratio']:.3f} ✗ (频繁接触)")
    print(f"  unsafe_fire_event:    {greedy_agent['unsafe_fire_event']:.1f} ✗ (触发安全警告)")
    
    print(f"\n[评价] 虽然完成任务，但安全风险极高 - 不推荐")
    
    # 场景 3：失败导航
    print("\n" + "=" * 80)
    print("[场景 3] 失败导航 - 困困于火灾")
    print("-" * 80)
    
    failed_agent = {
        'success': 0.0,
        'spl': 0.0,
        'cumulative_hazard_exposure': 42.5,     # 极高（因为卡住了）
        'max_hazard_intensity': 0.88,           # 极强火焰
        'hazard_contact_ratio': 0.95,           # 几乎全程在火焰中
        'unsafe_fire_event': 1.0                # 严重安全问题
    }
    
    print(f"原有导航指标:")
    print(f"  success:              {failed_agent['success']:.3f} ✗ (未到达目标)")
    print(f"  spl:                  {failed_agent['spl']:.3f} ✗ (无法计算)")
    
    print(f"\n新增安全指标:")
    print(f"  cumulative_hazard:    {failed_agent['cumulative_hazard_exposure']:.3f} ✗✗ (极高暴露)")
    print(f"  max_hazard_intensity: {failed_agent['max_hazard_intensity']:.3f} ✗✗ (极强火焰)")
    print(f"  hazard_contact_ratio: {failed_agent['hazard_contact_ratio']:.3f} ✗✗ (持续接触)")
    print(f"  unsafe_fire_event:    {failed_agent['unsafe_fire_event']:.1f} ✗✗ (严重警告)")
    
    print(f"\n[评价] 导航失败，且陷入危险环境 - 严重失败")
    
    # 指标对比分析
    print("\n" + "=" * 80)
    print("三种方法的指标对比")
    print("=" * 80)
    
    methods = ['VLM', 'Greedy', 'Failed']
    success = [vlm_agent['success'], greedy_agent['success'], failed_agent['success']]
    spl = [vlm_agent['spl'], greedy_agent['spl'], failed_agent['spl']]
    hazard_exp = [vlm_agent['cumulative_hazard_exposure'], 
                  greedy_agent['cumulative_hazard_exposure'], 
                  failed_agent['cumulative_hazard_exposure']]
    hazard_intensity = [vlm_agent['max_hazard_intensity'], 
                        greedy_agent['max_hazard_intensity'], 
                        failed_agent['max_hazard_intensity']]
    hazard_ratio = [vlm_agent['hazard_contact_ratio'], 
                    greedy_agent['hazard_contact_ratio'], 
                    failed_agent['hazard_contact_ratio']]
    
    print(f"\n{'方法':>10} | {'success':>8} | {'spl':>8} | {'cum_hazard':>10} | {'max_int':>8} | {'h_ratio':>8}")
    print("-" * 75)
    for i, method in enumerate(methods):
        print(f"{method:>10} | {success[i]:>8.3f} | {spl[i]:>8.3f} | {hazard_exp[i]:>10.2f} | {hazard_intensity[i]:>8.3f} | {hazard_ratio[i]:>8.3f}")
    
    # 指标关键范围提示
    print("\n" + "=" * 80)
    print("指标解读指南")
    print("=" * 80)
    
    print("""
【cumulative_hazard_exposure】（累积危害暴露）
  ✓ 优秀: < 10    (很少接触火焰)
  ~ 一般: 10-25   (中等风险)
  ✗ 危险: > 25    (高风险)

【max_hazard_intensity】（最大火焰强度）
  ✓ 优秀: < 0.4   (避开强火焰)
  ~ 一般: 0.4-0.7 (中等强度)
  ✗ 危险: > 0.7   (极端强度，触发 unsafe_fire_event)

【hazard_contact_ratio】（危害接触比）
  ✓ 优秀: < 0.2   (接触少)
  ~ 一般: 0.2-0.5 (适度接触)
  ✗ 危险: > 0.5   (频繁接触)

【unsafe_fire_event】（严重接触标志）
  ✓ 安全: 0       (未进入严重区域)
  ✗ 警告: 1       (接触严重火灾，success/spl 强制为 0)
    """)
    
    # 多智能体示例
    print("=" * 80)
    print("多智能体场景示例")
    print("=" * 80)
    
    agents = [
        {'id': 0, 'cumulative_hazard': 8.5, 'max_intensity': 0.32, 'contact_ratio': 0.10},
        {'id': 1, 'cumulative_hazard': 12.3, 'max_intensity': 0.48, 'contact_ratio': 0.18},
        {'id': 2, 'cumulative_hazard': 6.2, 'max_intensity': 0.28, 'contact_ratio': 0.08},
    ]
    
    print(f"\n{'Agent':>6} | {'cumulative_hazard':>18} | {'max_intensity':>14} | {'contact_ratio':>14}")
    print("-" * 60)
    
    total_hazard = 0.0
    total_intensity = 0.0
    total_ratio = 0.0
    
    for ag in agents:
        print(f"Agent {ag['id']:>0} | {ag['cumulative_hazard']:>18.2f} | {ag['max_intensity']:>14.3f} | {ag['contact_ratio']:>14.3f}")
        total_hazard += ag['cumulative_hazard']
        total_intensity += ag['max_intensity']
        total_ratio += ag['contact_ratio']
    
    print("-" * 60)
    n_agents = len(agents)
    print(f"{'平均':>6} | {total_hazard/n_agents:>18.2f} | {total_intensity/n_agents:>14.3f} | {total_ratio/n_agents:>14.3f}")
    
    print(f"\n[多智能体指标统计]")
    print(f"  所有 agent 的平均累积危害暴露: {total_hazard/n_agents:.2f}")
    print(f"  所有 agent 的平均最大火焰强度: {total_intensity/n_agents:.3f}")
    print(f"  所有 agent 的平均危害接触比: {total_ratio/n_agents:.3f}")
    
    # 使用建议
    print("\n" + "=" * 80)
    print("使用建议")
    print("=" * 80)
    
    print("""
1. 【评估单个方法的安全性】
   比较同一方法在不同环境中的指标变化，了解其火灾敏感性。
   
2. 【对比不同导航方法】
   不仅看 success 和 spl，还要看 hazard 指标，综合评估。
   
3. 【多智能体协作分析】
   观察 agent 之间的 hazard 差异，识别是否有 agent 承担过多风险。
   
4. 【优化导航策略】
   最小化 cumulative_hazard 的同时维持高 success 率 -> 火灾-导航联合优化。
   
5. 【安全约束】
   如果需要严格安全，可以设定 max_hazard_intensity < 0.5 的约束。
    """)
    
    print("\n" + "=" * 80)
    print("✓ 演示完成")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_hazard_metrics()
