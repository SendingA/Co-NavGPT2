# 火灾危害暴露指标（Cumulative Hazard Exposure Metrics）

## 概述

本文档说明了如何在保持原有多智能体导航的 **success rate** 和 **SPL** 不变的基础上，添加一套新的火灾安全指标来衡量环境中的火灾对 agent 的影响。

## 核心概念

### 原始指标（保持不变）
- **success**: 0/1，表示是否成功到达目标
- **spl**: 最短路径长度指标（Success weighted by Path Length），衡量路径效率
- **unsafe_fire_event**: 0/1，表示是否进入过严重火灾区域（强度 ≥ 0.7）

### 新增指标

#### 1. **cumulative_hazard_exposure** （累积危害暴露）
- **定义**: 每一步的火焰强度累加和
- **公式**: 
  ```
  cumulative_hazard_exposure = Σ(fire_intensity_t) for all steps t
  ```
- **单位**: 无量纲（0-N，取决于步数和火焰强度）
- **含义**: 
  - 衡量 agent 在整个导航过程中积累的火灾危害
  - 值越高表示暴露于火灾风险中越久，或环境火焰越强
  - 结合总步数可评估 agent 的轨迹安全性

#### 2. **max_hazard_intensity** （最大火焰强度）
- **定义**: 导航过程中遭遇的最高火焰强度
- **公式**:
  ```
  max_hazard_intensity = max(fire_intensity_t) for all steps t
  ```
- **范围**: [0, 1]
- **含义**:
  - 反映 agent 在最危险时刻的暴露程度
  - 值接近 1 表示曾进入极度危险区域
  - 值接近 0 表示环境火灾风险较低

#### 3. **hazard_contact_ratio** （危害接触比）
- **定义**: 与火焰接触（火焰强度 > 0.1）的步数占总步数的比例
- **公式**:
  ```
  hazard_contact_ratio = hazard_contact_steps / total_steps
  ```
- **范围**: [0, 1]
- **含义**:
  - 0 表示全程未接触火焰
  - 1 表示全程都在火焰中
  - 0.5 表示一半的步数在火焰中
  - 可用于评估导航路径的火灾避免效果

## 多智能体场景

在多智能体场景中，系统会计算并记录：

1. **全局指标**（所有 agent 的平均）
   - `cumulative_hazard_exposure`: 平均累积危害暴露
   - `max_hazard_intensity`: 平均最大火焰强度
   - `hazard_contact_ratio`: 平均危害接触比

2. **单个 agent 指标**
   - `agent_i/cumulative_hazard`: Agent i 的累积危害暴露
   - `agent_i/max_hazard_intensity`: Agent i 的最大火焰强度
   - `agent_i/hazard_contact_ratio`: Agent i 的危害接触比

## 代码实现

### VLM_Agent 初始化

```python
# 在 init_map_and_navigation_param() 中添加
self.cumulative_hazard_exposure = 0.0     # 累积危害暴露
self.step_hazard_exposure = 0.0            # 当前步危害暴露
self.max_hazard_intensity = 0.0            # 最大火焰强度
self.hazard_contact_steps = 0              # 与火焰接触的步数
self.total_steps = 0                       # 总步数
```

### main.py 中的每步计算

```python
# 在主循环中，对每个 agent：
curr_pos = np.array(agent_state.position, dtype=np.float32)
fire_intensity = fire_simulator.get_fire_intensity_at_position(curr_pos)

# 累积危害暴露
agent[i].step_hazard_exposure = fire_intensity
agent[i].cumulative_hazard_exposure += fire_intensity
agent[i].max_hazard_intensity = max(agent[i].max_hazard_intensity, fire_intensity)

# 统计接触步数
if fire_intensity > 0.1:
    agent[i].hazard_contact_steps += 1

agent[i].total_steps += 1
```

### main.py 中的 Metric 计算

```python
# 计算多智能体平均指标
if num_agents > 0:
    avg_cumulative_hazard = np.mean([a.cumulative_hazard_exposure for a in agent])
    avg_max_hazard_intensity = np.mean([a.max_hazard_intensity for a in agent])
    avg_hazard_contact_ratio = np.mean([
        a.hazard_contact_steps / max(a.total_steps, 1) for a in agent
    ])

# 添加到 metrics
metrics['cumulative_hazard_exposure'] = avg_cumulative_hazard
metrics['max_hazard_intensity'] = avg_max_hazard_intensity
metrics['hazard_contact_ratio'] = avg_hazard_contact_ratio

# 单个 agent 的详细指标
for i, ag in enumerate(agent):
    metrics[f'agent_{i}/cumulative_hazard'] = ag.cumulative_hazard_exposure
    metrics[f'agent_{i}/max_hazard_intensity'] = ag.max_hazard_intensity
    metrics[f'agent_{i}/hazard_contact_ratio'] = ag.hazard_contact_steps / max(ag.total_steps, 1)
```

## 使用示例

### 场景 1：比较两种导航方法的安全性

```python
# 假设运行了 VLM 方法和 Greedy 方法

# VLM 结果
vlm_metrics = {
    'success': 0.85,
    'spl': 0.92,
    'cumulative_hazard_exposure': 15.3,
    'max_hazard_intensity': 0.65,
    'hazard_contact_ratio': 0.35
}

# Greedy 结果
greedy_metrics = {
    'success': 0.80,
    'spl': 0.85,
    'cumulative_hazard_exposure': 28.7,  # 危害暴露更多
    'max_hazard_intensity': 0.78,        # 遭遇更强烈的火焰
    'hazard_contact_ratio': 0.62         # 更多时间在火焰中
}

# 结论：VLM 方法虽然成功率略高，但安全性指标均优于 Greedy
```

### 场景 2：评估火灾强度的影响

```python
# 低强度火灾环境
low_fire_metrics = {
    'cumulative_hazard_exposure': 8.5,
    'max_hazard_intensity': 0.25,
    'hazard_contact_ratio': 0.15
}

# 高强度火灾环境
high_fire_metrics = {
    'cumulative_hazard_exposure': 45.2,
    'max_hazard_intensity': 0.92,
    'hazard_contact_ratio': 0.68
}

# 结论：环境火灾强度对导航安全性有显著影响
```

## 指标解读指南

| 指标 | 含义 | 良好范围 | 警告范围 |
|------|------|---------|---------|
| cumulative_hazard_exposure | 累积危害 | < 15 | > 30 |
| max_hazard_intensity | 最大强度 | < 0.5 | > 0.8 |
| hazard_contact_ratio | 接触比例 | < 0.3 | > 0.6 |
| unsafe_fire_event | 严重接触 | 0 (未接触) | 1 (接触) |

## 与原有指标的关系

```
原有指标（导航效果）        新增指标（安全性）
    success                cumulative_hazard_exposure
       ↓                           ↓
    spl                 max_hazard_intensity
       ↓                           ↓
  unsafe_fire_event         hazard_contact_ratio
```

**关键点**：
- 原有指标决定了导航的**有效性**（能否到达目标、路径效率）
- 新增指标衡量了导航的**安全性**（火灾影响程度）
- 两套指标相互独立，共同评估多智能体导航的综合性能

## 日志输出示例

```
Episode 1 metrics:
  success: 0.900, spl: 0.950
  cumulative_hazard_exposure: 12.3, max_hazard_intensity: 0.55, hazard_contact_ratio: 0.28
  agent_0/cumulative_hazard: 12.3, agent_0/max_hazard_intensity: 0.55, agent_0/hazard_contact_ratio: 0.28
  unsafe_fire_event: 0.000 (no severe contact)

Episode 2 metrics:
  success: 0.800, spl: 0.920
  cumulative_hazard_exposure: 23.5, max_hazard_intensity: 0.72, hazard_contact_ratio: 0.45
  agent_0/cumulative_hazard: 23.5, agent_0/max_hazard_intensity: 0.72, agent_0/hazard_contact_ratio: 0.45
  unsafe_fire_event: 1.000 (severe contact detected)
```

## 扩展建议

### 可选增强：
1. **加权危害暴露**：考虑步长（距离）而非步数
   ```python
   weighted_hazard = Σ(fire_intensity_t × distance_t)
   ```

2. **危害衰减函数**：模拟火焰从高强度衰退
   ```python
   effective_hazard = fire_intensity_t * decay_factor(distance_to_source)
   ```

3. **多目标优化**：在最大化 success 的同时最小化 hazard
   ```python
   combined_score = w1 * success - w2 * cumulative_hazard_exposure
   ```

## 总结

新的 **Cumulative Hazard Exposure** 指标集合提供了一套全面的火灾安全评估框架，使得：

1. ✅ 保持原有导航效果指标（success, spl）不变
2. ✅ 新增安全性指标（hazard exposure），衡量火灾影响
3. ✅ 支持多智能体场景的细粒度分析
4. ✅ 便于不同导航方法的安全性对比
5. ✅ 可用于火灾-导航联合优化研究
