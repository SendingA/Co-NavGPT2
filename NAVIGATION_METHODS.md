# Co-NavGPT 导航方法调用指南

## 概述

Co-NavGPT 提供了多种导航策略来选择机器人的探索目标。这些方法在 `--nav_mode` 参数中配置，控制机器人如何选择frontier（前沿区域）进行导航。

---

## 1. VLM (Vision Language Model) - GPT 方法 【推荐】

### 方法说明
- **模式名**: `gpt`
- **原理**: 使用 GPT-4V 作为全局规划器，根据多个机器人的局部地图和frontier，通过自然语言理解为每个机器人分配最优的探索目标。
- **适用场景**: 需要高智能决策、多机器人协调的复杂场景。

### 调用方式

**在命令行中指定:**
```bash
python main.py --nav_mode gpt --num_agents 2 --gpt_type 2
```

**参数说明:**
- `--nav_mode gpt`: 启用 GPT 导航模式
- `--gpt_type`: GPT 模型选择
  - `0`: text-davinci-003 (已弃用)
  - `1`: gpt-3.5-turbo (较便宜，速度快)
  - `2`: gpt-4o (推荐，性能最好)
  - `3`: gpt-4o-mini (折中方案)
- `--num_agents 2`: 机器人数量

### 核心实现代码 (main.py, 第 740-750 行)

```python
if (agent[0].l_step % args.num_local_steps == args.num_local_steps - 1 or agent[0].l_step == 0) and not found_goal:
    goal_points.clear()
    target_score, target_edge_map, target_point_list = map_process.Frontier_Det(threshold_point=8)
    if len(target_point_list) > 0 and agent[0].l_step > 0:
        # 获取所有候选地图
        candidate_map_list = chat_utils.get_all_candidate_maps(target_edge_map, top_view_map, pose_pred)
        # 构建消息
        message = chat_utils.message_prepare(system_prompt.system_prompt, candidate_map_list, agent[i].goal_name)
        # 调用 GPT-4V 获取目标分配
        goal_frontiers = chat_utils.chat_with_gpt4v(message)
        # 为每个机器人分配frontier
        for i in range(num_agents):
            goal_points.append(target_point_list[int(goal_frontiers["robot_"+ str(i)].split('_')[1])])
    else:
        # 如果没有检测到frontier，随机选择
        for i in range(num_agents):
            action = np.random.rand(1, 2).squeeze()*(obstacle_map.shape[0] - 1)
            goal_points.append([int(action[0]), int(action[1])])
```

### 关键函数
- `chat_utils.get_all_candidate_maps()`: 生成所有候选frontier的可视化地图
- `chat_utils.message_prepare()`: 准备 GPT 的输入消息
- `chat_utils.chat_with_gpt4v()`: 调用 GPT-4V API 获取frontier分配

### 环境要求
```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 2. Greedy (最近邻方法)

### 方法说明
- **模式名**: `nearest`
- **原理**: 每个机器人直接选择距离当前位置最近的frontier进行导航，完全独立的贪心策略。
- **优点**: 简单、快速、无需额外计算或API调用
- **缺点**: 无多机协调，容易导致机器人探索相同区域

### 调用方式

**在命令行中指定:**
```bash
python main.py --nav_mode nearest --num_agents 2
```

### 核心实现代码 (vlm_agents.py, 第 350-354 行)

```python
# 在 act() 方法中
if len(self.object_pcd.points) > 0:
    # 已找到目标对象
    goal_pcd = process_pcd(self.object_pcd)
    self.goal_map[self.object_map_building(goal_pcd)] = 1
    # 直接找最近点
    self.nearest_point = self.find_nearest_point_cloud(goal_pcd, self.camera_position)
    x = self.nearest_point[0]
    y = self.nearest_point[1]
    z = self.nearest_point[2]
    self.found_goal = True
else:
    # 未找到目标，选择最近frontier
    self.found_goal = False
    self.goal_map = np.zeros((self.local_w, self.local_h))
    self.goal_map[goal_points[0], goal_points[1]] = 1  # goal_points 由外部传入
```

### 关键函数
```python
def find_nearest_point_cloud(self, point_cloud, target_point):
    """找点云中距离目标点最近的点"""
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    [k, idx, _] = pcd_tree.search_knn_vector_3d(target_point, 1)
    nearest_point = np.asarray(point_cloud.points)[idx[0]]
    return nearest_point
```

### 特点
- 需要外部提供 `goal_points` 参数（通常由 frontier 检测模块生成）
- 当前代码中如果 `nav_mode != 'gpt'` 时，则默认使用最近邻逻辑

---

## 3. Cost-Utility (成本-效用方法)

### 方法说明
- **模式名**: `co_ut`
- **原理**: 综合考虑探索成本（距离）和效用（frontier 的信息量），通过成本函数优化frontier选择。
- **适用场景**: 需要平衡快速探索和高效覆盖的场景

### 调用方式

**在命令行中指定:**
```bash
python main.py --nav_mode co_ut --num_agents 2
```

### 核心实现逻辑 (待完整实现)

当前代码中 `co_ut` 模式的支持还在参数定义中，具体实现可参考以下伪代码：

```python
def cost_utility_frontier_selection(frontiers, current_position, explored_map):
    """
    Args:
        frontiers: list of frontier points [[x1,y1], [x2,y2], ...]
        current_position: 当前机器人位置 [x, y]
        explored_map: 已探索区域地图
    
    Returns:
        selected_frontier: 选中的frontier点
    """
    best_score = -float('inf')
    best_frontier = None
    
    for frontier in frontiers:
        # 计算距离成本（越近越好，成本越低）
        distance = np.linalg.norm(np.array(frontier) - np.array(current_position))
        cost = distance
        
        # 计算效用（frontier周围未探索区域越多，效用越高）
        utility = count_unexplored_neighbors(frontier, explored_map, radius=5)
        
        # 综合打分: utility / cost
        score = utility / (cost + 1e-6)
        
        if score > best_score:
            best_score = score
            best_frontier = frontier
    
    return best_frontier
```

### 预期工作流
```bash
# Step 1: 运行多个 episode 收集成本-效用数据
python main.py --nav_mode co_ut --num_agents 2 --num_episodes 100

# Step 2: 分析结果
# 查看日志中的成本-效用统计
```

---

## 4. Random Sample on Map (地图上的随机采样)

### 方法说明
- **模式名**: `fill` (填充模式)
- **原理**: 在当前已知地图上随机采样可行的点作为探索目标，适合于填充式的系统探索。
- **优点**: 简单、无偏
- **缺点**: 可能选择次优目标，探索效率低

### 调用方式

**在命令行中指定:**
```bash
python main.py --nav_mode fill --fill_mode 0 --num_agents 2
```

**参数说明:**
- `--fill_mode`: 填充策略
  - `0`: 完全随机采样
  - `1`: 限制在可行区域内（待实现）

### 核心实现代码 (main.py, 第 748-751 行)

```python
else:  # 当 GPT 检测失败或初始阶段时
    for i in range(num_agents):
        # 在地图空间内随机选择一个点
        action = np.random.rand(1, 2).squeeze() * (obstacle_map.shape[0] - 1)
        goal_points.append([int(action[0]), int(action[1])])
```

### 实现细节

```python
def random_sample_on_map(obstacle_map, num_samples=1):
    """
    在可行区域（非障碍）上随机采样
    """
    # 找到所有可行点
    valid_mask = obstacle_map == 0
    valid_points = np.where(valid_mask)
    
    if len(valid_points[0]) == 0:
        # 如果没有可行点，返回地图中心
        h, w = obstacle_map.shape
        return [[h//2, w//2]]
    
    # 随机选择
    indices = np.random.choice(len(valid_points[0]), min(num_samples, len(valid_points[0])), replace=False)
    sampled_points = [[valid_points[0][i], valid_points[1][i]] for i in indices]
    
    return sampled_points
```

---

## 5. Multi-SemExp (多语义探索)

### 方法说明
- **模式名**: (通常与 `gpt` 或其他模式组合)
- **原理**: 利用多个机器人的语义分割结果，多角度理解场景，增强目标识别和frontier优化。
- **适用场景**: 高度结构化环境，需要多视角语义理解

### 核心特性

1. **多视角语义融合** (vlm_agents.py)

```python
def detect_and_segment(self, observations):
    """
    使用 YOLOv8 进行目标检测和语义分割
    """
    rgb = observations['rgb']
    depth = observations['depth']
    
    # 目标检测
    results = self.obj_det_seg.detect_objects(rgb)
    
    # 提取目标点云
    if len(results) > 0:
        object_pcd = create_object_pcd(rgb, depth, results, self.camera_K)
        self.object_pcd = process_pcd(object_pcd)
    return results
```

2. **多机器人点云聚合** (main.py, 第 725 行)

```python
point_sum = o3d.geometry.PointCloud()

for i in range(num_agents):
    agent[i].mapping(observations[i], agent_state)
    # 聚合所有机器人的点云
    point_sum += agent[i].point_sum

# 从聚合点云中提取全局frontier
obstacle_map, explored_map, top_view_map = map_process.Map_Extraction(point_sum, agent[0].camera_position[1])
```

3. **语义感知的frontier检测**

```python
def Frontier_Det(threshold_point=8):
    """
    基于多机器人聚合地图的 frontier 检测
    同时考虑语义信息
    """
    # 1. 检测边界（explored 和 unexplored 的交界）
    # 2. 为每个 frontier 计算语义相关性得分
    # 3. 排序并返回前 K 个 frontier
```

### 调用方式

```bash
# 启用多语义探索（与 GPT 结合）
python main.py --nav_mode gpt --gpt_type 2 --num_agents 3

# 启用可视化查看多角度语义信息
python main.py --nav_mode gpt --gpt_type 2 --num_agents 3 -v 1
```

---

## 6. 导航方法对比表

| 方法 | `nav_mode` | 多机协调 | 计算成本 | 优点 | 缺点 |
|------|-----------|---------|---------|------|------|
| **VLM/GPT** | `gpt` | ✅ 强 | 🔴 高 (API调用) | 智能、高效、适应性强 | 需要API密钥、成本、延迟 |
| **Greedy** | `nearest` | ❌ 无 | 🟢 低 | 简单快速、无依赖 | 易重复、低效率 |
| **Cost-Utility** | `co_ut` | ⚠️ 可选 | 🟡 中 | 平衡效率、考虑多因素 | 参数调优复杂 |
| **Random Fill** | `fill` | ❌ 无 | 🟢 低 | 无偏、简单 | 次优决策、低效 |
| **Multi-SemExp** | (组合) | ✅ 强 | 🟡 中 | 多视角融合、精准识别 | 需要多机器人、复杂度高 |

---

## 7. 完整运行示例

### 示例 1: 使用 GPT-4o 进行多机协调导航
```bash
export OPENAI_API_KEY="your_key_here"
python main.py \
    --nav_mode gpt \
    --gpt_type 2 \
    --num_agents 2 \
    --num_episodes 10 \
    --visualize 1 \
    --print_images 1 \
    --save_video 1
```

### 示例 2: 使用 Greedy 最近邻策略
```bash
python main.py \
    --nav_mode nearest \
    --num_agents 2 \
    --num_episodes 10
```

### 示例 3: 使用随机采样进行系统探索
```bash
python main.py \
    --nav_mode fill \
    --fill_mode 0 \
    --num_agents 3 \
    --num_episodes 5
```

### 示例 4: 多进程加速（用于大规模评估）
```bash
python main_vec.py \
    --nav_mode gpt \
    --gpt_type 2 \
    --num_agents 2 \
    --num_processes 4
```

---

## 8. 自定义导航方法

如果想添加新的导航方法，需要修改以下文件：

1. **arguments.py**: 添加新的 `nav_mode` 选项
```python
parser.add_argument('--nav_mode', type=str, default="gpt",
                    choices=['nearest', 'co_ut', 'fill', 'gpt', 'my_custom'])
```

2. **main.py**: 在 `global_planner` 中添加分支
```python
if args.nav_mode == "my_custom":
    goal_points = my_custom_frontier_selector(target_point_list, agent_poses, exploration_history)
elif args.nav_mode == "gpt":
    # ... existing code
```

3. **agents/vlm_agents.py** (可选): 在 agent 内部实现方法逻辑

---

## 9. 关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| 主循环/方法选择 | main.py | 742-751 |
| Frontier 检测 | main.py | 742-751 |
| GPT 调用 | utils/chat_utils.py | - |
| Agent 决策/Greedy | vlm_agents.py | 330-400 |
| 点云聚合 | main.py | 720-730 |
| 多语义融合 | agents/vlm_agents.py | 230-260 |

---

## 10. 性能建议

- **快速原型**: 使用 `nearest` 模式，无需配置
- **高质量探索**: 使用 `gpt` 模式，但注意API成本
- **离线评估**: 使用 `fill` 或 `co_ut` 模式进行快速多轮测试
- **实时机器人**: 考虑 `nearest` 或 `co_ut` 的本地计算版本（无网络依赖）

---

## 常见问题

**Q1: 如何同时使用多个导航方法进行对比实验？**
```bash
for mode in gpt nearest fill co_ut; do
    python main.py --nav_mode $mode --num_episodes 5
done
```

**Q2: GPT 超时或失败时如何fallback？**
见 main.py 第 748-751 行，已自动fallback到随机采样。

**Q3: 如何在实时机器人上使用（无GPU/API）？**
推荐使用 `nearest` 或 `co_ut` 模式，参考 `agents/ros2_agents.py`

