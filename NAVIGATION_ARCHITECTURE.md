# Co-NavGPT 导航方法实现架构图

## 1. 整体导航流程架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Main Episode Loop (main.py)                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │  Agent Mapping Phase     │
                    │  (每个agent独立)        │
                    │  - 处理RGB-D           │
                    │  - 语义分割            │
                    │  - 生成局部点云        │
                    └─────────────────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │  Global Aggregation     │
                    │  - 合并所有点云         │
                    │  - 融合多机器人地图     │
                    │  - 生成全局视图        │
                    └─────────────────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │  Frontier Detection     │
                    │  - 找边界(explored边)   │
                    │  - 生成候选区域列表     │
                    └─────────────────────────┘
                                ↓
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ↓                      ↓                      ↓
    ┌─────────┐           ┌─────────┐          ┌────────────┐
    │  GPT    │           │ Greedy  │          │Cost-Utility│
    │ (smart) │           │(nearest)│          │  (balanced)│
    └─────────┘           └─────────┘          └────────────┘
         │                      │                      │
         │ (VLM推理)           │ (最近邻)              │ (打分)
         │                     │                      │
         └──────────────────────┼──────────────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │  Goal Assignment        │
                    │  - goal_points[i] ←    │
                    │    frontier_list[idx]  │
                    └─────────────────────────┘
                                ↓
                    ┌─────────────────────────┐
                    │  Agent Planning Phase   │
                    │  (每个agent独立)        │
                    │  - 规划路径            │
                    │  - 执行导航            │
                    └─────────────────────────┘
                                ↓
                            执行动作
                                ↓
                            update observations
```

---

## 2. 各方法的详细决策树

### A. GPT 方法决策链

```
GPT 方法 (VLM-based Multi-Robot Coordination)
│
├─ Input:
│  ├─ 多个 frontier 候选点的图像
│  ├─ 机器人当前位置
│  ├─ 目标对象名称
│  └─ 已探索/未探索地图
│
├─ Processing:
│  ├─ 1. 生成每个 frontier 的"俯视地图"可视化
│  │     candidate_map_list = get_all_candidate_maps(frontier_list)
│  │
│  ├─ 2. 构建 prompt
│  │     message = message_prepare(
│  │         system_prompt,           # 系统指令
│  │         candidate_map_list,      # 候选地图图像
│  │         goal_name                # 目标对象名
│  │     )
│  │
│  ├─ 3. 调用 GPT-4V
│  │     response = chat_with_gpt4v(message)
│  │     # 返回: {"robot_0": "frontier_3", "robot_1": "frontier_5"}
│  │
│  └─ 4. 解析结果并映射回坐标
│      goal_points[i] = target_point_list[frontier_id]
│
└─ Output:
   goal_points = [
       [x0, y0],  # robot 0 目标
       [x1, y1],  # robot 1 目标
       ...
   ]
```

### B. Greedy 方法决策链

```
Greedy 方法 (Nearest Neighbor)
│
├─ For each robot i:
│  │
│  ├─ Input:
│  │  ├─ current_position[i]
│  │  ├─ frontier_list (所有候选点)
│  │  └─ found_goal (是否已找到目标)
│  │
│  ├─ Processing:
│  │  ├─ If found_goal:
│  │  │  └─ goal = closest_point_in(object_pcd, current_pos)
│  │  │      [使用KD树快速查询]
│  │  │
│  │  └─ Else:
│  │     └─ goal = frontier_list[
│  │            argmin(distance(frontier, current_pos))
│  │        ]
│  │
│  └─ Output per robot:
│     goal_points[i] = [x_goal, y_goal]
│
└─ Complexity: O(n) per step, 无GPU需求
```

### C. Cost-Utility 方法决策链

```
Cost-Utility 方法 (Exploration Efficiency)
│
├─ For each frontier f:
│  │
│  ├─ Calculate:
│  │  ├─ cost(f, robot_i) = distance(robot_pos, frontier)
│  │  │  ├─ 距离越近，成本越低
│  │  │  └─ 可选加权 (近距离优先)
│  │  │
│  │  ├─ utility(f) = unexplored_area_around(f)
│  │  │  ├─ Frontier 周围的未探索区域面积
│  │  │  ├─ 可选加权 (信息增益)
│  │  │  └─ 使用morphological ops 计算
│  │  │
│  │  └─ score(f) = utility(f) / (cost(f) + epsilon)
│  │      # 效用高、成本低 → 分数高
│  │
│  └─ Per robot:
│     best_frontier[i] = argmax(score)
│
└─ Output:
   goal_points[i] = best_frontier[i]
```

### D. Random Fill 方法决策链

```
Random Fill 方法 (Stochastic Exploration)
│
├─ Input:
│  ├─ obstacle_map (障碍地图)
│  └─ explored_map (已探索地图)
│
├─ Processing:
│  ├─ valid_positions = find_all(obstacle_map == 0)
│  │  # 所有可行区域
│  │
│  └─ For each robot i:
│     └─ goal_points[i] = random_sample_from(valid_positions)
│        使用 np.random.choice() 均匀采样
│
└─ Output:
   goal_points (随机但合法的点列表)
```

---

## 3. 关键数据结构对应

```
VLM_Agent 中的关键属性:
├─ point_sum                 → Open3D 点云 (局部映射累积)
├─ obstacle_map              → np.array (障碍地图)
├─ explored_map              → np.array (已探索地图)
├─ object_pcd                → Open3D 点云 (检测到的目标)
├─ goal_map                  → np.array (目标区域 mask)
├─ current_grid_pose         → [x, y] (当前网格坐标)
├─ camera_position           → [x, y, z] (世界坐标)
└─ found_goal                → bool (目标找到标志)

全局状态 (main.py):
├─ agent[]                   → VLM_Agent 列表 (所有机器人)
├─ point_sum (global)        → Open3D 点云 (所有机器人聚合)
├─ goal_points               → [[x0,y0], [x1,y1], ...] (分配的目标)
├─ target_point_list         → 所有检测到的 frontier 点
├─ target_edge_map           → frontier 的 2D 网格表示
└─ fire_simulator            → 火焰模拟器 (安全评估)
```

---

## 4. 导航方法代码流程对应表

| 方法 | 主要代码位置 | 关键函数 | 决策时间 |
|------|-----------|---------|---------|
| **GPT** | main.py:742-750 | `chat_with_gpt4v()` | 每 25 步 |
| **Greedy** | vlm_agents.py:350-354 | `find_nearest_point_cloud()` | 实时 |
| **Cost-Utility** | utils/explored_map_utils.py | `cost_utility_score()` (待实现) | 每 25 步 |
| **Random** | main.py:748-751 | `np.random.rand()` | 每 25 步 |

---

## 5. 多机协调融合图

```
Multi-Robot + Multi-Semantic Exploration
│
├─ Robot 0:
│  └─ Local View 0 → RGB-D → Segmentation → object_pcd_0 → point_sum_0
│
├─ Robot 1:
│  └─ Local View 1 → RGB-D → Segmentation → object_pcd_1 → point_sum_1
│
└─ Robot N:
   └─ Local View N → RGB-D → Segmentation → object_pcd_N → point_sum_N
   
        ↓ (聚合 Aggregation)
        
global_point_sum = point_sum_0 + point_sum_1 + ... + point_sum_N
global_map = extract_map(global_point_sum)

        ↓ (Frontier 检测)
        
frontier_list = detect_frontier(global_map)

        ↓ (VLM 决策或其他方法)
        
decision_algorithm(frontier_list, robot_positions) 
    → goal_points[0], goal_points[1], ..., goal_points[N]

        ↓ (分配)
        
robot_0.act(goal_points[0]) → action_0
robot_1.act(goal_points[1]) → action_1
...
robot_N.act(goal_points[N]) → action_N

        ↓
    执行并 step 到下一状态
```

---

## 6. GPT 调用时序图

```
Timeline:
─────────────────────────────────────────────────────────────

T0: Step 0-24      T1: 第 25 步           T2: GPT 处理       T3: 返回结果
┌──────────────┐  ┌─────────────┐        ┌──────────────┐   ┌────────────┐
│ Mapping      │→ │ Aggregation │→ ─ ─ ─→│ GPT-4V       │───→│Goal Point  │
│  & Navigation│  │ & Frontier  │        │ Processing  │   │ Assignment │
└──────────────┘  │ Detection   │        │ (~1-2 sec)  │   └────────────┘
                  └─────────────┘        └──────────────┘
                        ↓
                  ┌─────────────┐
                  │ message =   │
                  │ {img1, img2,│
                  │  goal_name} │
                  └─────────────┘
```

---

## 7. 性能特征对比

```
┌────────────────────────────────────────────────────────────────┐
│                 决策方法性能特征                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ GPT (VLM):                                                    │
│   速度: ████░░░░░░ (受网络限制，~1-2秒)                      │
│   精度: ██████████ (极高，基于语言理解)                       │
│   成本: ██████░░░░ (API 调用费用)                            │
│   协调: ██████████ (完美多机协调)                            │
│                                                                │
│ Greedy:                                                       │
│   速度: ██████████ (极快，O(n) KD树查询)                      │
│   精度: ████░░░░░░ (简单，易次优)                             │
│   成本: ██████████ (零成本)                                   │
│   协调: ░░░░░░░░░░ (无协调)                                   │
│                                                                │
│ Cost-Utility:                                                 │
│   速度: █████████░ (快速，O(n²)计算)                          │
│   精度: ██████░░░░ (较好，平衡取舍)                           │
│   成本: ██████████ (零成本)                                   │
│   协调: █████░░░░░ (部分协调)                                 │
│                                                                │
│ Random:                                                       │
│   速度: ██████████ (极快，O(1)采样)                           │
│   精度: ██░░░░░░░░ (无偏但常次优)                             │
│   成本: ██████████ (零成本)                                   │
│   协调: ░░░░░░░░░░ (无协调)                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. 选择决策树

```
                    选择导航方法
                        │
         ┌──────────────┼──────────────┐
         │              │              │
      需要多机       需要快速        成本
      协调吗？        决策吗？      敏感吗？
         │              │              │
      是/否             是/否          是/否
        │ │              │ │            │ │
      是 否             是 否          是 否
       │  │              │ │            │ │
       ↓  │              ↓ │            ↓ │
      GPT │          greedy │        greedy
         │              │   │            │
         │              │   ↓            ↓
         │              │ co_ut      nearest
         │              │   ↓
         │              fill
         │
         └─→ 高性能要求 → GPT
              有 API Key  ✓

     建议决策规则:
     ────────────────────
     1. 快速验证 → nearest
     2. 成本敏感 → fill 或 co_ut
     3. 要求高 → gpt-4o
     4. 平衡型 → gpt-4o-mini
     5. 实时应用 → nearest
```

---

## 9. 调试流程

```
问题排查树:

导航效果不好？
  ├─ 是否成功找到 frontier？
  │  ├─ NO → 检查 mapping 或 frontier detection
  │  └─ YES ↓
  │
  ├─ 选择的 frontier 是否合理？
  │  ├─ 如果用 GPT：
  │  │  ├─ 检查 prompt 质量 (message_prepare)
  │  │  ├─ 检查 GPT 返回值格式
  │  │  └─ 尝试 gpt_type=2 (gpt-4o)
  │  │
  │  ├─ 如果用 greedy：
  │  │  ├─ 检查是否计算了正确的距离
  │  │  └─ 可尝试 co_ut 加入效用考量
  │  │
  │  └─ 如果用 random：
  │     └─ 考虑改用其他算法
  │
  ├─ 路径规划是否正常？
  │  ├─ 检查 search_navigable_path()
  │  └─ 检查 FMM planner 输出
  │
  └─ 执行是否成功？
     ├─ 检查碰撞检测
     └─ 检查 greedy_follower_act()

快速诊断命令:
```bash
# 1. 单 agent, nearest, 3 episodes
python main.py --nav_mode nearest --num_agents 1 --num_episodes 3 -v 1

# 2. 保存日志和图像
python main.py --nav_mode gpt --num_agents 2 \
    --print_images 1 --log_interval 1

# 3. 多进程快速测试
python main_vec.py --nav_mode nearest --num_processes 4 --num_episodes 1
```

---

## 10. 扩展新方法模板

如要添加新的导航方法 `my_method`：

```python
# Step 1: 在 arguments.py 中添加
parser.add_argument('--nav_mode', 
                   choices=['nearest', 'co_ut', 'fill', 'gpt', 'my_method'])

# Step 2: 在 main.py 的全局规划部分添加
if args.nav_mode == 'my_method':
    goal_points = my_frontier_selector(
        frontier_list,
        agent_poses,
        global_map,
        args
    )
    
# Step 3: 实现函数
def my_frontier_selector(frontier_list, agent_poses, global_map, args):
    """
    Your custom frontier selection logic
    Return: goal_points = [[x0,y0], [x1,y1], ...]
    """
    pass

# Step 4: 测试
python main.py --nav_mode my_method --num_agents 2 --num_episodes 1
```

