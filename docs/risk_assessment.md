# Dynamic Risk Assessment：火焰、温度与烟雾风险评估

本文说明 `main.py` 中已经接通的动态风险评估与风险感知导航实现。它借鉴
[VULCAN](2604.12831v1.pdf) 的“感知风险图 → frontier 风险 → 风险调制
FMM → CHE”分层设计，但以当前仓库代码为准，并不是对论文公式的逐字复现。
FireWorld 的生成、传播与多模态渲染另见
[FireWorld pipeline](fire_world_pipeline.md)。

最重要的实验边界是：

- 规划器只能使用 `--risk_source` 指定的风险源；
- 只要 `--risk_enabled=1`，暴露指标始终由独立的 FireWorld ground-truth
  provider 计算；
- 因此 `none`、`oracle`、`sensed` 可以在同一 GT 风险标尺下比较，而
`sensed` 规划器不会因此获得 GT 地图。
风险层与 `objectgoal` category 解耦：`person`、chair、bed 等目标共享同一套
frontier/FMM/STOP 与 Habitat Success/SPL 链路，风险模块只改变路径代价和安全
约束，不为人物另设成功判定。

## 1. 运行模式与信息边界

`RiskConfig.effective_source` 同时由 `--risk_enabled` 和 `--risk_source`
决定。四种实际状态如下：

| 配置 | 规划器看到什么 | 风险指标 | 用途 |
| --- | --- | --- | --- |
| `--risk_enabled 0` | 不构造风险运行时，保留原始导航逻辑 | 不生成 | 完全关闭；向后兼容旧实验 |
| `--risk_enabled 1 --risk_source none` | 中性零风险图，且风险规划显式关闭 | 独立 GT evaluator | evaluator-only 基线：测量旧策略在同一动态火场中的暴露 |
| `--risk_enabled 1 --risk_source oracle` | 完整 FireWorld 场经 2D 投影后的 GT 风险图 | 独立 GT evaluator | privileged upper bound / 调试消融，不应作为主 benchmark 方法 |
| `--risk_enabled 1 --risk_source sensed` | 多机器人观测反投影并融合得到的共享 belief map | 独立 GT evaluator | 主要 benchmark 设置 |

`--risk_enabled=1` 要求同时启用 `--fire_world=1` 并提供有效的
`--fire_world_plan_id`。论文主表建议使用 `sensed + appearance_depth + step
clock`；`oracle` 和 privileged transmittance 应明确标作消融或上界。
这里的 `source=none` 只关闭 hazard-aware planning，不关闭 FireWorld 及其传感器
退化；它测量的是原导航策略在火场观测下的表现，而不是 clean-environment 基线。

数据流如下：

```text
FireWorld flame/smoke/temp(t)
        ├── privileged GT provider ───────────────► exposure evaluator
        │                                          CHE/step / critical steps
        │
        ├── source=oracle ─► floor projection ───► planner risk map
        │
        └── FireSensorSuite observations
              RGB(smoke), depth(clean/smoke), thermal
                    └── source=sensed
                         evidence back-projection
                         + shared temporal belief ─► planner risk map

planner risk map ─► frontier assignment / VLM safety guard
                 └► hazard-aware local FMM
```

## 2. 风险定义与单位

### 2.1 物理风险指数

烟雾 `S` 是 `[0,1]` 的 FireWorld/传感器风险量；温度 `T` 以摄氏度为单位。
温度首先归一化为：

```text
r_T(T) = clip((T - T_ref) / (T_hazard - T_ref), 0, 1)
```

随后计算不含不确定性的物理风险：

```text
H_phys = clip(w_T r_T + w_S S, 0, 1)
```

CLI 默认值为 `T_ref=35°C`、`T_hazard=150°C`，以及
`(w_T,w_S)=(0.60,0.40)`；两个权重必须非负且和为 `1`。
原始温度层仍保留 `T_ambient=25°C`，低于 `T_ref` 的读数不会为了归一化而被
伪装成 `35°C`。
这里的温度阈值和 `H_phys` 都是 **benchmark calibration**，不是医学上的
生存阈值、伤害概率或生理模型。

火焰不再进入连续加权和：在当前 FireWorld 中，火焰已经会推高温度，把 `F`
再次加权会重复计算同一危险。高强度火焰核心保留为第 2.2 节的硬约束；外围
温度和烟雾继续通过 `H_phys` 平滑递减，所以这不是忽略明火，而是把“不可穿越
的火焰核心”与“可连续累计的外围热/烟暴露”分开。
感知不确定性也只进入规划代价，不混入 GT 暴露指标，从而把危险本身和机器人
对危险的未知程度分开。

### 2.2 Hard 标签（当前仅用于诊断与评估）

连续风险以外，代码还构造 hard 标签：

```text
M_hard = dilate(F >= F_hard, ceil(d_core / map_resolution))
         OR temperature_hard_enabled · (T >= T_hard)
```

CLI 默认 `F_hard=0.80`、`d_core=0m` 且
`temperature_hard_enabled=0`，因此只有高强度火焰核心属于 hard unsafe；周围
温度/烟雾全部保留为连续风险。需要复现旧式保守安全环时，可以显式设置
`--risk_flame_safety_distance_m`，需要温度二值 veto 时再启用
`--risk_temperature_hard_enabled 1 --risk_temperature_hard_c 250`。
`M_hard` 不会被低风险样本平均掉，并继续用于 hazard report、可视化与 critical
violation 计数；当前 FMM/A* 以及 value-first global assignment 不再用它过滤
frontier、栅格或 task goal。PointNav/RL 未纳入本次行为修改。

## 3. Oracle 与 sensed 风险图

### 3.1 Oracle：完整 FireWorld 投影

`GroundTruthRiskProvider` 在共享导航栅格中对每个 `(x,z)` cell 反采样
FireWorld，并在当前楼层的 body-height band 内分别对 flame、smoke 和
temperature 做垂直最大值聚合。默认高度带为初始楼层高度以上
`[0.0m, 1.50m]`。随后计算 `H_phys` 和 `M_hard`。

Oracle 规划代价为：

```text
P_oracle = clip(max(H_GT, p_unknown (1-C_GT)), 0, 1)
```

FireWorld 覆盖范围内通常 `C_GT=1`。该模式直接读取完整模拟器状态，是明确的
privileged upper bound；GT provider 同时服务 evaluator，但 evaluator 与规划
接口是两个独立对象。

### 3.2 Sensed：观测证据与共享 belief

主要 benchmark 的 `sensed` 模式只从每个 agent 的传感器输出构造规划图：

- 火焰：`thermal_flame_mask`；
- 温度：`thermal_temperature`；人体热签名在没有与真实 flame 重叠时被压回
  `T_ref`，避免把人误判成火灾；
- 几何反投影：默认 `--risk_geometry_depth_source clean`，沿用主框架中作为
  smoke-robust radar/depth geometry surrogate 的 clean depth；可用 `smoke`
  选择退化视觉深度消融。Smoke appearance estimator 本身始终读取
  `depth_smoke`；
- 烟雾：默认使用 `rgb_smoke + depth_smoke` 的可观测 visibility proxy。

默认 smoke proxy 用去饱和、局部对比度损失、亮度门控和深度退化组合：

```text
A = clip(1 - saturation/0.45) · exp(-local_std/0.055)
    · clip((gray-0.08)/0.45)
S_hat = clip(0.80 A + 0.20 D_depth, 0, 1)
C_hat = clip(valid_depth · (1 - 0.55 S_hat), 0.05, 1)
```

其中无效 depth 的 `D_depth=1`，接近默认 `5m` 上限的 clipped depth 使用
`D_depth>=0.35`。这是一个确定性的基线估计器；灰色平滑墙面与烟雾存在歧义，
因此代码同时降低 confidence。`S_hat` 是能见度风险 proxy，不是毒气、CO、
烟尘剂量或人体毒性估计。

每个像素证据按 `--risk_sensor_stride` 下采样后反投影到 world frame，再由所有
机器人融合到同一个 `DynamicRiskMap`。同一栅格的物理量采用保守的“rise
fast, fall slowly”规则：新证据取 flame/smoke/temperature 最大值，confidence
按 `1-(1-C_old)(1-C_new)` 合并；uncertainty 首次取当前估计，之后保守取最大
值，避免一次乐观观测抹掉已有不确定性。陈旧证据按指数衰减：

```text
F,S             <- (F,S) exp(-dt/tau)
T               <- T_ambient + (T-T_ambient) exp(-dt/tau)
C               <- C exp(-dt/tau_C)
```

规划使用包含未知空间和不确定性惩罚的 belief cost：

```text
P_sensed = clip(max(H_belief, p_unknown (1-C)) + lambda_U U, 0, 1)
```

默认 `p_unknown=0.25`、`lambda_U=0.25`；未观测 cell 初始
`C=0,U=1`，因此默认规划代价为 `0.50`，但不会直接变成硬障碍。

### 3.3 Privileged transmittance 消融

设置 `--risk_smoke_source privileged_transmittance` 后，sensed 管线会读取
体渲染器直接输出的 scene transmittance，并以 Beer–Lambert 形式构造 smoke
proxy：

```text
S_hat = clip(-ln(transmittance) / (k d), 0, 1)
```

这是 **privileged simulator product**，代码要求显式开关才允许进入 evidence；
若任一 agent 缺少 transmittance 或传感器时间戳与 shared fire time 不一致，运行会
直接报错而不会静默退化为零烟雾。
它不属于 no-oracle sensed benchmark。当前 `k` 是 risk config 的内部系数，
而 `--fire_world_smoke_k_ext` 控制渲染器本身的消光，两者不要在论文中混写成
同一个可观测传感器参数。

## 4. 风险如何进入导航

### 4.1 Global frontier 评估与多机器人分配

每个 frontier 输出以下结构化字段：

- frontier 区域的 `mean_risk`、`p95_risk`、`max_risk`；
- 近似 approach route 的 `route_risk`（均值）与 `route_max_risk`；
- frontier 区域平均 `confidence`、`uncertainty=1-confidence`；
- `safe/moderate/dangerous` 严重度和不可覆盖的 `hard_blocked`。

当前运行时默认以 `0.25` 为 safe/moderate 分界，以
`--risk_danger_threshold`（默认 `0.55`）为 moderate/dangerous 分界。

用于连续 utility 的保守风险是：

```text
R_k = max(p95(frontier k), mean(approach route k))
```

`M_hard` 和 `--risk_hard_frontier_threshold` 只生成诊断标签，不再过滤
frontier。approach route 通过该 agent 自己的已探索自由空间执行八邻域风险加权搜索
（代价与 local A* 一致），
所以“目标方向的直线穿火，但实际可以绕火到达”不会被误判为必须穿火。仅在当前
地图中找不到可通行 approach route 时才退回直线诊断，并明确设置
`route_is_proxy=true`。风险模块不再重写 `nearest/co_ut/fill` 的策略公式。
每个 normal planner 先提供自己的原始 preference `B_a,k`，共享层先找出最佳值
`B*_a`，只把相对 regret 不超过 `epsilon` 的 frontier 视为“价值相当”：

```text
K_a = { k | B*_a - B_a,k <= epsilon * max(|B*_a|, median(|B_a|), 1) }
choice_a = argmin_(k in K_a) (R_k, -B_a,k, frontier_id)
```

`epsilon=--risk_frontier_value_tolerance`，默认 `0.10`。不包含独立 uncertainty
代价，也不因 hard label 拒绝候选。实际逐 action 绕火由 local FMM/A* 的
`--risk_alpha`（默认 `1.0`）负责。四个 classical planner
的 normal preference 与安全层关系如下：

| `nav_mode` | normal preference `B_a,k` | 共享 risk-awareness |
| --- | --- | --- |
| `nearest` | `-robot_frontier_distance` | 价值容差内以 `R_k` 打破近似平局 |
| `co_ut` | `frontier_size - lambda * distance` | 同上 |
| `fill` | `target_score`（缺失时使用 inverse distance） | 同上 |
| `random` | 可复现的 reachable map-goal sampling | 保持 normal sampling domain，不按风险拒绝 |

因此零风险、完整 confidence 且没有 hard cell 时，四种模式与各自 normal
planner 的目标完全一致；risk 只在近似平局中加入，不再把 `fill` 改写成另一套
size/distance/redundancy 策略。`random` 不使用 frontier utility，因此在每个
机器人的已探索可达自由空间内按原 seed 与 normal 模式完全相同地采样。

`gpt` 模式还会把同一 hazard report 作为 JSON 交给 VLM；提示明确要求首先考虑
语义、距离、探索价值和团队覆盖，只有价值相近才用连续风险打破平局。
`hard_blocked` 也是诊断信息，不再触发 deterministic hard veto；格式或 ID 无效
时仍回退到 value-first `co_ut` assignment。

当前 global `route_risk` 是已探索自由空间内的风险加权 approach proxy；找不到
该路线时才回退到单栅格宽直线并标记 `route_is_proxy=true`。它仍不是之后 FMM
实际执行的完整轨迹，而且共享 frontier 模式下不是逐 robot 路线。因此论文不能
把它称为真实路径累计暴露；真实执行暴露应使用第 6 节的 GT `CHE` 与逐步 trace。

### 4.2 Local hazard-aware FMM

火灾 none/oracle/sensed 的 FMM 对照统一使用共享栅格后端；none 使用普通 FMM，
oracle/sensed 使用 risk-aware FMM。普通无火实验的 `auto` 仍保留历史
navmesh-first。none/oracle 的 padded-grid 几何也保持一致：外层一格不可通行，
起点、目标和 STG 统一使用 `+1` 坐标偏移。连续风险通过速度场调制：

```text
v(x) = 1 / (1 + alpha · P(x))
```

其中 `alpha=--risk_alpha`，默认 `1.0`；风险越高，travel time 越大。
该形式对应 VULCAN Eq. (11)。FMM/A* 的主运行路径不再接收 `M_hard`，也不会生成
emergency escape、trapped holding point 或目标附近的 safe waypoint；真实任务
goal 始终保留。每个 action cycle 都用最新连续风险重算 travel time/path，因此
危险只会让局部单元更昂贵，不会使目标或通道不可达。PointNav 和 RL 的行为不在
本次修改范围内。

## 5. CLI 参考

以下默认值来自 `arguments.py`，而不是 dataclass 的直接构造默认值：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--risk_enabled` | `0` | 风险运行时总开关 |
| `--risk_source` | `sensed` | `none` / `oracle` / `sensed`；总开关关闭时 effective source 仍为 `none` |
| `--risk_weight_temperature` | `0.60` | `H_phys` 温度权重 |
| `--risk_weight_smoke` | `0.40` | `H_phys` smoke proxy 权重 |
| `--risk_temperature_ambient_c` | `25.0` | 原始温度层与衰减基线（°C） |
| `--risk_temperature_reference_c` | `35.0` | 温度风险开始上升的校准点（°C） |
| `--risk_temperature_hazard_c` | `150.0` | 温度归一化到 1 的校准点（°C） |
| `--risk_temperature_hard_c` | `250.0` | 启用温度硬 veto 时使用的阈值（°C） |
| `--risk_temperature_hard_enabled` | `0` | `1` 才启用温度二值硬封锁；默认温度为连续软风险 |
| `--risk_flame_hard_threshold` | `0.80` | 高强度火焰核心硬阈值 |
| `--risk_flame_safety_distance_m` | `0.0` | 可选火焰核心膨胀距离（m）；默认不扩张 |
| `--risk_danger_threshold` | `0.55` | danger 暴露与 frontier severity 阈值 |
| `--risk_critical_threshold` | `0.80` | critical step 的连续风险阈值 |
| `--risk_early_stop_enabled` | `1` | 任一 agent 的 action 后 GT risk 达阈值时立即结束 team episode，并把 Success/SPL 判为 0 |
| `--risk_early_stop_threshold` | `None` | safety early-stop 阈值；默认继承 `--risk_critical_threshold`，可显式设置为 `[0,1]` |
| `--risk_decay_tau_s` | `20.0` | sensed 物理证据衰减时间常数（s） |
| `--risk_confidence_decay_tau_s` | `30.0` | sensed confidence 衰减时间常数（s） |
| `--risk_unknown_risk_prior` | `0.25` | 未观测空间 prior |
| `--risk_uncertainty_weight` | `0.25` | sensed uncertainty 规划权重 |
| `--risk_sensor_stride` | `4` | 传感器图像反投影像素步长 |
| `--risk_floor_min_offset_m` | `0.0` | GT 垂直投影带下界，相对初始 floor y（m） |
| `--risk_floor_max_offset_m` | `1.50` | GT 垂直投影带上界，相对初始 floor y（m） |
| `--risk_smoke_source` | `appearance_depth` | `appearance_depth` / `privileged_transmittance` |
| `--risk_geometry_depth_source` | `clean` | `clean` smoke-robust geometry surrogate / `smoke` 退化深度消融 |
| `--risk_alpha` | `1.0` | FMM 风险速度惩罚强度 |
| `--risk_frontier_weight` | `0.5` | 兼容旧配置与 metadata；当前 value-first assignment 不使用该加权项 |
| `--risk_frontier_value_tolerance` | `0.10` | 允许风险打破近似平局的相对 normal-value regret |
| `--risk_hard_frontier_threshold` | `0.80` | frontier/approach 的诊断 severity 阈值，不参与拒绝 |
| `--risk_dump_dir` | `./outputs/risk_assessment` | 风险 artefact 根目录 |
| `--risk_save_every` | `10` | 每 N 个导航 step 保存 PNG；`0` 只关闭 PNG，不关闭 JSON trace/summary |
| `--risk_save_traces` | `1` | `1` 保存逐步 risk/action JSONL 和 `action_list.json`；`0` 只保留最终 `risk_summary.json` |
| `--risk_max_floor_deviation_m` | `0.75` | `sensed` 的共享 2D belief 约束：agent 离开 episode floor 超过该距离会 fail-fast；`oracle/none` 均按各 agent 当前高度独立投影，不使用此共享平面约束 |
| `--risk_run_id` | `default` | 输出 run 子目录；仅允许字母、数字、点、短横线、下划线 |
| `--risk_rank` | `0` | 并行运行的输出 rank 子目录 |

为可复现 benchmark，应使用 step clock；wallclock 会把模型/API 延迟变成火势演化
的一部分，只适合 latency stress test。

### 5.1 推荐 sensed benchmark

```bash
python main.py \
    --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_clock_mode step \
    --fire_steps_per_unit 5 --fire_seconds_per_unit 2.0 \
    --risk_enabled 1 --risk_source sensed \
    --risk_smoke_source appearance_depth \
    --risk_geometry_depth_source clean --risk_run_id sensed_seed1 \
    --risk_dump_dir outputs/risk_assessment --risk_save_every 10
```

### 5.2 同场景消融

保持 FireWorld plan、episode、seed、clock 和所有阈值不变，只替换风险源：

```bash
# evaluator-only：旧导航策略 + 独立 GT 风险测量
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source none

# oracle upper bound：完整 FireWorld 风险图参与规划
python main.py --num_agents 2 \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source oracle

# privileged smoke-only ablation；仍须明确标作 privileged
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source sensed \
    --risk_smoke_source privileged_transmittance
```

## 6. 输出与指标

每个 episode 写入：

```text
<risk_dump_dir>/<risk_run_id>/rank_000/
├── risk_config.json
└── ep_0000/
    ├── risk_steps.jsonl
    ├── actions.jsonl
    ├── action_list.json           # 三者由 --risk_save_traces 控制
    ├── risk_step_00000.png        # 由 --risk_save_every 控制
    ├── risk_step_00010.png
    └── risk_summary.json
```

设置 `--risk_save_traces 0 --risk_save_every 0` 时，每个 episode 目录只写
`risk_summary.json`；风险指标仍会进入主程序 stdout 和累计 metrics，规划与
评估语义不变。

### 6.1 Step trace 与可视化

`risk_steps.jsonl` 使用带 `record_type` 的事件记录，不能把 pre-action 地图和
post-action 暴露误当成同一个时刻：

- `planner_snapshot`：动作前风险图统计、source、frontier report，以及
  `frontier_computed_step`；frontier 不是每步重算时该字段可识别陈旧程度；
- `exposure`：`env.step` 后同一 action index 下每个 agent 的 GT 风险样本和
  planner status。

注意：在 evaluator-only `source=none` 中，顶层地图统计是中性 belief/零规划图，
不是偷偷暴露的 GT 地图；只有 `exposure` 的 `agents` 和最终 summary 来自独立
GT evaluator。Oracle 的地图统计是 GT，sensed 的地图统计是 belief。

PNG dashboard 为 2×3 panel：Flame risk、Temperature risk、Smoke risk、
Physical risk、Planning cost、Confidence；unknown cell 为灰色，hard-unsafe
为品红色，并叠加 agent/frontier 位置。

当 `--visualize 1` 或 `--print_images 1` 时，普通导航合成图左侧的
obstacle-map panel 会叠加 hazard：淡黄→橙→红表示风险从低到高，品红色区域
和深色轮廓表示 `hard_unsafe`。`source=sensed/oracle` 显示 planner 实际使用的
`planning_risk`；evaluator-only `source=none` 显示独立 GT evaluator 的
`physical_risk`，并明确标注 `GT display only`。后者只用于解释 risk-blind
轨迹穿过了什么危险，不会进入 Global/Local Planner。真实障碍物在混色后原样
重绘，右侧 top-view 保持不变。只有 `risk_enabled=0` 时不显示叠加层。

### 6.2 推荐 benchmark 指标

Evaluator 在 reset 后先 `prime` 初始位置，但不把 reset 算作 action sample；之后
每次 `env.step` 对所有 agent 在同一个 fire timestamp 采样一次。令
`H_a,t=H_GT(x_a,t)`，`A_t` 为当前联合 step 的 agent 数，`T` 为执行的联合
step 数：

```text
TeamRisk_t = (sum_a H_a,t) / A_t
CHE_per_step = (sum_t TeamRisk_t) / T
CriticalSteps = sum_t any_a(H_a,t >= risk_critical_threshold)
EarlyStop = any(H_a,t >= risk_early_stop_threshold)
FireSuccess = Habitat Success AND NOT EarlyStop
FireSPL = Habitat SPL if NOT EarlyStop else 0
SafeSuccess = FireSuccess * (1 - CHE_per_step)
```

从 `fireworld-risk-v4` 开始，公开键为 `risk/che_per_step`，范围为 `[0,1]`，
越低越好。它先在每个联合 step 内对 agent 求均值，再对已执行 step 求均值；
`risk_summary.json` 同时记录 `joint_steps` 和 `exposure_samples`。固定 agent 数时
它与 v3 的 agent-action 均值数值相同，但键名、SafeSuccess 和 critical 语义已经
改变，因此 v3/v4 checkpoint 不可混合续跑。reset 后的 prime 仍不计入样本。

EarlyStop 使用独立 GT evaluator，而不是 planner risk map；因此
`risk_source=none/oracle/sensed` 使用完全相同的安全裁判。默认阈值继承
`--risk_critical_threshold=0.80`。任一 agent action 后达到阈值，就立即结束整个
多智能体 episode，并在聚合前强制 `success=0`、`spl=0`。`soft_spl` 和
`distance_to_goal` 保留为 early-stop 位置的进度诊断。

`risk/critical_steps` 是显式主表指标：只要任一 agent 的连续 GT 风险达到
`--risk_critical_threshold`，该联合 step 计一次；同一 step 有多个 agent 触发也
只计一次。它不再混入 `M_hard`。`critical_agent_steps` 仍保存在 summary 中用于
诊断每个 agent 的贡献。

EarlyStop 仍是必要的运行控制和内部事件记录，但不再作为独立聚合指标显示：
它一旦触发就已经令 `success=0`、`spl=0`，并使连续 SafeSuccess 自动为 0。

推荐主表报告五项：

| 指标 | 方向 | 目的 |
| --- | --- | --- |
| Habitat `Success` | 越高越好 | 是否完成 ObjectNav 任务 |
| Habitat `SPL` | 越高越好 | 成功率与路径效率 |
| `risk/safe_success` | 越高越好 | 成功率按单位-step平均暴露连续折扣 |
| `risk/che_per_step` | 越低越好 | 每个联合 step 的平均 GT 暴露 |
| `risk/critical_steps` | 越低越好 | 出现严重连续风险的联合 step 数量 |

不再公开 `CHE_time`、`CHE_mean`、`path_risk`、`peak_risk`、
`danger_time_ratio` 等高度相关的 episode 指标，避免同一风险轨迹产生过多可
挑选的数字。逐步 flame/temperature/smoke/risk 和 hard 状态仍保留在 trace 中
用于诊断。

`risk_summary.json` 同时给出 `team` 与 `per_agent`，并记录
`metric_version=fireworld-risk-v4`、`joint_steps`、`exposure_samples`、
early-stop 阈值与
触发事件、run/rank/seed/plan/clock、planner/smoke/
geometry source（以及 FireScene 可用时的 scene id）。`planner_events` 还报告
safe refusal、emergency escape 和 trapped step；`main.py` 只把
`risk/che_per_step`、`risk/critical_steps` 与 `risk/safe_success` 加入 episode
聚合日志。

导航 `resume_state.json` 同时写入 `metric_contract=fireworld-risk-v4`。
旧 fire run 没有该字段或仍使用 v2/v3 时，续跑会明确报错；请更换 study/run id
从 episode 1 重跑。

### 6.3 与 Habitat Lab metric 的关系

当前实现是在 episode 结束后，把三个风险字段合并进 `main.py` 已取得的 Habitat
metrics；因此名字和聚合方式稳定，但它们还不是 Habitat registry 中真正的
`Measure`，也不会直接出现在裸 `env.get_metrics()` 结果中。

可以进一步原生化：注册 `RiskExposure` 与 `SafeSuccess` 两个 `Measure`，让前者
在 task measurement lifecycle 中持有独立 GT evaluator，后者依赖 Habitat
`Success` 和 `RiskExposure`。迁移时必须停用 `RiskRuntime` 中现有的 episode
累加器，避免同一个 action 被两套 evaluator 重复累计。原生化不应改写 Habitat
已有的 `Success`、`SPL`、`SoftSPL` 或 `DistanceToGoal`。

## 7. 当前限制与论文报告要求

1. **不是 CFD。** FireWorld 是确定性体素扩散/浮力/反应近似，不是经过验证的
   Navier–Stokes/燃烧 CFD，也不输出可用于真实消防决策的物理安全保证。
2. **阈值是 benchmark calibration。** `35/150°C` 与可选的 `250°C` hard
   veto 等阈值用于模拟
   实验分级，不代表人体可生存时间、烧伤阈值或装备认证界限。
3. **Smoke 不是毒性。** FireWorld smoke 是 `[0,1]` 的无量纲烟尘/消光场；
   `appearance_depth` 更只是能见度 proxy。当前没有 CO、O₂、毒性剂量或呼吸
   生理模型。
4. **Privileged transmittance 必须单列。** Renderer transmittance 来自完整
   模拟器光线积分，不能作为普通机器人可观测量；使用它的结果必须标为
   privileged ablation。
5. **风险层仍是按楼层投影的 2D 近似。** `oracle` 在每一步为每个 agent 以其
   当前 Habitat y 单独执行 body-height vertical max projection；非 GPT global
   planner 的 robot-local frontier score、各自 local planner 和 GT evaluator 都
   使用与该 agent 匹配的 RiskLayers，避免相同 x-z 位置的跨层 hazard 泄漏。
   GPT 仍有一个 shared frontier namespace，因此其 global hazard report 使用各
   agent floor 的保守并集，但 local execution 继续使用各自楼层。`none` 不把 GT
   图交给 planner，只按各 agent 当前 y 评估暴露。只有 `sensed` 仍是 episode
   初始 floor 上的一张共享融合 belief；agent 偏离该平面超过
   `--risk_max_floor_deviation_m` 会 fail-fast。这些都是多张 2D 投影，并非连续
   3D 风险规划或显式楼梯拓扑模型。
6. **Hazard depth 仍是近似。** Thermal product 以可见表面温度为主，但 sensed
   管线仍把温度/火焰证据投到观测 depth surface；尚未恢复完整的三维热源深度。
7. **Global 直线路径只是一阶提示。** Frontier `route_risk` 是最近机器人到
   frontier 的直线 approach proxy，可能穿过障碍，也不是 robot-specific 实际
   路径。执行安全性以 FMM 轨迹和 GT exposure metrics 为准。
8. **CHE 是终止前单位-step均值。** v4 CHE_per_step 不再随 episode action 数或
   agent 数线性增长，但 early-stop 会缩短观测窗口，所以低 CHE 仍不能单独证明
   方法更好；主表必须同时保留 SR、SPL、SafeSuccess 与 CriticalSteps，并固定
   FireWorld plan/seed、动作定义和 step clock。每个 episode 保存整数
   `critical_steps`；跨 episode 的 `aggregate.json` 显示其 episode 宏平均。
   wallclock 只适合作为单列 latency stress test。
9. **入口范围。** 完整 RiskRuntime 当前只接在 `main.py`。`main_vec.py` 与
   `ros_multi_nav.py` 会拒绝 `--risk_enabled=1`，避免生成没有同步风险图/GT
   evaluator 却被误标为 risk-aware 的结果。

推荐论文表格至少记录：FireWorld plan/seed、clock mode 与速率、risk source、
smoke source、两个风险权重、温度与 hard thresholds、`alpha`、地图分辨率和
agent 数。不要仅写“hazard-aware”，否则无法区分 evaluator-only、privileged
oracle 与真正的 sensed policy。

## 8. 代码索引与验证

| 路径 | 职责 |
| --- | --- |
| `utils/risk/config.py` | 单位、权重、阈值与 source contract |
| `utils/risk/projection.py` | 温度归一化、风险融合、GT 投影、sensor 反投影 |
| `utils/risk/estimation.py` | 非 privileged smoke visibility proxy |
| `utils/risk/map.py` | 多机器人 sensed belief、衰减与 planning cost |
| `utils/risk/providers.py` | privileged FireWorld GT provider |
| `utils/risk/frontier.py` | frontier report、team utility、VLM 后置安全 guard |
| `utils/risk/metrics.py` | 独立 GT exposure evaluator |
| `utils/risk/visualization.py` | 六 panel 风险 snapshot |
| `utils/risk/runtime.py` | episode orchestration、输出与 oracle/sensed 隔离 |
| `utils/fmm_planner.py` | 风险速度场、hard mask 与 emergency escape |
| `utils/visualization.py` | 导航 obstacle-map panel 的 hazard overlay 与图例 |
| `main.py` | sensor → risk → global planner → local planner → evaluator 接线 |

相关窄测试可用以下命令运行：

```bash
python -m unittest -v \
    tests.test_risk_core \
    tests.test_risk_estimation \
    tests.test_risk_frontier \
    tests.test_risk_metrics \
    tests.test_risk_planner \
    tests.test_risk_runtime \
    tests.test_risk_visualization \
    tests.test_hazard_overlay \
    tests.test_risk_integration
```
