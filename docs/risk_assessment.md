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
        │                                          CHE / critical violations
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
再次加权会重复计算同一危险。火焰仍保留为第 2.2 节的硬约束和安全膨胀源，
所以这不是忽略明火，而是把“不可穿越的火”与“可连续累计的热/烟暴露”分开。
感知不确定性也只进入规划代价，不混入 GT 暴露指标，从而把危险本身和机器人
对危险的未知程度分开。

### 2.2 硬不可通行区域

连续风险以外，代码还构造硬约束：

```text
M_hard = dilate(F >= F_hard, ceil(d_safe / map_resolution))
         OR (T >= T_hard)
```

CLI 默认 `F_hard=0.20`、火焰安全膨胀距离 `d_safe=0.45m`、
`T_hard=250°C`。`M_hard` 不会被低风险样本平均掉：它会过滤 frontier、
从 FMM 可通行域中移除对应栅格，并用于 critical violation 计数。

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

若 **frontier 区域本身**接触 `M_hard`，或其最大规划风险达到
`--risk_hard_frontier_threshold`，则先将它硬过滤。直线 route proxy 仍进入
连续风险代价，但因为它可能穿墙，明确设置 `route_is_proxy=true`，不作为硬 veto
依据。风险模块不再重写 `nearest/co_ut/fill` 的策略公式。每个 normal planner
先提供自己的原始 preference `B_a,k`，再由共享 `SharedRiskAwareness` 在每个
机器人内部将它单调归一化为 `B_bar_a,k`，并统一叠加安全代价：

```text
u_a,k = B_bar_a,k - w_R R_k - 0.5 (1-C_bar_k)
```

`w_R` 由 `--risk_frontier_weight` 控制，默认 `2.0`。四个 classical planner
的 normal preference 与安全层关系如下：

| `nav_mode` | normal preference `B_a,k` | 共享 risk-awareness |
| --- | --- | --- |
| `nearest` | `-robot_frontier_distance` | hard filter、`-w_R R_k`、`-0.5 uncertainty` |
| `co_ut` | `frontier_size - lambda * distance` | 同上 |
| `fill` | `target_score`（缺失时使用 inverse distance） | 同上 |
| `random` | 可复现的 reachable map-goal sampling | 同一模块先生成 safe sampling domain |

因此零风险、完整 confidence 且没有 hard cell 时，四种模式与各自 normal
planner 的目标完全一致；risk 只作为横切安全模块加入，不再把 `fill` 改写成
另一套 size/distance/redundancy 策略。`random` 不使用 frontier utility：它仍在
每个机器人的已探索可达自由空间内按原 seed 采样，只由共享模块提前剔除
hard-unsafe 和超过 danger threshold 的 cell。

`gpt` 模式还会把同一 hazard report 作为 JSON 交给 VLM，但安全性不依赖 VLM
服从提示：确定性 guard 会拒绝不存在、格式错误或 `hard_blocked` 的选择，并只
允许安全的 deterministic assignment 作为 fallback。若没有安全 frontier，系统
在已探索可导航区域内选择低风险 safety waypoint。

当前 global `route_risk` 有意只作为 **approach proxy**：`main.py` 从离该
frontier 最近的机器人到 frontier 画一条单栅格宽直线。这条线不是障碍约束路径、
不是 navmesh shortest path，也不是之后 FMM 实际执行的轨迹，而且 report 不是
逐 robot 路线。因此论文不能把它称为真实路径累计暴露；真实执行暴露应使用
第 6 节的 GT `CHE` 与逐步 exposure trace。

### 4.2 Local hazard-aware FMM

启用 oracle/sensed 风险规划后，agent 不再优先走 Habitat navmesh shortest
path，而是始终进入共享栅格上的 risk-aware FMM。连续风险通过速度场调制：

```text
v(x) = 1 / (1 + alpha · P(x))
```

其中 `alpha=--risk_alpha`，默认 `4.0`；风险越高，travel time 越大。
该形式对应 VULCAN Eq. (11)，但这里的 `P(x)` 明确包含本实现的温度/烟雾风险与
unknown/uncertainty planner penalty；火焰只通过 `M_hard` 进入。
`M_hard` cell 则直接从 traversible domain 中移除。如果 agent 已被动态更新的
hard region 包围，planner 会先建立一条局部 emergency escape corridor；如果
目标周围全部不安全，则转向最近安全 waypoint，且到达该 safety waypoint 不会
被当成 ObjectNav STOP。

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
| `--risk_temperature_hard_c` | `250.0` | 温度硬不可通行阈值（°C） |
| `--risk_flame_hard_threshold` | `0.20` | 火焰硬阈值 |
| `--risk_flame_safety_distance_m` | `0.45` | 火焰硬区域膨胀距离（m） |
| `--risk_danger_threshold` | `0.55` | danger 暴露与 frontier severity 阈值 |
| `--risk_critical_threshold` | `0.80` | critical violation 的连续风险阈值 |
| `--risk_decay_tau_s` | `20.0` | sensed 物理证据衰减时间常数（s） |
| `--risk_confidence_decay_tau_s` | `30.0` | sensed confidence 衰减时间常数（s） |
| `--risk_unknown_risk_prior` | `0.25` | 未观测空间 prior |
| `--risk_uncertainty_weight` | `0.25` | sensed uncertainty 规划权重 |
| `--risk_sensor_stride` | `4` | 传感器图像反投影像素步长 |
| `--risk_floor_min_offset_m` | `0.0` | GT 垂直投影带下界，相对初始 floor y（m） |
| `--risk_floor_max_offset_m` | `1.50` | GT 垂直投影带上界，相对初始 floor y（m） |
| `--risk_smoke_source` | `appearance_depth` | `appearance_depth` / `privileged_transmittance` |
| `--risk_geometry_depth_source` | `clean` | `clean` smoke-robust geometry surrogate / `smoke` 退化深度消融 |
| `--risk_alpha` | `4.0` | FMM 风险速度惩罚强度 |
| `--risk_frontier_weight` | `2.0` | global frontier utility 的风险权重 |
| `--risk_hard_frontier_threshold` | `0.80` | frontier/approach 最大规划风险硬过滤阈值；运行时至少不低于 danger threshold |
| `--risk_dump_dir` | `./outputs/risk_assessment` | 风险 artefact 根目录 |
| `--risk_save_every` | `10` | 每 N 个导航 step 保存 PNG；`0` 只关闭 PNG，不关闭 JSON trace/summary |
| `--risk_max_floor_deviation_m` | `0.75` | agent 离开当前 2D floor 超过该距离时 fail-fast |
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
    ├── risk_step_00000.png        # 由 --risk_save_every 控制
    ├── risk_step_00010.png
    └── risk_summary.json
```

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

### 6.2 推荐 benchmark 指标

Evaluator 在 reset 后先 `prime` 初始位置，但不把 reset 算作 action sample；之后
每次 `env.step` 对所有 agent 在同一个 fire timestamp 采样一次。令
`H_a,t=H_GT(x_a,t)`：

```text
CHE = sum_a sum_t H_a,t
SafeSuccess = Habitat Success AND critical_violations == 0
```

`CHE` 对应 VULCAN Eq. (14) 的离散累计形式，越低越好。
`critical_violations` 定义为 `H >= --risk_critical_threshold` 或采样位置落在
`M_hard` 的次数；它保留在 `risk_summary.json` 中用于解释 `SafeSuccess`，但不
作为主表独立列。

推荐主表只报告四项：

| 指标 | 方向 | 目的 |
| --- | --- | --- |
| Habitat `Success` | 越高越好 | 是否完成 ObjectNav 任务 |
| Habitat `SPL` | 越高越好 | 成功率与路径效率 |
| `risk/safe_success` | 越高越好 | 成功且全过程无 critical violation |
| `risk/che` | 越低越好 | 所有 agent、所有执行动作的累计 GT 暴露 |

不再公开 `CHE_time`、`CHE_mean`、`path_risk`、`peak_risk`、
`danger_time_ratio` 等高度相关的 episode 指标，避免同一风险轨迹产生过多可
挑选的数字。逐步 flame/temperature/smoke/risk 和 hard 状态仍保留在 trace 中
用于诊断。

`risk_summary.json` 同时给出 `team` 与 `per_agent`，并记录
`metric_version=fireworld-risk-v2`、run/rank/seed/plan/clock、planner/smoke/
geometry source（以及 FireScene 可用时的 scene id）。`planner_events` 还报告 safe refusal、emergency escape 和
trapped step；`main.py` 只把 `risk/che` 与 `risk/safe_success` 加入 episode
聚合日志。

### 6.3 与 Habitat Lab metric 的关系

当前实现是在 episode 结束后，把两个风险字段合并进 `main.py` 已取得的 Habitat
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
2. **阈值是 benchmark calibration。** `35/150/250°C` 等默认阈值用于模拟
   实验分级，不代表人体可生存时间、烧伤阈值或装备认证界限。
3. **Smoke 不是毒性。** FireWorld smoke 是 `[0,1]` 的无量纲烟尘/消光场；
   `appearance_depth` 更只是能见度 proxy。当前没有 CO、O₂、毒性剂量或呼吸
   生理模型。
4. **Privileged transmittance 必须单列。** Renderer transmittance 来自完整
   模拟器光线积分，不能作为普通机器人可观测量；使用它的结果必须标为
   privileged ablation。
5. **单楼层 2D 近似。** Runtime 以 episode 初始 agent y 固定一个导航平面；
   GT/oracle 只在配置的 body-height band 做垂直 max projection，sensed evidence
   也先按同一高度带过滤再写 x-z 栅格。当前没有多楼层状态或楼梯切换模型；任一
   robot 偏离初始 floor 超过 `--risk_max_floor_deviation_m` 会 fail-fast，而不是
   静默把楼上/楼下 hazard 合并。
6. **Hazard depth 仍是近似。** Thermal product 以可见表面温度为主，但 sensed
   管线仍把温度/火焰证据投到观测 depth surface；尚未恢复完整的三维热源深度。
7. **Global 直线路径只是一阶提示。** Frontier `route_risk` 是最近机器人到
   frontier 的直线 approach proxy，可能穿过障碍，也不是 robot-specific 实际
   路径。执行安全性以 FMM 轨迹和 GT exposure metrics 为准。
8. **CHE 依赖动作采样。** `CHE` 随 episode action 数和 agent 数增长；主
   benchmark 必须固定 agent 数、最大 action budget、动作定义、FireWorld plan/
   seed 和 step clock。wallclock 只适合作为单列的 latency stress test，不能与
   step-clock 主表混合。
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
    tests.test_risk_integration
```
