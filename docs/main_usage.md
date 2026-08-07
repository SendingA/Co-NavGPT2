# `main.py` 运行参数与键盘操控速查

> **Habitat-Lab 0.3.3 升级说明（2026-07）**：项目已从 Habitat 0.2.1 (YACS) 迁移到 Habitat-Lab 0.3.3 (Hydra + `omegaconf.DictConfig`)。所有 `--task_config` 现在指向 Hydra yaml；`arguments.load_config` 会把该文件 compose 成完整 DictConfig。多智能体不再依赖 `multi-robot-setting/` 补丁，改用官方 `agents_order` + `agents.<name>` schema。新增 `--num_humans`、`--robot_models_enabled`、`--robot_profiles` 等控制机器人+人体的开关。完整变更列表：`docs/habitat3_migration.md`。已废弃：`--exp_name`、`--log_interval`、`--agent`（早在 2026-06 就无人读；文档里保留仅为兼容记录）。
>
> 适用范围：Co-NavGPT2 仓库根目录的 `main.py`（多智能体 nav 主入口），以及 `scripts/keyboard_teleop.py` / `scripts/keyboard_teleop_fire.py`（手动遥操作）。
>
> 注意：`main.py` 本身**没有**键盘操控接口，跑的是全自动 nav 循环。键盘控制位于 `scripts/keyboard_teleop*.py`，本文一并整理。

---

## 1. 启动方式

```bash
# 多智能体 nav 自动循环（GPT/最近/合作/覆盖几种调度策略）
python main.py [选项...]

# 单智能体手动遥操作（无火灾叠加）
python scripts/keyboard_teleop.py --task-config configs/multi_objectnav_hm3d.yaml \
    --num-agents 1 --agent-id 0 --scene-id Nfvxx8J5NCo --show-depth 1

# 单智能体手动遥操作（带 FireWorld 体素火灾叠加）
python scripts/keyboard_teleop_fire.py --task-config configs/multi_objectnav_hm3d.yaml \
    --scene-id Nfvxx8J5NCo --plan-id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --steps-per-unit 5 --seconds-per-unit 2.0 --depth_use_clean 1
```加了内部行为提示（--visualize 起的是 Open3D GUI 子线程、--lidar_360 仅在 --fire_world=1 时生效、

---

## 2. `main.py` CLI 参数（来自 `arguments.py`）

### 2.1 基础

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--seed` | int, `1` | 随机种子 |
| `-d`, `--dump_location` | str, `./tmp` | 日志/模型 dump 目录 |
| `-v`, `--visualize` | int, `0` | `1` 渲染观测和预测语义图（会启动 Open3D GUI 线程） |
| `--print_images` | int, `0` | `1` 把可视化结果保存为图片 |
| `--max_episodes` | int, `0` | `main.py` 最多评测多少个 dataset episodes；`0` 表示全部。它不改变每个 episode 的 500-step action budget |

### 2.2 环境/数据集/episode

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `-fw`, `--frame_width` | int, `640` | 帧宽 |
| `-fh`, `--frame_height` | int, `480` | 帧高 |
| `--task_config` | str, `multi_objectnav_hm3d.yaml` | 在 `configs/` 下的任务 yaml |
| `--hfov` | float, `79.0` | 水平视场角（度） |

### 2.3 模型超参

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--num_local_steps` | int, `25` | 两次 global step 之间的 local 步数 |
| `-n`, `--num_processes` | int, `1` | 进程数 |
| `--rank` | int, `0` | 进程 rank |
| `--gpu_id` | int, `0` | Habitat-sim 使用的 GPU id |
| `--map_resolution` | int, `5` | 占据栅格分辨率（cm/格） |
| `--map_size_cm` | int, `2400` | 地图边长（cm） |
| `--map_height_cm` | int, `130` | 地图剖面高度（cm） |
| `--sem_threshold` | float, `0.85` | 语义检测置信阈值，超过即认为 found_goal |
| `--num_agents` | int, `2` | 同场景中的机器人 agent 数量 |

### 2.3.1 Habitat 3 新增：人形行人 + 可见机器人 URDF

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--num_humans` | int, 配置值 | 覆盖 `conav.num_humans`；普通配置默认 0，静态 person benchmark 默认 1 |
| `--robot_models_enabled` | int, `0` | `1` 在 nav agent 上叠一层可见的 Habitat 3 机器人 URDF |
| `--robot_profiles` | str | 逗号分隔的机器人 profile：`fetch,fetch_no_wheels,fetch_suction,spot,stretch`；多机器人按顺序循环 |
| `--robot_urdfs` | str | 可选：直接指定 URDF 路径覆盖 profile 默认路径 |
| `--dataset_path` | str | 覆盖 `habitat.dataset.data_path`（比如切到 `objectnav_hm3d_v2`） |
| `--scenes_dir` | str | 覆盖 `habitat.dataset.scenes_dir` |
| `--scene_dataset` | str | 覆盖 `habitat.simulator.scene_dataset` 指向的 `scene_dataset_config.json` |
| `--config` | str | 直接给 Hydra yaml 的完整路径，绕过 `--task_config` 的相对拼接 |

### 2.4 全局调度策略 / GPT

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--nav_mode` | `nearest` / `co_ut` / `fill` / `random` / `gpt`（默认 `gpt`） | 全局目标选择策略：最近前沿 / Cost-Utility / 高得分覆盖 / 随机长期目标 / GPT-4o 决策 |
| `--cost_utility_lambda` | float, `1.0` | `co_ut` 的距离系数；逐机器人最大化 `frontier_size - λ × robot_distance`，其中 size 和 distance 分别以 frontier cell 数和 map cell 为单位 |
| `--random_goal_min_distance_m` | float, `1.0` | `random` 优先采样的最小目标距离；有效候选不足时退化到同一可达自由空间内的任意 cell |
| `--fill_mode` | int, `0` | `fill` 模式细分（保留位） |
| `--gpt_type` | int, `2` | `0=text-davinci-003`，`1=gpt-3.5-turbo`，`2=gpt-4o`（默认），`3=gpt-4o-mini` |

### 2.5 火灾多模态传感器

传感器套件（含噪深度 / radar / lidar / thermal / dashboard）随 `--fire_world=1` 自动启用；RGB 与 Thermal 均由 FireWorld 体素 ray-march 生成。

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--fire_apply_to_obs` | int, `1` | `1` 把退化后的 RGB+Depth 写回 obs 供建图；`0` 仅保存退化数据但 nav 用干净 obs |
| `--smoke_density` | float, `0.6` | `[0,1]`，烟雾光学厚度。`0` 完全清晰，`1` 能见度 ≈1m |
| `--fire_dump_dir` | str, `./outputs/fire_sensors` | 每步传感器图像输出目录 |
| `--fire_save_every` | int, `1` | 每 N 步保存一次（`1` 每步都存） |
| `--fire_save_npz` | int, `0` | `1` 同时 dump 原始 numpy `.npz` |
| `--fire_show_window` | int, `0` | `1` 打开 OpenCV 2x4 实时仪表盘 |
| `--lidar_360` | int, `0` | `1` 给每个 agent 安装 4 个偏航深度相机（前/左/后/右），由 LIDAR 模块拼接 360° 点云（仅在 `--fire_world=1` 时生效） |
| `--lidar_resolution` | int, `320` | 每片深度图分辨率（HxW，越小越快） |

### 2.6 烟雾场景感知开关

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--depth_use_clean` | int, `-1` | 只控制**写回建图的深度**：`1` 用干净深度，`0` 用烟雾退化深度，`-1`(默认) 有火场时自动用干净深度。**不影响** RGB/Thermal 火焰渲染与 dashboard——它们始终对干净几何深度做 ray-march，所以 `0/1` 下火焰烟雾观感一致 |
| `--use_thermal_perception` | int, `1` | `1` 把热成像 flame 掩码注入观测，并让检测器从热像读 fire（烟不影响） |

### 2.7 FireWorld 运行时（3D 体素 ray-march 路径）

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--fire_world` | int, `0` | `1` 启用 FireWorld：以 3D 体素 ray-march 合成 RGB/Thermal，并自动挂上传感器套件 |
| `--fire_world_plan_id` | str, `None` | 可选的显式语义 plan id，格式为 `<scene>_<type>_<intensity>_<12位hash>`；省略或设为 `auto` 时按当前 episode scene 自动关联可播放 plan |
| `--fire_world_intensity` | `light\|medium\|severe`, `medium` | 自动关联 plan 时使用的严重程度；默认 `medium` |
| `--fire_world_fire_type` | `auto\|multi_origin\|kitchen_grease_fire\|bedroom_textile\|living_room_electric`, `auto` | 自动关联的火灾类型；`auto` 固定优先级为 multi-origin、kitchen、bedroom、living-room |
| `--fire_world_scenes_root` | str, `scenes` | inventory.json + plan.json 所在根目录 |
| `--fire_world_out_root` | str, `outputs/fire_world` | timeline.npz 根目录（`<root>/<scene>/<plan_id>/timeline.npz`） |
| `--fire_steps_per_unit` | int, `5` | 多少机器人步推进 1 单位“火灾时间”，越大火越慢 |
| `--fire_seconds_per_unit` | float, `2.0` | 每单位火灾时间消耗多少秒火灾时间线（默认 5 步 → 推进 2s） |
| `--fire_world_smoke_k_ext` | float, `4.0` | 烟雾体素的消光系数倍率（每米），越大越不透明 |
| `--fire_world_n_steps` | int, `24` | ray-march 每像素采样数 |
| `--fire_world_render_scale` | float, `0.5` | 体积积分渲染相对相机分辨率的比例，`0.5` 约 4× 加速，`1.0` 全分辨率 |
| `--fire_fast` | int, `1` | benchmark 快速模式：关闭 procedural noise，并把采样数/渲染比例限制到 `10`/`0.35`；`0` 启用有火舌纹理、颜色扰动和时间闪烁的高质量火焰 |
| `--fire_render_backend` | `auto\|numpy\|torch`, `auto` | `auto` 在 CUDA 可见时使用 Torch GPU ray-march，否则沿用 NumPy；可显式固定后端 |
| `--fire_render_device` | str, `auto` | FireWorld Torch 设备，如 `cuda:0`、`cuda:1` 或 `cpu`；与 Habitat-Sim 的 `--gpu_id` 独立 |
| `--fire_render_dtype` | `float16\|float32`, `float16` | GPU 常驻体素精度；透射率/颜色积分仍为 FP32，Torch CPU 路径自动使用 FP32 |
| `--fire_render_max_sample_points` | int, `2000000` | 每个 GPU tile 最多包含的射线采样点；显存不足时调低 |

Torch 后端只缓存当前 timeline frame，而不是把完整 `timeline.npz` 放入显存；同一场景、同一时间帧的多机器人共享该缓存。输出字典同时记录 `fire_render_backend`、`fire_render_device`、frame index 和 cache hit/upload 计数，便于确认 benchmark 实际使用的后端。若 CUDA 初始化或 kernel 执行失败，会发出一次警告并回退到 NumPy。

> 备注：`--gpu_id` 只控制 Habitat-Sim。FireWorld 使用独立的 `--fire_render_device`；`args.cuda` 仍由 `torch.cuda.is_available()` 自动推导。

RGB 火焰使用独立的烟雾散射/火焰辐射积分：高温火焰会局部降低烟雾消光，
火焰高光会保留色彩比例地压缩，并最多透出约 `13%` 的原始表面纹理。因此已
点燃的床、沙发或桌子仍能从火焰与烟雾中辨认，不会被替换为纯白光团。该
变化只影响 RGB 可视化，不会改变 timeline、thermal temperature、风险地图
或 benchmark ground truth。

烟雾散射色默认为灰黑色 `(72, 72, 72)`。`--print_images 1` 的检测图仍显示
`fire` 的 bounding box 和置信度，但不会再用绿色语义 mask 覆盖体积火焰。
如果需要与 teleoperation 相同的火舌纹理、边缘扰动和时间闪烁，请在导航命令
中使用 `--fire_fast 0`；默认的 `--fire_world_n_steps 24` 和
`--fire_world_render_scale 0.5` 会保留完整采样设置。benchmark 默认的
`--fire_fast 1` 优先速度。

### 2.8 生成 plan 时指定点火源数量

`utils.fire_world.planner` 支持 `--num_ignitions N`，用于严格指定 plan
中 **`t=0` 同时点燃的 initial ignition objects 数量**。planner 的职责到此
为止：它不会预选 secondary objects，不会写入 delay、parent-child 关系或
传播通道。之后哪些家具达到点燃条件，完全由 propagation solver 根据 voxel
温度、辐射、接触、可燃性和几何关系在积分过程中决定。因此显式设置 `N` 时，
plan 的 `ignitions` 数组严格只有 N 个条目。

不传该参数时仍然使用 initial-only 逻辑，只是 initial source 数量由
intensity preset 决定：light 1 个、medium 1–2 个、severe 2–3 个。

```bash
python -m utils.fire_world.planner \
    --scene Nfvxx8J5NCo \
    --fire_type bedroom_textile \
    --intensity severe \
    --seed 7 \
    --num_ignitions 4
```

生成器会打印新的 count-specific `plan_id`，并在 JSON 中写入
`num_initial_ignitions_requested`；所有 plan 都会记录实际的
`num_initial_ignitions`、`ignition_selection_mode = "initial_only"` 和
`ignition_selection_version`。每个点火条目均标记
`ignition_role = "initial"` 且 `ignite_time_s = 0`。如果 template 类别中
没有足够的 initial objects，命令会明确失败而不会静默减少数量。生成 plan
后必须使用新 `plan_id` 重新运行 propagation；单独修改或生成 JSON 不会改变
已有的 `timeline.npz`。

Template v10 默认使用低矮、连续且有硬边界的径向火焰场。可见 flame
column 的米制硬上限在 light/medium/severe 下分别为
`0.20/0.30/0.35 m`；由于实际高度必须是完整 voxel 层，在默认 `0.15 m`
网格上对应 `0.15/0.30/0.30 m`。地板火焰分别限制在距所属 source 约
`1.43/2.20/2.97 m` 内。`0.0033/0.0044/0.0055 m/s` 是各强度的最低扩张
速度；每个 source 还会根据剩余 `duration_s` 计算一个恒定有效速度，保证最迟
在剩余时长的 `90%` 处达到自己的最大半径。每个
solver step 都会先清除上一帧没有 fuel 支撑的可视 flame column，因此它
不会逐帧叠加到天花板，也不会作为额外辐射源继续放大火场。
相较 v9，v10 保留扩大的径向危险区域和 duration deadline，同时增加与
voxel 分辨率无关的垂直硬上限；它仍保留局部表面扩散、四 voxel 辐射预热、两 voxel
接触点燃和最大半径硬裁剪。地板前沿会先预热、
再渐显、最后进入 reaction；家具六连通前沿也会在相邻 voxel 层之间连续
插值，不再约每 `18.75s` 整层跳变。
物体自然点燃使用稍宽的动态局部域：light/medium/severe 最大半径约
`1.19/1.35/1.51 m`，扩张速度约 `0.0050/0.0066/0.0083 m/s`。这个域只限制
可参与 reaction 的空间，不指定 object id；域内究竟点燃 chair、pillow、
cabinet 或其他 fuel object，仍由 solver 的温度场决定。
物体首次出现满足温度和 flame 条件的 voxel 后，火焰会在该物体自己的
`object_id_field` mask 内以约 `0.006/0.008/0.010 m/s` 的速度进行
六连通逐层填充；medium 在 `0.15m` voxel 下约每 `18.75s` 推进一层。
动态局部域负责决定物体能否首次点燃；一旦点燃，front 可以完成该物体自身
低处的 mask，但不会跨到相邻 object ID，也不会据此点燃域外的新物体。
为避免 curtain/cabinet 等高大 AABB 把可见火焰带到顶部，该可视填充相对
物体最低 occupied voxel 的垂直范围被限制为 light/medium/severe
`0.55/0.75/0.90 m`。该限制不裁剪物理温度场或烟雾场。
`multi_origin` 的 initial 只从低矮落地家具
（床、沙发/座椅、桌子/书桌、地毯、脚凳）中选择，不再把高柜、灯、
墙面电视、枕头或 laptop 当成初始火点；这些对象仍可被传播器自然点燃。

`limit_flame_to_source_envelope = 1` 会把合成地板/空气 flame 限制在上述
source-centered XZ 包络中，并将没有物体 fuel 支撑的最高可见火焰限制为
source core 上方的米制 flame-column 上限。真实 inventory object voxel 不受这个
地板 XZ 裁剪：一旦传播器将局部域内的附近家具加热到
`ignition_temp_c`，其火焰在地板包络外仍然可见；物体火焰允许比短 plume
额外高 `0` 个 voxel。温度场和烟雾场不受包络或可见火焰高度限制。

在每个 source 的局部范围内，地板火焰强度使用高斯分布
`G_r(d,t)=exp(-d²/(2·(0.58·r(t))²))`：中心更亮，边缘随距离平滑变暗。
对 source `i`，设目标半径 `R_i = floor_max_spread_radius_m ×
floor_spread_scale_i`，有效速度为
`v_i = max(v_preset, (R_i-r_i(0))/(0.9×(duration_s-t_ignite_i)))`，
然后 `r_i(t)=min(R_i, r_i(0)+v_i×source_age)`。因此传播速度恒定、半径单调
增大，并保证在 duration 结束前到达硬边界。不存在 planner 构造的定向连接线；
家具之间是否传播仍由 reaction、conduction 和 local radiation 共同决定。

仓库中已按上述 v10 规则为 `Nfvxx8J5NCo` 生成一个四源 medium
`multi_origin` 示例 plan：`Nfvxx8J5NCo_multi_origin_medium_0641161af604`。历史 plan 与 timeline 不会被
覆盖；对应 timeline 已输出到
`outputs/fire_world/Nfvxx8J5NCo/Nfvxx8J5NCo_multi_origin_medium_0641161af604/timeline.npz`。需要重建时运行：

```bash
python -m utils.fire_world.propagation \
    --scene Nfvxx8J5NCo \
    --plan_id Nfvxx8J5NCo_multi_origin_medium_0641161af604 \
    --voxel_m 0.15 --dt 0.5 --save_dt 1.0
```

### 2.9 一键准备单场景或数据集 FireWorld assets

为一个场景生成四个 template × 三种 intensity 的完整资产：

```bash
python scripts/prepare_fire_world_scene.py \
    --scene Nfvxx8J5NCo \
    --fire-types all --intensities all --seeds 42
```

批量执行前可用只读 dry-run 检查可执行场景、类别缺失和空间上限：

```bash
python scripts/prepare_fire_world_dataset.py \
    --dataset-root data/scene_datasets/hm3d_v0.2 --splits val \
    --fire-types all --intensities all --seeds 42 \
    --jobs 1 --dry-run
```

正式执行并断点续跑：

```bash
python scripts/prepare_fire_world_dataset.py \
    --dataset-root data/scene_datasets/hm3d_v0.2 --splits val \
    --fire-types all --intensities all --seeds 42 \
    --jobs 2
```

脚本保留已有的 `scenes/<scene>/plans/<plan_id>.json` 和
`outputs/fire_world/<scene>/<plan_id>/timeline.npz` 路径，因此生成结果可以
直接传给 `--fire_world_plan_id`。有效 timeline 会被自动 resume；新文件先在
`.staging` 中烘焙和验证，再原子安装。每次批量运行的 task 状态、错误、
checksum、日志和汇总保存在 `outputs/fire_world/runs/<run_id>/`，每个场景的
可用 plan 索引保存在 `outputs/fire_world/<scene>/asset_index.json`。

省略 `--fire_world_plan_id` 时，导航会在创建 Habitat 环境前只保留具备
匹配 plan JSON 和 `timeline.npz` 的场景，并在每次 scene 切换后重新绑定
该 scene 的 plan。相同 fire type 内选择 `template_version` 最新的可播放
plan；显式 plan ID 则不会套用 intensity/type 筛选。当前本地资产中只有
`Nfvxx8J5NCo` 和 `TEEsavR23oF` 具备 medium timeline，因此 person 数据集
自动模式目前对应 40 个 episodes，其余场景需先运行上述 FireWorld pipeline。

person 数据集、GPT global planner、FMM local planner、sensed risk 和保存图像：

```bash
python main.py \
    --task_config person_objectnav_hm3d.yaml \
    --nav_mode gpt --local_planner fmm \
    --fire_world 1 --fire_world_intensity medium \
    --fire_clock_mode step \
    --risk_enabled 1 --risk_source sensed \
    --depth_use_clean 1 --use_thermal_perception 1 \
    --print_images 1 \
    --dump_location outputs/benchmarks/person_fire_gpt_fmm_medium/navigation \
    --fire_dump_dir outputs/benchmarks/person_fire_gpt_fmm_medium/fire_sensors \
    --risk_dump_dir outputs/benchmarks/person_fire_gpt_fmm_medium/risk \
    --risk_run_id person_fire_gpt_fmm_medium_seed1
```

---

## 3. `main.py` 内部行为约定

- `--visualize=1` 时会用 `threading` 启动一个 Open3D GUI 子线程（`utils.vis_gui.ReconstructionWindow`），主线程跑 `main()`，二者通过 `multiprocessing.Queue` 通信。GUI 自身不接受 WASD，是只读重建可视化。
- `--fire_world=1` 时自动挂上传感器套件（含噪深度 / radar / lidar / thermal / dashboard），RGB 与 Thermal 均来自体素 ray-march。
- `--lidar_360=1` 仅在 `--fire_world=1` 时生效（参考 `main.py` 中的条件判断）。
- 全局规划在每个 `args.num_local_steps` 步触发一次；`main.py` /
  `main_vec.py` 只负责重规划时机和 `Frontier_Det`，随后统一调用
  `utils.global_planners.create_global_planner(...).plan(...)`。`gpt` 实现
  内部才会调用 `chat_utils.chat_with_gpt4v`。
- `--nav_mode gpt` 使用当前 robot/frontier 数量动态生成严格 JSON Schema，
  completion 上限为 300 tokens。每次请求的完整状态会静默追加到
  `<dump_location>/logs/gpt/gpt_response_status.jsonl`，记录
  response/request ID、实际 model、`finish_reason`、`refusal`、返回内容、
  token usage 和校验结果（不记录 API key 或候选图像的 base64），不会再向
  stdout 打印 `[gpt-response-status]`。控制台只保留历史兼容的
  `gpt-4o response:` 和非空 JSON response 两行。`refusal` /
  `content_filter` 会立即作为显式失败；空内容、截断或非法 JSON/Schema
  最多重试 5 次。
- GPT 请求最终失败时会输出 `[gpt-fallback]` JSON 并自动使用 `co_ut`。
  normal case 直接使用普通 `co_ut`；risk case 使用 risk-aware `co_ut`
  的信息增益、距离、风险和 hard-risk 过滤语义。该 fallback 只捕获
  `GPTResponseError`，不会掩盖 planner 本身的程序错误。

### 3.1 架构（重要）

经过最近一次重构后，火灾仿真分两层：

- `utils/fire_world/`：**世界模型**。负责火焰/烟雾/温度的 3D 体素时间线、相机位姿换算、机器人步 → 火灾时间映射。核心类：`FireWorld`（数据）、`FireClock`（时钟）、`FireScene`（统一入口，`FireScene.from_args(args, config)` 一行构造好）。
- `utils/fire_sensors/`：**观测层**。负责"如何看见这个世界"——RGB/Thermal 的体素 ray-march（`VoxelSmokeSensor`，以 `FireScene` 为输入），以及含噪深度 / Radar / LIDAR / dashboard。`FireSensorSuite` 接受 `scene=FireScene, camera_K=...` 参数；RGB 与 Thermal 一律走体素相机。
- `utils/fire_pipeline.py::step_fire_observation()`：薄薄的胶水函数，负责调 `suite.process(...)` 并把结果写回 `observations`。

全局规划也采用独立接口：

- `utils/global_planners/base.py`：定义 `GlobalPlannerContext`、
  `RiskPlanningContext`、`GlobalPlannerResult` 和 `GlobalPlanner` 接口。
- `utils/global_planners/{nearest,co_ut,fill,random,gpt}.py`：五种 `--nav_mode`
  的具体实现。
- `utils/global_planners/risk_aware.py`：统一包装五种策略；只有 context
  包含 planning risk 时才进入风险评分、hard-risk 过滤和低风险回退。
- `utils/global_planners/factory.py`：唯一构造入口。增加新的 global
  planner 时，在独立文件实现接口并注册到 factory，不需要修改主循环。

`main.py` 仍然拥有地图更新、frontier 检测、PointNav 请求重规划的确认以及
risk report 的保存；planner 只返回每个机器人本轮的 frontier/map goal 和
风险报告元数据。因此 `--nav_mode` 的命令行用法、重规划周期与现有 benchmark
标签不变。

### 3.2 FireWorld 感知管线

主循环统一调用 `utils/fire_pipeline.py::step_fire_observation()`，`FireSensorSuite` 从绑定的 `FireScene` 上做体素 ray-march：

| `--fire_world` | RGB 来源 | Thermal 来源 | Depth | Radar/LIDAR/Dashboard |
| --- | --- | --- | --- | --- |
| 0 | clean | 无 | clean | 无 |
| 1 | FireWorld 体素 | 体素温度 | 套件噪声深度 | 套件 dashboard，RGB 那格已是 FireWorld 输出 |

要点：
- 套件消费**干净** RGB-D 做深度/雷达/激光退化，FireWorld 单独在体素场上算 RGB/Thermal 的 ray-march。
- RGB/Thermal 的 ray-march 始终使用**干净几何深度**作为光线终点，与 `--depth_use_clean` 无关。这样火焰/烟雾观感在 `--depth_use_clean 0` 和 `1` 下完全一致；该旗位只决定写回**建图**用的是干净还是烟雾退化深度。（此前二者耦合，烟雾退化深度会把光线截在能见度层之前，导致火焰烟雾从渲染中消失。）
- 想保护建图就配合 `--depth_use_clean=1`（写回干净 depth）+ `--use_thermal_perception=1`（让检测器从热像读 fire）。
- 当 `--fire_show_window=1` 时弹出的 viewer dashboard 已经以 FireWorld 渲染结果为基底，能直接看到 agent 实际感知。

---

## 4. 键盘操控

`main.py` 不读键盘。下面两个脚本提供了交互式 WASD 操控。

### 4.1 `scripts/keyboard_teleop.py`（无火灾叠加）

CLI：

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--task-config` | str, **required** | 任务 yaml 路径 |
| `--num-agents` | int, `1` | 场景中 agent 总数 |
| `--agent-id` | int, `0` | 当前键盘控制的 agent id（其它 agent 发送 `STOP`） |
| `--scene-id` | str, `None` | HM3D 短 id（如 `Nfvxx8J5NCo`），用于过滤到匹配 episode |
| `--show-depth` | int, `0` | `1` 在同窗口右侧并排显示彩色深度 |

按键（OpenCV 窗口必须在前台）：

| 按键 | 动作 |
| --- | --- |
| `W` | MOVE_FORWARD |
| `A` | TURN_LEFT |
| `D` | TURN_RIGHT |
| `S` | STOP |
| `Q` | LOOK_DOWN |
| `E` | LOOK_UP |
| `ESC` | 退出 |

未识别的键会被忽略。`POSSIBLE_ACTIONS` 中没有的动作名也会被跳过并打印提示。

### 4.2 `scripts/keyboard_teleop_fire.py`（FireWorld 体素叠加）

主要 CLI：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--task-config` | required | 任务 yaml |
| `--num-agents` | `1` | agent 总数 |
| `--agent-id` | `0` | 控制目标 |
| `--scene-id` | required | HM3D 短 id |
| `--plan-id` | required | FireWorld 语义 plan id（`<scene>_<type>_<intensity>_<12位hash>`），`outputs/fire_world/<scene>/<plan_id>/timeline.npz` 必须已存在 |
| `--scenes-root` | `scenes` | inventory/plan 所在根 |
| `--out-root` | `outputs/fire_world` | timeline 输出根 |
| `--steps-per-unit` | `5` | 多少步推进 1 单位火灾时间 |
| `--seconds-per-unit` | `2.0` | 每单位等价多少秒时间线 |
| `--smoke-k-ext` | `4.0` | 烟雾消光倍率 |
| `--n-steps` | `16` | ray-march 采样数（teleop 推荐 12~16） |
| `--render-scale` | `0.5` | 体积渲染比例 |
| `--depth_use_clean` | `1` | `1` 使用 Habitat 干净深度（推荐） |
| `--save-frames-to` | `None` | 给定路径则按步 dump PNG |
| `--enable-suite` | `1` | 已弃用占位；套件总是挂上（噪声 RGB-D / Radar / LIDAR / Thermal / dashboard） |
| `--smoke-density` | `0.6` | 套件含噪深度 / lidar / radar 的烟雾密度 |
| `--show-dashboard` | `1` | `1` 另开一窗显示套件 2x4 dashboard |
| `--save-npz` | `0` | `1` 同时保存原始 numpy `.npz` |

按键：

| 按键 | 动作 |
| --- | --- |
| `W` | MOVE_FORWARD |
| `A` | TURN_LEFT |
| `D` | TURN_RIGHT |
| `S` | STOP（不移动，但火灾时钟仍 tick） |
| `Q` | LOOK_DOWN |
| `E` | LOOK_UP |
| `R` | 重置 episode，`t_sim` 归零 |
| `ESC` | 退出 |

> 该脚本在窗口顶端叠了 HUD：当前步数、`t_sim`、平均透过率 `T_mean`、火焰像素占比 `flame`、以及 steps/unit 与 s/unit。

---

## 5. 速记常用组合

```bash
# 自动评测，GPT-4o 调度，单 agent，FireWorld 体素烟雾+热像
python main.py --num_agents 1 --nav_mode gpt --gpt_type 2 \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c --smoke_density 0.7 \
    --depth_use_clean 1 --use_thermal_perception 1

# 自动评测，FireWorld 体素，2 agent，可视化保存图片
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --fire_steps_per_unit 1 --fire_seconds_per_unit 5.0 \
    --fire_show_window 1


python main.py \
    --num_agents 2 \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --fire_speedup 2.0 \
    --print_images 1

# FireWorld GPU ray-march（benchmark 快速视觉）
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --fire_render_backend torch --fire_render_device cuda:0 \
    --fire_render_dtype float16 --fire_fast 1

# 高质量 GPU 火焰；显存紧张时降低 max sample points
python main.py --num_agents 1 --nav_mode nearest \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --fire_render_backend torch --fire_render_device cuda:0 \
    --fire_fast 0 --fire_render_max_sample_points 1000000

# 固定 NumPy 参考后端，用于 CPU/GPU benchmark 对照
python main.py --num_agents 1 --nav_mode nearest \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --fire_render_backend numpy



# # 自动评测，FireWorld 完整火灾感知
# python main.py --num_agents 1 --nav_mode gpt \
#     --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
#     --fire_steps_per_unit 1 --fire_seconds_per_unit 5.0 \
#     --smoke_density 0.6 \
#     --depth_use_clean 1 --use_thermal_perception 1 \
#     --fire_show_window 1

# 手动开 FireWorld + 套件，看着自己走、看火长、同时看 dashboard
python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --scene-id Nfvxx8J5NCo --plan-id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --steps-per-unit 1 --seconds-per-unit 5.0 \
    --depth_use_clean 1 --render-scale 0.5 --n-steps 16 \
    --smoke-density 0.6 \
    --show-dashboard 1

python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --scene-id Nfvxx8J5NCo --plan-id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --speedup 2.0 \
    --show-dashboard 1


python scripts/keyboard_teleop_full.py \
    --num-agents 1 --num-humans 2 \
    --robot-models-enabled 1 --robot-profiles spot \
    --scene-id Nfvxx8J5NCo --plan-id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --clock-mode wallclock --speedup 2.0 \
    --lidar-360 1 --lidar-resolution 320 \
    --snapshot-dir outputs/teleop_sensor_snapshots

# Full teleop defaults to four-slice true 360-degree LiDAR.
# Press V or click SAVE SENSOR PANELS in the dashboard header to save:
# clean/smoke RGB-D, thermal, LiDAR BEV, three radar products,
# the dashboard, and manifest.json.
# Use --lidar-360 0 only for the legacy forward-depth fallback.


conda activate co-nav3
cd /home/liushe10/Co-NavGPT2

# 单 agent 快速冒烟测试
python main.py --num_agents 1 --num_humans 0 --nav_mode nearest --fire_world 0 --dump_location /tmp/conav_smoke

# 双 agent + GPT frontier assignment（老功能）
python main.py --num_agents 2 --nav_mode gpt

# 双 agent + 2 个 humanoid pedestrian（新功能）
python main.py --num_agents 2 --num_humans 2

# 双 agent + 可见 Spot URDF 模型（新功能）
python main.py --num_agents 2 --num_humans 2 \
    --robot_models_enabled 1 --robot_profiles spot,fetch \
    --nav_mode nearest \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --fire_show_window 1


python main.py --task_config person_objectnav_hm3d.yaml \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --print_images 1


python main.py \
  --task_config person_objectnav_hm3d.yaml \
  --num_agents 2

```

### Navigation step metric

当前 ObjectNav 配置启用了 Habitat-Lab 原生的 `NumStepsMeasure`。每个 episode
的 `env.get_metrics()` 和最终平均指标中都会包含 `num_steps`：

- episode reset 后从 `0` 开始；
- 每次真正执行一次联合 `env.step(actions)` 增加 `1`，包括送入环境的终止
  `STOP` action；
- 多机器人在同一个 Habitat step 中同步执行，因此一次联合 step 仍然只计
  `1`，不会乘以 `num_agents`。

建议将 `num_steps` 与 `Success`、`SPL` 一起报告，作为与机器性能无关的导航
时间/动作预算代理。它不是 wall-clock 时间：VLM API 延迟、渲染和硬件速度
不会改变 `num_steps`。`main_vec.py` 中的 `episode_runtime`、`fps` 和
`avg_step_time` 可用于分析实际运行耗时。

### 风险评估 benchmark

推荐使用 sensed 风险图和 step clock：

```bash
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_6df964ec1f4c \
    --fire_clock_mode step \
    --risk_enabled 1 --risk_source sensed \
    --risk_smoke_source appearance_depth
```

连续风险只融合温度与烟雾，默认
`H_phys = 0.60 * temperature_risk + 0.40 * smoke_risk`。火焰不再使用连续
权重，但仍作为硬不可通行区域，并向外膨胀 `0.45m`，因此不要再传
`--risk_weight_flame`。

主 benchmark 报告 Habitat `Success`、Habitat `SPL`、Habitat `num_steps`、
`risk/safe_success` 和 `risk/che`。完整定义、实验边界和 Habitat 原生
metric 的迁移说明见 [Dynamic Risk Assessment](risk_assessment.md)。

### Local planner baselines

默认 `--local_planner fmm` 保留当前实现：risk-off 时依旧优先 Habitat
navmesh、失败后回退 FMM；risk-aware 时直接使用 risk-aware FMM。新增
`astar`、地图 PPO `rl` 和 Habitat 官方 DD-PPO `pointnav` 三个显式
baseline：

```bash
--local_planner fmm
--local_planner astar
--local_planner rl --rl_local_checkpoint <checkpoint.pth>
--local_planner pointnav \
  --pointnav_checkpoint data/ddppo-models/gibson-2plus-resnet50.pth \
  --pointnav_device cuda:0
```

`pointnav` 与原有 `rl` 不同：前者把 global planner 选择的 frontier
转换为 Habitat polar point-goal，使用独立的 checkpoint-exact policy
camera 和逐机器人 recurrent state；后者读取 occupancy/risk map crop 并
产生 grid waypoint。默认建图/检测/VLM 继续使用原始
`640×480/HFOV79` RGB-D，PointNav depth policy 单独读取
`pointnav_depth=256×256/HFOV90`，不会再切割或改变建图相机。PointNav
的局部 `STOP` 只触发 global replanning，不会结束 ObjectNav；只有检测到
目标后的原有完成路径可以发出 task `STOP`。

不再提供单独的 local-planner risk 开关。四种 planner 都直接跟随现有
`risk_enabled` 和 `risk_source`：

```bash
# 自动 risk-aware：global/local planner 都使用 sensed risk/hard-unsafe
--risk_enabled 1 --risk_source sensed

# 自动 risk-blind：planner 不使用 risk，evaluator 继续计算 SafeSuccess/CHE
--risk_enabled 1 --risk_source none

# 完全关闭 risk runtime
--risk_enabled 0
```

`sensed`/`oracle` 会自动选择 aware planner；`none` 和 risk-off 会自动选择
blind planner。PointNav 的 aware 版本是 frozen policy + risk-aware
frontier + hard-hazard action shield，不会伪装成经过 risk 训练的 neural
policy。官方 checkpoint 的下载、SHA-256、精确 sensor/action schema、
RL 训练命令及完整
benchmark 矩阵见 [Local-planner baselines](local_planner_baselines.md)。

## 6. 批量运行 normal/person planner baselines

使用 `scripts/run_baseline_benchmarks.py` 可以在原生 ObjectNav 和静态
person ObjectNav 上运行相同的 episode 数，并按 dataset/global/local
planner 隔离结果。请从安装了 Habitat 的 `co-nav3` 环境运行；脚本默认使用
启动它的 Python 解释器。

先做只读预检：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/run_baseline_benchmarks.py \
  --episodes 200 \
  --dry-run
```

默认 `--matrix controlled` 遵循正式实验的受控比较：

- 固定 `local_planner=fmm`，比较
  `nearest/co_ut/fill/random/gpt`；
- 固定 `nav_mode=co_ut`，比较
  `fmm/astar/pointnav`；
- `co_ut+fmm` 这个重复组合在矩阵中自动去重，所以每个 dataset 是 7 个 run，两个 dataset
  共 14 个 run、每个 run 200 episodes。

`random` 使用统一的 `--seed`，并把 episode index 和当前 replan step 混入
采样种子；相同实验从断点恢复时会重新得到相同的 long-term goal。普通模式只从
机器人所在的已探索、无障碍连通域采样；risk 模式还会排除
`hard_unsafe` 和超过 danger threshold 的 cell。

运行完整默认矩阵前，GPT 行需要 OpenAI key：

```bash
export OPENAI_API_KEY="your-key"

/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/run_baseline_benchmarks.py \
  --episodes 200 \
  --study-id normal_primary_200ep
```

建议先运行无需 API 的确定性行：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/run_baseline_benchmarks.py \
  --episodes 200 \
  --global-planners nearest co_ut fill \
  --study-id normal_deterministic_200ep
```

如果确实需要每个 global/local planner 的完整组合，可显式使用：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/run_baseline_benchmarks.py \
  --matrix cartesian \
  --episodes 200
```

默认 local planner 不包含尚未具备仓库 checkpoint 的地图 PPO `rl`。训练
完成后可显式加入：

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/run_baseline_benchmarks.py \
  --local-planners fmm astar pointnav rl \
  --rl-checkpoint outputs/local_planner_rl/policy.pth \
  --episodes 200
```

输出位于：

```text
outputs/benchmarks/<study_id>/
├── study_manifest.json
├── environment.json
├── runs/
│   └── <dataset>__normal__G-<global>__L-<local>__s-<seed>__n-200/
│       ├── manifest.json
│       ├── command.txt
│       ├── status.json
│       ├── status_history.jsonl
│       ├── stdout.log
│       ├── metrics/aggregate.json
│       └── navigation/
└── reports/completeness.json
```

重复执行同一个命令会跳过 fingerprint 一致且已经完成的 run；失败的 run
会增加 attempt 并自动从最近的有效 episode checkpoint 继续。`main.py`
每完成一个 episode 就原子写入
`<dump_location>/metrics/resume_state.json`，其中包含精确累计 metric sums、
最后 episode ID 和计划总数；launcher 会追加 `--start_episode` 与
`--resume_metrics_path`，并把上一 attempt 的 stdout 归档到
`runs/<run_id>/attempts/`。历史 run 如果只有三位小数的
`metrics/aggregate.json` 也可以迁移续跑，但会在 checkpoint 中标记
`precision=legacy_3_decimal_average`。使用 `--restart-incomplete` 可以显式
禁用续跑并从 episode 1 重启。只有 `main.py` 返回成功且最后明确报告
`---(200/200)` 时，run 才会标记为 `completed`。`--force` 会强制重跑已完成
的相同 run。`--main-args` 可在命令最后追加非核心参数，但不能覆盖 dataset、
planner、episode 数或输出目录。

如果 master study 已经存在，只想执行其中几个预先登记的 run，可使用
`--only-run-ids`。该模式保留原有 `study_manifest.json` 和完整 planned-run
范围，只更新指定 run 及统一 completeness；适合分批执行或迁移结果后续跑：

```bash
python scripts/run_baseline_benchmarks.py \
  --study-id normal_planner_baselines_200ep \
  --only-run-ids \
    objectnav__normal__G-fill__L-fmm__s-1__n-200 \
    objectnav__normal__G-fill__L-pointnav__s-1__n-200 \
    objectnav__normal__G-fill__L-astar__s-1__n-200
```

指定的 ID 必须已经存在于 master manifest；无需也不应使用 `--force` 来缩小
既有 study 的 matrix。
