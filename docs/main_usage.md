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
    --scene-id Nfvxx8J5NCo --plan-id 83679a07b632 \
    --steps-per-unit 5 --seconds-per-unit 2.0 --depth_use_clean 1
```加了内部行为提示（--visualize 起的是 Open3D GUI 子线程、--lidar_360 仅在 --fire_sensors=1 时生效、

---

## 2. `main.py` CLI 参数（来自 `arguments.py`）

### 2.1 基础

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--seed` | int, `1` | 随机种子 |
| `-d`, `--dump_location` | str, `./tmp` | 日志/模型 dump 目录 |
| `-v`, `--visualize` | int, `0` | `1` 渲染观测和预测语义图（会启动 Open3D GUI 线程） |
| `--print_images` | int, `0` | `1` 把可视化结果保存为图片 |

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
| `--num_humans` | int, `0` | 每次 reset 随机撒到场景里的 KinematicHumanoid 数量；`0` 关闭 |
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
| `--nav_mode` | `nearest` / `co_ut` / `fill` / `gpt`（默认 `gpt`） | 全局目标选择策略：最近前沿 / 合作分配 / 高得分覆盖 / GPT-4o 决策 |
| `--fill_mode` | int, `0` | `fill` 模式细分（保留位） |
| `--gpt_type` | int, `2` | `0=text-davinci-003`，`1=gpt-3.5-turbo`，`2=gpt-4o`（默认），`3=gpt-4o-mini` |

### 2.5 火灾多模态传感器（“2D 烟雾衰减”路径）

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--fire_sensors` | int, `0` | `1` 启用 smoke/depth-noise/radar/thermal 模拟，覆盖 RGB+Depth 并 dump 每步文件 |
| `--fire_apply_to_obs` | int, `1` | `1` 把退化后的 RGB+Depth 写回 obs 供建图；`0` 仅保存退化数据但 nav 用干净 obs |
| `--smoke_density` | float, `0.6` | `[0,1]`，烟雾光学厚度。`0` 完全清晰，`1` 能见度 ≈1m |
| `--fire_dump_dir` | str, `./outputs/fire_sensors` | 每步传感器图像输出目录 |
| `--fire_save_every` | int, `1` | 每 N 步保存一次（`1` 每步都存） |
| `--fire_save_npz` | int, `0` | `1` 同时 dump 原始 numpy `.npz` |
| `--fire_show_window` | int, `0` | `1` 打开 OpenCV 2x4 实时仪表盘 |
| `--lidar_360` | int, `0` | `1` 给每个 agent 安装 4 个偏航深度相机（前/左/后/右），由 LIDAR 模块拼接 360° 点云 |
| `--lidar_resolution` | int, `320` | 每片深度图分辨率（HxW，越小越快） |

### 2.6 烟雾场景感知开关

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--depth_use_clean` | int, `0` | `1` 烟下仍用 Habitat 干净深度做建图/导航（RGB 仍然是被烟衰减的版本） |
| `--use_thermal_perception` | int, `1` | `1` 在 `fire_sensors` 启用时把热成像 flame 掩码注入观测，并让检测器从热像而非烟雾 RGB 上的 HSV 里读 fire |
| `--rgb_dehaze` | int, `0` | `1` 在检测前对烟雾 RGB 做基于深度的逆 Beer-Lambert + CLAHE 去雾（建议配合 `--depth_use_clean=1`） |

### 2.7 FireWorld 运行时（3D 体素 ray-march 路径）

| 参数 | 类型/默认 | 说明 |
| --- | --- | --- |
| `--fire_world` | int, `0` | `1` 启用 FireWorld，覆盖 SmokeRGBSensor 的 RGB/Thermal 输出，改用 3D 体素合成 |
| `--fire_world_plan_id` | str, `None` | `scenes/<scene>/plans/` 下的 plan id（12 位 hex），`--fire_world=1` 时必填 |
| `--fire_world_scenes_root` | str, `scenes` | inventory.json + plan.json 所在根目录 |
| `--fire_world_out_root` | str, `outputs/fire_world` | timeline.npz 根目录（`<root>/<scene>/<plan_id>/timeline.npz`） |
| `--fire_steps_per_unit` | int, `5` | 多少机器人步推进 1 单位“火灾时间”，越大火越慢 |
| `--fire_seconds_per_unit` | float, `2.0` | 每单位火灾时间消耗多少秒火灾时间线（默认 5 步 → 推进 2s） |
| `--fire_world_smoke_k_ext` | float, `4.0` | 烟雾体素的消光系数倍率（每米），越大越不透明 |
| `--fire_world_n_steps` | int, `24` | ray-march 每像素采样数 |
| `--fire_world_render_scale` | float, `0.5` | 体积积分渲染相对相机分辨率的比例，`0.5` 约 4× 加速，`1.0` 全分辨率 |
| `--fire_world_compound_rgb` | int, `0` | `1` 当 `--fire_world=1` 且 `--fire_sensors=1` 同时开启时，在 FireWorld 体素 RGB 之上再叠一层全局 Beer-Lambert（用套件 `smoke_density` + 干净 depth）。适合“火灾在某个房间，整层楼也有环境烟”的场景。 |

> 备注：`args.cuda` 由 `torch.cuda.is_available()` 自动推导，无需显式传参。

---

## 3. `main.py` 内部行为约定

- `--visualize=1` 时会用 `threading` 启动一个 Open3D GUI 子线程（`utils.vis_gui.ReconstructionWindow`），主线程跑 `main()`，二者通过 `multiprocessing.Queue` 通信。GUI 自身不接受 WASD，是只读重建可视化。
- `--fire_sensors=1` 与 `--fire_world=1` **现在可以同时打开**（见下方“双管线组合”）。
- `--lidar_360=1` 仅在 `--fire_sensors=1` 时生效（参考 `main.py` 中的条件判断）。
- 全局规划在每个 `args.num_local_steps` 步触发一次，依据 `--nav_mode` 选目标点；`gpt` 模式下会调 `chat_utils.chat_with_gpt4v`。

### 3.1 架构（重要）

经过最近一次重构后，火灾仿真分两层：

- `utils/fire_world/`：**世界模型**。负责火焰/烟雾/温度的 3D 体素时间线、相机位姿换算、机器人步 → 火灾时间映射。核心类：`FireWorld`（数据）、`FireClock`（时钟）、`FireScene`（统一入口，`FireScene.from_args(args, config)` 一行构造好）。
- `utils/fire_sensors/`：**观测层**。负责"如何看见这个世界"——RGB/Depth 的烟雾衰减/退化，Radar/LIDAR/Thermal 的合成，以及 dashboard。新增 `VoxelSmokeSensor`：以 `FireScene` 为输入的体素烟雾相机；既有的 `SmokeRGBSensor`/`SmokeDepthSensor` 等保持不变。`FireSensorSuite` 接受 `scene=FireScene, camera_K=...` 参数，按 `cfg.rgb_source / cfg.thermal_source` 决定每帧 RGB/Thermal 是从体素相机出还是从 Beer-Lambert/HSV 出。
- `utils/fire_pipeline.py::step_fire_observation()`：薄薄的胶水函数，负责调 `suite.process(...)` 并把结果写回 `observations`。`compose_fire_step` 作为兼容别名保留。

`FireWorldController`（旧入口）保留为 back-compat shim：内部用 `FireScene` + `VoxelRenderParams` 实现，老脚本（`scripts/demo_fire_world_runtime.py`、`scripts/test_fire_world_*.py`）不需要改就能跑。

### 3.2 FireWorld + 烟雾感知降级 双管线组合

主循环统一调用 `utils/fire_pipeline.py::step_fire_observation()`，`FireSensorSuite` 内部按 `rgb_source / thermal_source` 切换：

| `--fire_world` | `--fire_sensors` | RGB 来源 | Thermal 来源 | Depth | Radar/LIDAR/Dashboard |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | clean | 无 | clean | 无 |
| 0 | 1 | Beer-Lambert | HSV | 套件噪声深度 | 套件 dashboard |
| 1 | 0 | FireWorld 体素 | 体素温度 | clean (套件未启用) | 无 |
| 1 | 1 | FireWorld 体素（可选 + 全局 Beer-Lambert，`--fire_world_compound_rgb=1`） | 体素温度 | 套件噪声深度 | 套件 dashboard，RGB 那格已是 FireWorld 输出 |

要点：
- 双开时套件总是消费**干净** RGB-D 做退化（这是它的本职），FireWorld 单独在体素场上算 ray-march。
- 默认 `--fire_world_compound_rgb=0`：体素 RGB 已经包含火焰本身的 ray-marched 烟雾衰减；只有想叠"场景级环境雾"才把它打开。
- 想保护建图就配合 `--depth_use_clean=1`（写回干净 depth）+ `--use_thermal_perception=1`（让检测器从热像读 fire），这两个旗子在双开时完全可用。
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
| `--plan-id` | required | FireWorld plan id（12 位 hex），`outputs/fire_world/<scene>/<plan_id>/timeline.npz` 必须已存在 |
| `--scenes-root` | `scenes` | inventory/plan 所在根 |
| `--out-root` | `outputs/fire_world` | timeline 输出根 |
| `--steps-per-unit` | `5` | 多少步推进 1 单位火灾时间 |
| `--seconds-per-unit` | `2.0` | 每单位等价多少秒时间线 |
| `--smoke-k-ext` | `4.0` | 烟雾消光倍率 |
| `--n-steps` | `16` | ray-march 采样数（teleop 推荐 12~16） |
| `--render-scale` | `0.5` | 体积渲染比例 |
| `--depth_use_clean` | `1` | `1` 使用 Habitat 干净深度（推荐） |
| `--save-frames-to` | `None` | 给定路径则按步 dump PNG |
| `--enable-suite` | `0` | `1` 同时挂上 `FireSensorSuite`（噪声 RGB-D / Radar / LIDAR / Thermal / dashboard） |
| `--smoke-density` | `0.6` | 套件 Beer-Lambert 烟雾密度，`--enable-suite=1` 或 `--compound-rgb=1` 时生效 |
| `--compound-rgb` | `0` | `1` 在 FireWorld 体素 RGB 上再叠一层全局 Beer-Lambert（需配合 `--enable-suite=1`） |
| `--show-dashboard` | `1` | `1` 当 `--enable-suite=1` 时另开一窗显示套件 2x4 dashboard |
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
# 自动评测，GPT-4o 调度，单 agent，开烟雾+热像
python main.py --num_agents 1 --nav_mode gpt --gpt_type 2 \
    --fire_sensors 1 --smoke_density 0.7 \
    --depth_use_clean 1 --use_thermal_perception 1

# 自动评测，FireWorld 体素，2 agent，可视化保存图片
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id 83679a07b632 \
    --fire_steps_per_unit 1 --fire_seconds_per_unit 5.0 \
    --fire_world_compound_rgb 1 \
    --fire_show_window 1


python main.py \
    --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id 83679a07b632 \
    --fire_speedup 2.0 \
    --fire_show_window 1



# # 自动评测，FireWorld + 烟雾 RGB-D 同开（最完整的火灾感知）
# python main.py --num_agents 1 --nav_mode gpt \
#     --fire_world 1 --fire_world_plan_id 83679a07b632 \
#     --fire)steps_per_unit 1 --fire_seconds_per_unit 5.0
#     --fire_sensors 1 --smoke_density 0.6 \
#     --fire_world_compound_rgb 1 \
#     --depth_use_clean 1 --use_thermal_perception 1 \
#     --fire_show_window 1

# 手动开 FireWorld + 套件，看着自己走、看火长、同时看 dashboard
python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --scene-id Nfvxx8J5NCo --plan-id 83679a07b632 \
    --steps-per-unit 1 --seconds-per-unit 5.0 \
    --depth_use_clean 1 --render-scale 0.5 --n-steps 16 \
    --smoke-density 0.6 --compound-rgb 1 \
    --show-dashboard 1

python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --scene-id Nfvxx8J5NCo --plan-id 83679a07b632 \
    --speedup 2.0 \
    --show-dashboard 1

```




