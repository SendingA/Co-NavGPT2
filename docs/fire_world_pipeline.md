# FireWorld Benchmark: Pipeline 概览与论文写作模板

> 本文同时作为工程总结与论文 Methodology 的写作模板。前半部分描述
> 仓库里 6 个阶段的具体实现；后半部分给出一段可直接搬到 paper 的描述
> （含图示、公式、可引用伪代码以及消融建议）。

---

## 0. 背景与动机

我们在 Habitat / HM3D 的多机器人 ObjectGoal Navigation 基础上，构建了一个
**动态火灾感知导航 benchmark — FireWorld**。它要回答一个问题：

> 当真实建筑环境中发生 (1) 时变的火源 / 烟雾扩散，且 (2) 机器人
> 视觉传感器同时受到烟雾遮蔽与噪声退化时，多智能体探索算法
> 应该怎样工作？

为此我们在仿真层显式地把"世界模型"（火灾物理）和"观测模型"（多模
态传感器降级）解耦：

| 层 | 责任 | 主要代码 |
| --- | --- | --- |
| 世界模型 `utils/fire_world` | 3D 体素场 (flame / smoke / temperature)、传播、相机位姿换算 | `scene_scan.py / planner.py / propagation.py / runtime.py / scene.py` |
| 观测模型 `utils/fire_sensors` | 体素相机、Beer-Lambert 烟雾 RGB、含噪深度、毫米波雷达、LIDAR、热像 | `voxel_render.py / sensors/* / suite.py` |

整个流水线分 6 个阶段，每一步输出可序列化、可缓存、确定性可复现：

```
HM3D scene
      │
      ▼
1) scene_scan ─────► scenes/<id>/inventory.json
      │                  + structural/{walls, floors, ceilings}.npy
      ▼
2) planner   ─────► scenes/<id>/plans/<plan_id>.json
      │
      ▼
3) propagation ───► outputs/fire_world/<id>/<plan_id>/timeline.npz
      │                  + timeline_meta.json
      ▼
4) FireScene (世界模型 facade，运行时加载 timeline + 时钟 + 位姿换算)
      │
      ▼
5) FireSensorSuite (观测层：体素相机 + Beer-Lambert + radar / lidar / thermal)
      │
      ▼
6) main.py / main_vec.py / scripts/keyboard_teleop_fire.py
   (机器人 nav loop、与 obs 集成、dashboard 可视化)
```

下面逐阶段展开。

---

## 1. Stage 1 — Scene scan：从 HM3D semantic.glb 重建结构化 inventory

**目标**：把 HM3D 一个场景里 600+ 个 instance（家具、墙、地板、门窗等）
解析成可被火灾仿真直接使用的 JSON 描述（schema v2 ``inventory.json``）。

**关键技术点**

1. **vertex-color → instance ID 的 sRGB 解码**
   HM3D 的 ``*.semantic.glb`` 用 ``COLOR_0`` 顶点属性（uint16 线性强度）
   编码 instance id；它必须经过 IEC 61966-2-1 sRGB OETF 才能与
   ``*.semantic.txt`` 里的 6-hex color 匹配。我们之前踩过的坑：朴素线性
   remap ``round(u16/65535*255)`` 命中率 0/209，正确做法命中
   ~187/209（剩余的来自一个 primitive 内多 instance 的 mesh 共享）。
2. **HM3D mesh primitive 不是按 instance 切分的**
   一个 primitive 可包含 5-50 种 instance 颜色。我们按面 (face) 颜色聚类
   重建 instance 几何；然后对每个 instance 取顶点云的 AABB / 质心 /
   面数，写到 inventory。
3. **坐标系转换**：GLB 是 +Z up，Habitat 是 +Y up；我们用
   ``(x, y, z)_glb → (x, z, -y)_habitat`` 把所有 AABB 转到 Habitat 世界系
   （见 `hm3d_semantic.aggregate_instances` 中 `apply_habitat_axis_fix`）。
4. **物理属性映射**：每个 category 查 `MATERIAL_TABLE` 得到
   ``(flammability, smoke_yield) ∈ [0, 1]^2``。例如 bed=(0.75, 0.80)，
   curtain=(0.85, 0.75)，stove=(0.85, 0.70)，sink=(0.05, 0.05)。结构件
   （wall/floor/ceiling/door/...）一律 (0, 0)。
5. **结构体素栅格化**：把 wall / floor / ceiling 的 AABB 烤进
   ``(Nx, Ny, Nz) bool`` 数组（``voxel_m=0.10`` 默认），存为 ``.npy``。
   这层 mask 在 propagation 和 mapping 里都被当作"不可穿透边界"。
6. **Floor clustering**：1D k-means-lite，用 instance y_min 把同层物体
   聚成一个 floor，输出 `floors[i] = {y, y_min, y_max}`，用来限制火灾
   只在同层蔓延（避免楼下 chair 引燃楼上 bed）。

**论文写作要点**：强调 "we recover **all** instance categories
(rather than only the 5 ObjectGoal targets) and ground the world
model in HM3D's semantic ground truth"。

---

## 2. Stage 2 — Planner：fire-type / intensity / seed → 可复现的 plan.json

**目标**：在 inventory 的基础上，决定哪几个 object 起火、何时起火、
源温度多少、烟雾产率多少；并给传播器一组规则。

**确定性 ID 哈希**：
``plan_id = sha1("scene|fire_type|intensity|seed|template_version")[:12]``
保证同一组输入永远生成同一个 plan_id 和同一份 ignitions。

**Template-based 选源**（见 `templates.py`）：

| Template | 主要源 | 副源（同一楼层、距离 ≤ R） |
| --- | --- | --- |
| `kitchen_grease_fire` | stove / hood，fallback chair / sofa | 距离 ≤3.5m，按 flammability/距离 加权采样 |
| `bedroom_textile`     | bed / sofa / couch | 距离 ≤4.0m |
| `living_room_electric`| tv / monitor / computer | 距离 ≤3.0m |
| `multi_origin`        | 两个尽量远的高 flammability 节点 | — |

**强度预设**（`IntensityPreset`，3 档）：

| intensity | n_ignitions_max | source_temp_c | fuel_kg | duration_s |
| --- | --- | --- | --- | --- |
| light  | 1 | 550 | 2  | 300 |
| medium | 2 | 750 | 5  | 600 |
| severe | 3 | 950 | 10 | 900 |

**Plan schema** (`plans/<plan_id>.json`)：
- 顶层：`scene_id, world_aabb, fire_type, intensity, seed, template_version, duration_s`
- `ignitions[i]`：`object_id, category, position, ignite_time_s,
  source_radius_m, source_temp_c, fuel_kg, smoke_yield`
- `propagation_rules`：`flammable_threshold (0.4), ignition_temp_c (350),
  spread_speed_m_per_s (0.04 * intensity_scale), ceiling_jet_speed_m_per_s,
  buoyancy_v_m_per_s, thermal_diffusivity (0.05), ambient_temp_c (25)`

**论文要点**：强调 (a) 确定性 + 哈希命名 = full reproducibility，
(b) template 设计映射到 NFPA / NIST 真实场景类别，
(c) plan.json 是人类可读、人类可手编的（如手动添加 chair 起火点）。

---

## 3. Stage 3 — Propagation：3D 体素反应-扩散-浮力-顶板射流积分

**目标**：把 plan 推进 ``duration_s`` 秒，得到逐帧
``(flame, smoke, temperature)`` 体素时间线 ``timeline.npz``。

**State per voxel**：
- ``fuel  ∈ [0, 1]``  无量纲燃料质量分数（来自 inventory 的 flammability，
  按 AABB 烤进体素）
- ``temp ∈ ℝ``  ℃
- ``flame ∈ [0, 1]``  火焰强度
- ``smoke ∈ [0, 1]``  烟雾光学厚度

**单步 (dt=0.5s) 6+1 阶段**：

1. **热扩散** (CFL-substepped)：``T ← T + α · Δt · ∇²T``，墙/天花板/地板
   零通量边界（强制设回 ambient）；α=`thermal_diffusivity`。
2. **垂直浮力**：把 hot mask 的 ``T - T_ambient`` 与 smoke 沿 +y 平移
   ``frac = clip(v_buoy · dt / v, 0, 0.95)``；天花板反射防"穿顶"。
3. **顶板射流**：天花板下 4 层做 3×3 XZ box-blur，权重
   ``frac_cj = clip(cj · dt / v, 0, 0.95)``。
4. **反应**：where ``fuel > flammable_threshold`` AND ``T > T_ignite``：
   ``Δflame = k · fuel · dt``，``Δfuel = -Δflame``，``ΔT = Q · Δflame``，
   ``Δsmoke = 0.5 · Δflame``。
   表面火焰扩散：``flame ← flame + spread · dt · ∇²flame · 𝟙(fuel>0)``。
5. **衰减**：fuel 耗尽时 ``flame *= 1 - 5%`` per step；smoke 半衰期 ~140s。
6. **Cooling**：``T ← T_ambient + (T - T_ambient) · exp(-0.005 · dt)``。
7. **Sustained sources**：每个未燃尽的 ignition 在球内 pin
   ``T ≥ T_ambient + (T_src - T_ambient) · falloff``，并按
   ``0.10 · falloff · dt`` 注入 smoke。

**输出**：``timeline.npz`` 包含 ``flame / smoke / temp``（fp16，压缩）+
``times``（fp32 秒）+ ``meta_json``（unicode array，跨 numpy 版本可读）；
同步写 ``timeline_meta.json`` 作为 sidecar。

**论文要点**：
- 这是**仿真级**（simulation-grade）而非研究级 CFD：目的不是预测
  真实火场，而是为 nav 算法提供物理一致、确定性、廉价的时间-空间
  结构化挑战。
- 引用 NIST FDS / OpenFOAM 的简化策略：reaction-diffusion + buoyancy +
  ceiling jet 已能产生符合 occupants 视感的烟柱演化。
- 在文中给一张 log-time × room 的蔓延曲线，证明火源 / 烟雾在不同
  房间是有时间梯度地"到达"的。

---

## 4. Stage 4 — FireScene：世界模型运行时 facade

**目标**：把 timeline + clock + camera-pose 换算封装成一个
"per-episode 世界对象"，让下游传感器只需要 ``query(t_sim)`` +
``camera_pose(agent_state)``。

**时间语义**：因为 Habitat 没有 wall-clock，我们把 robot step → fire-time
seconds 用两个整数固定：

```
t_sim = floor(N_steps / steps_per_unit) * seconds_per_unit + base_t0
```

默认 `5 / 2.0`：每 5 步推进 2 秒火灾时间。Teleop 调成 `1 / 5.0` 可加速
观察。

**位姿换算**：`habitat_agent_state_to_cam` 把 Habitat agent.sensor_states
里的 quaternion + position → ``(cam_pos_world, R_cam2world)``，列向量
为相机 right / up / -forward。

**论文要点**：强调把"世界何时演化"与"机器人何时观测"显式解耦，
使得控制 fire 时间尺度与机器人运动时间尺度的比值（fire-to-robot
clock ratio）成为可消融的实验维度。

---

## 5. Stage 5 — FireSensorSuite：体素相机 + Beer-Lambert + 多模态降级

观测层把 FireScene 渲染 + 多种主动/被动传感器统一在一个 suite 里，
``rgb_source / thermal_source ∈ {auto, voxel, beer_lambert/hsv}`` 决定
每帧 RGB / Thermal 来自哪条路径。

### 5.1 体素相机 `VoxelSmokeSensor`（cmd: `cfg.rgb_source="voxel"`)

逐像素 ray-march，front-to-back emission-absorption 合成。每条 ray
维护**两个**透过率累加器：

- ``T_scene``：对 scene RGB 与 smoke scatter 用，每步乘
  ``exp(-(σ_smoke + σ_flame) · ds)``，``σ_smoke = k_smoke · smoke_voxel``
- ``T_flame``：对火焰发射用，每步乘
  ``exp(-((1 - p_pass) · σ_smoke + σ_flame) · ds)``，
  ``p_pass`` (= `flame_smoke_passthrough`, 默认 0.95) 表示火焰自发光
  对烟雾消光的"穿透系数"（来自 Starr & Lattimer 2014 Fig. 7：
  可见波段火焰在 medium-thick smoke 中仍可见）。

发射颜色用一条 6 段火焰 LUT（深红 → 橙 → 黄 → 近白核），按 voxel
flame intensity 做线性插值。``flame_threshold = 0.04`` 容许 trilinear 插值
得到的边缘 voxel（值在 0.05~0.30 之间）参与渲染，避免之前
"两个像素亮点"的 bug。``flame_emission_gain = 8`` 让发射在浓烟里仍可见。

可选 ``compound_rgb=True``：在 voxel RGB 上叠一层全局 Beer-Lambert
（密度 = `smoke_density`），但用 voxel flame_mask（高斯扩张过的）
作 guard，避免 fog 把火焰二次抹掉。

### 5.2 Beer-Lambert RGB `SmokeRGBSensor`（兜底，无 FireScene 时使用）

按 Jin's visibility 公式 ``V = 2.3 / k``，``k = density · k_max``：
``T = exp(-k · d_pixel)``，``rgb_smoky = T · rgb_clean + (1 - T) · smoke_color``。
火焰像素 HSV 检测后获得 ``passthrough · flame_mask`` 的额外透过率。

### 5.3 烟雾化深度 `SmokeDepthSensor`（参考 Starr & Lattimer Fig. 5）

按真实 LIDAR 在烟下退化路径：

1. range/density 依赖高斯噪声：``σ = σ_base + σ_range · d + σ_smoke · density · d``
2. cm 级量化
3. 概率丢点：``p_drop = density · 0.5``
4. 能见度截断：``d > 2.3/k`` 的像素 clip 到能见度层 + 小抖动

### 5.4 mmWave Radar / LIDAR / Thermal

- **Radar** (`RadarSensor`)：256 距离 × 64 方位的 range-azimuth heatmap，
  含 sinc-side lobe，0.18 阈值得到点云；``mode=learned`` 时用 0.10m 高斯
  噪声 + 4×stride 抽样模拟 RadarHD post-training 误差；mmWave 在烟雾中
  几乎不衰减（论文 Table 3）。
- **LIDAR** (`LidarSensor`)：360° 拼接 4 个偏航深度相机（启动时给
  Habitat config 注入 4 个 sensor，HFOV=90°），smoke-aware 噪声 + 丢点。
- **Thermal** (`ThermalSensor` 或体素温度通道)：FLIR-like 灰阶
  + INFERNO blend，用 LWIR 7.5–14 μm 在烟雾中几乎不衰减的事实证明
  thermal 是 fire 检测的"primary sensor"；火焰边界 + 邻接 surface 的
  halo 模拟辐射加热。

### 5.5 Dashboard

2×4 网格 + 1 辅助：clean RGB / clean depth / thermal / lidar BEV / smoky
RGB / smoky depth / radar BEV / radar range-azimuth + radar range-elevation。
Title 同时打 ``step`` 与 ``t_sim``，便于截图标注。

---

## 6. Stage 6 — Robot integration & visualization

`utils/fire_pipeline.step_fire_observation()`：每帧
1. 抓 ``obs['rgb' / 'depth']`` → metric depth；
2. 调 ``suite.process(rgb, depth, agent_state, robot_step)``，得到合成
   sensors dict；
3. ``apply_clean_depth_and_thermal`` 把：smoky/voxel RGB → ``obs['rgb']``，
   noisy/clean depth → ``obs['depth']``（按 ``--depth_use_clean``），
   thermal_image / thermal_flame_mask → ``obs['thermal*']``。

调用方：

| 入口 | 用途 |
| --- | --- |
| `main.py` / `main_vec.py` | 自动评测，读 `arguments.py` 全套旗位 |
| `scripts/keyboard_teleop_fire.py` | WASD 手动驾驶，左右两窗（agent 第一人称 + suite dashboard），可调 `flame_smoke_passthrough` 等 |

---

## 7. 论文版方法描述模板（可直接复用）

> 你可以把下面这一节抄进 paper 的 §3 (Methodology) 或 §4 (Benchmark
> Construction)。变量名和图序号要按你的 paper 对齐。

### 7.1 Overview

We introduce **FireWorld**, a benchmark that augments Habitat–HM3D
ObjectGoal Navigation with a deterministic, time-varying 3D fire and
smoke field, paired with a multi-modal sensor degradation model. The
benchmark is built by a six-stage pipeline (Fig. X) that decouples a
*world model* of fire propagation from an *observation model* of
sensor degradation, so each can be ablated independently. Every
intermediate artefact is content-addressed by a SHA1-truncated
``plan_id``, making complete experimental runs bit-for-bit
reproducible.

### 7.2 Scene parsing and material assignment

Given an HM3D scene, we recover a *full* instance-level inventory by
walking each ``*.semantic.glb`` primitive, decoding its per-vertex
linear ``COLOR_0`` (uint16) through the IEC 61966-2-1 sRGB OETF, and
matching the resulting 24-bit color to ``*.semantic.txt`` instance
IDs. Because HM3D primitives are built for view-frustum culling
rather than per-instance grouping, a single primitive routinely
contains 5-50 distinct instance colors; we therefore aggregate at face
granularity rather than at mesh granularity. Each recovered instance is
augmented with (i) an axis-aligned bounding box in Habitat world
coordinates (after the GLB-to-Habitat axis fix), (ii) a floor index
obtained by 1D clustering of instance y-extents with a 1.5 m gap
threshold, and (iii) two material parameters, *flammability* and
*smoke yield*, looked up from a 30-entry table indexed by HM3D
category. Structural categories (walls, floors, ceilings, doors,
windows, columns) are tagged separately and rasterised into a
``0.10 m`` voxel mask used as zero-flux boundaries by the propagation
solver.

### 7.3 Plan generation

A *plan* selects ignitions and propagation rules from the inventory
under a (``fire_type``, ``intensity``, ``seed``) triple. We expose four
template generators (kitchen-grease, bedroom-textile,
living-room-electric, multi-origin) that pick a primary ignition by
flammability-weighted sampling restricted to the relevant categories,
then add up to *N* secondaries on the same floor with distance- and
flammability-weighted sampling. Three intensity presets (light /
medium / severe) control the primary source temperature
(550 / 750 / 950 °C), the per-ignition fuel mass (2 / 5 / 10 kg) and
the simulated duration (300 / 600 / 900 s). The output is a JSON
``plan.json`` containing every ignition's world position, source
radius, source temperature, ignition delay, and the global propagation
rule set (``flammable_threshold``, ``ignition_temp_c``,
``spread_speed_m_per_s``, ``ceiling_jet_speed_m_per_s``,
``buoyancy_v_m_per_s``, ``thermal_diffusivity``, ``ambient_temp_c``).
Each plan is content-addressed by a SHA1-truncated hash of its inputs.

### 7.4 Voxel propagation

Each plan is integrated forward by a custom 3D voxel solver whose state
per voxel is ``(fuel, temperature, flame, smoke)``. One step advances
the field by ``dt = 0.5 s`` through a six-stage update: (1)
CFL-substepped Laplacian heat diffusion with zero-flux structural
masks, (2) buoyancy-driven advection of the supra-ambient temperature
and smoke fields along +y, (3) a ceiling-jet 3×3 XZ box blur applied
to the top four voxel layers, (4) a reaction step where voxels with
``fuel > φ_f`` and ``T > T_ig`` consume fuel and emit flame, smoke and
sensible heat, with surface flame spread modelled as a Laplacian
operator masked by ``𝟙(fuel > 0)``, (5) flame and smoke decay, (6)
exponential cooling toward ``T_ambient``. After step (6), each
non-expired ignition pins its source neighbourhood through a
spherical falloff, mimicking sustained heat release. The integrator
runs at ``voxel = 0.15 m`` and writes a compressed
``timeline.npz``: a sequence of ``(flame, smoke, temperature)`` fp16
volumes plus the corresponding fire-time stamps. A 900 s scene at
``142 × 28 × 78`` voxels takes ~25 s wall-clock to bake.

### 7.5 World–observation decoupling at runtime

At evaluation time, we do *not* re-run the propagator. Instead a
``FireScene`` object loads the cached timeline and exposes two
operators: ``query(t_sim) → (flame, smoke, temperature)`` and
``camera_pose(state) → (position, rotation)``. Robot–wall-clock is
intentionally meaningless in the benchmark; instead we map robot steps
to fire-time seconds via two integers,
``t_sim = ⌊N_step / s⌋ · τ + t_0``, where ``s`` (``steps_per_unit``)
and ``τ`` (``seconds_per_unit``) are reported per experiment. This
makes the fire-to-robot clock ratio a first-class ablation axis.

### 7.6 Sensor model

Every modality the agent has access to is produced by a
``FireSensorSuite`` that observes the FireScene from the agent's
viewpoint:

* **Volumetric RGB.** A ray-march integrator composites the
  flame / smoke / temperature voxel field on top of the scene's clean
  RGB. To preserve flame radiance through dense smoke, the integrator
  maintains *two* transmittance accumulators: a scene-channel
  ``T_scene = ∏ exp(-(σ_smoke + σ_flame) · ds)`` and a flame-channel
  ``T_flame = ∏ exp(-((1 - p) · σ_smoke + σ_flame) · ds)``, where
  ``p = 0.95`` is a passthrough coefficient consistent with the
  visible-band imagery reported in [Starr & Lattimer, 2014, Fig. 7].
  The flame emission is colored by a 6-stop LUT and added with weight
  ``T_flame``; smoke scatter and the residual scene RGB are weighted by
  ``T_scene``.
* **Smoke-degraded depth.** Following [Starr & Lattimer, 2014, Fig. 5],
  the depth sensor is degraded by range- and density-dependent Gaussian
  noise, cm-level quantisation, density-proportional dropout, and a
  visibility cutoff at ``V = 2.3 / k`` Jin's law.
* **Thermal IR.** Voxel temperatures are tone-mapped through a FLIR-
  style auto-stretch with optional INFERNO color blending. LWIR
  (7.5–14 µm) is treated as smoke-invariant.
* **mmWave radar / LIDAR.** A 256×64 range-azimuth heatmap with
  sinc-shaped sidelobes, optionally post-processed to a 3D point cloud
  via a 0.10 m Gaussian-noised LIDAR-like cloud (mode ``learned``,
  matching RadarHD's reported median Hausdorff error). LIDAR uses four
  yaw-rotated depth sensors stitched into a 360° point cloud, with
  smoke-aware noise and dropout identical to the depth sensor.

A 2×4 dashboard composites all eight modalities into a single image
for offline inspection.

### 7.7 Determinism, reproducibility, and configurability

Every artefact in the pipeline is deterministic given its inputs:
``inventory.json`` is a function of HM3D content, ``plan.json`` is a
function of (fire_type, intensity, seed), ``timeline.npz`` is a
function of (plan, voxel_m, dt), and per-step observations are a
function of (timeline, agent_state, step_index). We also support
hand-edited plans (e.g. injecting a kitchen ignition + chair
ignitions on top of an auto-generated bedroom plan); a stale-cache
detector compares plan and timeline modification times and warns the
user if the propagation step needs re-running. The full pipeline is
configurable through 30+ CLI flags exposed by ``arguments.py``,
grouped into perception (depth_use_clean, use_thermal_perception,
rgb_dehaze), volumetric rendering (n_steps, render_scale,
flame_smoke_passthrough), fire-time mapping (steps_per_unit,
seconds_per_unit) and degradation (smoke_density,
fire_world_compound_rgb).

### 7.8 Suggested ablations (paper §5)

| Axis | Lever | Measures the effect of |
| --- | --- | --- |
| Fire-time scale | `--fire_steps_per_unit, --fire_seconds_per_unit` | how aggressively the fire grows w.r.t. agent action rate |
| Voxel RGB ↔ Beer-Lambert | `cfg.rgb_source` | whether geometric fidelity of smoke matters for the agent |
| Depth path | `--depth_use_clean` 0/1 | how much navigation cost the noisy depth model adds |
| Thermal | `--use_thermal_perception` 0/1 | whether LWIR rescues fire detection in dense smoke |
| Compound fog | `--fire_world_compound_rgb` 0/1 | environment smoke beyond the active fire room |
| Flame penetration | `flame_smoke_passthrough` 0–1 | sensitivity of detection to a physically-grounded radiance model |

---

## 附录 A — 关键文件速查

| 路径 | 作用 |
| --- | --- |
| `utils/fire_world/scene_scan.py` | Stage 1: build inventory |
| `utils/fire_world/hm3d_semantic.py` | sRGB OETF + glb parsing helpers |
| `utils/fire_world/templates.py` | Stage 2: ignition templates + intensity presets |
| `utils/fire_world/planner.py` | Stage 2 CLI |
| `utils/fire_world/propagation.py` | Stage 3: 6+1-step solver |
| `utils/fire_world/voxel_world.py` | Stage 3 helper: VoxelWorld 数据结构 |
| `utils/fire_world/runtime.py` | Stage 4: FireWorld data loader (含 stale-cache 警告) |
| `utils/fire_world/scene.py` | Stage 4 facade: FireScene + FireClock |
| `utils/fire_world/controller.py` | 兼容 shim：legacy FireWorldController → FireScene |
| `utils/fire_sensors/voxel_render.py` | Stage 5: ray-march + flame LUT + dual-T accumulator |
| `utils/fire_sensors/sensors/voxel_smoke.py` | Stage 5: voxel RGB sensor 包装 |
| `utils/fire_sensors/sensors/{rgb_smoke, depth_smoke, radar, lidar, thermal}.py` | Stage 5: per-modality |
| `utils/fire_sensors/suite.py` | Stage 5: FireSensorSuite 总编排 |
| `utils/fire_pipeline.py` | Stage 6: step_fire_observation glue |
| `main.py / main_vec.py` | Stage 6: 自动评测入口 |
| `scripts/keyboard_teleop_fire.py` | Stage 6: 手动驾驶 + dashboard |
| `arguments.py` | 全部 CLI 旗位 |
| `docs/main_usage.md` | 旗位 + 键盘速查 |

## 附录 B — 复现一个新 plan 的最小步骤

```bash
# 1) 建 inventory（每个场景只跑一次）
python -m utils.fire_world.scene_scan --scene Nfvxx8J5NCo

# 2) 自动生成 plan（或手编 plans/<id>.json）
python -m utils.fire_world.planner \
    --scene Nfvxx8J5NCo \
    --fire_type bedroom_textile --intensity severe --seed 7

# 3) 跑 propagation（必须用与 nav 相同的 conda env）
python -m utils.fire_world.propagation \
    --scene Nfvxx8J5NCo --plan_id 83679a07b632 \
    --voxel_m 0.15

# 4) 自动评测
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id 83679a07b632 \
    --depth_use_clean 1

# 4') 或者手动巡视检验
python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --scene-id Nfvxx8J5NCo --plan-id 83679a07b632 \
    --steps-per-unit 1 --seconds-per-unit 5.0 \
    --depth_use_clean 1 --show-dashboard 1
```
