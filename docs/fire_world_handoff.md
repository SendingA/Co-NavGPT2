# Fire-World Pipeline — Handoff Notes

> Branch: **`vulcan`** (pushed to `origin`).
> Remote: https://github.com/SendingA/Co-NavGPT2.git
> Latest HEAD: `74f7c44 scripts: keyboard teleop with FireWorld overlay`

This doc explains what is in the repo, why each piece exists, and the
exact commands needed to reproduce every artefact for the next agent
who picks the work up.

---

## 1. What problem are we solving?

Co-NavGPT2 navigates HM3D scenes. The original "fire scenario" was a
post-processing layer (`SmokeRGBSensor`) that multiplied the agent's
RGB by a global smoke density. That has two large limitations:

1. The fire scene was **2D**: smoke and flame did not exist as 3D
   structure; agents could not look at flames from different angles
   and see consistent geometry.
2. The fire was **coupled to the agent**: each agent saw its own smoke
   model; there was no shared "the building is on fire" world state.

The fire-world pipeline rebuilds the navigation environment as a
**3D voxel world that evolves independently of the agent**. The agent
only consumes that world via its sensors (RGB, depth, thermal). The
fire propagates with **robot-step time**, not wall-clock time.

---

## 2. End-to-end pipeline (5 stages)

```
                 (offline / cached)                           (runtime)
 ┌──────────────────────────────────────────────────┐  ┌──────────────────────┐
 │ 1) scene_scan ─► inventory.json                  │  │ 5) FireWorldRenderer │
 │ 2) planner    ─► scenes/<id>/plans/<plan>.json   │  │   ray-marches the    │
 │ 3) propagation─► outputs/.../timeline.npz        │──►   timeline against   │
 │ 4) topdown_video ─► topdown.mp4 (validation)     │  │   the agent's pose   │
 └──────────────────────────────────────────────────┘  └──────────────────────┘
```

Stage 1–3 are deterministic. Same inputs always produce the same files.
Stage 4 is purely a sanity-check render. Stage 5 is the live hook.

### Layout (per scene)

```
scenes/<scene_short>/
├── inventory.json              # schema v2, every HM3D instance, structural masks paths
├── plans/
│   └── <plan_id>.json          # scene_type_intensity_12hex, deterministic
└── structural/                 # generated alongside inventory.json
    ├── walls.npy               # boolean (Nx, Ny, Nz)
    ├── floors.npy
    └── ceilings.npy

outputs/fire_world/<scene_short>/<plan_id>/
├── timeline.npz                # (T, Nx, Ny, Nz) flame / smoke / temp + meta
├── timeline_meta.json
├── topdown.mp4 / topdown.png   # stage-4 validation video
├── topdown_summary.json
└── runtime_demo/               # stage-5 demo PNGs / mp4 (if generated)
```

The repo gitignores `outputs/`, `*.npy`, `*.npz`, `*.mp4` etc., so only
`inventory.json` and the small `plan.json` files are committed. They
are tiny (~0.2–1 MB) and review-friendly.

### Available scene fixtures

| scene short id | source split | floors | instances |
|---|---|---|---|
| `TEEsavR23oF` | val_mini | 2 | 628/660 |
| `Nfvxx8J5NCo` | val      | 1 | 359/360 |

### Available plans

| plan id        | scene         | fire_type             | intensity | seed |
|----------------|---------------|-----------------------|-----------|------|
| `d4f8b9c253ab` | TEEsavR23oF   | kitchen_grease_fire   | medium    | 42   |
| `549afa3c5305` | TEEsavR23oF   | bedroom_textile       | severe    | 7    |
| `b2fc76fae83d` | Nfvxx8J5NCo   | kitchen_grease_fire   | medium    | 42   |
| `Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632` | Nfvxx8J5NCo   | bedroom_textile       | severe    | 7    |

---

## 3. Modules added under `utils/fire_world/`

| File | Purpose |
|---|---|
| `__init__.py`        | Package skeleton + roadmap docstring |
| `hm3d_semantic.py`   | GLB parser (no trimesh / habitat-sim), HM3D-specific sRGB OETF colour decoder, per-instance triangle aggregation |
| `scene_scan.py`      | Stage 1: `build_inventory(scene_id) -> dict` + `write_inventory()` CLI. Schema v2: `instances` (every HM3D object), `floors`, `structural.{wall,floor,ceiling}_voxel_path` |
| `templates.py`       | Stage 2 building blocks: `INTENSITIES`, category groups for kitchen/bedroom/living-room, plus area-stratified multi_origin sampling (4–6 regions, 1–2 flammable sources each) |
| `planner.py`         | Stage 2 driver: `plan_id = sha1(scene\|fire_type\|intensity\|seed\|tplV)[:12]`, `build_plan(inventory, ...) -> plan dict`, CLI |
| `voxel_world.py`     | `VoxelWorld.from_aabb`, `attach_structural_masks`, `stamp_object_aabbs`, `kindle_ignition` (returns slice + falloff for sustained sources) |
| `propagation.py`     | Stage 3: 6+1-step integrator (sub-stepped diffusion, buoyancy, ceiling jet, reaction, surface spread, decay, sustained-source pinning). Walls / floors / ceilings act as zero-flux barriers |
| `runtime.py`         | Stage 4: `FireWorld.load(scene, plan_id)` — pure voxel-timeline data loader (+ stale-cache warning). Rendering lives on the sensor side (`utils/fire_sensors/voxel_render.py`). |
| `scene.py`           | Stage 4 facade: `FireScene` + `FireClock` (`FireClock` translates robot-step → `t_sim`; `FireScene.from_args(args, config)`). Consumed by `FireSensorSuite`. |

### Why two GLB parsing facts mattered for stage 1

HM3D `*.semantic.glb` has two non-obvious quirks that were the reason
v1 of `scene_scan.py` could only see goal-category objects:

1. **Vertex colour encoding**: the COLOR_0 attribute is a **linear
   uint16 intensity**. To match the 6-hex IDs in `*.semantic.txt` you
   must apply the **sRGB OETF** before discretising to 8-bit. The naive
   `round(u16 / 65535 * 255)` hits 0/209 primitives in our test scene;
   the sRGB OETF hits 187/209.
2. **Per-primitive instance multiplexing**: HM3D splits meshes by
   *geometry chunk*, not by instance. A single primitive routinely
   carries 5–50 distinct instance colours. So you must group **per
   triangle** by colour, not "one mesh = one instance".

Plus a Habitat-axis fix: GLB is `+Z up`; Habitat is `+Y up`. Apply a
−90° rotation about X (`(x, y, z)_glb → (x, z, −y)_habitat`) to make
the recovered AABBs match ObjectGoal positions to a few centimetres.

### Perception path (voxel-only)

RGB and Thermal are produced exclusively by the voxel renderer, driven
by `--fire_world`. The legacy 2D Beer-Lambert RGB / HSV thermal sensors
and the dehaze path have been removed.

| File | Purpose |
|---|---|
| `utils/fire_sensors/` | voxel RGB + Thermal camera + noisy depth + radar + 360 LIDAR + dashboard |
| `utils/smoke_perception.py` | helpers: `apply_clean_depth_and_thermal`, `thermal_mask_to_detections` |
| `utils/detection_segmentation.py` | `Object_Detection_and_Segmentation.detect()` reads `thermal_flame_mask` so fire detections come from the (smoke-invariant) voxel thermal channel |

### Args added across the pipeline (in `arguments.py`)

```
--fire_apply_to_obs {0,1}                     # write degraded RGB/Depth back to obs
--smoke_density 0..1                          # depth / lidar / radar smoke noise
--fire_dump_dir / --fire_save_every / --fire_save_npz / --fire_show_window
--lidar_360 {0,1} / --lidar_resolution N      # only when --fire_world=1

--depth_use_clean {0,1}                       # keep clean Habitat depth
--use_thermal_perception {0,1}                # default 1

--fire_world {0,1}                            # enable the voxel fire suite
--fire_world_plan_id <scene_type_intensity_hash> # which plan to load
--fire_world_scenes_root scenes
--fire_world_out_root outputs/fire_world
--fire_steps_per_unit 5                       # NEW: every N robot steps -> 1 fire-time unit
--fire_seconds_per_unit 2.0                   # NEW: 1 unit = X seconds of timeline
--fire_world_smoke_k_ext 4.0                  # ray-march extinction multiplier
--fire_world_n_steps 24                       # ray-march samples per pixel
```

### Files modified outside `utils/fire_world/`

* `arguments.py` — all the new switches above.
* `main.py` / `main_vec.py` — when `--fire_world=1`, build a
  `FireScene` + `FireSensorSuite` and call `step_fire_observation` in
  the per-step loop **before** the agent's `mapping(...)` call; result
  goes through `apply_clean_depth_and_thermal`.
* `agents/vlm_agents.py`, `agents/vlm_multi_agents.py` — pass
  `obs['thermal_flame_mask']` into the detector.
* `utils/detection_segmentation.py` — `detect(image, thermal_flame_mask=...)`.
* `utils/smoke_perception.py` — new module.
* `.gitignore` — added `outputs/`, `*.glb`, `*.mp4`, `*.pdf`,
  `*.npz`, `*.npy`, `*.zip`, `=*`. (We had a near-miss where 9.5k
  generated artefacts (~2.6 GiB) were almost pushed; the rewrite
  history step is in commit `82fce04`.)

### Scripts (in `scripts/`)

| Script | Purpose |
|---|---|
| `build_inventory.py`              | Friendly wrapper to scan one or more scenes (single id, `--scene_dir`, or `--all val`). Stage-1 driver. |
| `test_scene_scan.py`              | Smoke tests for stage-1 (material table, build_inventory, write_inventory). |
| `test_fire_planner.py`            | Smoke tests for stage-2 (plan_id determinism, all 4×3 templates×intensities, and multi-origin area-count invariants). |
| `test_fire_propagation.py`        | Smoke tests for stage-3 (end-to-end run, determinism, smoke growth+decay, T@source > 600 °C). |
| `keyboard_teleop_fire.py`         | WASD/QE/S/R teleop with FireWorld overlay. Walk through the scene and watch the fire evolve as you take steps. |
| `keyboard_teleop.py`              | Legacy teleop (no fire overlay), kept for reference. |
| `test_detect_with_thermal.py`     | End-to-end check: gray RGB + thermal mask still yields a fire detection. |
| `test_fire_sensors.py`            | Multi-sensor suite test (voxel passthrough + 360° LIDAR stitching). |
| `test_radar_depth_reproject.py`   | Radar reprojection check. |
| `compare_radar_depth_ep0.py`      | Radar vs depth visual diff. |

---

## 4. Standard recipes (commands you'll actually run)

All commands below assume the `co-nav` conda env. Replace
`Nfvxx8J5NCo` with any HM3D scene short id you have on disk.

### A) Build inventory + plan + propagation + top-down video for a scene

```bash
# 1. Scan the scene → scenes/<id>/inventory.json + structural/*.npy
python scripts/build_inventory.py --scene 00880-Nfvxx8J5NCo

# 2. Generate a plan → scenes/<id>/plans/<plan_id>.json
python -m utils.fire_world.planner \
    --scene Nfvxx8J5NCo \
    --fire_type bedroom_textile --intensity severe --seed 7

# 3. Propagate → outputs/fire_world/<id>/<plan_id>/timeline.npz
python -m utils.fire_world.propagation \
    --scene Nfvxx8J5NCo --plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --voxel_m 0.15 --dt 0.5 --save_dt 2.0
```

对已经烘焙的 timeline 生成完整场景最终状态俯视图，可使用正交纹理底图
渲染器。它取 timeline 最后一帧，将垂直最大 flame、地面至 1.5 m 的平均
smoke 和所有 ignition source 按真实世界坐标叠加：

```bash
python scripts/render_fire_final_topdown.py \
    --scene 00880-Nfvxx8J5NCo \
    --plan-id Nfvxx8J5NCo_multi_origin_medium_24e63421b9fa \
    --width 1600 --gpu-device-id 0
```

默认输出为
`outputs/fire_world/<scene>/<plan_id>/final_topdown.png`，并在同目录写入
包含最终时刻、画面范围、火焰/烟雾覆盖率和火源像素坐标的 JSON 元数据。
俯视叠加使用带多尺度纹理的灰黑烟雾、低透明度热晕和半透明火焰核心；即使
两种 hazard 都达到最高显示强度，也会保留底图家具纹理，不把火焰画成不透明
的 segmentation/heatmap 色块。

如果要在 person ObjectNav 场景中显示数据集原生静态目标，可同时提供 person
scene shard 和 episode id。渲染器会从 `goals_by_category` 读取固定 person 世界
坐标，并把它写入 PNG 与同名 JSON，而不是人工猜测人物位置：

```bash
python scripts/render_fire_final_topdown.py \
    --scene 00880-Nfvxx8J5NCo \
    --plan-id Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
    --person-dataset-path \
      data/processed/fire_route_scenarios/\
Nfvxx8J5NCo_person_ep10_fork_detour_three_source/content/Nfvxx8J5NCo.json.gz \
    --person-episode-id 10 \
    --output outputs/fire_world/Nfvxx8J5NCo/\
Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab/final_topdown_person.png
```

Available `--fire_type` values: `kitchen_grease_fire`, `bedroom_textile`,
`living_room_electric`, `multi_origin`. `--intensity`: `light`, `medium`,
`severe`.

### B) Run all smoke tests

```bash
python scripts/test_scene_scan.py
python scripts/test_fire_planner.py
python scripts/test_fire_propagation.py
python scripts/test_fire_sensors.py
python scripts/test_detect_with_thermal.py
```

All scripts exit `0` and print `ALL OK` on success.

### C) Synthetic camera demo (no Habitat needed)

The standalone synthetic-camera demo has been removed. To view the
voxel render interactively, use the keyboard teleop below (section D),
which drives the same `VoxelSmokeSensor` through `FireSensorSuite`.

### D) Keyboard teleop with live fire overlay (interactive)

```bash
python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --num-agents 1 --agent-id 0 \
    --scene-id Nfvxx8J5NCo \
    --plan-id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --steps-per-unit 5 --seconds-per-unit 2.0 \
    --depth_use_clean 1
```

Window shows three panels: smoky RGB · thermal · depth-colormap. The
header shows step / `t_sim` / mean transmittance / flame_px. Press
`R` to reset back to `t_sim=0`.

Keys: `W` forward · `A` left · `D` right · `Q` look down · `E` look up
· `S` stop · `R` reset · `Esc` quit.

### E) Full navigation run with fire-world

```bash
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --nav_mode gpt --num_agents 1 \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_steps_per_unit 5 --fire_seconds_per_unit 2.0 \
    --depth_use_clean 1 --use_thermal_perception 1 \
    --visualize 0 --print_images 1
```

Note: `--task_config` must point at a config whose dataset shard
contains episodes for the target scene. If it doesn't, use
`scripts/keyboard_teleop_fire.py` (it picks the first matching
episode by scene short-id).

---

## 5. Time semantics ⚠️ important

The fire engine has **no notion of wall-clock time**. We translate
robot steps to "fire-time" via `FireClock`:

```
t_sim = (robot_step // steps_per_unit) * seconds_per_unit + base_t0
```

So `--fire_steps_per_unit 5 --fire_seconds_per_unit 2.0` means:
*every 5 robot steps the simulated fire advances by 2 s of timeline.*
At robot_step=0 the timeline is at `base_t0` (default 0 s, the start
of the npz); at robot_step=240 it's at 96 s.

`base_t0` defaults to the first time stamp in the timeline; pass a
non-zero value to start "in medias res".

The clock saturates at the last frame of the timeline (`fw.times[-1]`,
typically 600 s or 900 s depending on intensity). After that the fire
stays in the burnt-out state.

---

## 6. Known issues + gotchas

### 6.1 EGL crash with Open3D + habitat-sim

Symptom (you'll see this in stderr):

```
Platform::WindowlessEglContext: cannot make context current: EGL_SUCCESS
Engine::shutdown() called from the wrong thread!
```

Root cause: `main.py` starts an **Open3D GUI thread** when
`--visualize 1`. Open3D internally creates a Filament EGL context. On
WSL2 / NVIDIA, habitat-sim's EGL initialiser cannot grab another
context in the same process, so `eglMakeCurrent` silently fails.

Workarounds (try in order):
1. **Run with `--visualize 0`** (this is the easy fix; the keyboard
   teleop scripts do this by default).
2. Pin the EGL platform: `export EGL_PLATFORM=surfaceless`.
3. If the above don't work, move the Open3D GUI to a `multiprocessing.Process`
   so it has its own EGL.

### 6.2 v1 → v2 inventory schema migration

`scenes/<id>/inventory.json` is **schema v2** (629 instances,
structural masks, etc.). The legacy v1 schema only had `objects` (the
6 ObjectGoal categories). Every downstream module already supports
both via `_inventory_pool` (see `utils/fire_world/templates.py`).
If you hand a v1 inventory to the new propagation, it still runs but
without wall barriers (because there's no `structural.*_voxel_path`).

### 6.3 Smoke / flame is contained by the room

If the fire is in a closed room the plume **stays in that room**
(walls are zero-flux). This is correct physics but can look like the
fire "didn't spread". To get scene-wide smoke, use the
`multi_origin` template or a `severe` intensity so several rooms
ignite simultaneously.

### 6.4 Two abandoned experiments worth keeping note of

- `e9445bf` (reverted by `77981a6`): smoke RGB turbulence field. We
  tried replacing the pixel-iid noise in `SmokeRGBSensor` with a
  multi-octave Gaussian-blurred field, but the result looked noisier,
  not smoother. Reverted; current `SmokeRGBSensor` is back to the
  original noise model. Don't redo this without a better visual test.
- The `--rgb_dehaze` flag exists but is off by default. It works
  best with `--depth_use_clean=1`; otherwise the inverse
  Beer-Lambert blows up at far range.

### 6.5 `view_height_off` clamping

`scripts/demo_fire_world_runtime.py` clamps the eye to the world AABB
to keep rays inside the building. If you increase `--radius_m`
beyond the room's half-extent, the camera ends up against the AABB
edge and rays start essentially perpendicular to the ignition - the
Beer-Lambert integration becomes degenerate. Stay within `radius_m
<= 4.0` for typical HM3D scenes.

### 6.6 `t_sim` autocalibration tip

When you don't know the right `t_sim`, run with `--t_sim -1` (auto-
peak in the demo script). The orchestrator picks the frame where the
total smoke field is maximal, which usually correlates with the most
visually interesting moment.

---

## 7. Material / propagation parameters in one place

```python
# utils/fire_world/scene_scan.py
MATERIAL_TABLE = {
    "bed":         (0.75, 0.80),    # (flammability, smoke_yield)
    "couch":       (0.80, 0.85),
    "stove":       (0.85, 0.70),
    "oven and stove": (0.85, 0.70),
    "ventilation hood": (0.20, 0.30),
    ...                              # exact HM3D semantic.txt categories
    "_default":    (0.30, 0.30),
}
STRUCTURAL_CATEGORIES = {"wall", "floor", "ceiling", "door", "window", "stairs", ...}
```

```python
# utils/fire_world/templates.py - intensity presets
INTENSITIES = {
    "light":  IntensityPreset(n_min=1, n_max=1, T_src=550, fuel=2.0,  duration=300),
    "medium": IntensityPreset(n_min=1, n_max=2, T_src=750, fuel=5.0,  duration=600),
    "severe": IntensityPreset(n_min=2, n_max=3, T_src=950, fuel=10.0, duration=900),
}
# Template v2 category literals are exact semantic.txt spellings.
# Primary and secondary selection are both category-restricted; there is
# no arbitrary-object fallback.
# Templates: kitchen_grease_fire, bedroom_textile, living_room_electric,
# multi_origin.
```

```python
# utils/fire_world/propagation.py defaults (consumed via plan["propagation_rules"])
{
    "flammable_threshold":      0.4,
    "ignition_temp_c":          350.0,
    "spread_speed_m_per_s":     0.04,
    "ceiling_jet_speed_m_per_s": 0.30,
    "buoyancy_v_m_per_s":       0.5,
    "thermal_diffusivity":      0.05,
    "ambient_temp_c":           25.0,
    "k_burn_per_s":             1/240,
    "q_release_c":              350.0,
}
```

---

## 8. Roadmap / what is **not** yet done

- [ ] **LLM-driven planner** (`planner.py --use_llm`). Today the
      planner picks ignitions via four hand-written templates. The
      schema (`plan.json`) was designed to also be the output of an
      LLM that grounds on the inventory's `semantic_summary`. The
      hash-cache key (`plan_id`) is already template-version aware,
      so the LLM-mode plans will live alongside the template-mode
      plans without collision.
- [ ] **Burn-through / collapse**: when a wall instance's fuel goes
      to 0, it currently keeps blocking heat. We'd want to stop
      blocking once it's destroyed (and update the agent's obstacle
      map).
- [ ] **Multi-floor plumes**: ceiling jets currently spread on the
      top voxel layer of the world AABB. With staircases / open
      voids we should let smoke bleed to the floor above. The
      `floors` array already separates levels; needs a small
      propagation tweak.
- [ ] **mp4 writers without OpenCV**: the topdown / runtime mp4 are
      written by `cv2.VideoWriter` with `mp4v`. On some Linux builds
      the codec is missing and you get an empty file. If that happens,
      switch to `XVID` + `.avi` or use `imageio[ffmpeg]`.
- [ ] **Smoothing the runtime renderer** with trilinear (instead of
      nearest-neighbour) voxel sampling. The aliasing is mostly
      hidden by the Beer-Lambert integration but is visible at
      voxel boundaries when the camera is close to a flame voxel.
- [ ] **EGL isolation**: see issue 6.1. Open3D should run in its
      own subprocess so habitat-sim is the sole owner of EGL.
- [ ] **Auto inventory for all val scenes**: `build_inventory.py
      --all val` works but takes ~30 s per scene; nobody has run
      it across the full 100-scene split yet.

---

## 9. Quick commit map (for `git log`)

```
74f7c44 keyboard_teleop_fire.py
bb513a1 main.py / main_vec.py wired to FireWorldController
87273da stage 5: runtime + ray-march + demo
67b4671 'benchmark' (00880 inventory + plans + build_inventory.py)
d5b4d3b stage 1 v2: full HM3D inventory, sRGB OETF colour fix,
        structural barriers in propagation
9a26e4a stage 4: top-down validation video
d4c6edd stage 3: voxel propagation engine
aa2287f stage 2: deterministic plan.json from templates
55c2119 stage 1 v1 (deprecated by d5b4d3b)
77981a6 revert turbulence field
e9445bf turbulence field (reverted)
c7f02c9 smoke perception layer (clean depth + thermal-driven detection)
82fce04 sensor customization (the original 2D fire suite)
```

---

## 10. TL;DR

```bash
# 1. Pick / generate an inventory + plan + timeline:
python scripts/build_inventory.py --scene 00880-Nfvxx8J5NCo
python -m utils.fire_world.planner --scene Nfvxx8J5NCo \
       --fire_type bedroom_textile --intensity severe --seed 7
python -m utils.fire_world.propagation --scene Nfvxx8J5NCo \
       --plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632
python -m utils.fire_world.topdown_video --scene Nfvxx8J5NCo \
       --plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632

# 2. See the fire from the agent's eye, manually:
python scripts/keyboard_teleop_fire.py \
       --task-config configs/multi_objectnav_hm3d.yaml \
       --scene-id Nfvxx8J5NCo --plan-id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
       --depth_use_clean 1

# 3. Run the full navigation stack with a live fire:
python main.py --task_config multi_objectnav_hm3d.yaml \
       --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
       --depth_use_clean 1 --use_thermal_perception 1 \
       --visualize 0
```

That's the whole loop. If anything is unclear, the per-stage scripts
under `scripts/test_*.py` are the fastest way to confirm what works.
