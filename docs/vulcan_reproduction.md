# Reproducing the VULCAN Branch

This guide reproduces the current `vulcan` branch of Co-NavGPT2: Habitat 3
multi-robot ObjectNav, VLM frontier assignment, FireWorld generation and
rendering, thermal/radar/LIDAR sensing, static-person ObjectNav, and dynamic
risk-aware navigation.

The commands assume Linux x86-64, a Bash-compatible shell, and that they are
run from the repository root unless stated otherwise.

## 1. Reproducibility scope

Record the exact project revision before running an experiment:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

The branch must be `vulcan`. A commit hash identifies only committed files;
uncommitted changes and generated `outputs/` are not reproducible from Git.
Commit or archive the final experiment state before publishing results.

The validated software baseline is:

| Component | Version |
| --- | --- |
| Python | 3.9.19 |
| PyTorch | 2.0.1 + CUDA 11.8 |
| torchvision | 0.15.2 + CUDA 11.8 |
| torchaudio | 2.0.2 + CUDA 11.8 |
| Habitat-Sim | 0.3.3, Bullet build |
| Habitat-Lab | tag `v0.3.3`, commit `094d6be2f9d057e4781a68ae792132895fd4d3d0`, plus the VULCAN patch below |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0 |
| Open3D | 0.19.0 |
| scikit-fmm | 2023.4.2 |
| scikit-image | 0.24.0 |
| Hydra / OmegaConf | 1.3.4 / 2.3.1 |
| OpenAI Python client | 2.44.0 |
| Ultralytics | 8.4.88 |
| Supervision | 0.19.0 |
| numpy-quaternion | 2023.0.4 |

An NVIDIA GPU is required by the current navigation agent because the
YOLO-World and MobileSAM models are placed on `cuda:<gpu_id>`. Habitat RGB-D
rendering also requires a working OpenGL/EGL setup. CPU-only execution is
supported for the focused unit tests and most synthetic FireWorld sensor tests,
but not for a full `main.py` episode without code changes.

## 2. Clone and pin the project

```bash
git clone --branch vulcan https://github.com/SendingA/Co-NavGPT2.git
cd Co-NavGPT2

PROJECT_ROOT="$PWD"
git rev-parse HEAD
```

For a paper artifact, replace the moving branch with the published commit:

```bash
git checkout <VULCAN_COMMIT>
```

## 3. System and Conda environment

On Ubuntu, install the common native dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

Create the environment:

```bash
conda create -n co-nav3 python=3.9 cmake=3.14.0 -y
conda activate co-nav3

conda install -y \
    pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    pytorch-cuda=11.8 \
    -c pytorch -c nvidia

conda install -y habitat-sim=0.3.3 withbullet \
    -c conda-forge -c aihabitat
```

Install the repository dependencies, followed by the versions used for the
validated snapshot:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install \
    numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    open3d==0.19.0 \
    scikit-image==0.24.0 \
    hydra-core==1.3.4 \
    omegaconf==2.3.1 \
    openai==2.44.0 \
    ultralytics==8.4.88 \
    supervision==0.19.0 \
    numpy-quaternion==2023.0.4
```

### 3.1 Required Habitat-Lab patch

Stock Habitat-Lab 0.3.3 is not sufficient for this branch. The active runtime
uses a patch that provides the classic `Sim-v0` multi-agent list-of-actions and
list-of-observations contract, per-agent episode start states, direct
`LOOK_UP`/`LOOK_DOWN` simulator actions, and the matching multi-agent
measurement lifecycle.

Install the exact patched runtime:

```bash
HABITAT_LAB_ROOT="/path/to/habitat-lab-0.3.3"

git clone https://github.com/facebookresearch/habitat-lab.git \
    "$HABITAT_LAB_ROOT"
git -C "$HABITAT_LAB_ROOT" checkout \
    094d6be2f9d057e4781a68ae792132895fd4d3d0
git -C "$HABITAT_LAB_ROOT" apply \
    "$PROJECT_ROOT/ref/habitat_lab_0.3.3_vulcan.patch"

python -m pip install -e "$HABITAT_LAB_ROOT/habitat-lab"
python -m pip install -e "$HABITAT_LAB_ROOT/habitat-baselines"
```

Verify that Python imports the intended checkout and that the patch is present:

```bash
python - <<'PY'
import habitat
import habitat_sim
from habitat.config.default_structured_configs import SimulatorConfig

print("habitat:", habitat.__version__, habitat.__file__)
print("habitat_sim:", habitat_sim.__version__, habitat_sim.__file__)
assert habitat.__version__ == "0.3.3"
assert habitat_sim.__version__ == "0.3.3"
assert hasattr(SimulatorConfig(), "tilt_angle")
PY
```

Do not install a second Habitat-Lab package after this step; otherwise Python
may silently import the wrong checkout.

The editable `habitat-baselines` install is required only by
`--local_planner pointnav`, but installing it here keeps the environment
complete for every documented baseline.

Download the official PointNav DD-PPO checkpoint:

```bash
mkdir -p "$PROJECT_ROOT/data/ddppo-models"
wget --continue \
  https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth \
  -P "$PROJECT_ROOT/data/ddppo-models"
sha256sum \
  "$PROJECT_ROOT/data/ddppo-models/gibson-2plus-resnet50.pth"
```

Expected SHA-256:

```text
a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad
```

## 4. Detection model assets

The navigation agent loads both files from the repository root:

```text
Co-NavGPT2/
├── mobile_sam.pt
└── yolov8l-world.pt
```

Ultralytics can download them when they are first resolved:

```bash
python - <<'PY'
from ultralytics import SAM, YOLO

SAM("mobile_sam.pt")
YOLO("yolov8l-world.pt")
PY
```

Hashes from the validated snapshot:

```text
6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f  mobile_sam.pt
8bdfaef999116760247d6fb0b0f8fca064b43e94598b3d4a807ebae9bcf0cdd5  yolov8l-world.pt
```

Check them with:

```bash
sha256sum mobile_sam.pt yolov8l-world.pt
```

`weights/clip/ViT-B-32.pt` is a historical local artifact and is not read by
the current VULCAN navigation path.

## 5. HM3D scenes and ObjectNav episodes

HM3D is license-controlled and is not distributed by this repository. Follow
the Habitat HM3D download instructions and place the v0.2 assets under
`data/`. A minival download can be initiated with:

```bash
python -m habitat_sim.utils.datasets_download \
    --username <HM3D_TOKEN_ID> \
    --password <HM3D_TOKEN_SECRET> \
    --uids hm3d_minival_v0.2 \
    --data-path data/
```

The full VULCAN evaluation requires the HM3D v0.2 scenes used by the selected
ObjectNav split and the ObjectNav HM3D v2 episode files. The expected layout
is:

```text
data/
├── scene_datasets/
│   └── hm3d_v0.2/
│       ├── hm3d_annotated_basis.scene_dataset_config.json
│       ├── val/
│       │   └── 00880-Nfvxx8J5NCo/
│       │       ├── Nfvxx8J5NCo.basis.glb
│       │       ├── Nfvxx8J5NCo.basis.navmesh
│       │       ├── Nfvxx8J5NCo.semantic.glb
│       │       └── Nfvxx8J5NCo.semantic.txt
│       └── val_mini/
└── datasets/
    └── objectnav_hm3d_v2/
        ├── val/
        │   ├── val.json.gz
        │   └── content/*.json.gz
        └── val_mini/
            ├── val_mini.json.gz
            └── content/*.json.gz
```

Check the canonical VULCAN scene:

```bash
test -f \
  data/scene_datasets/hm3d_v0.2/val/00880-Nfvxx8J5NCo/Nfvxx8J5NCo.basis.glb
test -f data/datasets/objectnav_hm3d_v2/val/val.json.gz
```

### 5.1 Optional humanoid and robot assets

Random pedestrians and the static-person benchmark require Habitat humanoids:

```bash
python -m habitat_sim.utils.datasets_download \
    --uids habitat_humanoids --data-path data/
```

Visible Spot and Stretch overlays require:

```bash
python -m habitat_sim.utils.datasets_download \
    --uids hab_spot_arm --data-path data/
python -m habitat_sim.utils.datasets_download \
    --uids hab_stretch --data-path data/
```

Fetch assets normally arrive with Habitat-Sim. These URDF models are visual
kinematic overlays; the ObjectNav policy still controls the cylinder agents.

## 6. Configuration files

All first-party task configurations under `configs/` are listed below.

| File | Status | Purpose |
| --- | --- | --- |
| `configs/multi_objectnav_hm3d.yaml` | Current, required | Habitat 0.3.3 Hydra configuration for multi-agent HM3D ObjectNav. Defines RGB-D sensors, `Sim-v0`, 0.25 m forward steps, 30-degree turns, 0.2 m Success distance, and the `conav` humanoid/robot settings. |
| `configs/person_objectnav_hm3d.yaml` | Current, optional | Extends the multi-agent config with the generated static-person dataset and `static_person_goal=True`. |
| `configs/rl_local_planner_ppo.yaml` | Current, optional | PPO training defaults for separate risk-blind and risk-aware map-based RL local-planner checkpoints. |
| `configs/objectnav_hm3d.yaml` | Legacy reference | Habitat 0.2.1 YACS syntax. Do not pass it to the current Hydra `load_config()` path. |
| `configs/objectnav_gibson.yaml` | Legacy reference | Old Gibson/YACS configuration; not validated with the Habitat 0.3.3 VULCAN runtime. |
| `configs/human.yaml` | Legacy reference | Old two-agent YACS configuration; superseded by `multi_objectnav_hm3d.yaml` plus `--num_humans`. |

`arguments.py` applies CLI overrides after Hydra composition. The most
important experiment controls are:

```text
--seed
--task_config
--num_agents
--num_humans
--nav_mode {nearest,co_ut,fill,random,gpt}
--cost_utility_lambda 1.0
--random_goal_min_distance_m 1.0
--fire_world
--fire_world_plan_id
--fire_clock_mode {step,wallclock}
--risk_enabled
--risk_source {none,oracle,sensed}
```

Use `step` fire time, a fixed seed, a fixed plan ID, a fixed agent count, and
the same action budget for comparable benchmark results.

## 7. Environment verification

Run the CPU-capable repository suite before launching Habitat:

```bash
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/tmp/conav-matplotlib \
python -m unittest discover -s tests -v
```

Then verify the CLI and configuration imports:

```bash
python main.py --help
python scripts/build_inventory.py --help
python -m utils.fire_world.planner --help
python -m utils.fire_world.propagation --help
```

A minimal full navigation smoke test without OpenAI is:

```bash
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 1 --num_humans 0 \
    --nav_mode nearest \
    --fire_world 0 \
    --dump_location /tmp/conav-clean-smoke
```

## 8. Reproducing navigation experiments

### 8.1 Clean multi-agent baseline

```bash
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 \
    --num_humans 0 \
    --nav_mode co_ut \
    --seed 1 \
    --fire_world 0 \
    --dump_location outputs/clean_co_ut
```

### 8.2 VLM frontier assignment

Only `nav_mode=gpt` needs an OpenAI key:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 \
    --nav_mode gpt \
    --seed 1 \
    --fire_world 0 \
    --dump_location outputs/clean_gpt
```

The current API path is hard-coded to `gpt-4o`; `--gpt_type` does not switch
models. The VLM assigns global frontiers, while deterministic local planners
produce the robot actions.

### 8.3 Multiprocess evaluation

```bash
python main_vec.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 \
    --nav_mode co_ut \
    --risk_enabled 0 \
    -n 2 \
    --dump_location outputs/vector_co_ut
```

`main_vec.py` supports FireWorld rendering but intentionally refuses
`--risk_enabled=1`. Use `main.py` for synchronized risk maps and exposure
evaluation.

## 9. Reproducing FireWorld

FireWorld has three deterministic preparation stages:

```text
HM3D semantic scene
  -> scenes/<scene>/inventory.json + structural masks
  -> scenes/<scene>/plans/<plan_id>.json
  -> outputs/fire_world/<scene>/<plan_id>/timeline.npz
```

### 9.1 Build the semantic inventory

```bash
python scripts/build_inventory.py \
    --scene Nfvxx8J5NCo \
    --scene_dataset_root data/scene_datasets/hm3d_v0.2 \
    --objectgoal_root data/datasets/objectnav_hm3d_v2 \
    --out_root scenes \
    --voxel_m 0.10
```

### 9.2 Select or generate a plan

The active experiment uses the committed, frozen template-v1 plan:

```text
scenes/Nfvxx8J5NCo/plans/83679a07b632.json
```

It contains the corrected real HM3D `object_id=48`, category
`oven and stove`, rather than the old synthetic grease label:

```bash
python - <<'PY'
import json

path = "scenes/Nfvxx8J5NCo/plans/83679a07b632.json"
plan = json.load(open(path, encoding="utf-8"))
assert plan["template_version"] == 1
assert any(
    ignition["object_id"] == 48
    and ignition["category"] == "oven and stove"
    for ignition in plan["ignitions"]
)
print(path, "OK")
PY
```

Use this committed JSON directly when reproducing the existing
`83679a07b632` benchmark.

The current generator is template v2. Running the same semantic scenario now
creates a deliberately different plan:

```bash
python -m utils.fire_world.planner \
    --scene Nfvxx8J5NCo \
    --fire_type bedroom_textile \
    --intensity severe \
    --seed 7 \
    --scenes_root scenes \
    --plans_root scenes
```

Its expected ID is `020ff4f13cd0`, with the current bedroom-template
ignitions. This is a new benchmark condition: generate a matching timeline
and do not compare it to results labeled `83679a07b632`.

### 9.3 Generate the fire timeline

```bash
python -m utils.fire_world.propagation \
    --scene Nfvxx8J5NCo \
    --plan_id 83679a07b632 \
    --scenes_root scenes \
    --out_root outputs/fire_world \
    --voxel_m 0.15 \
    --dt 0.5 \
    --save_dt 1.0
```

The result is:

```text
outputs/fire_world/Nfvxx8J5NCo/83679a07b632/timeline.npz
```

This file can be hundreds of megabytes and is generated rather than committed.

### 9.4 Run FireWorld navigation

```bash
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 \
    --nav_mode co_ut \
    --seed 1 \
    --fire_world 1 \
    --fire_world_plan_id 83679a07b632 \
    --fire_clock_mode step \
    --fire_steps_per_unit 5 \
    --fire_seconds_per_unit 2.0 \
    --depth_use_clean 1 \
    --use_thermal_perception 1 \
    --fire_save_every 10 \
    --fire_dump_dir outputs/fire_sensors
```

## 10. Reproducing risk-aware navigation

The main sensed benchmark uses only sensor-derived belief for planning and an
independent FireWorld ground-truth provider for evaluation:

```bash
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 \
    --nav_mode co_ut \
    --seed 1 \
    --fire_world 1 \
    --fire_world_plan_id 83679a07b632 \
    --fire_clock_mode step \
    --fire_steps_per_unit 5 \
    --fire_seconds_per_unit 2.0 \
    --risk_enabled 1 \
    --risk_source sensed \
    --risk_smoke_source appearance_depth \
    --risk_geometry_depth_source clean \
    --risk_run_id sensed_seed1 \
    --risk_dump_dir outputs/risk_assessment \
    --risk_save_every 10
```

The continuous score is:

```text
H_phys = 0.60 * normalized_temperature + 0.40 * smoke
```

Flame is not a third continuous weight; it remains a hard exclusion mask with
safety dilation. Report the following primary benchmark columns:

```text
success
spl
risk/safe_success
risk/che
```

For controlled ablations, keep the scene, episode, seed, plan, agent count,
clock, and thresholds fixed and change only:

```bash
# Evaluator-only: legacy planner, independent GT exposure.
--risk_enabled 1 --risk_source none

# Privileged planner upper bound.
--risk_enabled 1 --risk_source oracle
```

See `docs/risk_assessment.md` for the information boundary and metric
definitions.

## 11. Reproducing the static-person benchmark

Download the humanoid assets first, then generate and validate the dataset:

```bash
python scripts/build_person_objectnav_dataset.py --split val_mini
python scripts/build_person_objectnav_dataset.py \
    --split val_mini --validate-only

python scripts/build_person_objectnav_dataset.py --split val
python scripts/build_person_objectnav_dataset.py \
    --split val --validate-only
```

Run it through the normal ObjectNav pipeline:

```bash
python main.py \
    --task_config person_objectnav_hm3d.yaml \
    --num_agents 2 \
    --nav_mode co_ut \
    --seed 1
```

`person` uses the same point-cloud goal map, FMM/greedy STOP path, and Habitat
Success/SPL measurements as the other ObjectNav categories.

## 12. Optional pedestrians and visible robot models

```bash
# Two random humanoid pedestrians.
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 --num_humans 2 \
    --nav_mode nearest

# Visible Spot and Fetch overlays.
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 \
    --robot_models_enabled 1 \
    --robot_profiles spot,fetch \
    --nav_mode nearest
```

## 13. Teleoperation

OpenCV windows must be focused for keyboard input. On a headless server, use a
working X11 forwarding or EGL/virtual-display setup.

```bash
# Clean Habitat RGB-D.
PYTHONPATH="$PROJECT_ROOT" python scripts/keyboard_teleop.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --num-agents 1 \
    --agent-id 0 \
    --scene-id Nfvxx8J5NCo \
    --show-depth 1

# FireWorld overlay and sensor dashboard.
python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --num-agents 1 \
    --scene-id Nfvxx8J5NCo \
    --plan-id 83679a07b632 \
    --clock-mode step \
    --steps-per-unit 5 \
    --seconds-per-unit 2.0 \
    --show-dashboard 1

# Unified fire, humans, and visible robot dashboard.
python scripts/keyboard_teleop_full.py \
    --task-config multi_objectnav_hm3d.yaml \
    --num-agents 1 \
    --num-humans 2 \
    --robot-models-enabled 1 \
    --robot-profiles spot \
    --scene-id Nfvxx8J5NCo \
    --plan-id 83679a07b632 \
    --clock-mode wallclock \
    --lidar-360 1 --lidar-resolution 320 \
    --snapshot-dir outputs/teleop_sensor_snapshots \
    --show-dashboard 1
```

Controls:

```text
W/A/D       forward/left/right
Q/E         look down/up
S or Space  stop
R           reset
Tab, 1..N   switch active robot in keyboard_teleop_full.py
P           pause/resume wall-clock fire
V           save all individual sensor panels and the dashboard
Mouse       click SAVE SENSOR PANELS in the dashboard header
Esc         quit
```

The unified teleoperation entrypoint enables four-slice 360-degree LiDAR by
default. Each manual snapshot is written below
`--snapshot-dir/<scene>/agent_<id>/step_<step>_<timestamp>/` with the nine
sensor panels, the composed dashboard, and a JSON manifest.

## 14. Script inventory

Every file directly under `scripts/` is included here.

| Script | Purpose and representative invocation |
| --- | --- |
| `scripts/_diag_depth_effect.py` | Empty historical placeholder. It has no executable behavior and must not be used as a benchmark command. |
| `scripts/build_inventory.py` | Builds FireWorld inventories and structural masks. Run `python scripts/build_inventory.py --scene Nfvxx8J5NCo`. |
| `scripts/build_person_objectnav_dataset.py` | Generates or validates the static-person ObjectNav dataset. Run `python scripts/build_person_objectnav_dataset.py --split val_mini`. |
| `scripts/compare_radar_depth_ep0.py` | Reprocesses dumped `ep_0000` sensor arrays and writes radar/depth comparison panels and CSV summaries. Run after a run with `--fire_save_npz 1`: `python scripts/compare_radar_depth_ep0.py`. Paths are currently fixed to `outputs/fire_sensors/agent_0/ep_0000/agent_0`. |
| `scripts/keyboard_teleop.py` | Clean Habitat 0.3.3 keyboard teleoperation. Use the command in Section 13; `PYTHONPATH` avoids its workstation-specific compatibility path. |
| `scripts/keyboard_teleop_fire.py` | Keyboard teleoperation with a precomputed FireWorld timeline and optional dashboard. |
| `scripts/keyboard_teleop_full.py` | Unified multi-agent, humanoid, robot-model, FireWorld, thermal, radar, and LIDAR teleoperation dashboard. |
| `scripts/render_thermal_validation.py` | Writes a deterministic thermal validation montage and JSON statistics under `outputs/thermal_validation/`. Run `python scripts/render_thermal_validation.py`. |
| `scripts/train_rl_local_planner.py` | Trains the checkpointed map-based PPO local planner on randomized connected occupancy/risk grids. See `docs/local_planner_baselines.md`; one-update smoke runs are not benchmark checkpoints. |
| `scripts/test_detect_with_thermal.py` | Stubs YOLO/SAM and verifies thermal-only fire detection. Run `python scripts/test_detect_with_thermal.py`. |
| `scripts/test_fire_planner.py` | Smoke-tests deterministic FireWorld plan creation against the committed TEEsav fixture. Run `python scripts/test_fire_planner.py`. |
| `scripts/test_fire_propagation.py` | Runs deterministic propagation and physical-range checks on the TEEsav fixture. Run `python scripts/test_fire_propagation.py`. |
| `scripts/test_fire_sensors.py` | Synthetic FireSensorSuite smoke test, including 360-degree LIDAR stitching. Run `python scripts/test_fire_sensors.py`. |
| `scripts/test_human_thermal.py` | Verifies physical person temperature and thermal overlay behavior. Run `python scripts/test_human_thermal.py`. |
| `scripts/test_radar_depth_reproject.py` | Generates a synthetic radar/depth reconstruction panel under `outputs/radar_depth_recon/`. Run `python scripts/test_radar_depth_reproject.py`. |
| `scripts/test_scene_scan.py` | Tests HM3D semantic inventory schema and material lookup using the TEEsav fixture. Run `python scripts/test_scene_scan.py`. |

FireWorld also exposes these module entry points:

| Module | Purpose |
| --- | --- |
| `python -m utils.fire_world.hm3d_semantic --glb <semantic.glb> --txt <semantic.txt>` | Inspect semantic mesh primitives and instance annotations. |
| `python -m utils.fire_world.scene_scan --scene <scene>` | Single-scene inventory builder; `scripts/build_inventory.py` is the more convenient multi-scene wrapper. |
| `python -m utils.fire_world.planner --scene <scene> --fire_type <type>` | Deterministic plan generator. |
| `python -m utils.fire_world.propagation --scene <scene> --plan_id <id>` | Voxel timeline generator. |

Top-level runtime scripts:

| File | Purpose |
| --- | --- |
| `main.py` | Primary single-process Habitat/VLM/FireWorld/risk benchmark entry point. |
| `main_vec.py` | Multiprocess scene evaluation; risk mode is intentionally unsupported. |
| `ros_multi_nav.py` | ROS 2 multi-robot runtime; FireWorld risk mode is intentionally unsupported because no real-sensor risk provider is implemented. |
| `ros_single_nav.py` | ROS 2 single-robot validation runtime. |
| `multi_lidar_icp.py` | Real-robot multi-LIDAR registration utility. |

`arguments.py`, `constants.py`, and `system_prompt.py` are shared control files,
not standalone experiment scripts.

## 15. Outputs and metrics

| Output | Default location |
| --- | --- |
| Navigation logs | `<dump_location>/logs/<nav_mode>/` |
| Navigation maps/images | `<dump_location>/dump/<nav_mode>/` |
| Fire sensor frames | `outputs/fire_sensors/` |
| Fire timelines | `outputs/fire_world/<scene>/<plan_id>/` |
| Risk traces and summaries | `outputs/risk_assessment/<run_id>/rank_000/` |
| Thermal validation | `outputs/thermal_validation/` |
| Radar reconstruction diagnostics | `outputs/radar_depth_recon/` and `outputs/radar_vs_depth_ep0/` |

For the risk benchmark, the primary table is Habitat `Success`, Habitat `SPL`,
`risk/safe_success`, and `risk/che`. Keep diagnostic traces, but do not
substitute correlated peak/time/path-risk variants into the primary table.

## 16. Troubleshooting

### `AssertionError: No action 4/5 in action space`

The VULCAN Habitat patch was not applied, or Python imported another Habitat
checkout. Repeat Section 3.1 and print `habitat.__file__`.

### `main.py` returns one observation dict instead of one dict per robot

This is also a missing or shadowed Habitat patch. Stock Habitat-Lab 0.3.3 does
not provide the exact classic multi-agent `Sim-v0` contract used here.

### `OPENAI_API_KEY` is missing

Use `--nav_mode nearest`, `co_ut`, `fill`, or `random` for an offline run, or
export the key before `--nav_mode gpt`. `random` is reproduced by the same
`--seed`; `co_ut` evaluates `frontier_size - cost_utility_lambda ×
robot_grid_distance`.

### Model files download at runtime

Pre-download `mobile_sam.pt` and `yolov8l-world.pt` as described in Section 4
and record their SHA-256 hashes.

### OpenCV/Open3D windows fail on a headless server

Run without `--visualize`, `--fire_show_window`, or teleoperation windows.
For interactive runs, configure X11 forwarding, VirtualGL, or a suitable
virtual display.

### Risk mode fails in `main_vec.py` or ROS

This is intentional. The synchronized `RiskRuntime` is currently connected
only to `main.py`.

## 17. Experiment record

Archive the following with every reported run:

```text
project commit and git status
Habitat-Lab commit and patch SHA-256
conda/pip package export
GPU and driver information
HM3D/ObjectNav split
scene and episode IDs
FireWorld inventory and plan JSON
FireWorld plan ID, seed, timeline parameters, and clock mode
navigation mode and agent count
risk source and thresholds
model asset hashes
OpenAI model name when VLM assignment is enabled
```

Useful capture commands:

```bash
git rev-parse HEAD
git status --short
git -C "$HABITAT_LAB_ROOT" rev-parse HEAD
sha256sum ref/habitat_lab_0.3.3_vulcan.patch
conda env export --no-builds > outputs/conda_environment.yml
python -m pip freeze > outputs/pip_freeze.txt
```
