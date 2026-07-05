# Habitat 0.2.1 → 0.3.3 Migration

> Date: 2026-07-04
> Scope: whole repository
> Target: **habitat-sim 0.3.3 + habitat-lab 0.3.3**, Python 3.9
> Reference: `Co-NavGPTv3/` (self-contained demo of a Habitat 3 multi-robot + humanoid ObjectNav rollout)

The main project used to depend on a monkey-patched fork of habitat-lab
0.2.1 shipped under `multi-robot-setting/habitat/` — that fork bolted
multi-agent support onto Habitat 2 by adding `SIMULATOR.NUM_AGENTS` and
`SIMULATOR.AGENTS` YACS keys and by hacking `HabitatSim.create_sim_config`
to replicate the same `AgentConfiguration` across agents. Habitat-Lab
0.3.3 supports multi-agent natively via `habitat.simulator.agents_order`
+ `habitat.simulator.agents.<name>`, so the whole patched fork became
dead code. It also introduces `KinematicHumanoid` + humanoid
controllers, which let us put pedestrians in the ObjectNav scene.

The migration was intentionally surgical: the VLM planner, FMM planner,
frontier assignment logic, mapping stack, fire pipeline, and CLI
surface all stayed the same. Only the plumbing that talks to Habitat
changed.

## 1. Files removed

| Path | Why |
| --- | --- |
| `multi-robot-setting/` (whole tree) | H2-patched habitat-lab package superseded by upstream 0.3.3 multi-agent support. |
| `utils/shortest_path_follower.py` (H2 version) | Rewritten below. |
| `=0.29.1`, `Co-NavGPTv3(1).zip`, `data.zip` | Stray build artefacts at the repo root. |

## 2. Files added

| Path | Purpose |
| --- | --- |
| `envs/__init__.py` | Re-exports `RandomHumanoidWalker` + `RobotModelManager`. |
| `envs/random_humanoid.py` | Habitat 3 humanoid pedestrians (KinematicHumanoid + HumanoidRearrangeController) spawned as random walkers. Ported from `Co-NavGPTv3/envs/random_humanoid.py`. |
| `envs/robot_models.py` | Loads visible Habitat 3 robot URDFs (Fetch / Spot / Stretch) as kinematic models synced to each nav agent. Ported from `Co-NavGPTv3/envs/robot_models.py`. |
| `docs/habitat3_migration.md` | This document. |

## 3. Files rewritten

| Path | What changed |
| --- | --- |
| `arguments.py` | New `load_config(args)` composes a Hydra `DictConfig` via `habitat.config.default.get_config`, replicates the template agent into `--num_agents` entries, applies CLI overrides on top. Adds humanoid + robot-model + dataset-path flags. All existing flags kept for backwards compatibility. |
| `configs/multi_objectnav_hm3d.yaml` | YACS → Hydra `@package _global_` config that inherits `/benchmark/nav/objectnav/objectnav_hm3d` and overrides two agents at 0.88 m with the RGB + Depth stack we've been using. Adds a `conav` group with humanoid and robot-model knobs. |
| `main.py` | Config load through `arguments.load_config`; sim-agents count read from `config.conav.num_robots`; humanoid + robot-model managers wired in; per-frame flow tolerates the H3 sim returning a single dict for single-agent runs and a list for multi-agent runs. |
| `main_vec.py` | Same edits as `main.py`, plus per-worker `content_scenes` + `scene` overrides via `habitat.config.read_write`. Uses `make_dataset` on the DictConfig-native dataset spec. |
| `utils/shortest_path_follower.py` | Rewritten as an H3.3 compat wrapper: same public API (`get_next_action(goal_pos, current_grid_pose, angle, stg_x, stg_y)` + `get_path_points`) but talks to `sim.habitat_config.forward_step_size`, `sim.habitat_config.turn_angle`, and `HabitatSimActions.{stop,move_forward,turn_left,turn_right}` singletons. |
| `utils/fire_pipeline.py` | `step_fire_observation` now resolves the depth-sensor max_depth / normalize_depth from either the H3 DictConfig or the legacy YACS config so unit tests keep passing. |
| `utils/fire_world/scene.py` | `FireScene.from_args` accepts either `config.habitat.simulator.scene` (H3) or `config.SIMULATOR.SCENE` (H2). |
| `utils/fire_world/controller.py` | Same DictConfig-vs-YACS dispatch when reading `max_depth`. |
| `utils/fire_sensors/lidar_360.py` | `install_lidar_depth_sensors` rewritten around the H3 DictConfig schema (add new `sim_sensors.<uuid>` entries under every agent). Must be called inside `habitat.config.read_write`. |
| `scripts/keyboard_teleop.py` | Uses `arguments.load_config`, lowercase `HabitatSimActions` names, tolerant of list-vs-dict step returns. |
| `scripts/keyboard_teleop_fire.py` | Same treatment; camera intrinsics pulled from the DictConfig sensors. |
| `README.md` | Installation section now targets habitat-sim / habitat-lab 0.3.3 and documents the humanoid + Spot/Stretch asset downloads. |
| `docs/main_usage.md` | New top banner explaining the migration, plus section 2.3.1 for the humanoid + robot-model flags. |

## 4. Config schema diff

| Habitat 0.2.1 (YACS) | Habitat 0.3.3 (Hydra + OmegaConf) |
| --- | --- |
| `config.SIMULATOR.SCENE` | `config.habitat.simulator.scene` |
| `config.SIMULATOR.NUM_AGENTS` | `len(config.habitat.simulator.agents_order)` |
| `config.SIMULATOR.AGENTS = ["AGENT_0", ...]` | `config.habitat.simulator.agents_order = ["main_agent", "agent_1", ...]` |
| `config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]` | `config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor / depth_sensor` (map keyed by sensor name) |
| `config.SIMULATOR.RGB_SENSOR.WIDTH / HEIGHT / HFOV` | `config.habitat.simulator.agents.<name>.sim_sensors.rgb_sensor.width / height / hfov` |
| `config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH / MAX_DEPTH` | `... .sim_sensors.depth_sensor.min_depth / max_depth` |
| `config.SIMULATOR.FORWARD_STEP_SIZE`, `config.SIMULATOR.TURN_ANGLE` | `config.habitat.simulator.forward_step_size / turn_angle` |
| `config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID` | `config.habitat.simulator.habitat_sim_v0.gpu_device_id` |
| `config.DATASET.TYPE / DATA_PATH / SCENES_DIR / CONTENT_SCENES / SPLIT` | `config.habitat.dataset.type / data_path / scenes_dir / content_scenes / split` |
| `config.TASK.POSSIBLE_ACTIONS` | Discrete cylinder actions live under `config.habitat.task.actions`; keyboard scripts hardcode the singleton `{stop: 0, move_forward: 1, turn_left: 2, turn_right: 3, look_up: 4, look_down: 5}`. |
| `config.ENVIRONMENT.MAX_EPISODE_STEPS` | `config.habitat.environment.max_episode_steps` |
| `config.defrost() / .freeze()` | `with habitat.config.read_write(config): ...` |

## 5. Runtime API diff

| Habitat 0.2.1 | Habitat 0.3.3 |
| --- | --- |
| `HabitatSimActions.MOVE_FORWARD` (uppercase enum) | `HabitatSimActions.move_forward` (lowercase singleton) |
| `env.step(actions_list)` returns `List[Observations]` (**only via the H2 patched fork**) | Same shape — upstream 0.3.3 now natively returns a list for multi-agent runs, a single `Observations` dict for single-agent. Our code guards with `if not isinstance(observations, list): observations = [observations]`. |
| `env.reset()` | Same guard as above. |
| `sim.get_agent_state(i)` → `AgentState(position=np.ndarray, rotation=numpy.quaternion, sensor_states=Dict[str, SixDOFPose])` | Same. `AgentState.sensor_states["depth"]` still works because our sensors use the bare `"rgb"` / `"depth"` uuids from the `RGBSensor` / `DepthSensor` base classes. |
| `sim.get_straight_shortest_path_points(a, b)` | Same. |
| `ShortestPathFollowerCompat(sim, radius, return_one_hot, agent_id)` (custom shim) | Kept the same public API but the wrapper now uses `HabitatSimActions.stop / move_forward / turn_left / turn_right` and reads `sim.habitat_config.forward_step_size / turn_angle`. |

## 6. New Habitat 3 features exposed

Both are optional and off by default so `python main.py` keeps behaving
exactly like the H2 version once the environment is upgraded.

* **Humanoid pedestrians** (`--num_humans N`) via
  `envs.random_humanoid.RandomHumanoidWalker`. Each humanoid loads a
  URDF from `data/humanoids/humanoid_data/*/`, a `KinematicHumanoid`
  articulated agent, and a `HumanoidRearrangeController`. On every
  `env.step()` the walker (a) picks a fresh random navigable target
  when the current one is reached, (b) asks the pathfinder for a
  waypoint, (c) drives the controller one motion frame, (d) applies
  the resulting joint pose + root transform back onto the humanoid.
  Cycles through `conav.human_urdfs` if `N > len(list)`.

* **Visible robot URDF models** (`--robot_models_enabled 1
  --robot_profiles spot,fetch`) via `envs.robot_models.RobotModelManager`.
  Loads a `SpotRobot` / `FetchRobot` / `StretchRobot` / etc. URDF as a
  **kinematic** articulated agent, then syncs its `base_pos` / `base_rot`
  to the corresponding classic nav agent after every step. The
  navigation policy still runs on the nav agent — the URDF is purely
  a visual overlay.

## 7. Running the migrated project

```bash
# clean 2 robots on hm3d val
python main.py

# 2 robots + 3 humanoid pedestrians
python main.py --num_humans 3

# 2 robots + visible Spot on agent 0 and Fetch on agent 1
python main.py --robot_models_enabled 1 --robot_profiles spot,fetch

# switch dataset without touching the yaml
python main.py --dataset_path data/datasets/objectnav_hm3d_v2/{split}/{split}.json.gz \
               --scenes_dir data/scene_datasets \
               --scene_dataset data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json

# fire-scene evaluation (unchanged)
python main.py --fire_world 1 --fire_world_plan_id 83679a07b632 --fire_sensors 1

# vectorised eval (2 workers)
python main_vec.py -n 2

# manual teleop
python scripts/keyboard_teleop.py --task-config configs/multi_objectnav_hm3d.yaml \
    --num-agents 1 --agent-id 0 --scene-id Nfvxx8J5NCo --show-depth 1
```

## 8. Verification

The workstation used to draft this migration has habitat-lab 0.2.1
installed under `/home/liushe10/miniconda3/envs/co-nav/` and no
habitat-sim/lab 0.3.3, so I could not launch `main.py` end-to-end
before shipping the change. The following static checks passed:

* `python -m py_compile arguments.py main.py main_vec.py envs/*.py utils/shortest_path_follower.py utils/fire_pipeline.py utils/fire_world/scene.py utils/fire_world/controller.py utils/fire_sensors/lidar_360.py scripts/keyboard_teleop.py scripts/keyboard_teleop_fire.py`
  (see `todo/todo.json::T034` for details).
* `grep` sweep confirms no remaining `config.SIMULATOR.*` / `config.TASK.*`
  / `config.DATASET.*` / `config.ENVIRONMENT.*` reads outside of the
  legacy-YACS fallback branches in `utils/fire_*/*.py`.
* `grep` sweep confirms no remaining references to
  `multi-robot-setting/` or `ShortestPathFollowerCompat` outside the
  new implementation.

Runtime validation must be done in a habitat 0.3.3 environment. The
recommended smoke test is:

```bash
python main.py --num_agents 1 --num_humans 0 --nav_mode nearest \
    --fire_sensors 0 --fire_world 0 --dump_location /tmp/conav_smoke
```

which should reset an episode, take a few `env.step()` calls, and
produce metrics without exceptions.
