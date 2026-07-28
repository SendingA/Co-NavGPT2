# Benchmark experiment plan

This document freezes the experimental questions, comparison boundaries, run
matrix, metrics, and artifact layout before large-scale evaluation starts.
It distinguishes what the repository already implements from the work that is
still required.

## 1. Research questions

The benchmark should answer four separate questions:

1. Does the navigation stack generalize from the six native ObjectNav
   categories to the category-extensible `person` goal?
2. Which global and local planner choices work best in normal and dynamic
   fire/smoke environments?
3. How much do clean-depth multimodal fusion and risk-aware navigation
   contribute under the same FireWorld episodes?
4. At lower priority, how do VLM visual layout and textual/visual hazard
   representations affect frontier assignment?

These questions must be evaluated in separate controlled blocks. A single
full Cartesian product would be expensive, difficult to interpret, and would
change multiple modules at once.

## 2. Verified repository state

### 2.1 Datasets

| Dataset ID | Task config | Goal categories | Episodes | Scenes |
| --- | --- | --- | ---: | ---: |
| `objectnav` | `multi_objectnav_hm3d.yaml` | chair, bed, plant, toilet, tv_monitor, sofa | 1000 | 36 |
| `person` | `person_objectnav_hm3d.yaml` | person | 662 | 36 |

The datasets use the same 36 HM3D validation scenes but contain different
episodes and task distributions. Results must be reported separately. Do not
pool their episodes into one overall SR or SPL.

The `person` task follows the native ObjectNav chain: dataset category,
`objectgoal`, ordinary detection/navigation, Habitat STOP, Success,
DistanceToGoal, and SPL. It is not a parallel human-only score.

### 2.2 Global planner selectors

The active selector is `--nav_mode`:

| Value | Current behavior without planning risk |
| --- | --- |
| `nearest` | Each robot selects its nearest frontier; sharing is allowed |
| `co_ut` | Greedy cooperative assignment that avoids duplicate frontiers when possible |
| `fill` | Selects the frontier with the highest frontier score |
| `gpt` | Sends separate candidate-map images to GPT-4o for assignment |

Global replanning occurs every `--num_local_steps` navigation steps, default
`25`.

When `risk_source=sensed` or `oracle`, all four modes enter the common
risk-aware frontier assignment path. Their names then select different
utility weights and sharing behavior; they are no longer byte-for-byte the
risk-blind algorithms. Report these rows as, for example,
`nearest+sensed-risk`, not simply `nearest`.

### 2.3 Local planner selectors

The active selector is `--local_planner`:

| Value | Current status | Benchmark interpretation |
| --- | --- | --- |
| `fmm` | Implemented | Historical navmesh-first/FMM fallback when blind; risk-aware FMM when risk is available |
| `astar` | Implemented | Grid A* with blind and risk-aware forms |
| `rl` | Implemented but not benchmark-ready | Map-based PPO waypoint policy trained on randomized grids |
| PointNav DD-PPO | Not integrated | Proposed standard pretrained RL local-control baseline |

The existing `rl` backend must not be described as Habitat PointNav. It uses
an egocentric occupancy/goal crop and produces a grid waypoint. It is a useful
supplementary learned-planner baseline only after converged blind and aware
checkpoints are trained without validation-scene leakage.

### 2.4 Current VLM input

The current `gpt` path uses one separate candidate-map image per frontier.
In risk-aware mode it adds a structured textual hazard report.

The active code does **not** draw the hazard field into each candidate map.
It therefore currently corresponds to:

```text
separate candidate images + textual hazard report
```

`get_all_candidate_full_maps` combines one map with one frontier-facing image;
it is not an all-candidate aggregated montage and is not used by `main.py`.
The prompt text still mentions frontier-direction images even though the
active candidate builder supplies maps only. This must be reconciled before
the prompt study.

The API call also hard-codes `gpt-4o`; `--gpt_type` does not currently change
the requested model. Model selection and exact request logging are required
before publishing a VLM comparison.

### 2.5 Current FireWorld coverage

| Scene | Plan | Scenario | Intensity | Timeline |
| --- | --- | --- | --- | --- |
| `Nfvxx8J5NCo` | `83679a07b632` | bedroom textile | severe | ready |
| `Nfvxx8J5NCo` | `b2fc76fae83d` | kitchen grease | medium | **missing** |
| `TEEsavR23oF` | `549afa3c5305` | bedroom textile | severe | ready |
| `TEEsavR23oF` | `d4f8b9c253ab` | kitchen grease | medium | ready |

The two current fire scenes contain:

| Dataset | Episodes per one matched no-fire pass | Episode-plan pairs for four plans |
| --- | ---: | ---: |
| `objectnav` | 56 | 112 |
| `person` | 40 | 80 |

The existing two-scene set is appropriate for a pilot and engineering
ablation, not for a dataset-wide fire generalization claim. A final paper
claim should preregister a larger hazard subset, ideally all 36 scenes. At
minimum, report every fire result by scene and plan.

`main.py` accepts one `--fire_world_plan_id` at a time and automatically
restricts the episode iterator to that plan's scene. A fire launcher must
therefore execute and aggregate plans separately.

## 3. Fixed experimental controls

Unless a factor is explicitly being ablated, freeze:

| Control | Value |
| --- | --- |
| Number of robots | `--num_agents 2` |
| Episode limit | Habitat `max_episode_steps=500` |
| Global replan interval | `--num_local_steps 25` |
| Map resolution | `--map_resolution 5` cm |
| Forward step / turn | `0.25` m / `30` degrees |
| Fire clock | `--fire_clock_mode step` |
| Fire time scale | `--fire_steps_per_unit 5 --fire_seconds_per_unit 2.0` |
| Fire render | `--fire_fast 1` for benchmark throughput |
| Thermal perception | `--use_thermal_perception 1` |
| Sensed smoke source | `--risk_smoke_source appearance_depth` |
| Default clean geometry | `--depth_use_clean 1 --risk_geometry_depth_source clean` |
| RL inference | deterministic, frozen checkpoint |
| Seeds | `1, 2, 3` after the one-seed pilot |

Use step-clock fire for every comparison. Wall-clock fire would make the
hazard evolve faster for methods with higher VLM/API or rendering latency and
would confound navigation quality with machine/API speed.

Before the final run, freeze a clean commit or tag and record the commit,
working-tree diff hash, Python environment, task config, FireWorld timeline
hash, model/checkpoint hash, and full command for every run.

## 4. Environment conditions

Use these registered conditions:

| Condition | Required flags | Meaning |
| --- | --- | --- |
| `normal` | `--fire_world 0 --risk_enabled 0` | Native clean ObjectNav |
| `fire-none` | `--fire_world 1 --risk_enabled 1 --risk_source none` | Fire perception active, planner risk-blind, CHE evaluator active |
| `fire-sensed` | `--fire_world 1 --risk_enabled 1 --risk_source sensed` | Main deployable risk-aware method |
| `fire-oracle` | `--fire_world 1 --risk_enabled 1 --risk_source oracle` | Privileged upper bound |

The primary baseline fire row is `fire-sensed`. `fire-none` and
`fire-oracle` belong to the risk-awareness ablation.

Do not use `risk_enabled=0` for a reported fire row that needs CHE: that turns
off the independent risk evaluator. `risk_source=none` is the correct blind
control because it hides risk from global/local planning while retaining
ground-truth exposure measurement.

For every fire episode-plan pair, use the same dataset episode and seed as its
no-fire control. The stable episode key is:

```text
dataset + scene_id + episode_id
```

`episode_id` alone is not unique because IDs repeat across HM3D content
shards.

## 5. Controlled baseline design

### 5.1 Global-planner block

Freeze:

```text
local_planner = fmm
prompt design = current production design for gpt
```

Compare:

```text
nearest
co_ut
fill
gpt
```

Run all four on both datasets in `normal` and `fire-sensed`. Extract the
matched normal subset by scene/episode when comparing against FireWorld;
do not compare a 36-scene normal aggregate directly with a two-scene fire
aggregate.

### 5.2 Local-planner block

Freeze the deterministic global policy:

```text
nav_mode = co_ut
```

This removes VLM service variance while isolating local path execution.

Compare:

```text
fmm
astar
pointnav-ddppo
```

Keep the existing map-PPO `rl` row in supplementary experiments only after a
converged checkpoint exists.

### 5.3 PointNav DD-PPO adapter requirements

A pretrained PointNav policy cannot be dropped into the current local-planner
factory without an adapter. The implementation must:

1. Convert the selected frontier grid/world position into the point-goal
   coordinate convention expected by the checkpoint.
2. Match the checkpoint's RGB/depth, GPS/compass, normalization, action, and
   camera schemas exactly.
3. Maintain independent recurrent hidden state and masks for every robot.
4. Reset or update local-goal state when the global frontier changes.
5. Intercept PointNav `STOP`: it means the local frontier was reached and
   should trigger global replanning. It must not terminate the ObjectNav
   episode. Only the existing detected-object STOP path may end ObjectNav.
6. Record checkpoint URL/source, SHA-256, training dataset, architecture, and
   inference determinism.

The pretrained policy is risk-blind. If a with-risk variant is required,
label it explicitly as `pointnav-ddppo+shield`: keep the frozen policy and
veto one-step actions that enter hard-unsafe cells, with a deterministic safe
escape behavior. Do not claim that the pretrained neural policy itself is
risk-aware. A truly risk-conditioned RL policy requires retraining with risk
observations and a preregistered reward.

### 5.4 Primary matrix size

After all four fire timelines are ready:

| Block | Normal cells/seed | Fire cells/seed | Total/seed |
| --- | ---: | ---: | ---: |
| Global: 2 datasets × 4 global planners | 8 | 32 | 40 |
| Local: 2 datasets × 3 local planners | 6 | 24 | 30 |
| Overlap: `co_ut + fmm` | 2 | 8 | -10 |
| Unique primary cells |  |  | **60** |

Three seeds produce 180 run invocations. A fire invocation contains only the
episodes belonging to its plan's scene; a normal invocation covers the full
dataset and can be filtered offline to the matched fire scenes.

## 6. Ablation studies

### 6.1 Clean-depth multimodal fusion

The requested two-condition ablation is:

| Variant | Mapping depth | Risk evidence geometry |
| --- | --- | --- |
| `mpaf-full` | `--depth_use_clean 1` | `--risk_geometry_depth_source clean` |
| `mpaf-no-clean` | `--depth_use_clean 0` | `--risk_geometry_depth_source smoke` |

Run under the same `fire-sensed` episodes, plans, seeds, global planner and
local planner. Keep thermal perception and appearance-based smoke estimation
on.

Changing both switches is intentional for the end-to-end no-clean condition:
otherwise the mapper would receive smoky depth while the risk projector still
used clean geometry, leaking the ablated modality through a second path.

This experiment isolates access to clean geometry; by itself it is not a
complete ablation of all multimodal perception and fusion. If the paper makes
a broader MPAF claim, add a later `use_thermal_perception=0/1` factor or rename
the table to “clean-depth fusion ablation.”

### 6.2 Dynamic risk-aware navigation

Freeze FireWorld perception, depth settings, thresholds, global/local planner,
episode-plan pairs and seeds. Compare:

```text
risk_source = none
risk_source = sensed
risk_source = oracle
```

Interpretation:

- `none`: risk-blind navigation control with independent CHE evaluation;
- `sensed`: deployable belief-map method;
- `oracle`: privileged upper-bound planner, never the main claimed method.

This is a module ablation: the factor changes both global frontier risk and
local risk-aware path execution. A later component analysis may independently
disable global risk or local risk, but it should not be mixed into the primary
three-row table.

## 7. Low-priority VLM prompt study

Treat visual layout and hazard representation as two separate questions.

### 7.1 Layout study

With the full hazard input fixed, compare:

```text
separate = one ordered candidate image per frontier
aggregated = one indexed montage containing all candidates
```

Preserve the same candidate ordering, map content, pixel budget, JPEG quality,
prompt text, model, temperature, and maximum tokens as far as possible.

### 7.2 Hazard-input factorial

The original list contains a duplicated “report + overlay” condition. Until
revised, interpret the missing condition as `report only` and use the clean
2×2 design:

| ID | Hazard overlay | Hazard report |
| --- | --- | --- |
| `H00` | no | no |
| `H10` | yes | no |
| `H01` | no | yes |
| `H11` | yes | yes |

The current implementation is closest to `H01`, not `H11`.

Run this study after deterministic baselines, on a frozen fire subset with
the same precomputed candidates where possible. Save exact request messages,
candidate images, prompt hashes, raw responses, parsed assignments, guard
rejections, API retries, model identifier, token usage, and latency.

## 8. Metrics

### 8.1 Primary tables

| Metric | Key | Direction | Scope |
| --- | --- | --- | --- |
| Number of steps | `num_steps` | lower | All runs |
| Final distance to goal | `distance_to_goal` | lower | All runs |
| Success rate | mean of `success` | higher | All runs |
| SPL | `spl` | higher | All runs |
| Cumulative hazard exposure | `risk/che` | lower | Fire only |
| Safe success | `risk/safe_success` | higher | Fire only |

`num_steps` is the number of joint Habitat action cycles, including terminal
STOP. It is an action-budget/navigation-time proxy, not wall-clock runtime.

CHE must be shown as `N/A` for normal runs, not zero. CHE alone can reward a
method that fails early and stops accumulating exposure, so every fire table
must retain SR and SafeSuccess beside CHE.

### 8.2 Frozen SPL semantics

The benchmark keeps the repository's current Habitat `spl` implementation
unchanged for continuity with all previous runs. No additional path-efficiency
metric is introduced. Use the same `spl` key and calculation in every baseline
and ablation row.

### 8.3 Aggregation and uncertainty

- Report `objectnav` and `person` separately.
- For ObjectNav, additionally report per-category results and a category-macro
  average so frequent categories do not dominate.
- Keep per-episode records and compute paired deltas by stable episode key.
- For full 36-scene normal results, use a scene-clustered bootstrap for 95%
  confidence intervals.
- For a two-scene fire pilot, report every scene/plan explicitly; do not make a
  strong cross-scene confidence claim from only two clusters.
- Keep raw CHE cumulative. Do not replace it with correlated peak/mean/time
  variants in the primary table.

## 9. Unified output layout

Current outputs are fragmented across `dump_location`, `fire_dump_dir`, and
`risk_dump_dir`. The benchmark launcher should bind them under:

```text
outputs/benchmarks/<study_id>/
├── study_manifest.yaml
├── episode_manifests/
├── runs/
│   └── <run_id>/
│       ├── manifest.json
│       ├── command.txt
│       ├── resolved_config.yaml
│       ├── environment.txt
│       ├── status.json
│       ├── stdout.log
│       ├── metrics/
│       │   ├── episodes.jsonl
│       │   └── aggregate.json
│       ├── navigation/
│       │   ├── logs/
│       │   └── maps/
│       ├── fire_sensors/
│       ├── risk/
│       └── vlm/
│           ├── requests.jsonl
│           └── candidate_maps/
└── reports/
    ├── completeness.json
    ├── main_table.csv
    ├── ablation_table.csv
    └── figures/
```

Recommended run ID:

```text
<dataset>__<condition>__G-<global>__L-<local>__D-<depth>__P-<prompt>__s-<seed>__plan-<id>
```

Always retain:

- resolved configuration and exact command;
- code, checkpoint, dataset-manifest, prompt and FireWorld timeline hashes;
- one machine-readable row per episode;
- risk step JSONL and episode risk summary for fire runs;
- VLM request/response artifacts for `gpt` runs;
- completion/failure status and restart provenance.

For storage control, save full fire dashboards and per-step PNGs only for a
preregistered diagnostic subset. Do not duplicate large `timeline.npz` files
inside every run; reference their absolute/relative path and hash.

## 10. Execution order

### Phase 0: blocking infrastructure

1. Freeze the repository revision and resolve the existing risk-prompt test
   failure.
2. Generate dataset/episode manifests.
3. Add per-episode JSONL metrics while preserving the current Habitat SPL.
4. Implement the unified launcher, run status, resume and completeness checks.
5. Rebake and validate the missing medium Nfvxx8J5NCo timeline.
6. Integrate and validate PointNav DD-PPO.

### Phase 1: pilot

Use one seed and a small fixed subset:

1. `normal`, `fire-none`, `fire-sensed`, and `fire-oracle`;
2. both datasets;
3. `co_ut + fmm`, `co_ut + astar`, and
   `co_ut + pointnav-ddppo`;
4. one medium and one severe plan.

Validate metric ranges, episode pairing, PointNav STOP handling, fire timing,
disk usage, artifact completeness and resume behavior.

### Phase 2: deterministic main baselines

Run non-VLM global and local rows first for seeds 1, 2 and 3. Generate a
preliminary completeness report and inspect outlier episodes before spending
API budget.

### Phase 3: end-to-end VLM row

Freeze model/prompt settings and run `gpt + fmm`. Preserve all request and
response artifacts. Do not mix results from a changed hosted model or prompt
under the same method ID.

### Phase 4: ablations

Run the clean-depth fusion and `none/sensed/oracle` risk-source studies on
exactly paired fire episode-plan manifests.

### Phase 5: prompt study

Only after the main method is stable, run layout and `H00/H10/H01/H11` on a
frozen subset.

### Phase 6: aggregation

Check that every registered cell contains the exact expected episode keys
before computing tables. Produce per-dataset, per-category, per-scene and
per-plan summaries plus paired deltas.

## 11. Current manual command templates

Normal:

```bash
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 --seed 1 \
    --nav_mode co_ut --local_planner fmm \
    --fire_world 0 --risk_enabled 0 \
    --dump_location <run_root>/navigation
```

Fire sensed:

```bash
python main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --num_agents 2 --seed 1 \
    --nav_mode co_ut --local_planner fmm \
    --fire_world 1 --fire_world_plan_id 83679a07b632 \
    --fire_clock_mode step \
    --fire_steps_per_unit 5 --fire_seconds_per_unit 2.0 \
    --depth_use_clean 1 --use_thermal_perception 1 \
    --risk_enabled 1 --risk_source sensed \
    --risk_smoke_source appearance_depth \
    --risk_geometry_depth_source clean \
    --dump_location <run_root>/navigation \
    --fire_dump_dir <run_root>/fire_sensors \
    --risk_dump_dir <run_root>/risk \
    --risk_run_id run
```

These commands are useful for a smoke test, but they do not yet provide the
manifest, per-episode metrics, completeness checks, overwrite protection or
automatic matrix expansion required for the final benchmark.
