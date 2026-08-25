# Local-planner baselines

The formal local-planner selector is:

```text
--local_planner fmm | astar | rl | pointnav
```

`fmm` is the default. Normal no-fire `auto` mode preserves the historical
navmesh-first/grid-FMM fallback. Fire evaluations deliberately resolve to grid
FMM for `risk_source=none`, `oracle`, and `sensed`, so none/oracle comparisons
use the same exploration and detected-target-completion backend. The resolved
backend is printed at startup and can be fixed with `--fmm_backend`.
Backend parity also covers padded-grid geometry: both sources use the same
blocked one-cell outer boundary and `+1` coordinate offset. Normal no-fire
`auto` keeps its historical navmesh-first fallback behavior.

`astar` and `rl` are explicit grid backends. They never invoke Habitat's
navmesh. Both consume the same inflated occupancy grid, visited/collision
state, dilated goal and shared low-level turn/forward controller used by the
existing FMM fallback.

`pointnav` is a separate, agent-scoped backend for Habitat's official
pretrained PointNav DD-PPO policy. It consumes a dedicated checkpoint-shaped
depth/RGB policy sensor and a frontier-relative point goal, maintains recurrent
policy state, and emits Habitat actions directly. It is not an alias for the
map-based `rl` backend.

## Automatic local risk awareness

Local risk awareness has no separate CLI switch. Every local planner follows
the existing risk configuration:

- `--risk_enabled 1 --risk_source sensed|oracle`: automatically risk-aware.
- `--risk_enabled 1 --risk_source none`: risk-blind, while the independent
  ground-truth evaluator still records exposure.
- `--risk_enabled 0`: risk-blind and the risk runtime is completely disabled.

Use `risk_source=none` for a risk-blind navigation control that still produces
SafeSuccess and CHE:

```bash
# Risk-aware A*
python main.py --local_planner astar \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source sensed

# Risk-blind A*, with the same risk evaluator still measuring exposure
python main.py --local_planner astar \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source none
```

The same rule applies to `fmm`, `rl`, and `pointnav`. Risk availability changes
the entire navigation stack consistently: a sensed/oracle map is available to
the global frontier logic and the selected local planner, while `source=none`
hides it from both. The frozen PointNav network remains risk-blind; its aware
form adds the common risk-aware frontier selection plus a one-step hard-hazard
action shield outside the policy.

## A* definition

The A* backend uses eight-connected motion, forbids diagonal corner cutting
and uses Euclidean multi-goal heuristics. Its risk-aware edge cost is:

```text
edge_cost = geometric_length
            * (1 + risk_alpha * mean(endpoint_risk))
```

This matches the cost semantics of FMM's
`speed = 1 / (1 + risk_alpha * risk)`. In the FMM/A* agent execution path,
`hard_unsafe` is diagnostic only: cells are not excluded, the real goal is not
replaced by a temporary safety waypoint, and emergency escape is disabled.
The latest continuous-risk grid is reconsidered on every action cycle.

## RL definition and training

The RL backend is a checkpointed map-based PPO waypoint policy. It rolls out
learned moves on the grid to produce a short-term goal; the existing action
controller converts that waypoint into Habitat actions.

Risk-blind observations contain:

```text
traversible, goal, rollout-visited
```

Risk-aware observations additionally contain:

```text
planning_risk, hard_unsafe
```

Train separate checkpoints because checkpoint metadata enforces the awareness
mode and crop size:

```bash
# Risk-blind policy
python scripts/train_rl_local_planner.py \
    --config configs/rl_local_planner_ppo.yaml \
    --risk-aware 0 \
    --output outputs/local_planner_rl/blind.pth

# Risk-aware policy
python scripts/train_rl_local_planner.py \
    --config configs/rl_local_planner_ppo.yaml \
    --risk-aware 1 \
    --output outputs/local_planner_rl/aware.pth
```

Evaluate a frozen checkpoint:

```bash
python main.py --local_planner rl \
    --rl_local_checkpoint outputs/local_planner_rl/aware.pth \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source sensed
```

The runtime fails when a checkpoint is absent or its risk mode, crop size,
action ordering or schema version does not match. It never evaluates random
weights or silently falls back to FMM.

For the blind RL checkpoint, replace the checkpoint path with `blind.pth` and
use `--risk_source none`. Checkpoint awareness is therefore selected and
validated automatically from the same risk configuration as the planner.

The supplied trainer uses randomized connected occupancy/risk grids. A
one-update smoke run proves only that PPO, serialization and inference work:

```bash
python scripts/train_rl_local_planner.py \
    --risk-aware 0 --updates 1 --num-envs 2 --rollout-steps 4 \
    --grid-size 15 --crop-size 15 --hidden-size 32 \
    --ppo-epochs 1 --mini-batch-size 8 \
    --output /tmp/rl_local_smoke.pth
```

Do not report a smoke-trained policy as a benchmark baseline. A reported RL
result needs a converged, frozen checkpoint, fixed seed/config, and training
data that do not include validation/test scenes.

## Habitat PointNav DD-PPO

Install `habitat-baselines` from the same Habitat-Lab 0.3.3 checkout used by
the project:

```bash
python -m pip install -e "$HABITAT_LAB_ROOT/habitat-baselines"
```

Download the official Gibson 2+ depth ResNet50 + LSTM512 checkpoint:

```bash
mkdir -p data/ddppo-models
wget --continue \
  https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-2plus-resnet50.pth \
  -P data/ddppo-models
sha256sum data/ddppo-models/gibson-2plus-resnet50.pth
```

The validated official file is 49,853,716 bytes and has SHA-256:

```text
a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad
```

Run the baseline with:

```bash
python main.py \
    --local_planner pointnav \
    --pointnav_checkpoint data/ddppo-models/gibson-2plus-resnet50.pth \
    --pointnav_device cuda:0
```

The official `.pth` contains weights but no embedded training config. The
adapter recognizes Habitat's published checkpoint filenames, loads the
matching packaged `ddppo_pointnav.yaml`, and strictly loads the state dict.
The default checkpoint contract is:

```text
training data     Gibson 2+
policy            PointNavResNetPolicy, ResNet50 + 2-layer LSTM512
visual input      normalized depth, 256 x 256 x 1, float32 in [0, 1]
goal input        pointgoal_with_gps_compass, polar [distance, angle]
policy camera     256 x 256, HFOV 90 degrees
actions           STOP, MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
motion            0.25 m forward, 10 degree turn
```

The policy camera is separate from the project's mapping camera. The default
ObjectNav RGB-D stream remains `640 x 480, HFOV 79` for point-cloud mapping,
detection, VLM input, FireWorld perception and dashboards. A second
`pointnav_depth` sensor renders the checkpoint-exact `256 x 256, HFOV 90`
observation directly; it is not cropped or resized from the mapping image.
RGB checkpoints similarly receive `pointnav_rgb`, and RGB-D checkpoints
receive aligned policy-only RGB and depth sensors. If a different checkpoint
has an embedded config, that config drives its policy sensors. For an
unrecognized weights-only checkpoint, pass both `--pointnav_config` and
`--pointnav_observation_mode`; schema or weight mismatches fail instead of
silently falling back.

Each robot has independent LSTM hidden state, previous action, mask, and local
frontier state. A changed frontier resets only that robot. PointNav `STOP`
means that the local frontier was reached: it is intercepted, its local state
is cleared, and global frontier replanning is requested. It is never sent to
the ObjectNav environment. Only the existing detected-object completion path
may issue the task-level STOP.

`--pointnav_deterministic 0` samples actions, matching Habitat's published
evaluation procedure. Set it to `1` for modal actions in deterministic
engineering runs and record the choice with benchmark artifacts.

For fire experiments:

```bash
# Frozen PointNav plus risk-aware frontier assignment and hard-hazard shield
python main.py --local_planner pointnav \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --risk_enabled 1 --risk_source sensed

# Frozen PointNav without access to planning risk; evaluator remains active
python main.py --local_planner pointnav \
    --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
    --risk_enabled 1 --risk_source none
```

Report the first as `pointnav-ddppo+shield`, not as a risk-conditioned neural
policy. The shield never emits STOP and never adds a hazard channel to the
checkpoint observation.

## Fair benchmark matrix

Hold the dataset episodes, seeds, FireWorld plans, global `nav_mode`, risk
thresholds and evaluator configuration constant. Use `risk_source=sensed` for
the aware rows and `risk_source=none` for their blind controls:

```text
fmm   x aware
fmm   x blind
astar x aware
astar x blind
rl    x aware
rl    x blind
pointnav-ddppo+shield x aware
pointnav-ddppo        x blind
```

The primary metrics remain Habitat Success, Habitat SPL, SafeSuccess and CHE.
Planner latency, collision count, replanning rate and RL invalid-action-mask
interventions are diagnostics rather than primary benchmark columns.
