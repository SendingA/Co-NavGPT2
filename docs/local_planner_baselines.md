# Local-planner baselines

The formal local-planner selector is:

```text
--local_planner fmm | astar | rl
```

`fmm` is the default and deliberately preserves the repository's existing
implementation. In risk-blind mode that implementation first asks Habitat's
navmesh for a shortest path and falls back to grid FMM when the navmesh path is
unavailable. In risk-aware mode it goes directly to risk-aware grid FMM. This
historical hybrid remains labelled `fmm` for continuity with existing runs.

`astar` and `rl` are explicit new backends. They never invoke Habitat's
navmesh. Both consume the same inflated occupancy grid, visited/collision
state, dilated goal and shared low-level turn/forward controller used by the
existing FMM fallback.

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
    --fire_world 1 --fire_world_plan_id 83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source sensed

# Risk-blind A*, with the same risk evaluator still measuring exposure
python main.py --local_planner astar \
    --fire_world 1 --fire_world_plan_id 83679a07b632 \
    --fire_clock_mode step --risk_enabled 1 --risk_source none
```

The same rule applies to `fmm` and `rl`. Risk availability changes the entire
navigation stack consistently: a sensed/oracle map is available to the global
frontier logic and the selected local planner, while `source=none` hides it
from both.

## A* definition

The A* backend uses eight-connected motion, forbids diagonal corner cutting
and uses Euclidean multi-goal heuristics. Its risk-aware edge cost is:

```text
edge_cost = geometric_length
            * (1 + risk_alpha * mean(endpoint_risk))
```

This matches the cost semantics of FMM's
`speed = 1 / (1 + risk_alpha * risk)`. `hard_unsafe` cells are excluded and
an agent already inside a newly unsafe region receives the same
obstacle-respecting emergency escape behavior as the risk-aware FMM path.

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
    --fire_world 1 --fire_world_plan_id 83679a07b632 \
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
```

The primary metrics remain Habitat Success, Habitat SPL, SafeSuccess and CHE.
Planner latency, collision count, replanning rate and RL invalid-action-mask
interventions are diagnostics rather than primary benchmark columns.
