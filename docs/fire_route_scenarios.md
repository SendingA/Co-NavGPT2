# FireWorld risk-aware route scenarios

These two curated scenarios isolate the behavior that a navigation comparison
needs:

- the risk-blind route minimizes geometric distance and crosses the fire zone;
- the risk-aware route uses the same start and goal, but minimizes
  `distance * (1 + risk_alpha * mean_risk)`, treats high-intensity flame cores
  as hard obstacles, and retains surrounding heat/smoke as continuous cost;
- the expensive FireWorld timeline, rather than only the radial screening
  surrogate, must pass the route-contrast contract at 20%, 50%, and 80% of
  the episode duration.

## Curated scenarios

| Profile | Scene and target | Fire source | Final plan |
| --- | --- | --- | --- |
| Stable strong detour | `Nfvxx8J5NCo`, episode `5`, bed | trashcan 51 + chair 339 | `Nfvxx8J5NCo_route_contrast_stable_0ddce73046be` |
| Dynamic efficient detour | `TEEsavR23oF`, episode `4`, sofa | chair instance 319 | `TEEsavR23oF_route_contrast_dynamic_d8b5f25bd8ae` |

The stable profile requires a route at least 15% longer. The dynamic profile
requires at least 5% additional distance because its parallel safe corridor is
geometrically efficient; it still requires at least 70% exposure reduction,
20% path divergence, blind maximum risk at least 0.60, and aware maximum risk
at most 0.35.

Measured from the baked oracle-risk fields:

| Profile | Times (s) | Detour ratio | Exposure reduction | Blind max risk | Aware max risk | Path divergence |
| --- | --- | --- | --- | --- | --- | --- |
| Stable | 60 / 150 / 240 | 1.352 / 1.370 / 1.513 | 85.6%–96.3% | 1.000 | 0.046–0.221 | 0.978–0.979 |
| Dynamic | 84 / 210 / 336 | 1.060 at all samples | 70.2%–77.7% | 0.778–0.783 | 0.136–0.190 | 0.752 |

The dynamic plan deliberately bounds both the floor front and object spread.
It disables visual object-AABB flame filling, which otherwise caused late
secondary flame to close every corridor. These settings are plan-local and do
not change FireWorld's global propagation defaults.

The strengthened stable plan replaces the previous single `0.42m` trashcan
source with two sustained `0.58m` sources. Under the same current risk
projection, its mean physical-risk footprint is about 3.6x, 4.6x and 5.6x
larger at 60s, 150s and 240s, while the aware route remains below the `0.35`
maximum-risk acceptance bound at every sample.

## Rebuild and validate

Use the project's Habitat environment:

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  /home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/tune_fire_route_scenarios.py \
  --scenarios Nfvxx8J5NCo:stable TEEsavR23oF:dynamic \
  --max-episodes 20 --top-k 8 \
  --write-plans --write-datasets --bake
```

If the plan and timeline already exist, rescore them and regenerate the route
overlays without running the voxel solver:

```bash
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
  /home/liushe10/miniconda3/envs/co-nav3/bin/python \
  scripts/tune_fire_route_scenarios.py \
  --scenarios Nfvxx8J5NCo:stable TEEsavR23oF:dynamic \
  --max-episodes 20 --top-k 8 \
  --write-datasets --validate-existing
```

Candidate and actual-field reports are written under
`outputs/fire_route_tuning/<scene>_<profile>/`. In the PNG overlays, blue is
the risk-blind shortest route, green is the risk-aware route, yellow points
are the ignitions, purple is the goal, and red intensity is physical risk.

## End-to-end comparison

Each generated dataset contains exactly the selected episode. Keep all other
planner settings and the seed identical between the two runs.
Pass the package's root `val.json.gz` shown below, not its
`content/<scene>.json.gz` shard.

Stable risk-blind run:

```bash
python main.py \
  --task_config multi_objectnav_hm3d.yaml \
  --dataset_path data/processed/fire_route_scenarios/Nfvxx8J5NCo_route_contrast_stable_0ddce73046be/val.json.gz \
  --max_episodes 1 \
  --num_agents 2 \
  --nav_mode co_ut \
  --local_planner fmm \
  --fire_world 1 \
  --fire_world_plan_id Nfvxx8J5NCo_route_contrast_stable_0ddce73046be \
  --fire_clock_mode step \
  --fire_fast 0 \
  --fire_world_n_steps 24 \
  --fire_world_render_scale 0.5 \
  --fire_render_backend torch \
  --fire_render_device cuda:0 \
  --risk_enabled 1 \
  --risk_source none \
  --print_images 1
```

Stable oracle risk-aware run:

```bash
python main.py \
  --task_config multi_objectnav_hm3d.yaml \
  --dataset_path data/processed/fire_route_scenarios/Nfvxx8J5NCo_route_contrast_stable_0ddce73046be/val.json.gz \
  --max_episodes 1 \
  --num_agents 2 \
  --nav_mode co_ut \
  --local_planner fmm \
  --fire_world 1 \
  --fire_world_plan_id Nfvxx8J5NCo_route_contrast_stable_0ddce73046be \
  --fire_clock_mode step \
  --fire_fast 0 \
  --fire_world_n_steps 24 \
  --fire_world_render_scale 0.5 \
  --fire_render_backend torch \
  --fire_render_device cuda:0 \
  --risk_enabled 1 \
  --risk_source oracle \
  --risk_alpha 4 \
  --print_images 1
```

Replace both occurrences of the stable plan ID and dataset with the dynamic
ones from the table to run the second pair. Oracle risk is the deterministic
planning upper bound used by the route-contract validator. For the
perception-facing benchmark, change `--risk_source oracle` to
`--risk_source sensed`; that additionally measures thermal/smoke observation
coverage and should be reported separately from the oracle route result.

The dynamic dataset root is:
`data/processed/fire_route_scenarios/TEEsavR23oF_route_contrast_dynamic_d8b5f25bd8ae/val.json.gz`.
