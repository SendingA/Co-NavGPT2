# FireWorld risk-aware route scenarios

These two curated scenarios isolate the behavior that a navigation comparison
needs:

- the risk-blind route minimizes geometric distance and crosses the fire zone;
- the risk-aware route uses the same start and goal, but minimizes
  `distance * (1 + risk_alpha * mean_risk)` and treats flame/high temperature
  cells as hard obstacles;
- the expensive FireWorld timeline, rather than only the radial screening
  surrogate, must pass the route-contrast contract at 20%, 50%, and 80% of
  the episode duration.

## Curated scenarios

| Profile | Scene and target | Fire source | Final plan |
| --- | --- | --- | --- |
| Stable strong detour | `Nfvxx8J5NCo`, episode `5`, bed | trashcan instance 51 | `Nfvxx8J5NCo_route_contrast_stable_e6a4ad2abbf4` |
| Dynamic efficient detour | `TEEsavR23oF`, episode `4`, sofa | chair instance 319 | `TEEsavR23oF_route_contrast_dynamic_d8b5f25bd8ae` |

The stable profile requires a route at least 15% longer. The dynamic profile
requires at least 5% additional distance because its parallel safe corridor is
geometrically efficient; it still requires at least 70% exposure reduction,
20% path divergence, blind maximum risk at least 0.60, and aware maximum risk
at most 0.35.

Measured from the baked oracle-risk fields:

| Profile | Times (s) | Detour ratio | Exposure reduction | Blind max risk | Aware max risk | Path divergence |
| --- | --- | --- | --- | --- | --- | --- |
| Stable | 60 / 150 / 240 | 1.352 at all samples | 94.8%–97.8% | 0.979–0.987 | 0.017–0.028 | 0.978 |
| Dynamic | 84 / 210 / 336 | 1.060 at all samples | 70.2%–77.7% | 0.778–0.783 | 0.136–0.190 | 0.752 |

The dynamic plan deliberately bounds both the floor front and object spread.
It disables visual object-AABB flame filling, which otherwise caused late
secondary flame to close every corridor. These settings are plan-local and do
not change FireWorld's global propagation defaults.

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
the risk-blind shortest route, green is the risk-aware route, yellow is the
ignition, purple is the goal, and red intensity is physical risk.

## End-to-end comparison

Each generated dataset contains exactly the selected episode. Keep all other
planner settings and the seed identical between the two runs.
Pass the package's root `val.json.gz` shown below, not its
`content/<scene>.json.gz` shard. The root-plus-content layout is required so
Habitat's `content_scenes` filter and the FireWorld short scene ID agree.

Stable risk-blind run:

```bash
python main.py --task_config multi_objectnav_hm3d.yaml \
  --dataset_path data/processed/fire_route_scenarios/Nfvxx8J5NCo_route_contrast_stable_e6a4ad2abbf4/val.json.gz \
  --max_episodes 1 --num_agents 1 --nav_mode nearest --local_planner fmm \
  --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_route_contrast_stable_e6a4ad2abbf4 \
  --fire_clock_mode step --risk_enabled 0
```

Stable oracle risk-aware run:

```bash
python main.py --task_config multi_objectnav_hm3d.yaml \
  --dataset_path data/processed/fire_route_scenarios/Nfvxx8J5NCo_route_contrast_stable_e6a4ad2abbf4/val.json.gz \
  --max_episodes 1 --num_agents 1 --nav_mode nearest --local_planner fmm \
  --fire_world 1 --fire_world_plan_id Nfvxx8J5NCo_route_contrast_stable_e6a4ad2abbf4 \
  --fire_clock_mode step --risk_enabled 1 --risk_source oracle --risk_alpha 4
```

Replace both occurrences of the stable plan ID and dataset with the dynamic
ones from the table to run the second pair. Oracle risk is the deterministic
planning upper bound used by the route-contract validator. For the
perception-facing benchmark, change `--risk_source oracle` to
`--risk_source sensed`; that additionally measures thermal/smoke observation
coverage and should be reported separately from the oracle route result.

The dynamic dataset root is:
`data/processed/fire_route_scenarios/TEEsavR23oF_route_contrast_dynamic_d8b5f25bd8ae/val.json.gz`.
