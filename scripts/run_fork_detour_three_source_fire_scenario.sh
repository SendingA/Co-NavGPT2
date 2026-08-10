#!/usr/bin/env bash
set -euo pipefail

scenario="${1:-all}"
python_bin="${CO_NAV_PYTHON:-/home/liushe10/miniconda3/envs/co-nav3/bin/python}"
root="outputs/fire_cost_experiments/fork_detour_nfv_three_source"
dataset="data/processed/fire_route_scenarios/Nfvxx8J5NCo_fork_detour_stable_three_source_separated_agents/val.json.gz"
plan_id="Nfvxx8J5NCo_route_contrast_stable_50d97ff18bd9"
timeline="outputs/fire_world/Nfvxx8J5NCo/${plan_id}/timeline.npz"

build_scenario() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1 \
    "${python_bin}" scripts/build_fork_detour_scenario.py \
    --candidate-report outputs/fire_route_tuning/Nfvxx8J5NCo_stable/rank_01_ep_5_obj_51.json \
    --plan-path "scenes/Nfvxx8J5NCo/plans/${plan_id}.json" \
    --dataset-output-dir "${dataset%/val.json.gz}" \
    --report-output-dir "${root}/geometry"
}

bake_timeline() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1 \
    "${python_bin}" -m utils.fire_world.propagation \
    --scene Nfvxx8J5NCo \
    --plan_id "${plan_id}" \
    --voxel_m 0.15 \
    --dt 0.5 \
    --save_dt 1.0
}

ensure_timeline() {
  if [[ ! -r "${timeline}" ]]; then
    bake_timeline
  fi
}

common=(
  --task_config multi_objectnav_hm3d.yaml
  --dataset_path "${dataset}"
  --max_episodes 1
  --num_agents 2
  --nav_mode co_ut
  --local_planner fmm
  --fire_world 1
  --fire_world_plan_id "${plan_id}"
  --fire_clock_mode step
  --fire_world_n_steps 24
  --fire_world_render_scale 0.5
  --fire_render_backend torch
  --fire_render_device cuda:0
  --fire_save_every 0
  --risk_enabled 1
  --risk_alpha 4
  --risk_frontier_weight 0.5
  --risk_save_every 10
  --fire_fast 0
  --print_images 1
)

run_oracle() {
  ensure_timeline
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MPLCONFIGDIR=/tmp/matplotlib-cache \
    PYTHONDONTWRITEBYTECODE=1 "${python_bin}" main.py \
    "${common[@]}" \
    --risk_source oracle \
    --risk_run_id fork-detour-three-source-oracle-alpha4-images \
    --dump_location "${root}/oracle/navigation" \
    --fire_dump_dir "${root}/oracle/fire" \
    --risk_dump_dir "${root}/oracle/risk"
}

run_none() {
  ensure_timeline
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MPLCONFIGDIR=/tmp/matplotlib-cache \
    PYTHONDONTWRITEBYTECODE=1 "${python_bin}" main.py \
    "${common[@]}" \
    --risk_source none \
    --risk_run_id fork-detour-three-source-none-images \
    --dump_location "${root}/risk_none/navigation" \
    --fire_dump_dir "${root}/risk_none/fire" \
    --risk_dump_dir "${root}/risk_none/risk"
}

analyze() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1 \
    "${python_bin}" scripts/analyze_fork_detour_run.py \
    --candidate-report outputs/fire_route_tuning/Nfvxx8J5NCo_stable/rank_01_ep_5_obj_51.json \
    --plan-path "scenes/Nfvxx8J5NCo/plans/${plan_id}.json" \
    --oracle-actions "${root}/oracle/risk/fork-detour-three-source-oracle-alpha4-images/rank_000/ep_0000/action_list.json" \
    --oracle-risk-summary "${root}/oracle/risk/fork-detour-three-source-oracle-alpha4-images/rank_000/ep_0000/risk_summary.json" \
    --oracle-log "${root}/oracle/navigation/logs/co_ut/output.log" \
    --baseline-actions "${root}/risk_none/risk/fork-detour-three-source-none-images/rank_000/ep_0000/action_list.json" \
    --baseline-risk-summary "${root}/risk_none/risk/fork-detour-three-source-none-images/rank_000/ep_0000/risk_summary.json" \
    --baseline-log "${root}/risk_none/navigation/logs/co_ut/output.log" \
    --output-dir "${root}/comparison"
}

build_scenario
case "${scenario}" in
  all)
    run_none
    run_oracle
    analyze
    ;;
  none|risk_none)
    run_none
    ;;
  oracle)
    run_oracle
    ;;
  analyze)
    analyze
    ;;
  bake)
    bake_timeline
    ;;
  *)
    echo "usage: $0 [all|none|risk_none|oracle|analyze|bake]" >&2
    exit 2
    ;;
esac
