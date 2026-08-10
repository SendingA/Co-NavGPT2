#!/usr/bin/env bash
set -euo pipefail

scenario="${1:-all}"
python_bin="${CO_NAV_PYTHON:-/home/liushe10/miniconda3/envs/co-nav3/bin/python}"
root="outputs/fire_cost_experiments/fork_detour_nfv"
dataset="data/processed/fire_route_scenarios/Nfvxx8J5NCo_fork_detour_stable_separated_agents/val.json.gz"
plan_id="Nfvxx8J5NCo_route_contrast_stable_0ddce73046be"

build_scenario() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1 \
    "${python_bin}" scripts/build_fork_detour_scenario.py \
    --candidate-report outputs/fire_route_tuning/Nfvxx8J5NCo_stable/rank_01_ep_5_obj_51.json \
    --plan-path scenes/Nfvxx8J5NCo/plans/${plan_id}.json \
    --dataset-output-dir "${dataset%/val.json.gz}" \
    --report-output-dir "${root}/geometry"
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
)

run_oracle() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MPLCONFIGDIR=/tmp/matplotlib-cache \
    PYTHONDONTWRITEBYTECODE=1 "${python_bin}" main.py \
    "${common[@]}" \
    --fire_fast 0 \
    --risk_source oracle \
    --risk_run_id fork-detour-oracle-alpha4-images \
    --dump_location "${root}/final_oracle/navigation" \
    --fire_dump_dir "${root}/final_oracle/fire" \
    --risk_dump_dir "${root}/final_oracle/risk" \
    --print_images 1
}

run_baseline() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MPLCONFIGDIR=/tmp/matplotlib-cache \
    PYTHONDONTWRITEBYTECODE=1 "${python_bin}" main.py \
    "${common[@]}" \
    --fire_fast 1 \
    --risk_source none \
    --risk_run_id fork-detour-known-goal-none \
    --dump_location "${root}/baseline_none/navigation" \
    --fire_dump_dir "${root}/baseline_none/fire" \
    --risk_dump_dir "${root}/baseline_none/risk" \
    --print_images 0
}

analyze() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1 \
    "${python_bin}" scripts/analyze_fork_detour_run.py \
    --candidate-report outputs/fire_route_tuning/Nfvxx8J5NCo_stable/rank_01_ep_5_obj_51.json \
    --plan-path scenes/Nfvxx8J5NCo/plans/${plan_id}.json \
    --oracle-actions "${root}/final_oracle/risk/fork-detour-oracle-alpha4-images/rank_000/ep_0000/action_list.json" \
    --oracle-risk-summary "${root}/final_oracle/risk/fork-detour-oracle-alpha4-images/rank_000/ep_0000/risk_summary.json" \
    --oracle-log "${root}/final_oracle/navigation/logs/co_ut/output.log" \
    --baseline-actions "${root}/baseline_none/risk/fork-detour-known-goal-none/rank_000/ep_0000/action_list.json" \
    --baseline-risk-summary "${root}/baseline_none/risk/fork-detour-known-goal-none/rank_000/ep_0000/risk_summary.json" \
    --baseline-log "${root}/baseline_none/navigation/logs/co_ut/output.log" \
    --output-dir "${root}/comparison"
}

build_scenario
case "${scenario}" in
  all)
    run_baseline
    run_oracle
    analyze
    ;;
  oracle)
    run_oracle
    ;;
  baseline)
    run_baseline
    ;;
  analyze)
    analyze
    ;;
  *)
    echo "usage: $0 [all|oracle|baseline|analyze]" >&2
    exit 2
    ;;
esac
