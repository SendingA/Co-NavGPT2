#!/usr/bin/env bash
set -euo pipefail

scenario="${1:-all}"
python_bin="${CO_NAV_PYTHON:-/home/liushe10/miniconda3/envs/co-nav3/bin/python}"

common=(
  --max_episodes 1
  --num_agents 2
  --nav_mode co_ut
  --local_planner fmm
  --fire_world 1
  --fire_clock_mode step
  --fire_fast 0
  --fire_world_n_steps 24
  --fire_world_render_scale 0.5
  --fire_render_backend torch
  --fire_render_device cuda:0
  --fire_save_every 0
  --risk_enabled 1
  --risk_source oracle
  --risk_alpha 1
  --risk_save_every 10
  --print_images 1
)

run_person() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MPLCONFIGDIR=/tmp/matplotlib-cache \
    "${python_bin}" main.py \
    --task_config person_objectnav_hm3d.yaml \
    --dataset_path data/processed/fire_cost_scenarios/a8BtkwhxdRV_person_ep12/val.json.gz \
    --fire_world_plan_id a8BtkwhxdRV_multi_origin_medium_6bac9d86987c \
    --risk_run_id cost-oracle-person-a8b-ep12-images \
    --dump_location outputs/fire_cost_experiments/selected_person_a8b_ep12/navigation \
    --fire_dump_dir outputs/fire_cost_experiments/selected_person_a8b_ep12/fire \
    --risk_dump_dir outputs/fire_cost_experiments/selected_person_a8b_ep12/risk \
    "${common[@]}"
}

run_sofa() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MPLCONFIGDIR=/tmp/matplotlib-cache \
    "${python_bin}" main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --dataset_path data/processed/fire_route_scenarios/TEEsavR23oF_route_contrast_dynamic_d8b5f25bd8ae/val.json.gz \
    --fire_world_plan_id TEEsavR23oF_route_contrast_dynamic_d8b5f25bd8ae \
    --risk_run_id cost-oracle-sofa-tees-images \
    --dump_location outputs/fire_cost_experiments/selected_sofa_tees_images/navigation \
    --fire_dump_dir outputs/fire_cost_experiments/selected_sofa_tees_images/fire \
    --risk_dump_dir outputs/fire_cost_experiments/selected_sofa_tees_images/risk \
    "${common[@]}"
}

case "${scenario}" in
  all)
    run_person
    run_sofa
    ;;
  person)
    run_person
    ;;
  sofa)
    run_sofa
    ;;
  *)
    echo "usage: $0 [all|person|sofa]" >&2
    exit 2
    ;;
esac
