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
    --dataset_path data/processed/fire_cost_scenarios/Nfvxx8J5NCo_person_ep18_multi11/val.json.gz \
    --fire_world_plan_id Nfvxx8J5NCo_multi_origin_medium_262b437b2d6d \
    --risk_run_id cost-oracle-original-person-nfv-multi11-images \
    --dump_location outputs/fire_cost_experiments/original_multi_origin/person_nfv_multi11/navigation \
    --fire_dump_dir outputs/fire_cost_experiments/original_multi_origin/person_nfv_multi11/fire \
    --risk_dump_dir outputs/fire_cost_experiments/original_multi_origin/person_nfv_multi11/risk \
    "${common[@]}"
}

run_chair() {
  MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet MPLCONFIGDIR=/tmp/matplotlib-cache \
    "${python_bin}" main.py \
    --task_config multi_objectnav_hm3d.yaml \
    --dataset_path data/processed/fire_cost_scenarios/Nfvxx8J5NCo_chair_ep3_multi8/val.json.gz \
    --fire_world_plan_id Nfvxx8J5NCo_multi_origin_severe_021be03d8a84 \
    --risk_run_id cost-oracle-original-chair-nfv-multi8-images \
    --dump_location outputs/fire_cost_experiments/original_multi_origin/chair_nfv_multi8/navigation \
    --fire_dump_dir outputs/fire_cost_experiments/original_multi_origin/chair_nfv_multi8/fire \
    --risk_dump_dir outputs/fire_cost_experiments/original_multi_origin/chair_nfv_multi8/risk \
    "${common[@]}"
}

case "${scenario}" in
  all)
    run_person
    run_chair
    ;;
  person)
    run_person
    ;;
  chair)
    run_chair
    ;;
  *)
    echo "usage: $0 [all|person|chair]" >&2
    exit 2
    ;;
esac
