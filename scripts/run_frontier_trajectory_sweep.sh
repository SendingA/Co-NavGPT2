#!/usr/bin/env bash
set -euo pipefail

python_bin="${CO_NAV_PYTHON:-/home/liushe10/miniconda3/envs/co-nav3/bin/python}"
root="${TRAJECTORY_OUTPUT_ROOT:-outputs/fire_cost_experiments/frontier_trajectory_strategy_sweep}"
torch_device="${TRAJECTORY_TORCH_DEVICE:-cuda:0}"
print_images="${TRAJECTORY_PRINT_IMAGES:-0}"
fire_fast="${TRAJECTORY_FIRE_FAST:-1}"
render_scale="${TRAJECTORY_RENDER_SCALE:-0.35}"
experiment_seed="${TRAJECTORY_SEED:-1}"
num_local_steps="${TRAJECTORY_NUM_LOCAL_STEPS:-25}"

runtime_env=(
  env
  MAGNUM_LOG=quiet
  HABITAT_SIM_LOG=quiet
  MPLCONFIGDIR=/tmp/conav-matplotlib
  PYTHONDONTWRITEBYTECODE=1
)

wait_for_existing_run() {
  local wait_pid="${2:-${WAIT_FOR_PID:-}}"
  if [[ -z "${wait_pid}" ]]; then
    return
  fi
  while kill -0 "${wait_pid}" 2>/dev/null; do
    echo "[trajectory-sweep] waiting for existing benchmark pid=${wait_pid}"
    sleep 30
  done
  echo "[trajectory-sweep] existing benchmark finished; starting GPU sweep"
}

run_case() {
  local target="$1"
  local task_config="$2"
  local dataset="$3"
  local plan_id="$4"
  local planner="$5"
  shift 5

  local label="${TRAJECTORY_RUN_LABEL:-${target}__${planner}}"
  local case_root="${root}/runs/${label}"
  local run_id="trajectory-sweep-${label}"
  local action_list="${case_root}/risk/${run_id}/rank_000/ep_0000/action_list.json"
  local risk_summary="${case_root}/risk/${run_id}/rank_000/ep_0000/risk_summary.json"
  local metrics="${case_root}/navigation/metrics/resume_state.json"
  if [[ -r "${action_list}" && -r "${risk_summary}" && -r "${metrics}" ]]; then
    echo "[trajectory-sweep] skip complete case=${label}"
    return
  fi

  echo "[trajectory-sweep] start case=${label}"
  "${runtime_env[@]}" "${python_bin}" main.py \
    --task_config "${task_config}" \
    --dataset_path "${dataset}" \
    --max_episodes 1 \
    --seed "${experiment_seed}" \
    --num_local_steps "${num_local_steps}" \
    --num_agents 2 \
    --nav_mode "${planner}" \
    --local_planner fmm \
    --visualize 0 \
    --print_images "${print_images}" \
    --dump_location "${case_root}/navigation" \
    --fire_world 1 \
    --fire_world_plan_id "${plan_id}" \
    --fire_clock_mode step \
    --fire_world_n_steps 24 \
    --fire_world_render_scale "${render_scale}" \
    --fire_render_backend torch \
    --fire_render_device "${torch_device}" \
    --fire_render_max_sample_points 750000 \
    --fire_fast "${fire_fast}" \
    --fire_save_every 0 \
    --fire_save_npz 0 \
    --fire_show_window 0 \
    --fire_dump_dir "${case_root}/fire" \
    --risk_enabled 1 \
    --risk_source oracle \
    --risk_alpha 4 \
    --risk_frontier_weight 0.5 \
    --risk_save_every 0 \
    --risk_save_traces 1 \
    --risk_run_id "${run_id}" \
    --risk_dump_dir "${case_root}/risk" \
    "$@"

  test -r "${action_list}"
  test -r "${risk_summary}"
  test -r "${metrics}"
  echo "[trajectory-sweep] complete case=${label}"
}

run_screen() {
  local bed_dataset="data/processed/fire_route_scenarios/Nfvxx8J5NCo_fork_detour_stable_three_source_separated_agents/val.json.gz"
  local bed_plan="Nfvxx8J5NCo_route_contrast_stable_50d97ff18bd9"
  local person_dataset="data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_fork_detour_three_source/val.json.gz"
  local person_plan="Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab"

  run_case bed multi_objectnav_hm3d.yaml "${bed_dataset}" "${bed_plan}" nearest
  run_case bed multi_objectnav_hm3d.yaml "${bed_dataset}" "${bed_plan}" fill
  run_case bed multi_objectnav_hm3d.yaml "${bed_dataset}" "${bed_plan}" random \
    --random_goal_min_distance_m 2.0
  run_case person person_objectnav_hm3d.yaml "${person_dataset}" "${person_plan}" nearest
  run_case person person_objectnav_hm3d.yaml "${person_dataset}" "${person_plan}" fill
  run_case person person_objectnav_hm3d.yaml "${person_dataset}" "${person_plan}" random \
    --random_goal_min_distance_m 2.0
}

run_one() {
  local target="${1:?target is required}"
  local planner="${2:?planner is required}"
  local -a extra=()
  if [[ "${planner}" == "random" ]]; then
    extra+=(--random_goal_min_distance_m "${3:-2.0}")
  fi

  case "${target}" in
    bed)
      run_case bed multi_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_fork_detour_stable_three_source_separated_agents/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_50d97ff18bd9 \
        "${planner}" "${extra[@]}"
      ;;
    person)
      run_case person person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_fork_detour_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    bed_ep4)
      run_case bed_ep4 multi_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_bed_ep4_fork_detour_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_50d97ff18bd9 \
        "${planner}" "${extra[@]}"
      ;;
    bed_open)
      run_case bed_open multi_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_bed_ep4_open_secondary_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_50d97ff18bd9 \
        "${planner}" "${extra[@]}"
      ;;
    bed_open_egress)
      run_case bed_open_egress multi_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_bed_ep4_open_egress_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_50d97ff18bd9 \
        "${planner}" "${extra[@]}"
      ;;
    person_open)
      run_case person_open person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_open_secondary_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    person_open_egress)
      run_case person_open_egress person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_open_egress_candidate/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    person_open_forward)
      run_case person_open_forward person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_open_forward_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    person_east_start)
      run_case person_east_start person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_east_start_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    person_north_forward)
      run_case person_north_forward person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_north_forward_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    person_north_yaw90)
      run_case person_north_yaw90 person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_north_yaw90_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    person_north_yaw180)
      run_case person_north_yaw180 person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_north_yaw180_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    person_north_yawm90)
      run_case person_north_yawm90 person_objectnav_hm3d.yaml \
        data/processed/fire_route_scenarios/Nfvxx8J5NCo_person_ep10_north_yawm90_three_source/val.json.gz \
        Nfvxx8J5NCo_route_contrast_stable_6898210fe2ab \
        "${planner}" "${extra[@]}"
      ;;
    *)
      echo "unknown target: ${target}" >&2
      exit 2
      ;;
  esac
}

run_distance_screen() {
  TRAJECTORY_RUN_LABEL=bed_ep4__random_d4 run_one bed_ep4 random 4.0
  TRAJECTORY_RUN_LABEL=bed_ep4__random_d6 run_one bed_ep4 random 6.0
  TRAJECTORY_RUN_LABEL=person__random_d4 run_one person random 4.0
  TRAJECTORY_RUN_LABEL=person__random_d6 run_one person random 6.0
}

run_open_start_screen() {
  run_one bed_open random 2.0
  run_one person_open random 2.0
}

run_open_start_strategies() {
  for planner in nearest co_ut fill; do
    run_one bed_open "${planner}"
  done
  for planner in nearest co_ut fill; do
    run_one person_open "${planner}"
  done
}

case "${1:-screen}" in
  screen)
    wait_for_existing_run "${1:-}" "${2:-}"
    run_screen
    ;;
  one)
    run_one "${2:-}" "${3:-}" "${4:-}"
    ;;
  distances)
    run_distance_screen
    ;;
  open-start)
    run_open_start_screen
    ;;
  open-strategies)
    run_open_start_strategies
    ;;
  *)
    echo "usage: $0 screen [wait_pid] | distances | open-start | open-strategies | one <bed|bed_ep4|bed_open|bed_open_egress|person|person_open|person_open_egress|person_open_forward|person_east_start|person_north_forward|person_north_yaw90|person_north_yaw180|person_north_yawm90> <nearest|co_ut|fill|random> [random_min_distance_m]" >&2
    exit 2
    ;;
esac
