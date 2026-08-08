#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/tmp/conav-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"

mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

case "${1:-}" in
  preflight)
    shift
    exec python /workspace/scripts/docker_preflight.py "$@"
    ;;
  test)
    shift
    exec python -m unittest \
      tests.test_docker_packaging \
      tests.test_global_planners \
      tests.test_risk_planner \
      tests.test_local_planner_baselines \
      tests.test_baseline_benchmark_launcher \
      -v "$@"
    ;;
  --*)
    exec python /workspace/main.py "$@"
    ;;
  "")
    exec python /workspace/scripts/docker_preflight.py \
      --mode navigation --strict
    ;;
  *)
    exec "$@"
    ;;
esac
