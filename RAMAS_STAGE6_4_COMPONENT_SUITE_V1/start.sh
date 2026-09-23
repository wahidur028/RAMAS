#!/usr/bin/env bash
set -euo pipefail
RAMAS_SUITE_PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
export RAMAS_PROJECT_ROOT
RAMAS_LAUNCH_LOG="${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_4_RUN.log"
if [[ -e "${RAMAS_LAUNCH_LOG}" ]]; then
  RAMAS_LAUNCH_LOG="${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_4_RUN_$(date -u +%Y%m%dT%H%M%S%NZ).log"
fi
nohup bash "${RAMAS_SUITE_PACKAGE}/run_server.sh" > "${RAMAS_LAUNCH_LOG}" 2>&1 &
printf 'PID=%s\nLOG=%s\n' "$!" "${RAMAS_LAUNCH_LOG}"
printf '%s\n' "${RAMAS_LAUNCH_LOG}" > "${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_4_LAST_LOG.txt"
