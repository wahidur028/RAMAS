#!/usr/bin/env bash
set -euo pipefail
RAMAS_PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
export RAMAS_PROJECT_ROOT
RAMAS_LOG="${RAMAS_PROJECT_ROOT}/RAMAS_STAGE7_RUN.log"
if [[ -e "${RAMAS_LOG}" ]]; then
  RAMAS_LOG="${RAMAS_PROJECT_ROOT}/RAMAS_STAGE7_RUN_$(date -u +%Y%m%dT%H%M%S%NZ).log"
fi
nohup bash "${RAMAS_PACKAGE}/run_server.sh" >"${RAMAS_LOG}" 2>&1 &
printf 'PID=%s\nLOG=%s\n' "$!" "${RAMAS_LOG}"
printf '%s\n' "${RAMAS_LOG}" >"${RAMAS_PROJECT_ROOT}/RAMAS_STAGE7_LAST_LOG.txt"
