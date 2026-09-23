#!/usr/bin/env bash
set -euo pipefail
RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
RAMAS_STAGE63_PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${RAMAS_STAGE63_PACKAGE}"
sha256sum -c PACKAGE_MANIFEST.sha256
RAMAS_LAUNCH_LOG="${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_3_$(date -u +%Y%m%dT%H%M%S%NZ).log"
nohup bash "${RAMAS_STAGE63_PACKAGE}/run_server.sh" > "${RAMAS_LAUNCH_LOG}" 2>&1 < /dev/null &
RAMAS_LAUNCH_PID=$!
ln -sfn "${RAMAS_LAUNCH_LOG}" "${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_3_RUN.log"
echo "PID=${RAMAS_LAUNCH_PID}"
echo "LOG=${RAMAS_LAUNCH_LOG}"
echo "FOLLOW_LOG=${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_3_RUN.log"
