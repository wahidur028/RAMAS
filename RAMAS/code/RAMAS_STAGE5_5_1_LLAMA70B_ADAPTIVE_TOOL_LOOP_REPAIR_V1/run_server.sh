#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
PACKAGE_ROOT="${PROJECT_ROOT}/RAMAS_STAGE5_5_1_LLAMA70B_ADAPTIVE_TOOL_LOOP_REPAIR_V1"
PROVIDER="${RAMAS_PROVIDER:-ollama}"
MODEL="${RAMAS_MODEL:-llama3.3:70b}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${PROJECT_ROOT}/EXPERIMENT_BRANCHES/RAMAS_STAGE5_5_1_LLAMA70B_ADAPTIVE_TOOL_LOOP_REPAIR_V1/artifacts/${TIMESTAMP}"

if [[ "${PROVIDER}" == "ollama" && "${MODEL}" != "llama3.3:70b" ]]; then
  echo "ERROR: this experiment is locked to llama3.3:70b; received ${MODEL}" >&2
  exit 64
fi

echo "RAMAS_PROJECT_ROOT=${PROJECT_ROOT}"
echo "RAMAS_STAGE5_5_1_PACKAGE=${PACKAGE_ROOT}"
echo "RAMAS_PROVIDER=${PROVIDER}"
echo "RAMAS_MODEL=${MODEL}"

cd "${PACKAGE_ROOT}"
sha256sum -c PACKAGE_SHA256SUMS.txt
python3 tests/run_all.py

ARGS=(
  --project-root "${PROJECT_ROOT}"
  --provider "${PROVIDER}"
  --model "${MODEL}"
  --output "${RUN_ROOT}"
)

if [[ -n "${RAMAS_EVENT_CSV:-}" || -n "${RAMAS_EVENT_MANIFEST:-}" ]]; then
  ARGS+=(--event-csv "${RAMAS_EVENT_CSV:-}" --event-manifest "${RAMAS_EVENT_MANIFEST:-}")
fi

python3 run_event_agent.py "${ARGS[@]}"

RESULT_ARCHIVE="${RUN_ROOT}.tar.gz"
tar -C "$(dirname "${RUN_ROOT}")" -czf "${RESULT_ARCHIVE}" "$(basename "${RUN_ROOT}")"
sha256sum "${RESULT_ARCHIVE}" > "${RESULT_ARCHIVE}.sha256"

echo "RESULT_ARCHIVE=${RESULT_ARCHIVE}"
echo "RESULT_SHA256=${RESULT_ARCHIVE}.sha256"
