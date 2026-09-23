#!/usr/bin/env bash
set -euo pipefail
RAMAS_PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
RAMAS_MODE="${RAMAS_STAGE7_MODE:-full}"
exec 8>"${RAMAS_PROJECT_ROOT}/RAMAS_STAGE7_PROCESS.lock"
flock -n 8 || { echo 'ERROR=ANOTHER_STAGE7_PROCESS_IS_RUNNING' >&2; exit 2; }
cd "${RAMAS_PACKAGE}"
sha256sum -c PACKAGE_MANIFEST.sha256
PYTHONPATH="${RAMAS_PACKAGE}/RAMAS_STAGE6_4_COMPONENT_SUITE_V1${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -m unittest discover -s tests -p 'test_*.py' -v
if [[ -n "${RAMAS_RESUME_DIR:-}" ]]; then
  RAMAS_RUN_ROOT="${RAMAS_RESUME_DIR}"
else
  RAMAS_RUN_ROOT="${RAMAS_PROJECT_ROOT}/EXPERIMENT_BRANCHES/${RAMAS_STAGE7_EXPERIMENT_DIR:-RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1}/artifacts/$(date -u +%Y%m%dT%H%M%S%NZ)"
fi
mkdir -p "${RAMAS_RUN_ROOT}"
exec 9>"${RAMAS_RUN_ROOT}/SERVER.lock"
flock -n 9 || { echo 'ERROR=RESULT_ALREADY_LOCKED' >&2; exit 2; }
printf '%s\n' "${RAMAS_RUN_ROOT}" >"${RAMAS_PROJECT_ROOT}/RAMAS_STAGE7_LAST_RUN.txt"
ARGS=(--project-root "${RAMAS_PROJECT_ROOT}" --output "${RAMAS_RUN_ROOT}" --mode "${RAMAS_MODE}")
if [[ -n "${RAMAS_FULL_DATASET:-}" ]]; then ARGS+=(--full-dataset "${RAMAS_FULL_DATASET}"); fi
archive_run() {
  local rc=$?
  trap - EXIT
  local archive="${RAMAS_RUN_ROOT}.tar.gz"
  if [[ -e "${archive}" ]]; then archive="${RAMAS_RUN_ROOT}__snapshot_$(date -u +%Y%m%dT%H%M%S%NZ).tar.gz"; fi
  tar -C "$(dirname "${RAMAS_RUN_ROOT}")" -czf "${archive}.pending" "$(basename "${RAMAS_RUN_ROOT}")"
  mv "${archive}.pending" "${archive}"
  (cd "$(dirname "${archive}")" && sha256sum "$(basename "${archive}")" >"$(basename "${archive}").sha256")
  printf 'RESULT_ARCHIVE=%s\nRESULT_SHA256=%s\nEXIT_CODE=%s\n' "${archive}" "${archive}.sha256" "${rc}"
  exit "${rc}"
}
trap archive_run EXIT
python3 -u run_stage7.py "${ARGS[@]}"
