#!/usr/bin/env bash
set -euo pipefail
RAMAS_SUITE_PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
RAMAS_SUITE_MODE="${RAMAS_SUITE_MODE:-full}"
exec 8>"${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_4_PROCESS.lock"
if ! flock -n 8; then
  echo 'ERROR=ANOTHER_STAGE6_4_PROCESS_IS_RUNNING' >&2
  exit 2
fi
cd "${RAMAS_SUITE_PACKAGE}"
sha256sum -c PACKAGE_MANIFEST.sha256
python3 -m unittest discover -s tests -v
RAMAS_ARGS=(--project-root "${RAMAS_PROJECT_ROOT}" --mode "${RAMAS_SUITE_MODE}")
if [[ -n "${RAMAS_RESUME_DIR:-}" ]]; then
  RAMAS_RUN_ROOT="${RAMAS_RESUME_DIR}"
  RAMAS_ARGS+=(--resume)
else
  RAMAS_RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
  RAMAS_RUN_ROOT="${RAMAS_PROJECT_ROOT}/EXPERIMENT_BRANCHES/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/${RAMAS_RUN_ID}"
fi
mkdir -p "${RAMAS_RUN_ROOT}"
exec 9>"${RAMAS_RUN_ROOT}/SERVER.lock"
if ! flock -n 9; then echo 'ERROR=RESULT_ALREADY_LOCKED' >&2; exit 2; fi
RAMAS_ARGS+=(--output "${RAMAS_RUN_ROOT}")
if [[ -n "${RAMAS_FULL_DATASET:-}" ]]; then RAMAS_ARGS+=(--full-dataset "${RAMAS_FULL_DATASET}"); fi
printf '%s\n' "${RAMAS_RUN_ROOT}" > "${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_4_LAST_RUN.txt"
printf 'MODE=%s\nOUTPUT=%s\nMODEL=llama3.3:70b\nQWEN_USED=FALSE\n' "${RAMAS_SUITE_MODE}" "${RAMAS_RUN_ROOT}"
archive_run() {
  local rc=$?
  trap - EXIT
  local result_archive="${RAMAS_RUN_ROOT}.tar.gz"
  if [[ -e "${result_archive}" ]]; then result_archive="${RAMAS_RUN_ROOT}__snapshot_$(date -u +%Y%m%dT%H%M%S%NZ).tar.gz"; fi
  tar -C "$(dirname "${RAMAS_RUN_ROOT}")" -czf "${result_archive}.pending" "$(basename "${RAMAS_RUN_ROOT}")"
  mv "${result_archive}.pending" "${result_archive}"
  (cd "$(dirname "${result_archive}")" && sha256sum "$(basename "${result_archive}")" > "$(basename "${result_archive}").sha256")
  printf 'RESULT_ARCHIVE=%s\nRESULT_SHA256=%s\nEXIT_CODE=%s\n' "${result_archive}" "${result_archive}.sha256" "${rc}"
  exit "${rc}"
}
trap archive_run EXIT
python3 -u run_suite.py "${RAMAS_ARGS[@]}"
