#!/usr/bin/env bash
set -euo pipefail
RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
RAMAS_STAGE63_PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${RAMAS_PROJECT_ROOT}"
exec 8>"${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_3_PROCESS.lock"
if ! flock -n 8; then
  echo "ERROR=STAGE6_3_ALREADY_RUNNING_NO_DUPLICATE_STARTED" >&2
  exit 2
fi
cd "${RAMAS_STAGE63_PACKAGE}"
sha256sum -c PACKAGE_MANIFEST.sha256
python3 tests/run_all.py

RAMAS_ARGS=()
if [[ -n "${RAMAS_RESUME_DIR:-}" ]]; then
  RAMAS_RUN_ROOT="${RAMAS_RESUME_DIR}"
  RAMAS_ARGS+=(--resume)
else
  RAMAS_RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
  RAMAS_RUN_ROOT="${RAMAS_PROJECT_ROOT}/EXPERIMENT_BRANCHES/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/artifacts/${RAMAS_RUN_ID}"
fi
mkdir -p "${RAMAS_RUN_ROOT}"
exec 9>"${RAMAS_RUN_ROOT}/SERVER.lock"
if ! flock -n 9; then
  echo "ERROR=ANOTHER_SERVER_WRAPPER_IS_USING_THIS_RUN" >&2
  exit 2
fi
printf '%s\n' "${RAMAS_RUN_ROOT}" > "${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_3_LAST_RUN.txt"
echo "RAMAS_PROJECT_ROOT=${RAMAS_PROJECT_ROOT}"
echo "RAMAS_STAGE63_PACKAGE=${RAMAS_STAGE63_PACKAGE}"
echo "RAMAS_MODEL=llama3.3:70b"
echo "OUTPUT=${RAMAS_RUN_ROOT}"
echo "QWEN_USED=FALSE"
if [[ -n "${RAMAS_FULL_DATASET:-}" ]]; then
  RAMAS_ARGS+=(--full-dataset "${RAMAS_FULL_DATASET}")
fi
if [[ "${RAMAS_AUDIT_ONLY:-0}" == "1" ]]; then
  RAMAS_ARGS+=(--audit-only)
fi
archive_run() {
  local result_code=$?
  trap - EXIT
  if [[ -d "${RAMAS_RUN_ROOT}" ]]; then
    local archive="${RAMAS_RUN_ROOT}.tar.gz"
    if [[ -e "${archive}" ]]; then
      archive="${RAMAS_RUN_ROOT}__snapshot_$(date -u +%Y%m%dT%H%M%S%NZ).tar.gz"
    fi
    tar -C "$(dirname "${RAMAS_RUN_ROOT}")" -czf "${archive}.pending" "$(basename "${RAMAS_RUN_ROOT}")"
    mv "${archive}.pending" "${archive}"
    (cd "$(dirname "${archive}")" && sha256sum "$(basename "${archive}")" > "$(basename "${archive}").sha256")
    echo "RESULT_ARCHIVE=${archive}"
    echo "RESULT_SHA256=${archive}.sha256"
  fi
  exit "${result_code}"
}
trap archive_run EXIT
python3 -u run_experiment.py --project-root "${RAMAS_PROJECT_ROOT}" --output "${RAMAS_RUN_ROOT}" "${RAMAS_ARGS[@]}"
echo "RAMAS_STAGE6_3_SERVER_STATUS=PASS"
