#!/usr/bin/env bash
set -euo pipefail
RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
RAMAS_STAGE62_PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${RAMAS_STAGE62_PACKAGE}"
sha256sum -c PACKAGE_MANIFEST.sha256
python3 tests/run_all.py

RAMAS_ARGS=()
if [[ -n "${RAMAS_RESUME_DIR:-}" ]]; then
  RAMAS_RUN_ROOT="${RAMAS_RESUME_DIR}"
  RAMAS_ARGS+=(--resume)
else
  RAMAS_RUN_ID="$(date -u +%Y%m%dT%H%M%S%NZ)"
  RAMAS_RUN_ROOT="${RAMAS_PROJECT_ROOT}/EXPERIMENT_BRANCHES/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/artifacts/${RAMAS_RUN_ID}"
fi
mkdir -p "${RAMAS_RUN_ROOT}"
exec 9>"${RAMAS_RUN_ROOT}/SERVER.lock"
if ! flock -n 9; then
  echo "ERROR=ANOTHER_SERVER_WRAPPER_IS_USING_THIS_RUN" >&2
  exit 2
fi
printf '%s\n' "${RAMAS_RUN_ROOT}" > "${RAMAS_PROJECT_ROOT}/RAMAS_STAGE6_2_LAST_RUN.txt"
echo "RAMAS_PROJECT_ROOT=${RAMAS_PROJECT_ROOT}"
echo "RAMAS_STAGE62_PACKAGE=${RAMAS_STAGE62_PACKAGE}"
echo "RAMAS_MODEL=llama3.3:70b"
echo "OUTPUT=${RAMAS_RUN_ROOT}"
echo "QWEN_USED=FALSE"
if [[ -n "${RAMAS_FULL_DATASET:-}" ]]; then
  RAMAS_ARGS+=(--full-dataset "${RAMAS_FULL_DATASET}")
fi
archive_run() {
  local result_code=$?
  trap - EXIT
  if [[ -d "${RAMAS_RUN_ROOT}" ]]; then
    local archive="${RAMAS_RUN_ROOT}.tar.gz"
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
