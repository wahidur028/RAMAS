#!/usr/bin/env bash
set -euo pipefail

RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
RAMAS_PACKAGE="${RAMAS_PACKAGE:-${RAMAS_PROJECT_ROOT}/RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1}"

if [[ ! -d "${RAMAS_PROJECT_ROOT}" ]]; then
  echo "ERROR=PROJECT_ROOT_NOT_FOUND:${RAMAS_PROJECT_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${RAMAS_PACKAGE}" ]]; then
  echo "ERROR=PACKAGE_NOT_FOUND:${RAMAS_PACKAGE}" >&2
  exit 2
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR=OLLAMA_NOT_FOUND" >&2
  exit 3
fi

cd "${RAMAS_PACKAGE}"
RAMAS_MODEL="$(python3 -c 'import json; print(json.load(open("config.json", encoding="utf-8"))["provider"]["model"])')"
if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "${RAMAS_MODEL}"; then
  echo "ERROR=OLLAMA_MODEL_NOT_FOUND:${RAMAS_MODEL}" >&2
  exit 3
fi
if [[ ! -f PACKAGE_MANIFEST.sha256 ]]; then
  echo "ERROR=PACKAGE_MANIFEST_NOT_FOUND" >&2
  exit 2
fi

echo "RAMAS_PROJECT_ROOT=${RAMAS_PROJECT_ROOT}"
echo "RAMAS_PACKAGE=${RAMAS_PACKAGE}"
echo "RAMAS_MODEL=${RAMAS_MODEL}"
sha256sum -c PACKAGE_MANIFEST.sha256
python3 tests/run_all.py

RAMAS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAMAS_BRANCH_ROOT="${RAMAS_PROJECT_ROOT}/EXPERIMENT_BRANCHES/RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1"
RAMAS_RUN_ROOT="${RAMAS_BRANCH_ROOT}/artifacts/${RAMAS_TIMESTAMP}"
RAMAS_PREFLIGHT_DIR="${RAMAS_RUN_ROOT}/interface_preflight"
RAMAS_RESULTS_DIR="${RAMAS_RUN_ROOT}/results"
mkdir -p "${RAMAS_RUN_ROOT}"

{
  echo "RUN_DIR=${RAMAS_RUN_ROOT}"
  echo "PREFLIGHT_DIR=${RAMAS_PREFLIGHT_DIR}"
  echo "RESULTS_DIR=${RAMAS_RESULTS_DIR}"
  echo "MODEL=${RAMAS_MODEL}"
  echo "START_DATE=$(date '+%Y/%m/%d')"
  echo "START_TIME=$(date '+%H:%M:%S')"
} > "${RAMAS_BRANCH_ROOT}/CURRENT_RUN.env"

archive_run() {
  local archive="${RAMAS_RUN_ROOT}.tar.gz"
  tar -C "$(dirname "${RAMAS_RUN_ROOT}")" -czf "${archive}" "$(basename "${RAMAS_RUN_ROOT}")"
  sha256sum "${archive}" > "${archive}.sha256"
  echo "RESULT_ARCHIVE=${archive}"
  echo "RESULT_SHA256=${archive}.sha256"
}

set +e
python3 -u run_residual_agent.py \
  --project-root "${RAMAS_PROJECT_ROOT}" \
  --provider ollama \
  --preflight-only \
  --output "${RAMAS_PREFLIGHT_DIR}"
RAMAS_PREFLIGHT_RC=$?
set -e

if [[ ${RAMAS_PREFLIGHT_RC} -ne 0 ]]; then
  echo "RAMAS_STAGE5_3_STATUS=STOP_PREFLIGHT_FAILED"
  archive_run
  exit "${RAMAS_PREFLIGHT_RC}"
fi

set +e
python3 -u run_residual_agent.py \
  --project-root "${RAMAS_PROJECT_ROOT}" \
  --provider ollama \
  --preflight-evidence "${RAMAS_PREFLIGHT_DIR}/00_PREFLIGHT.json" \
  --output "${RAMAS_RESULTS_DIR}"
RAMAS_FULL_RC=$?
set -e

archive_run
if [[ ${RAMAS_FULL_RC} -ne 0 ]]; then
  echo "RAMAS_STAGE5_3_STATUS=FAILED_WITH_ARCHIVE"
  exit "${RAMAS_FULL_RC}"
fi

echo "RAMAS_STAGE5_3_STATUS=PASS"
