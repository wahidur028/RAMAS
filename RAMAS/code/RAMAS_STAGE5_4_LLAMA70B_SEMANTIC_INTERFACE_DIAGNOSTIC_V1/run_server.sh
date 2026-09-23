#!/usr/bin/env bash
set -euo pipefail

RAMAS_PROJECT_ROOT="${RAMAS_PROJECT_ROOT:-/home/infonet/wahid/leader_router_fresh}"
RAMAS_PACKAGE="${RAMAS_PACKAGE:-${RAMAS_PROJECT_ROOT}/RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1}"
RAMAS_MODEL="${RAMAS_MODEL:-llama3.3:70b}"
RAMAS_STAGE53_RESULTS="${RAMAS_STAGE53_RESULTS:-}"

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
if ! ollama list | awk 'NR>1 {print $1}' | grep -Fxq "${RAMAS_MODEL}"; then
  echo "ERROR=OLLAMA_MODEL_NOT_FOUND:${RAMAS_MODEL}" >&2
  exit 3
fi

if [[ -z "${RAMAS_STAGE53_RESULTS}" ]]; then
  RAMAS_STAGE53_RESULTS="$(find "${RAMAS_PROJECT_ROOT}/EXPERIMENT_BRANCHES/RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1/artifacts" -mindepth 2 -maxdepth 2 -type d -name results -print 2>/dev/null | sort | tail -n 1)"
fi
RAMAS_CALLS="${RAMAS_STAGE53_RESULTS}/06_AGENT_CALLS.jsonl"
if [[ ! -f "${RAMAS_CALLS}" ]]; then
  echo "ERROR=STAGE53_CALLS_NOT_FOUND:${RAMAS_CALLS}" >&2
  exit 2
fi

cd "${RAMAS_PACKAGE}"
sha256sum -c PACKAGE_MANIFEST.sha256
python3 test_diagnostic.py

RAMAS_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAMAS_BRANCH_ROOT="${RAMAS_PROJECT_ROOT}/EXPERIMENT_BRANCHES/RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1"
RAMAS_OUTPUT="${RAMAS_BRANCH_ROOT}/artifacts/${RAMAS_TIMESTAMP}"
mkdir -p "${RAMAS_BRANCH_ROOT}/artifacts"

python3 -u diagnose_llama70b.py \
  --calls "${RAMAS_CALLS}" \
  --model "${RAMAS_MODEL}" \
  --output "${RAMAS_OUTPUT}"

RAMAS_ARCHIVE="${RAMAS_OUTPUT}.tar.gz"
tar -C "$(dirname "${RAMAS_OUTPUT}")" -czf "${RAMAS_ARCHIVE}" "$(basename "${RAMAS_OUTPUT}")"
sha256sum "${RAMAS_ARCHIVE}" > "${RAMAS_ARCHIVE}.sha256"

echo "RAMAS_STAGE5_4_STATUS=PASS"
echo "RESULT_ARCHIVE=${RAMAS_ARCHIVE}"
echo "RESULT_SHA256=${RAMAS_ARCHIVE}.sha256"
