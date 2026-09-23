#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/infonet/wahid/leader_router_fresh}"
EXPLICIT_SOURCE="${2:-}"
PACKAGE_DIR="$PROJECT_ROOT/RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1"
SOURCE_BRANCH="$PROJECT_ROOT/EXPERIMENT_BRANCHES/RAMOE_STAGE5_1_SINGLE_LLM_AGENT_INTERFACE_REPAIR_V1/artifacts"
OUTPUT_BRANCH="$PROJECT_ROOT/EXPERIMENT_BRANCHES/RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1/artifacts"

if [[ ! -d "$PACKAGE_DIR" ]]; then
  echo "ERROR=PACKAGE_NOT_FOUND:$PACKAGE_DIR" >&2
  exit 2
fi

python3 "$PACKAGE_DIR/tests/run_all.py"

if [[ -n "$EXPLICIT_SOURCE" ]]; then
  STAGE5_SOURCE="$EXPLICIT_SOURCE"
else
  STAGE5_SOURCE="$(find "$SOURCE_BRANCH" -maxdepth 1 -type f -name '*.tar.gz' -print 2>/dev/null | sort | tail -n 1)"
fi

if [[ -z "$STAGE5_SOURCE" || ! -e "$STAGE5_SOURCE" ]]; then
  echo "ERROR=STAGE5_1_RESULT_NOT_FOUND" >&2
  echo "Pass its archive as the second argument." >&2
  exit 3
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$OUTPUT_BRANCH/$RUN_ID"
mkdir -p "$OUTPUT_BRANCH"

python3 "$PACKAGE_DIR/run_agent_residual_audit.py" \
  --stage5-source "$STAGE5_SOURCE" \
  --config "$PACKAGE_DIR/config.json" \
  --output "$RUN_DIR"

RESULT_ARCHIVE="$RUN_DIR.tar.gz"
tar -C "$OUTPUT_BRANCH" -czf "$RESULT_ARCHIVE" "$RUN_ID"
sha256sum "$RESULT_ARCHIVE" > "$RESULT_ARCHIVE.sha256"

echo "RAMAS_STAGE5_2_STATUS=PASS"
echo "RESULT_ARCHIVE=$RESULT_ARCHIVE"
echo "RESULT_SHA256=$RESULT_ARCHIVE.sha256"
