#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${RAMAS_RESOLVED_CONFIG:-/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT/RESOLVED_CONFIG.json}"
RUN_DIR="${RAMAS_RUN_DIR:-$(cat /home/infonet/wahid/leader_router_fresh/RAMAS_FULL_PIPELINE_ISOLATION_LAST_RUN.txt)}"
VERIFY_DIR="${RAMAS_VERIFY_DIR:-$RUN_DIR/independent_verification}"

python "$PACKAGE_DIR/RAMAS_VERIFY_RESULTS.py" \
  --run "$RUN_DIR" \
  --config "$CONFIG" \
  --output "$VERIFY_DIR"

echo "VERIFICATION_DIR=$VERIFY_DIR"
