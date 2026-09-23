#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${RAMAS_CONFIG:-$PACKAGE_DIR/config/experiment.json}"
OUTPUT="${RAMAS_PREFLIGHT_DIR:-/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT}"

python "$PACKAGE_DIR/self_test.py" --config "$CONFIG"
python "$PACKAGE_DIR/RAMAS_FULL_PIPELINE_ISOLATION.py" plan \
  --config "$CONFIG" \
  --models "${RAMAS_MODELS:-llama3.3:70b}" \
  --seeds "${RAMAS_SEEDS:-42}"
python "$PACKAGE_DIR/RAMAS_FULL_PIPELINE_ISOLATION.py" preflight \
  --config "$CONFIG" \
  --output "$OUTPUT"

echo "PREFLIGHT_REPORT=$OUTPUT/PREFLIGHT.json"
echo "RESOLVED_CONFIG=$OUTPUT/RESOLVED_CONFIG.json"
