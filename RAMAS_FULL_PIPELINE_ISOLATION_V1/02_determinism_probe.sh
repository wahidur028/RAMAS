#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${RAMAS_RESOLVED_CONFIG:-/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT/RESOLVED_CONFIG.json}"
OUTPUT="${RAMAS_PROBE_DIR:-/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/DETERMINISM_PROBE}"

python "$PACKAGE_DIR/RAMAS_FULL_PIPELINE_ISOLATION.py" probe \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --models "${RAMAS_MODELS:-llama3.3:70b,qwen3-coder:30b,glm-4.7-flash,qwen3:8b}" \
  --seeds "${RAMAS_SEEDS:-42,2024,3407,7777}"

echo "PROBE_REPORT=$OUTPUT/DETERMINISM_PROBE.json"
