#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${RAMAS_RESOLVED_CONFIG:-/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT/RESOLVED_CONFIG.json}"
PREFLIGHT="${RAMAS_PREFLIGHT_REPORT:-/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT/PREFLIGHT.json}"
PROBE="${RAMAS_PROBE_REPORT:-/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/DETERMINISM_PROBE/DETERMINISM_PROBE.json}"
MODELS="${RAMAS_MODELS:-llama3.3:70b}"
SEEDS="${RAMAS_SEEDS:-42}"

if [[ -n "${RAMAS_RUN_DIR:-}" ]]; then
  RUN_DIR="$RAMAS_RUN_DIR"
  RESUME_FLAG="--resume"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  RUN_DIR="/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/artifacts/$RUN_ID"
  RESUME_FLAG=""
fi

EXTRA=()
if [[ -f "$PROBE" ]]; then EXTRA+=(--probe "$PROBE"); fi
if [[ "${RAMAS_ALLOW_REDUNDANT_SEEDS:-0}" == "1" ]]; then EXTRA+=(--allow-redundant-seeds); fi
if [[ -n "$RESUME_FLAG" ]]; then EXTRA+=("$RESUME_FLAG"); fi

python "$PACKAGE_DIR/RAMAS_FULL_PIPELINE_ISOLATION.py" plan \
  --config "$CONFIG" --models "$MODELS" --seeds "$SEEDS"
python "$PACKAGE_DIR/RAMAS_FULL_PIPELINE_ISOLATION.py" run \
  --config "$CONFIG" --preflight "$PREFLIGHT" --output "$RUN_DIR" \
  --models "$MODELS" --seeds "$SEEDS" "${EXTRA[@]}"

echo "$RUN_DIR" > /home/infonet/wahid/leader_router_fresh/RAMAS_FULL_PIPELINE_ISOLATION_LAST_RUN.txt
echo "RUN_DIR=$RUN_DIR"
