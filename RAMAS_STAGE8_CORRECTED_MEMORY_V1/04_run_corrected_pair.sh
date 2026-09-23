#!/usr/bin/env bash
# Execute the corrected memory / no-memory pair.
# ~3,216 Llama calls. Observed rate ~4.4 s/call, so roughly 4 hours for both arms.
#
# Writes a durable console log next to the artifacts so progress survives losing
# this terminal. Watch it with ./05_monitor.sh or by tailing the log it prints.
#
# DO NOT START until 01_preflight.sh and 02_validate_legacy_equivalence.sh pass
# and 00_check_safe_to_start.sh reports CLEAR.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUN_ID="${RAMAS_STAGE8_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/console.log"
echo "$RUN_DIR" > "/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/LAST_RUN.txt"
echo "$LOG"     > "/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/LAST_LOG.txt"

echo "run dir : $RUN_DIR"
echo "log     : $LOG"
echo "monitor : $(pwd)/05_monitor.sh"
echo

exec > >(tee -a "$LOG") 2>&1
/home/infonet/anaconda3/envs/wahid_test/bin/python RAMAS_CORRECTED_MEMORY.py run \
  --preflight "/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/PREFLIGHT/PREFLIGHT.json" \
  --output "$RUN_DIR" \
  "$@"
