#!/usr/bin/env bash
# launch_extension.sh — extension_v1_1 only (no re-run of the protocol main run): pilot -> run -> analyze/export.
# Usage: nohup bash launch_extension.sh <models,comma,separated> <pilot_dir> <run_dir> >> runs/extension_chain.log 2>&1 &
set -uo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
PY="${RAMAS_PYTHON:-/home/infonet/anaconda3/envs/wahid_test/bin/python}"
MODELS="${1:?models}"; PILOT="${2:?pilot dir}"; RUN="${3:?run dir}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$PILOT" "$RUN"
echo "[$(stamp)] extension pilot for ${MODELS} -> ${PILOT}"
"$PY" -u runner.py pilot --output "$PILOT" --models "$MODELS" --label extension_v1_1_additional_models >> "$PILOT/server.log" 2>&1; echo "[$(stamp)] extension pilot exited rc=$?"
if [ -f "$PILOT/complete.json" ] && grep -q COMPLETE_ALL_VALID "$PILOT/complete.json"; then
  echo "[$(stamp)] extension run starting -> ${RUN} (log: ${RUN}/server.log)"
  "$PY" -u runner.py extension --output "$RUN" --pilot "$PILOT" --models "$MODELS" --label extension_v1_1_additional_models >> "$RUN/server.log" 2>&1; echo "[$(stamp)] extension run exited rc=$?"
  if [ -f "$RUN/complete.json" ]; then
    "$PY" runner.py analyze --output "$RUN" > "$RUN/analysis_stdout.json" 2>> "$RUN/server.log" && echo "[$(stamp)] extension analysis written"
    "$PY" runner.py export --output "$RUN" && echo "[$(stamp)] extension exported"
  fi
else
  echo "[$(stamp)] extension pilot did not pass (all-valid gate); run skipped"
fi
echo "[$(stamp)] extension chain end"
