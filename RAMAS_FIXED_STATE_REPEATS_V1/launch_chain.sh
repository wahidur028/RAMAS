#!/usr/bin/env bash
# launch_chain.sh — server addition (not in the archive's protocol files).
# Runs, in order and resume-safely:  main protocol run (after a passed pilot) -> analyze/export
# -> optional extension pilot + extension run on additional installed models -> analyze/export.
# Usage:  nohup bash launch_chain.sh [ext_model1,ext_model2,...] > runs/chain.log 2>&1 &
set -uo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
PY="${RAMAS_PYTHON:-/home/infonet/anaconda3/envs/wahid_test/bin/python}"
EXT_MODELS="${1:-}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[$(stamp)] chain start; extension models: ${EXT_MODELS:-none}"
"$PY" - <<'EOF' || { echo "[$(date -u +%FT%TZ)] pilot gate failed; chain stopped"; exit 1; }
import json
r = json.load(open('runs/pilot/complete.json'))
assert r['status'] == 'COMPLETE_ALL_VALID' and r['valid'] == 48, r
print('PILOT_PASS', r['valid'], 'valid of', r['planned'])
EOF
mkdir -p runs/main
echo "[$(stamp)] main run starting (log: runs/main/server.log)"
"$PY" -u runner.py run --output runs/main --pilot runs/pilot >> runs/main/server.log 2>&1; rc=$?
echo "[$(stamp)] main run exited rc=$rc"
"$PY" runner.py status --output runs/main
if [ -f runs/main/complete.json ]; then
  "$PY" runner.py analyze --output runs/main > runs/main/analysis_stdout.json 2>> runs/main/server.log && echo "[$(stamp)] main analysis written"
  "$PY" runner.py export  --output runs/main && echo "[$(stamp)] main exported"
else
  echo "[$(stamp)] main run not complete (paused/blocked); resume with: bash launch_chain.sh ${EXT_MODELS}"; exit 3
fi
if [ -n "$EXT_MODELS" ]; then
  mkdir -p runs/extension_pilot runs/extension
  echo "[$(stamp)] extension pilot starting for ${EXT_MODELS}"
  "$PY" -u runner.py pilot --output runs/extension_pilot --models "$EXT_MODELS" --label extension_v1_1_additional_models >> runs/extension_pilot/server.log 2>&1; echo "[$(stamp)] extension pilot exited rc=$?"
  if [ -f runs/extension_pilot/complete.json ] && grep -q COMPLETE_ALL_VALID runs/extension_pilot/complete.json; then
    echo "[$(stamp)] extension run starting (log: runs/extension/server.log)"
    "$PY" -u runner.py extension --output runs/extension --pilot runs/extension_pilot --models "$EXT_MODELS" --label extension_v1_1_additional_models >> runs/extension/server.log 2>&1; echo "[$(stamp)] extension run exited rc=$?"
    if [ -f runs/extension/complete.json ]; then
      "$PY" runner.py analyze --output runs/extension > runs/extension/analysis_stdout.json 2>> runs/extension/server.log && echo "[$(stamp)] extension analysis written"
      "$PY" runner.py export --output runs/extension && echo "[$(stamp)] extension exported"
    fi
  else
    echo "[$(stamp)] extension pilot did not pass; extension run skipped"
  fi
fi
echo "[$(stamp)] chain end"
