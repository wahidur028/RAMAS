#!/usr/bin/env bash
# finalize_and_push.sh — idempotent: for every COMPLETED run directory, make sure analysis and export
# exist, then commit the completed run(s) to the monorepo and push. Incomplete runs are left alone.
set -uo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
PY="${RAMAS_PYTHON:-/home/infonet/anaconda3/envs/wahid_test/bin/python}"
repo="$(cd "$dir/.." && pwd)"
completed=(); pending=()
for rundir in runs/*/; do
  run="$(basename "$rundir")"; [ -f "runs/$run/contract.json" ] || continue
  if [ -f "runs/$run/complete.json" ]; then
    [ -f "runs/$run/analysis/summary.json" ] || "$PY" runner.py analyze --output "runs/$run" > "runs/$run/analysis_stdout.json" 2>>"runs/$run/server.log"
    [ -f "runs/${run}_results.zip" ] || "$PY" runner.py export --output "runs/$run" >/dev/null
    completed+=("$run")
  else
    pending+=("$run")
  fi
done
echo "completed runs: ${completed[*]:-none} | still running/blocked: ${pending[*]:-none}"
cd "$repo"
spec=(RAMAS_FIXED_STATE_REPEATS_V1)
for run in "${pending[@]}"; do spec+=(":!RAMAS_FIXED_STATE_REPEATS_V1/runs/$run"); done
git add "${spec[@]}" ':!RAMAS_FIXED_STATE_REPEATS_V1/runs/chain.pid'
if git diff --cached --quiet; then echo "nothing new to commit"; exit 0; fi
summary=""
for run in "${completed[@]}"; do
  summary+="$run: $("$PY" -c "import json;d=json.load(open('RAMAS_FIXED_STATE_REPEATS_V1/runs/$run/complete.json'));print(d['status'],d['valid'],'/',d['planned'])"); "
done
git -c user.name=wahidur028 -c user.email=sm.wahidur@gm.gsit.ac.kr commit -q -m "Fixed-state repeats: results — ${summary}

Call records, analysis and exports of the completed run(s) of RAMAS_FIXED_STATE_REPEATS_V1
(PROTOCOL.md v1: Llama-3.3-70B + Qwen3-8B, exposed/hidden retrieval, 3 repeats x 1,244 states;
extension_v1_1: additional installed models).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push -q origin main && echo "pushed $(git rev-parse --short HEAD): ${summary}"
