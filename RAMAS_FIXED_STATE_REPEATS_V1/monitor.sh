#!/usr/bin/env bash
# monitor.sh — one-screen status of pilot / main / extension runs, then follow the live log.
#   bash monitor.sh        status once      |   bash monitor.sh -f   follow the active server.log
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; cd "$dir"
PY="${RAMAS_PYTHON:-/home/infonet/anaconda3/envs/wahid_test/bin/python}"
for run in pilot main extension_pilot extension; do
  [ -d "runs/$run" ] || continue
  echo "=== runs/$run"
  "$PY" - "runs/$run" <<'EOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
def j(n):
    f = p / n
    return json.loads(f.read_text()) if f.exists() else None
c, pr, done, bl = j('contract.json'), j('progress.json'), j('complete.json'), j('blocked.json')
if c: print(f"  planned {c['planned_calls']} calls, models {c['models']}, repeats {c['repeats']}, started {c['created_at']}")
if pr: print(f"  recorded {pr['recorded']}/{pr['planned']}  valid {pr['valid']}  remaining {pr['remaining']}  eta~{pr['estimated_remaining_hours_from_observed_latency']} h  (as of {pr['at']})")
if done: print(f"  COMPLETE: {done['status']}  valid {done['valid']}/{done['planned']} at {done['completed_at']}")
if bl: print(f"  BLOCKED: {bl['latest']['status']} — {bl['latest']['detail'][:120]}")
EOF
  echo "  calls on disk: $(ls runs/$run/calls 2>/dev/null | wc -l)"
done
if [ "${1:-}" = "-f" ]; then
  for run in extension extension_pilot main pilot; do
    if [ -f "runs/$run/server.log" ] && [ ! -f "runs/$run/complete.json" ]; then echo "--- following runs/$run/server.log (Ctrl-C stops following, not the run)"; tail -n 5 -f "runs/$run/server.log"; exit; fi
  done
  echo "no active run to follow"
fi
