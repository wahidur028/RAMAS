#!/usr/bin/env bash
# Live monitor for the Stage 8 corrected-memory run.
#
# Reads only durable artifacts, so it works no matter how the run was launched
# and survives losing the original terminal. Safe to run any number of times.
#
#   ./05_monitor.sh              watch the newest run, refresh every 20s
#   ./05_monitor.sh -n 5         refresh every 5s
#   ./05_monitor.sh -1           print once and exit
#   ./05_monitor.sh -f           scrolling log: one line per progress step (tail -f style)
#   ./05_monitor.sh -r <run_id>  watch a specific run
set -uo pipefail

ROOT=/home/infonet/wahid/leader_router_fresh
ARTIFACTS="$ROOT/EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts"
PY=/home/infonet/anaconda3/envs/wahid_test/bin/python
INTERVAL=20
ONCE=0
FOLLOW=0
RUN_ID=""

while [ $# -gt 0 ]; do
  case "$1" in
    -n) INTERVAL="$2"; shift 2 ;;
    -1) ONCE=1; shift ;;
    -f) FOLLOW=1; shift ;;
    -r) RUN_ID="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$RUN_ID" ]; then
  RUN_ID=$(ls -1 "$ARTIFACTS" 2>/dev/null | sort | tail -1)
fi
RUN="$ARTIFACTS/$RUN_ID"
if [ ! -d "$RUN" ]; then
  echo "No run found under $ARTIFACTS" >&2
  exit 1
fi

render() {
  clear 2>/dev/null || true
  echo "RAMAS Stage 8 - corrected episodic memory"
  echo "run   : $RUN_ID"
  echo "time  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local pids
  pids=$(pgrep -af 'RAMAS_CORRECTED_MEMORY\.py run' | grep -E '^[0-9]+ [^ ]*python' || true)
  if [ -n "$pids" ]; then
    echo "status: RUNNING (pid $(echo "$pids" | awk '{print $1}' | tr '\n' ' '))"
  elif [ -f "$RUN/RUN_COMPLETE.json" ]; then
    echo "status: COMPLETE"
  elif [ -f "$RUN/FAILURE.json" ]; then
    echo "status: FAILED"
  else
    echo "status: NOT RUNNING (no completion record - interrupted?)"
  fi
  echo

  "$PY" - "$RUN" <<'PYEOF'
import json, sys, time
from pathlib import Path
run = Path(sys.argv[1])
arms = sorted((run / "arms").glob("*")) if (run / "arms").is_dir() else []
if not arms:
    print("  waiting for the first arm to start (source and core audits run first)...")
total_new = total_reused = 0
for arm in arms:
    if not arm.is_dir():
        continue
    name = arm.name
    progress = arm / "PROGRESS.json"
    complete = arm / "RUN_COMPLETE.json"
    calls = len(list((arm / "calls").glob("*.json"))) if (arm / "calls").is_dir() else 0
    if complete.is_file():
        rec = json.loads(complete.read_text())
        total_new += int(rec.get("new_calls", 0)); total_reused += int(rec.get("reused_calls", 0))
        print(f"  {name:22s} COMPLETE   1608/1608  wealth={rec.get('final_wealth',0):.6f}  "
              f"new={rec.get('new_calls',0)} reused={rec.get('reused_calls',0)}")
        continue
    if not progress.is_file():
        print(f"  {name:22s} starting...")
        continue
    d = json.loads(progress.read_text())
    done, tot = int(d["completed_days"]), int(d["total_days"])
    pct = 100.0 * done / tot
    bar = "#" * int(pct / 2.5) + "." * (40 - int(pct / 2.5))
    age = time.time() - progress.stat().st_mtime
    started = min((p.stat().st_mtime for p in (arm / "calls").glob("*.json")), default=None) if calls else None
    rate = (calls / (progress.stat().st_mtime - started)) if started and progress.stat().st_mtime > started else None
    eta = f"{(tot-done)/rate/3600:.1f}h" if rate and rate > 0 else "?"
    sec = f"{1/rate:.1f}s/call" if rate and rate > 0 else "?"
    print(f"  {name:22s} [{bar}] {done:5d}/{tot} {pct:5.1f}%")
    print(f"  {'':22s} valid={d['valid']} wealth={d['wealth']:.6f} last_return={d['last_return_date']}")
    print(f"  {'':22s} calls={calls} {sec} eta={eta} updated {age:.0f}s ago")
    spec = d.get("spec", {})
    print(f"  {'':22s} label={spec.get('label')} retrieval={spec.get('retrieval')} "
          f"per_action={spec.get('per_action')} regime_filter={spec.get('regime_filter')}")
for name in ("FAILURE.json",):
    p = run / name
    if p.is_file():
        f = json.loads(p.read_text())
        print(f"\n  {f.get('error_type')}: {str(f.get('error'))[:300]}")
PYEOF

  if [ -f "$RUN/COMPARISONS.json" ]; then
    echo
    echo "  --- primary contrast ---"
    "$PY" -c "
import json
c=json.load(open('$RUN/COMPARISONS.json'))
for x in c.get('contrasts',[]):
    print(f\"  {x['arm_a']} minus {x['arm_b']}: {x['mean_daily_net_log_difference_bps']:+.6f} bps/day\")
    print(f\"    95% CI [{x['ci95_lower_bps']:+.6f}, {x['ci95_upper_bps']:+.6f}]  {x['interpretation']}\")
    print(f\"    compounded gap {x['net_compounded_return_gap_percentage_points']:+.4f} pp over {x['paired_days']} days\")
" 2>/dev/null || true
  fi
}

if [ "$ONCE" = "1" ]; then
  render
  exit 0
fi

if [ "$FOLLOW" = "1" ]; then
  # Scrolling log. Emits a line only when an arm's progress actually advances,
  # so the output reads like the run's own console output.
  echo "following $RUN_ID - Ctrl-C to stop (does not affect the run)"
  if [ -f "$RUN/console.log" ]; then
    echo "(this run has a durable console log; showing it directly)"
    exec tail -n 50 -F "$RUN/console.log"
  fi
  declare -A SEEN=()
  while :; do
    for pfile in "$RUN"/arms/*/PROGRESS.json; do
      [ -f "$pfile" ] || continue
      arm=$(basename "$(dirname "$pfile")")
      line=$("$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"STAGE8_ARM={d['arm']} PROGRESS={d['completed_days']}/{d['total_days']} \"
      f\"VALID={d['valid']} WEALTH={d['wealth']:.6f} LAST_RETURN={d['last_return_date']}\")
" "$pfile" 2>/dev/null) || continue
      if [ "${SEEN[$arm]:-}" != "$line" ]; then
        echo "$(date -u +%H:%M:%SZ) $line"
        SEEN[$arm]="$line"
      fi
    done
    for done_file in "$RUN"/arms/*/RUN_COMPLETE.json; do
      [ -f "$done_file" ] || continue
      arm=$(basename "$(dirname "$done_file")")
      key="done:$arm"
      if [ "${SEEN[$key]:-}" != "1" ]; then
        echo "$(date -u +%H:%M:%SZ) ARM=$arm STATUS=COMPLETE"
        SEEN[$key]=1
      fi
    done
    if [ -f "$RUN/RUN_COMPLETE.json" ]; then
      echo "$(date -u +%H:%M:%SZ) RUN STATUS=COMPLETE"
      exit 0
    fi
    if [ -f "$RUN/FAILURE.json" ]; then
      echo "$(date -u +%H:%M:%SZ) RUN STATUS=FAILED"
      "$PY" -c "import json;d=json.load(open('$RUN/FAILURE.json'));print(' ',d['error_type'],':',d['error'][:300])"
      exit 1
    fi
    sleep 10
  done
fi

echo "monitoring $RUN_ID every ${INTERVAL}s - Ctrl-C to stop (does not affect the run)"
while :; do
  render
  echo
  echo "refreshing every ${INTERVAL}s - Ctrl-C to stop"
  sleep "$INTERVAL" || break
done
