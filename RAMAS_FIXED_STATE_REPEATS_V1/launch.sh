#!/usr/bin/env bash
set -euo pipefail
ramas_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ramas_python="${RAMAS_PYTHON:-python3}"
ramas_mode="${1:-}"
ramas_run="${2:-}"
if [[ "$ramas_mode" != "pilot" && "$ramas_mode" != "run" ]] || [[ -z "$ramas_run" ]]; then
  echo 'Usage: bash launch.sh pilot RUN_DIRECTORY | bash launch.sh run RUN_DIRECTORY PILOT_DIRECTORY' >&2
  exit 2
fi
mkdir -p -- "$ramas_run"
ramas_run="$(cd -- "$ramas_run" && pwd)"
ramas_args=("$ramas_dir/runner.py" "$ramas_mode" --output "$ramas_run")
if [[ "$ramas_mode" == "run" ]]; then
  ramas_pilot="${3:?Full run requires the completed pilot directory}"
  ramas_args+=(--pilot "$ramas_pilot")
fi
nohup "$ramas_python" -u "${ramas_args[@]}" >> "$ramas_run/server.log" 2>&1 < /dev/null &
ramas_pid=$!
printf '%s\n' "$ramas_pid" > "$ramas_run/launcher.pid"
printf 'Started PID %s. Log: %s/server.log\n' "$ramas_pid" "$ramas_run"
printf 'Resume with the same command and directory if interrupted.\n'
