#!/usr/bin/env bash
# Confirm no other RAMAS experiment holds the GPU before starting Stage 8.
set -euo pipefail
ACTIVE=$(pgrep -af 'RAMAS_FULL_PIPELINE_ISOLATION\.py' | grep -E '^[0-9]+ [^ ]*python' || true)
if [ -n "$ACTIVE" ]; then
  echo "BLOCKED: the isolation factorial is still running:"
  echo "$ACTIVE"
  exit 1
fi
echo "CLEAR: no isolation run detected; Stage 8 may start."
