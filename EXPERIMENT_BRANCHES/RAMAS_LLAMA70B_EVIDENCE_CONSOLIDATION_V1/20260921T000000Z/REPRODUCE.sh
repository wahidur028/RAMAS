#!/usr/bin/env bash
# Regenerate this entire consolidation from the archived runs. Read-only, no model calls.
set -euo pipefail
# Resolve this script's own directory BEFORE changing directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /home/infonet/wahid/leader_router_fresh/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1
exec /home/infonet/anaconda3/envs/wahid_test/bin/python consolidate.py \
  --sources sources.json \
  --output "$HERE"
