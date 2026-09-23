#!/usr/bin/env bash
# Prove this engine reproduces a frozen Stage 6.4 arm byte-for-byte.
# Replays archived responses only. No model calls, no GPU. Must PASS before 03.
set -euo pipefail
cd "$(dirname "$0")"
/home/infonet/anaconda3/envs/wahid_test/bin/python RAMAS_CORRECTED_MEMORY.py validate \
  --archived-arm "/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/20260909T114202240674964Z/arms/llama_memory_current_year" \
  --output "/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/validation"
