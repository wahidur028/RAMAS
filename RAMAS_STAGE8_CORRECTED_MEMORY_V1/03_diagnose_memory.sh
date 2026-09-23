#!/usr/bin/env bash
# Reproduce the retrieval-collapse and credit-label measurements. No model calls.
set -euo pipefail
cd "$(dirname "$0")"
/home/infonet/anaconda3/envs/wahid_test/bin/python RAMAS_CORRECTED_MEMORY.py diagnose \
  --archived-arm "/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/20260909T114202240674964Z/arms/llama_memory_current_year" \
  --output "/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/diagnostics"
