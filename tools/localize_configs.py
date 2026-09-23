#!/usr/bin/env python3
"""Write *.local.json copies of the pinned experiment configs pointing at THIS clone.

The shipped configs pin the absolute path of the original machine and are left
byte-identical so their hashes match the archived runs. Two ways to run here:

  A) make the pinned path point at this clone (nothing changes, all hashes match):
       sudo mkdir -p /home/infonet/wahid && sudo ln -s "$PWD" /home/infonet/wahid/leader_router_fresh
  B) run this script, then pass the generated *.local.json with --config.
     The run identity hash will differ from the archived runs; that is expected.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OLD = "/home/infonet/wahid/leader_router_fresh"
RAW_OLD = "/home/infonet/wahid/projects/leader_fresh/full_data_set/full_data_set.csv"
targets = [ROOT / "RAMAS_STAGE8_CORRECTED_MEMORY_V1/config/experiment.json",
           ROOT / "RAMAS_FULL_PIPELINE_ISOLATION_V1/config/experiment.json",
           ROOT / "RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/sources.json"]
for t in targets:
    text = t.read_text().replace(RAW_OLD, str(ROOT / "data/raw/full_data_set.csv")).replace(OLD, str(ROOT))
    json.loads(text)  # must still be valid JSON
    local = t.with_suffix(".local.json"); local.write_text(text)
    print(f"wrote {local.relative_to(ROOT)}")
print("\nNote: RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/consolidate.py hard-codes the Stage 6.4 path;")
print("use option A (symlink) to run the consolidation, or edit its STAGE64 constant locally.")
