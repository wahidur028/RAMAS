# RAMAS — episodic memory in a bounded LLM Bitcoin allocator

Code, frozen inputs, archived model calls and results for the RAMAS manuscript.
Everything needed to inspect or re-run the experiments is here; nothing was
edited after the runs — the executable packages carry `PACKAGE_MANIFEST.sha256`
files that still match, and every run directory carries its own completion
record with per-file hashes.

## What RAMAS is, in one paragraph

A deterministic numerical allocator (the *pre-agent RAMoE control*: a regime
posterior, a cash/Bitcoin policy blend, a CVaR/turnover risk projection and
10-bps cost accounting) is extended with a bounded Llama-3.3-70B advisor that
may vote BTC / CASH / ABSTAIN each day. The advisor's vote is blended into the
exposure with a small trust weight and then re-projected by the same risk
layer. An **episodic memory** shows the advisor its own completed past decisions
in similar states. The experiments measure whether that memory changes the
decisions, and whether the changes reach the executed portfolio.

## Repository map

| path | contents |
|---|---|
| `RAMAS_STAGE6_4_COMPONENT_SUITE_V1/` | the frozen engine: daily loop, projection, metrics, tests; vendors the original Stage 6.1 agent under `vendor/stage61/` |
| `RAMAS_FULL_PIPELINE_ISOLATION_V1/` | the 2×2×2 factorial (memory × trust authority × controller) orchestrator |
| `RAMAS_STAGE8_CORRECTED_MEMORY_V1/` | the corrected-retrieval experiment (balanced per-action retrieval) |
| `RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/` | consolidates every Llama-70B arm; bootstraps every memory contrast |
| `RAMAS/code/` | completed code packages for Stages 5.2 – 7 |
| `RAMAS/scripts/`, `RAMAS/results/` | one-off analyses (no-controller counterfactual, advisor ablations) and their outputs |
| `EXPERIMENT_BRANCHES/RAMAS_*/`, `EXPERIMENT_BRANCHES/RAMOE_STAGE5_*/` | every run: contracts, source/clock audits, per-arm `DAILY_LEDGER.csv`, `episodes.json`, `trust_events.json`, `RUN_COMPLETE.json`, and the full per-call LLM journals under `calls/` |
| `EXPERIMENT_BRANCHES/RAMOE_RETURN_CLOCK_REPAIR_AND_ECONOMIC_RERUN_V1/` | the **frozen input source**: the numerical control's router-probability and expert-proposal stream, its return-clock audit, and the small numerical package (`base_source/…/src/`) whose accounting, risk projection and trust update the RAMAS engine imports. Pinned by manifest; required to run anything |
| `data/raw/full_data_set.csv` | the daily BTC/USD dataset (sha256 `b69f17a1…`), pinned by the pipelines |
| `docs/RAMAS_TECHNICAL_REPORT.md` | **read this first**: the technical report — architecture, data, every experiment with its results and hashes, reproduction commands, supplementary-table recipes |
| `tools/localize_configs.py` | writes clone-local copies of the pinned configs (see *Running*) |
| `MANIFEST.sha256` | hash of every file in this repository |

The archives (`*.tar.gz`, `*.zip`) referenced by some completion records are not
included; their `.sha256` sidecars are, so identities can still be checked.

## Where the manuscript's numbers live

`docs/RAMAS_TECHNICAL_REPORT.md` walks through every experiment, table and hash; the short map below points at the primary files.

| result | file |
|---|---|
| Stage 6.4 component suite, 21 arms, 22 bootstrap contrasts | `EXPERIMENT_BRANCHES/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/20260909T114202240674964Z/{daily_ledger.csv,COMPARISONS.json,all_metrics.csv}` |
| 2×2×2 factorial, 8 Llama arms + 2 numerical controls | `EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/artifacts/20260920T084300Z/cells/llama3.3_70b__seed_42/arms/*/DAILY_LEDGER.csv` |
| corrected-retrieval experiment (Stage 8) | `EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts/20260922T014916Z/{COMPARISONS.json,arms/*/DAILY_LEDGER.csv}` |
| all Llama-70B memory contrasts, pooled | `EXPERIMENT_BRANCHES/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/20260921T000000Z/{MEMORY_CONTRASTS.csv,POOLED_MEMORY_EFFECT.json,REPLICATION_TRANSMISSION.json}` |
| retrieval-collapse measurement | recompute from `…/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/…/arms/llama_memory_current_year/calls/*.json` (`request.memory.summary`) |

Conventions: net return after 10-bps proportional trading costs; annualization
365.25; evaluation window return dates 2022-01-01 to 2025-05-28 (1,244 days);
paired circular moving-block bootstrap, block 30, 5,000 resamples, seed 16062.
`RAMAS_STAGE6_4_COMPONENT_SUITE_V1/METRIC_DEFINITIONS.md` has every formula.

## Running

**Runtime is pinned and matters:** python 3.11.9, numpy 1.26.4, pandas 2.2.3.
A different numpy BLAS build changes the last bit of the retrieval distance,
which changes request digests and breaks journal matching.

The configs pin the original machine's path. Either make that path point at
your clone (nothing changes, all hashes match):

```bash
sudo mkdir -p /home/infonet/wahid /home/infonet/wahid/projects/leader_fresh/full_data_set
sudo ln -s "$PWD" /home/infonet/wahid/leader_router_fresh
sudo ln -s "$PWD/data/raw/full_data_set.csv" /home/infonet/wahid/projects/leader_fresh/full_data_set/full_data_set.csv
```

or generate clone-local configs with `python tools/localize_configs.py` and pass
them via `--config` (run identities will then differ from the archived runs).

No model is needed for: preflight, the legacy-equivalence gate, the memory
diagnostics, the consolidation, or the test suites:

```bash
cd RAMAS_STAGE6_4_COMPONENT_SUITE_V1 && python -m pytest tests -q          # 31 tests
cd RAMAS_STAGE8_CORRECTED_MEMORY_V1 && ./01_preflight.sh && ./02_validate_legacy_equivalence.sh
cd RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1 && python consolidate.py --output /tmp/consolidation
```

Re-running the LLM arms additionally needs Ollama 0.23.0 serving `llama3.3:70b`,
digest `a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa`,
temperature 0. Durable journals mean an interrupted run resumes without
re-sampling any saved response.

## Scope of the evidence

All results are retrospective, reused-history mechanism diagnostics on one
asset and one model; 2024–2025 is not fresh out-of-sample; the regime posterior
is a state estimate conditioned on prior-day information, not a validated
next-day forecast; decisions use and fill at the same close. The manuscript's
Limitations section states these; the run contracts record them as
`claim_boundaries`.

## License and citation

MIT — see `LICENSE`. Cite the accompanying manuscript; `CITATION.cff` has the
software citation.
