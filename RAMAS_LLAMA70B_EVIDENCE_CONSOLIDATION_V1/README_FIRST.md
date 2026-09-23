# Llama-70B evidence consolidation

Read-only. Gathers every completed `llama3.3:70b` RAMAS arm across all runs into
one normalized table, recomputes all metrics with the frozen Stage 6.4 metrics
module, bootstraps every memory contrast, and summarizes agreement across
repeated realizations. **Makes no model calls and writes nothing outside its
output directory.**

## Run it

```bash
/home/infonet/anaconda3/envs/wahid_test/bin/python consolidate.py \
  --output /home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/<run_id>
```

The pinned interpreter is required (python 3.11.9 / numpy 1.26.4 / pandas 2.2.3).
A different numpy BLAS build changes the last bit of derived quantities.

## Sources

`sources.json` declares every run, its file format, the sampling seed, the prompt
variant, and the factor settings of each arm. Three archived formats are handled:

| format | runs | file |
|---|---|---|
| `single_trace` | 6.1 | `03_DAILY_TRACE.csv`, one arm per file |
| `paired_trace` | 6.2, 6.3 | `03_DAILY_TRACE.csv` with `memory_*` / `no_memory_*` columns |
| `arm_ledgers` | 6.4, 7, ISO | one `DAILY_LEDGER.csv` per arm directory |

Stage 6.4 republishes the Stage 6.2/6.3 trajectories under new names in its
consolidated ledger. This script reads arms from their own `arms/` directories,
so those pairs are counted once, from their original runs.

## Outputs

| file | contents |
|---|---|
| `ALL_ARMS_DAILY.csv` | long format, one row per arm-day, all 24 arms |
| `ARM_INVENTORY.csv` | arm to factor-settings map with provenance paths |
| `ARM_METRICS.csv` | per-arm return, Sharpe, Sortino, MDD, ES95, turnover, cost |
| `MEMORY_CONTRASTS.csv` | every memory-vs-no-memory contrast with bootstrap CI |
| `POOLED_MEMORY_EFFECT.json` | descriptive pooling across realizations |
| `REPLICATION_TRANSMISSION.json` | independent-replication action-flip absorption |
| `CROSS_RUN_REPLICATION.csv` | matched configurations averaged across runs |
| `CONSOLIDATED_REPORT.md` | the readable report |
| `SOURCE_MANIFEST.json` | SHA-256 of every input consumed and output written |

## Reading the pooled number

The realizations replay the **same 1,244 days** of market history. They are
repeated implementations, not independent samples. The pooled mean summarizes
agreement across implementations; no pooled confidence interval is reported and
the spread across realizations is not a standard error. Each realization's own
bootstrap interval is the inferential statement.

## Averaging caveat

`cross_run_average_is_matched` is `false` wherever a group mixes prompt variants.
The isolation run uses `model_neutralized_frozen_v1` ("bounded language-model
expert"); all earlier runs use `frozen_llama_role` ("bounded Llama expert").
Averaging across that boundary mixes two prompts and should be reported as a
robustness range, not a point estimate.
