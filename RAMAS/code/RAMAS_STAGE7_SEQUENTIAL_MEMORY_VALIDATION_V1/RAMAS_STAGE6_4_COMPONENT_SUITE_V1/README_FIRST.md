# RAMAS Stage6.4 — run the component suite once

The main metric is **net return after trading costs**. Daily net log return, compounded net return, volatility, Sharpe, Sortino, maximum drawdown, CAGR when applicable, tail loss, turnover and cost are reported using one documented convention. Annualization uses 365.25, matching the prior RAMAS reports.

This is an executable suite for the Pass2/Pass3 component questions. It reuses four verified Llama paths, calculates three external benchmarks, then runs seven numerical and seven new Llama portfolios in order. It does not stop because a year loses money. Read SCIENTIFIC_PROTOCOL.md for the exact interventions and questions this data cannot answer.

## Run on your existing server environment

Put the archive and its checksum in `/home/infonet/wahid/leader_router_fresh`. Activate the same `wahid_test` environment used for Stage6.2/6.3, then:

```bash
cd /home/infonet/wahid/leader_router_fresh
sha256sum -c RAMAS_STAGE6_4_COMPONENT_SUITE_V1_CODE.tar.gz.sha256 &&
tar -xzf RAMAS_STAGE6_4_COMPONENT_SUITE_V1_CODE.tar.gz &&
bash RAMAS_STAGE6_4_COMPONENT_SUITE_V1/start.sh
```

Default mode is `full`: no need to launch each variant. The launcher prints PID and exact log path. Follow the latest launch:

```bash
tail -f "$(cat /home/infonet/wahid/leader_router_fresh/RAMAS_STAGE6_4_LAST_LOG.txt)"
```

The expensive stage can require approximately 25 hours of inference for 11,256 calls at ~8 seconds/call, plus overhead. This is an estimate, not a deadline. Numerical results are written first; new-arm reports update after each completed portfolio. Inference remains sequential to avoid competing 70B jobs and changing the serving environment. Keep the existing Ollama server/model running.

## Resume an interrupted run

Keep the same code, data, mode and model. The launcher refuses a duplicate active Stage6.4 process.

```bash
cd /home/infonet/wahid/leader_router_fresh
export RAMAS_RESUME_DIR="$(cat RAMAS_STAGE6_4_LAST_RUN.txt)"
bash RAMAS_STAGE6_4_COMPONENT_SUITE_V1/start.sh
```

An already complete run is verified and exits with zero new calls. Saved responses are not resampled. A stopped partial arm reconstructs state using those responses, then continues. If the original run used a custom raw path or mode, repeat those same environment variables. Do not change mode on resume; use a separate output for a distinct mode. Unset RAMAS_RESUME_DIR before intentionally starting another new experiment.

## Optional limited runs

The default full launch is sufficient. For separately identified outputs, `RAMAS_SUITE_MODE=archive` only checks the completed results and standardizes reports, while `RAMAS_SUITE_MODE=numeric` additionally runs all seven numerical portfolios without contacting Llama. Pending Llama tests remain explicitly pending. The full run includes both phases; running numeric separately first duplicates only inexpensive work.

Example:

```bash
RAMAS_SUITE_MODE=numeric bash RAMAS_STAGE6_4_COMPONENT_SUITE_V1/start.sh
```

## Required frozen inputs

- Stage6.1 result: `EXPERIMENT_BRANCHES/RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/artifacts/20260907T012534Z/continuous_2021_2025`
- Stage6.2 result: `EXPERIMENT_BRANCHES/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/artifacts/20260908T055052711433762Z`
- Stage6.3 result: `EXPERIMENT_BRANCHES/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/artifacts/20260909T013716356200839Z`
- Corrected source: `EXPERIMENT_BRANCHES/RAMOE_RETURN_CLOCK_REPAIR_AND_ECONOMIC_RERUN_V1/artifacts/20260901T024730Z`
- Raw BTC CSV: `/home/infonet/wahid/projects/leader_fresh/full_data_set/full_data_set.csv`
- Raw SHA256: `b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e`

All relative paths are beneath the project root. The exact raw file is required for the earliest 252-day risk window. Its path can be overridden by `RAMAS_FULL_DATASET=/absolute/path/full_data_set.csv`, but the checksum must match. No substitute prices, estimated missing history or saved risk previews are accepted as equivalent inputs.

The matched environment is Python 3.11.9, NumPy 1.26.4, pandas 2.2.3, Ollama 0.23.0, model `llama3.3:70b`, model digest `a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa`. No automatic dependency changes, model download or Qwen fallback.

## What to read and upload

Results go under:
`/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/<run_id>`

- `FINAL_STATUS.json`: what actually completed and remaining scientific limits.
- `TEST_STATUS.json`: completed/pending/unresolved tests; completion is not automatic scientific validation.
- `full_period_metrics.csv`: complete-period and post2021 statistics.
- `yearly_metrics.csv`, `monthly_metrics.csv`, `regime_metrics.csv`: breakdowns.
- `daily_ledger.csv`: every date/portfolio, costs, wealth and daily log return.
- `COMPARISONS.json`: primary/secondary paired effects and uncertainty.
- `COMPONENT_TRANSMISSION.csv`: changed advice, desired allocation, final allocation and return counts.
- `CORE_DAILY_STATE.csv`, source/clock audits: per-regime W, original/altered allocations and scientific qualifications.
- `arms/<name>/calls`, `episodes.json`, `trust_events.json`, `ARM_STATE.json`: reproducible individual decisions and learning state.

The wrapper prints `RESULT_ARCHIVE` and `RESULT_SHA256` on exit, including partial stopped runs. Upload those exact files. A source/transport failure should be inspected, not turned into a favorable economic result.

## Local checks versus actual results

`LOCAL_VALIDATION_RECORD.json` distinguishes controlled integration, historical accounting and real missing-source limitations. A synthetic full-suite test verifies mechanics; it is not a market-performance result. The original forecast/fill chronology and historical reference seam remain explicit open research issues even if every runnable variant completes.
