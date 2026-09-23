# RAMAS Stage 7 — sequential memory validation

This package is the next scientific test, not a profitability contest. It asks
whether completed episodic memory changes a bounded Llama-70B decision and, in
an independent closed loop, whether that change is useful after costs.

## What it runs

`full` runs four fresh arms over the complete 2021–2025 source horizon:

1. `closed_loop_memory`: Llama 70B, fixed beta 0.05, expanding completed memory.
2. `closed_loop_no_memory`: the same model, state stream and beta contract, with
   all episodic retrieval disabled.
3. `state_controlled_memory`: a diagnostic pair in which both arms receive the
   same current state stream; only completed memory visibility differs.
4. `state_controlled_no_memory`: the state-controlled no-memory control.

The first pair is the primary sequential-learning test. The second pair is an
action-sensitivity diagnostic, not a closed-loop performance estimate.

All arms use the same active Cash/B&H policy core, standard risk projection,
next-day return interval, 0.10% turnover cost and exact three-action contract.
The Llama weights are not updated; “learning” means post-outcome episodic
adaptation. No Qwen fallback is permitted.

## Server launch

Place this directory in `/home/infonet/wahid/leader_router_fresh`, restore the
exact Stage 6.4 source paths, and run:

```bash
cd /home/infonet/wahid/leader_router_fresh
sha256sum -c RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1_CODE.tar.gz.sha256 && \
tar -xzf RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1_CODE.tar.gz && \
bash RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/start.sh
```

The launcher prints the PID and log path. Resume an interrupted run with the
same package, source, model and output directory:

```bash
export RAMAS_RESUME_DIR="$(cat RAMAS_STAGE7_LAST_RUN.txt)"
bash RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/start.sh
```

Use `RAMAS_STAGE7_MODE=closed_loop` or `RAMAS_STAGE7_MODE=state_controlled`
for one phase. `--demo` is available only for local mechanical checks and is
never financial evidence:

```bash
python3 RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/run_stage7.py \
  --project-root "$PWD" --output /tmp/ramas_stage7_demo --mode full --demo
```

## Outputs to inspect first

- `FINAL_STATUS.json`: completion and claim boundaries.
- `MEMORY_ELIGIBILITY_AUDIT.json`: future-episode and no-memory leakage checks.
- `COMPARISONS.json`: paired daily net-log-return effect and block interval.
- `TRANSMISSION.csv`: advice, desired-exposure and final-exposure changes.
- `full_period_metrics.csv`, `yearly_metrics.csv`, `monthly_metrics.csv`,
  `regime_metrics.csv`: one standardized after-cost accounting convention.
- `daily_ledger.csv` and each arm's `calls/`, `episodes.json` and
  `ARM_STATE.json`: audit trail.

This package intentionally reports `continuous_learning_established=false`
until the timing/provenance audit passes and a pre-registered effect survives
the paired uncertainty analysis. A positive return gap alone is insufficient.
