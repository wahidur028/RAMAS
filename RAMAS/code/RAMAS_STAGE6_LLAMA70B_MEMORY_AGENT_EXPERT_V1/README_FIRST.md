# RAMAS Stage 6 — Llama 70B Memory Agent Expert

This package tests a different role for the LLM. Llama does not directly control the portfolio and does not change the existing RAMAS experts. It emits one bounded signal: `BTC`, `CASH`, or `ABSTAIN`.

The signal is blended with the unchanged RAMAS desired exposure using a separate regime-specific trust weight. Trust starts at 0.05 and cannot exceed 0.20. The unchanged risk layer is applied after blending and remains the final authority.

## Why this replaces Stage 5.5.1

Stage 5.5.1 asked the model to keep exposure when evidence was missing or conflicting, but also required an intervention to pass. Llama returned valid and safe `KEEP` decisions, so that preflight could not distinguish model weakness from rational caution. Stage 6 first uses unambiguous semantic cases and then measures economic value on real development history.

## Scientific timeline

- 2015–2017: historical market observations can warm up/train the causal regime router.
- 2018–2020: valid agent training requires corresponding frozen RAMAS base traces. Those traces are not present in the currently frozen source, so this package does not invent them.
- 2021: development pilot using the earliest verified overlap of corrected RAMAS exposure, next-day returns, and causal router probabilities.
- 2022–2025: not used by this package. They remain for a later frozen, year-by-year diagnostic only if 2021 passes.

The source CSVs contain later dates, so the package verifies their complete file hashes. It parses and uses only the first 364 portfolio rows and only market values through 2021-12-31.

## Server run

From `/home/infonet/wahid/leader_router_fresh`:

```bash
sha256sum -c RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1_CODE.tar.gz.sha256
tar -xzf RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1_CODE.tar.gz

nohup bash RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1/run_server.sh \
  > RAMAS_STAGE6_RUN.log 2>&1 &

echo $!
tail -f RAMAS_STAGE6_RUN.log
```

Stop following the log with `Ctrl+C`; this does not stop the experiment.

## Result interpretation

`RAMAS_STAGE6_SERVER_STATUS=PASS` means the code completed. It is not the research conclusion.

The research conclusion is the `DECISION=` line:

- `CONTINUE_TO_STAGE6_LOCK_AND_YEARLY_DIAGNOSTIC`: the 2021 development gate passed; freeze the design before any later-year run.
- `REVISE_STAGE6_2021_DEVELOPMENT_GATE_FAILED_NO_POST2021`: the mechanism ran, but it did not improve the predeclared combination of profit, Sharpe, drawdown, tail loss, and deterministic comparison. Do not open later years.

The result archive and checksum paths are printed at the end.
