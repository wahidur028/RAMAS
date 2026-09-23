# RAMAS Stage 6.3 — fixed-trust memory experiment

Two fresh Llama 3.3 70B agents run with **fixed agent trust beta 0.05**. One receives eligible completed experiences; the other receives none. This tests episodic information when adaptive trust cannot reduce influence to zero. The main RAMAS architecture still uses adaptive trust; this is an ablation, not a claim that 0.05 is optimal.

## Start on the original server

Use the existing `wahid_test` environment. Place the archive and checksum in `/home/infonet/wahid/leader_router_fresh`, then run:

```bash
cd /home/infonet/wahid/leader_router_fresh

sha256sum -c RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1_CODE.tar.gz.sha256 &&
tar -xzf RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1_CODE.tar.gz &&
bash RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/start.sh
```

Follow progress:

```bash
tail -f /home/infonet/wahid/leader_router_fresh/RAMAS_STAGE6_3_RUN.log
```

Ctrl+C exits the log viewer; the background experiment continues. At completion, upload the files printed as `RESULT_ARCHIVE` and `RESULT_SHA256`.

Every normal launch chooses a new timestamped result directory. A process lock prevents simultaneous Stage 6.3 wrappers. Earlier code/results are not edited. Do not extract over a modified package you want to preserve.

## What runs

1. Verify the package manifest and offline tests.
2. Reconstruct the original market/features/risk stream from exact historical sources and raw BTC data.
3. Audit the completed Stage 6.2 reference, source ordering, return alignment and known timing/reference limitations before model calls.
4. Check the same Llama digest, Ollama serving metadata and Python/NumPy/pandas versions as Stage 6.2.
5. Run 12 shared semantic checks. These test output/instruction handling, not trading skill.
6. Make 1,608 independent daily calls per arm, starting with empty experience and separate all-cash portfolios. Carry each arm's state across years.
7. Report the fixed-trust memory contrast, passive controls, monthly/yearly/forecast-regime metrics, behavior and computation, plus a supplementary interaction with the archived adaptive-trust pair.

2021 is initialization/warmup. Primary evaluation is 2022-01-01 through 2025-05-28; 2025 is partial. No annual profit gate or calendar state reset is introduced. No Qwen, model substitution/download, model weight fine-tuning or exchange trading occurs.

## Required existing evidence

```text
/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/artifacts/20260907T012534Z/continuous_2021_2025

/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/artifacts/20260908T055052711433762Z

/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMOE_RETURN_CLOCK_REPAIR_AND_ECONOMIC_RERUN_V1/artifacts/20260901T024730Z
```

Complete raw dataset SHA-256:

```text
b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e
```

The loader checks its recorded default locations. If the exact raw file moved, set `RAMAS_FULL_DATASET` to its absolute path. A wrong explicit file fails instead of falling back to a different dataset.

Matched runtime: **Python 3.11.9, NumPy 1.26.4, pandas 2.2.3, Ollama 0.23.0**. Keep the original environment. Model `llama3.3:70b`, Q4_K_M, digest:

```text
a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa
```

Source or serving mismatch stops before new trading calls. Restore the named historical dependency; do not bypass a checksum or replace the pinned digest to force execution.

## Resume after interruption

Use unchanged code, data, runtime and model. To resume, use:

```bash
cd /home/infonet/wahid/leader_router_fresh

RAMAS_RESUME_DIR="$(cat RAMAS_STAGE6_3_LAST_RUN.txt)" \
  bash RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/start.sh
```

Completed saved responses reconstruct each arm and are not resampled. An interrupted call without a saved response can require a new request. A previous result archive is preserved; the wrapper writes a separate timestamped snapshot on subsequent archival.

## Optional audit with zero model calls

```bash
cd /home/infonet/wahid/leader_router_fresh

RAMAS_AUDIT_ONLY=1 \
  bash RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/start.sh
```

This writes source/scientific audits and `AUDIT_ONLY_COMPLETE.json` without contacting Ollama. It is not a trading experiment and cannot be resumed as one. The normal run already performs these source checks; the separate audit is optional.

## Interpretation limits

The archived regime posterior's target differs from the intended next holding interval. Original fit times and feasible close-derived decision/fill timing remain incompletely established. The imported 2024 reference seam is preserved as common legacy evidence. The package records these facts and permits only the declared **legacy mechanism diagnostic**. It does not relabel forecasts or claim realistic execution.

A clock/reference repair requires a separate version and matched corrected comparisons. This fixed-beta result cannot validate adaptive trust or prove monotonically improving memory. Post-2021 observations remain reused historical diagnostics. Synthetic controlled-provider tests are labelled non-economic.
