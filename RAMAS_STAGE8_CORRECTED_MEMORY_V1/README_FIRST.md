# RAMAS Stage 8 — corrected episodic memory

Tests whether the inconclusive memory effect in Stages 6.2–6.4 is a property of
the memory *implementation* rather than of LLM episodic memory in general.

Read `EXPERIMENT_CONTRACT.md` before running anything.

## Safety

This package **imports** the frozen `RAMAS_STAGE6_4_COMPONENT_SUITE_V1` read-only
and verifies its manifest on every command. It never writes inside that package
or inside any existing artifact directory. All output goes to
`EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/`.

It is safe to run `preflight`, `validate` and `diagnose` while another
experiment is running: none of them makes a model call.

`run` needs the GPU. Check first:

```bash
./00_check_safe_to_start.sh
```

## Runtime — this matters

Every command refuses to start unless the interpreter is the pinned
Stage 6.x environment:

    python 3.11.9   numpy 1.26.4   pandas 2.2.3
    /home/infonet/anaconda3/envs/wahid_test/bin/python

This is not pedantry. A different numpy BLAS build changes the last bit of the
retrieval distance, which changes the request digest, which breaks every journal
match. The system default `python3` (anaconda base, 3.11.7 / pandas 2.1.4) does
**not** reproduce the archived arms. The launcher scripts hard-code the correct
interpreter. `--allow-runtime-mismatch` exists for exploratory work only and
will not reproduce anything.

## Order of operations

```bash
./00_check_safe_to_start.sh          # no other experiment on the GPU
./01_preflight.sh                    # source/model identity audit      (no GPU)
./02_validate_legacy_equivalence.sh  # MUST print PASS before step 04   (no GPU, ~5 min)
./03_diagnose_memory.sh              # failure-mode measurements        (no GPU, seconds)
./04_run_corrected_pair.sh           # the experiment                   (GPU, ~7 h)
```

Step 02 is the gate. It replays an archived Stage 6.4 arm through this engine
with the corrections switched off and compares every column. Current status:

    PASS_ENGINE_REPRODUCES_ARCHIVED_ARM
    1608 / 1608 archived request digests matched
    52 columns compared, 0 mismatched
    terminal wealth difference 0.0

If step 02 ever fails, the engine has drifted from the frozen design and no GPU
time should be spent until it is explained.

## Resuming

`04_run_corrected_pair.sh` writes to a fresh timestamped directory. To resume an
interrupted run, pass the same output directory and `--resume`:

```bash
/home/infonet/anaconda3/envs/wahid_test/bin/python RAMAS_CORRECTED_MEMORY.py run \
  --preflight <...>/PREFLIGHT/PREFLIGHT.json \
  --output <...>/artifacts/<existing_run_id> --resume
```

Durable journalled responses are never resampled. A resume replays the arm from
day one and reuses saved calls; if any rebuilt payload differs from the archived
one it stops rather than silently continuing.

## What comes out

Per arm: `DAILY_LEDGER.csv`, `episodes.json`, `trust_events.json`,
`ARM_STATE.json`, `RUN_COMPLETE.json`, and the durable `calls/` journal.
Run level: `ALL_DAILY_LEDGER.csv`, `COMPARISONS.json`, `01_SOURCE_AUDIT.json`,
`02_CORE_AUDIT.json`.

Every ledger keeps the frozen Stage 6.4 columns unchanged and adds
`counterfactual_log_advantage`, `active_credit_label`, `credit_label_mode`,
`retrieval_mode`, `retrieved_action_diversity`, `core_only_exposure` and
`label_action_exposure`, so a corrected run stays directly comparable to the
archived arms.

## Interpreting the result

The primary estimand is the paired mean daily net log return after trading
costs, `corrected_memory` minus `corrected_no_memory`, 2022-01-01 to 2025-05-28.

**A null result here is a valid and useful outcome.** It closes the objection
that the Stage 6.x null was an implementation artifact, and broadens the
failure-mode claim rather than weakening it. Do not rerun with different
settings in search of a positive effect.

This experiment does not resolve the router forecast clock, the fill clock, or
fresh out-of-sample confirmation. Those claim boundaries stay false.
